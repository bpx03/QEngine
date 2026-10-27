// Qwen3.8-Flash-Next worker: loading of one card's slice and the decode forward (see flash.h).
#include "flash.h"
#include <cerrno>
#include <atomic>
#include <cstring>
#include "flash_kernels.cuh"
#include "flash_pfq.cuh"
#include "../kernels/watchdog.cuh"
#include "../kernels/ops.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <chrono>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace flash {

char* pin_host_file(const std::string& path, size_t off, size_t n, const void* src, int threads, const std::vector<int>& numa) {
    constexpr size_t PG = (size_t)2 << 20, BLK = (size_t)8 << 20, AL = 4096;
    const size_t lead = off % AL, foff = off - lead;  // the buffer starts at file offset foff (O_DIRECT alignment)
    const size_t tot = lead + n, cap = (tot + PG - 1) / PG * PG;
    char* p = nullptr;
    CHECK(posix_memalign((void**)&p, PG, cap) == 0 && p, "pin_host_file: out of host memory (%.1f GB)", cap / 1e9);
    // 4 KB pages by default (as cudaHostAlloc gives): 2 MB pages (FLASH_PIN_THP=1) fault with direct compaction
    // under memory pressure (sys time 70-80 %, the fill dropped to ~1 GB/s next to a 31 GB process)
    static const bool thp = getenv("FLASH_PIN_THP") && atoi(getenv("FLASH_PIN_THP"));
    if (thp) madvise(p, cap, MADV_HUGEPAGE);
    numa_place(p, cap, numa);  // before the first touch
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    CHECK(fd >= 0, "pin_host_file: cannot open %s", path.c_str());
    const char* msrc = (const char*)src - lead;  // mapping address of file offset foff
    const long pgsz = sysconf(_SC_PAGESIZE);
    threads = (int)std::max<size_t>(1, std::min<size_t>(threads, cap / PG));
    std::vector<std::thread> th;
    std::vector<cudaError_t> err(threads, cudaSuccess);
    std::vector<size_t> hit(threads, 0);
    for (int t = 0; t < threads; t++)
        th.emplace_back([&, t] {
            const size_t c0 = cap / PG * t / threads * PG, c1 = cap / PG * (t + 1) / threads * PG;
            std::vector<unsigned char> vec(BLK / pgsz + 1);
            for (size_t b = c0; b < c1; b += BLK) {
                const size_t e = std::min(c1, b + BLK), ve = std::min(e, tot);
                if (ve <= b) { memset(p + b, 0, e - b); continue; }
                // the mapping address must be page aligned for mincore: msrc + b is (foff, b multiples of 4 KB)
                bool res = mincore((void*)(msrc + b), ve - b, vec.data()) == 0;
                for (size_t i = 0; res && i < (ve - b + pgsz - 1) / pgsz; i++) res = vec[i] & 1;
                if (res) {
                    memcpy(p + b, msrc + b, ve - b);
                    hit[t] += ve - b;
                } else {
                    const size_t rl = (ve - b + AL - 1) / AL * AL;  // whole 4 KB units (the buffer extends to cap)
                    for (size_t o = 0; o < rl;) {
                        ssize_t r = pread(fd, p + b + o, rl - o, foff + b + o);
                        {  // test: FLASH_TEST_PREAD_FAIL=n makes the n-th read fail (EIO) once
                            static const long fail_at = getenv("FLASH_TEST_PREAD_FAIL") ? atol(getenv("FLASH_TEST_PREAD_FAIL")) : 0;
                            static std::atomic<long> nread{0};
                            if (fail_at > 0 && ++nread == fail_at) { r = -1; errno = EIO; }
                        }
                        if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;
                        // a failed read must not leave the table silently wrong (it used to be zero-filled as if it
                        // were the end of the file)
                        CHECK(r >= 0, "pin_host_file: read of %s at %zu failed: %s", path.c_str(), (size_t)(foff + b + o),
                              strerror(errno));
                        if (r == 0) {  // end of the file: the rest of the last block is padding
                            CHECK(b + o >= tot, "pin_host_file: %s ended at %zu, expected %zu bytes", path.c_str(),
                                  (size_t)(foff + b + o), (size_t)(foff + tot));
                            memset(p + b + o, 0, rl - o);
                            break;
                        }
                        o += r;
                    }
                }
                if (e > ve) memset(p + ve, 0, e - ve);
            }
            err[t] = cudaHostRegister(p + c0, c1 - c0, cudaHostRegisterMapped | cudaHostRegisterPortable);
        });
    for (auto& x : th) x.join();
    for (auto e : err) CHECK(e == cudaSuccess, "pin_host_file: cudaHostRegister: %s", cudaGetErrorString(e));
    size_t h = 0;
    for (size_t x : hit) h += x;
    posix_fadvise(fd, foff, tot, POSIX_FADV_DONTNEED);
    ::close(fd);
    fprintf(stderr, "[flash] pinned %.1f GB of %s: %.1f GB from the page cache, the rest read directly\n", n / 1e9,
            path.c_str(), h / 1e9);
    return p + lead;
}

char* pin_host(const void* src, size_t n, int threads, bool chunked, const std::vector<int>& numa) {
    constexpr size_t PG = (size_t)2 << 20;
    const size_t cap = (n + PG - 1) / PG * PG;
    char* p = nullptr;
    CHECK(posix_memalign((void**)&p, PG, cap) == 0 && p, "pin_host: out of host memory (%.1f GB)", cap / 1e9);
    madvise(p, cap, MADV_HUGEPAGE);
    numa_place(p, cap, numa);  // before the first touch
    threads = (int)std::max<size_t>(1, std::min<size_t>(threads, cap / PG));
    std::vector<std::thread> th;
    std::vector<cudaError_t> err(threads, cudaSuccess);
    for (int t = 0; t < threads; t++)
        th.emplace_back([&, t] {
            const size_t c0 = cap / PG * t / threads * PG, c1 = cap / PG * (t + 1) / threads * PG;
            const size_t z = src ? std::max(c0, std::min(c1, n)) : c0;  // [c0, z) copied, [z, c1) zeroed
            if (z > c0) memcpy(p + c0, (const char*)src + c0, z - c0);
            if (c1 > z) memset(p + z, 0, c1 - z);
            if (chunked) err[t] = cudaHostRegister(p + c0, c1 - c0, cudaHostRegisterMapped | cudaHostRegisterPortable);
        });
    for (auto& x : th) x.join();
    if (!chunked) err[0] = cudaHostRegister(p, cap, cudaHostRegisterMapped | cudaHostRegisterPortable);
    for (auto e : err) CHECK(e == cudaSuccess, "pin_host: cudaHostRegister: %s", cudaGetErrorString(e));
    return p;
}
// launch KERNEL<NT> with NT = 8-token tiles of m rows
#define NT_LAUNCH(m, KERNEL, GRID, BLOCK, ...)                                                    \
    do {                                                                                          \
        const int nt_ = ((m) + 7) / 8;                                                            \
        if (nt_ == 1) KERNEL<1><<<GRID, BLOCK, 0, st_>>>(__VA_ARGS__);                            \
        else if (nt_ == 2) KERNEL<2><<<GRID, BLOCK, 0, st_>>>(__VA_ARGS__);                       \
        else if (nt_ == 3) KERNEL<3><<<GRID, BLOCK, 0, st_>>>(__VA_ARGS__);                       \
        else KERNEL<4><<<GRID, BLOCK, 0, st_>>>(__VA_ARGS__);                                     \
    } while (0)
// W16 GEMV over any row count (<= 64 rows per call; ksplit: partial slices at y + s * ystride)
static const bool g_rows_gemm = !getenv("FLASH_ROWS_GEMM") || atoi(getenv("FLASH_ROWS_GEMM"));  // smem-staged GEMM (N % 64 == 0)
static const int g_rows_min = getenv("FLASH_ROWS_GEMV") ? atoi(getenv("FLASH_ROWS_GEMV")) : MAXM + 1;  // 0: off
static void gemv_w16_rows(const W16& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st, int ksplit = 1,
                          size_t ystride = 0) {
    if (g_rows_min > 0 && m >= g_rows_min && ksplit == 1) {
        if (g_rows_gemm && w.n % 64 == 0 && w.k % 128 == 0 && ldx % 8 == 0) {
            k_rows_gemm<false><<<dim3(w.n / 64, (m + 31) / 32), 128, 0, st>>>(w.w, nullptr, w.k, w.n, x, ldx, y, ldy, m);
            return;
        }
        if (w.n / 16 < 8)  // few tiles (DeltaNet a/b: 2): 8-row groups, 8 warps over K
            k_rows_gemv<false, 1, 8><<<dim3(w.n / 16, (m + 7) / 8), 256, 0, st>>>(w.w, nullptr, w.k, x, ldx, y, ldy, m);
        else
            k_rows_gemv<false><<<dim3(w.n / 16, (m + 31) / 32), 128, 0, st>>>(w.w, nullptr, w.k, x, ldx, y, ldy, m);
        return;
    }
    for (int r0 = 0; r0 < m; r0 += kMaxBatchRows)
        gemv_w16(w, x + (size_t)r0 * ldx, ldx, y + (size_t)r0 * ldy, ldy, std::min(kMaxBatchRows, m - r0), st, ksplit, ystride);
}
// W8 GEMV over any row count (<= 64 rows per call)
static void gemv_w8_rows(const W8& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st) {
    if (g_rows_min > 0 && m >= g_rows_min) {
        if (g_rows_gemm && w.n % 16 == 0 && w.n >= 256 && w.k % 128 == 0 && ldx % 8 == 0) {
            if (w.n >= 4096)  // wide (QSA qkv): 64-row blocks halve the weight re-reads
                k_rows_gemm<true, 64><<<dim3((w.n + 63) / 64, (m + 63) / 64), 128, 0, st>>>(w.q, w.s, w.k, w.n, x, ldx, y, ldy, m);
            else
                k_rows_gemm<true><<<dim3((w.n + 63) / 64, (m + 31) / 32), 128, 0, st>>>(w.q, w.s, w.k, w.n, x, ldx, y, ldy, m);
            return;
        }
        k_rows_gemv<true><<<dim3(w.n / 16, (m + 31) / 32), 128, 0, st>>>(w.q, w.s, w.k, x, ldx, y, ldy, m);
        return;
    }
    for (int r0 = 0; r0 < m; r0 += kMaxBatchRows)
        gemv_w8(w, x + (size_t)r0 * ldx, ldx, y + (size_t)r0 * ldy, ldy, std::min(kMaxBatchRows, m - r0), st);
}

// W8A8 row GEMM (prefill-r3): activations quantized per (row, 128-k group) into the MoE activation buffer (idle
// outside the MoE gate_up -> down window, every user on st_), then k_rows_q
void Worker::rows8(const W8& w, const bf16* x, int ldx, float* y, int ldy, int m, cudaStream_t st) {
    static const bool on = !getenv("FLASH_ROWS_Q") || atoi(getenv("FLASH_ROWS_Q"));
    const size_t need = (size_t)m * w.k + (size_t)m * (w.k / 128) * 8;
    if (!on || !pf_ || m < 64 || st != st_ || w.k % 128 || w.k > RQ_KMAX || ldx % 8 || need > pfact_bytes_ || (qsa_split_ && need > ((size_t)4 << 20))) {
        gemv_w8_rows(w, x, ldx, y, ldy, m, st);
        return;
    }
    if (!rq_attr_) CUDA_CHECK(cudaFuncSetAttribute(k_rows_q, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)RQ_SMEM));
    rq_attr_ = true;
    int8_t* xq = reinterpret_cast<int8_t*>(pfact_);
    float2* xs = reinterpret_cast<float2*>(xq + (size_t)m * w.k);
    k_x_to_r8<<<m, w.k / 16, 0, st>>>(x, ldx, w.k, xq, xs);
    k_rows_q<<<dim3((m + RQ_TB - 1) / RQ_TB, (w.n + RQ_NB - 1) / RQ_NB), 256, RQ_SMEM, st>>>(w.q, w.s, w.k, w.n, xq, xs, y, ldy, m);
    (void)0;
}
static const bool g_kchk = getenv("FLASH_KCHK") && atoi(getenv("FLASH_KCHK"));
#define KCHK(name) do { if (g_kchk) { cudaError_t e_ = cudaDeviceSynchronize(); if (e_ != cudaSuccess) { fprintf(stderr, "kernel %s: %s\n", name, cudaGetErrorString(e_)); exit(1); } } } while (0)

std::vector<Plan> make_plan(int ngpu, const std::vector<double>& wts) {
    if ((ngpu != 2 && ngpu != 3) || (int)wts.size() != ngpu)
        throw std::runtime_error("flash: 2 or 3 GPUs and as many expert-group weights expected");
    static_assert(NQ % 6 == 0 && LNV % 6 == 0 && LNV % LNK == 0, "head splits over 2 and 3 cards");
    // expert groups (e * 5 + g, e = 0..512 with 512 the shared expert) spread by a largest-deficit interleave in
    // proportion to the weights: consecutive groups of one expert land on different cards (per-token balance)
    const int ng = (NE + 1) * NGRP;
    double tot = 0;
    for (double w : wts) tot += w;
    std::vector<int> card(ng), cnt(ngpu, 0);
    for (int k = 0; k < ng; k++) {
        int best = 0;
        double bd = -1e30;
        for (int c = 0; c < ngpu; c++) {
            const double d = wts[c] / tot * (k + 1) - cnt[c];
            if (d > bd + 1e-12) { bd = d; best = c; }
        }
        card[k] = best;
        cnt[best]++;
    }
    std::vector<Plan> v(ngpu);
    for (int r = 0; r < ngpu; r++) {
        Plan& p = v[r];
        p.rank = r;
        p.ngpu = ngpu;
        p.gcard = card;
        // DeltaNet: LNV / ngpu value heads, the key heads they use (3 value heads per key head; 3 cards: shared
        // key heads at the boundaries); QSA: NQ / ngpu q heads (gate / o_proj / combine)
        p.v0 = r * (LNV / ngpu);
        p.v1 = (r + 1) * (LNV / ngpu);
        p.k0 = p.v0 / (LNV / LNK);
        p.k1 = (p.v1 - 1) / (LNV / LNK) + 1;
        p.q0 = r * (NQ / ngpu);
        p.q1 = (r + 1) * (NQ / ngpu);
        const int g = VOCAB / 128, a = g * r / ngpu, b = g * (r + 1) / ngpu;
        p.voc0 = a * 128;
        p.voc1 = b * 128;
        p.period = 512 * ngpu;
        p.own0 = 512 * r;
        p.own = 512;
    }
    return v;
}

std::vector<double> default_group_weights(int ngpu) {
    if (ngpu == 3) return {176, 160, 176};
    return std::vector<double>(std::max(ngpu, 1), 1.0);
}

int plan_groups(const Plan& p) {
    int n = 0;
    for (int c : p.gcard) n += c == p.rank;
    return n;
}

namespace {
const std::string P = "model.language_model.";

struct Scratch {  // device scratch for loading (freed after load)
    void* p = nullptr;
    size_t n = 0;
    void* get(size_t b) {
        if (b > n) {
            if (p) CUDA_CHECK(cudaFree(p));
            CUDA_CHECK(cudaMalloc(&p, b));
            n = b;
        }
        return p;
    }
    void release() {
        if (p) CUDA_CHECK(cudaFree(p));
        p = nullptr;
        n = 0;
    }
};
thread_local Scratch g_s1, g_s2, g_s3;
struct HostScratch {  // pinned host staging for loading (freed after load)
    char* p = nullptr;
    size_t n = 0;
    char* get(size_t b) {
        if (b > n) {
            if (p) CUDA_CHECK(cudaFreeHost(p));
            CUDA_CHECK(cudaHostAlloc((void**)&p, b, cudaHostAllocPortable));
            n = b;
        }
        return p;
    }
    void release() {
        if (p) CUDA_CHECK(cudaFreeHost(p));
        p = nullptr;
        n = 0;
    }
};
thread_local HostScratch g_hs;
thread_local double g_tg = 0, g_tw = 0, g_tl = 0;  // load profile: gather, wait for the buffer, launches (s)
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

const TensorInfo& T(const SafeTensors& st, const std::string& n) { return st.get(n); }

struct RowSel {
    std::string name;  // base name (without .weight_packed)
    int r0, r1;
};
}  // namespace

void* Worker::dalloc(size_t bytes) {
    void* p;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    // on the work stream + wait: a legacy-stream cudaMemset is asynchronous and does not order with the
    // non-blocking stream, so it could land after the repack kernels that fill the buffer
    CUDA_CHECK(cudaMemsetAsync(p, 0, bytes, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    wbytes_ += bytes;
    allocs_.push_back({p, bytes});
    return p;
}

__global__ void k_csum(const uint32_t* p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        s += (unsigned long long)p[i] * (1 + (i & 255));
    atomicAdd(out, s);
}
unsigned long long Worker::checksum(int first, int last) {
    CUDA_CHECK(cudaSetDevice(dev_));
    unsigned long long* d;
    CUDA_CHECK(cudaMalloc(&d, 8));
    CUDA_CHECK(cudaMemset(d, 0, 8));
    for (int i = first; i < last && i < (int)allocs_.size(); i++)
        k_csum<<<256, 256, 0, st_>>>((const uint32_t*)allocs_[i].first, allocs_[i].second / 4, d);
    unsigned long long h;
    CUDA_CHECK(cudaMemcpyAsync(&h, d, 8, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaFree(d));
    return h;
}

Worker::Worker(const Plan& p, int device, const Comm& comm, int max_ctx) : p_(p), dev_(device), max_ctx_(max_ctx), comm_(comm) {
    // the GEMM autotuner picks per process by timing (choices and so the rounding differ between processes); the cost
    // model's choice is as fast for the flash shapes (text 928 prefill 169.6 vs 169.7 ms) and the same in every
    // process: pinned unless QE_GEMM_TUNE is set explicitly
    setenv("QE_GEMM_TUNE", "0", 0);
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    qsa_a_ = getenv("FLASH_QSA_A") && atoi(getenv("FLASH_QSA_A"));
    qe_wd_init(p_.rank, dev_);  // collective watchdog (budget QE_COLL_TIMEOUT_MS; uninitialized it trips on long waits)
    gemv_workspace_init();
    CUDA_CHECK(cudaFuncSetAttribute(k_qsa_attn_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TA_SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(k_qsa_attn_tc2<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TB_SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(k_qsa_attn_tc2<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TB_SMEM));
    ::dn_chunk_init();
    ctr_.seq = (unsigned*)dalloc(4);
    ctr_.done = (unsigned*)dalloc(8);
    ctr_.go = (unsigned*)dalloc(4);
    ctr_.bar = (unsigned*)dalloc(8);
    ctr_.ss = (float*)dalloc(kCommBlocks * kMaxBatchRows * 4);
    ctr_.epoch = (unsigned*)dalloc(4);
    ctr_.ready = (unsigned*)dalloc(4096 * 4);
    CUDA_CHECK(cudaFuncSetAttribute(k_dn_rec, cudaFuncAttributeMaxDynamicSharedMemorySize, (128 * 128 + 256) * 4));
}

size_t Worker::free_bytes() const {
    CUDA_CHECK(cudaSetDevice(dev_));
    size_t f, t;
    CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return f;
}

// ---------------------------------------------------------------- loaders (host -> device)
namespace {
// rows of compressed-tensors INT4 matrices (weight_packed int32 [N][K/8], weight_scale 16-bit [N][K/128]), K columns
// [c0, c1) (multiple of 128), stacked; GLU: rows are [gate i][up i] halves to interleave per 16-row tile
W4 load_w4(const SafeTensors& st, const std::vector<RowSel>& rows, int c0, int c1, bool glu, Worker* w,
           void* (Worker::*alloc)(size_t), cudaStream_t s) {
    const TensorInfo& t0 = T(st, rows[0].name + ".weight_packed");
    const int K = (int)t0.shape[1] * 8;
    if (c1 < 0) { c0 = 0; c1 = K; }
    const int kl = c1 - c0;
    int n = 0;
    for (auto& r : rows) n += r.r1 - r.r0;
    CHECK(n % 16 == 0 && kl % 128 == 0 && c0 % 128 == 0, "w4 %s n=%d k=%d", rows[0].name.c_str(), n, kl);
    std::vector<int32_t> hq((size_t)n * kl / 8);
    std::vector<uint16_t> hs((size_t)n * kl / 128);
    int off = 0;
    for (auto& r : rows) {
        const TensorInfo& q = T(st, r.name + ".weight_packed");
        const TensorInfo& sc = T(st, r.name + ".weight_scale");
        CHECK((int)q.shape[1] * 8 == K, "K mismatch %s", r.name.c_str());
        for (int i = r.r0; i < r.r1; i++, off++) {
            memcpy(&hq[(size_t)off * kl / 8], q.data + ((size_t)i * (K / 8) + c0 / 8) * 4, kl / 2);
            memcpy(&hs[(size_t)off * kl / 128], sc.data + ((size_t)i * (K / 128) + c0 / 128) * 2, kl / 128 * 2);
        }
    }
    int32_t* dq0 = (int32_t*)g_s1.get(hq.size() * 4);
    bf16* ds0 = (bf16*)g_s2.get(hs.size() * 2);
    // H2D on the stream of the repack below (qfn_prefill_r4): a plain cudaMemcpy from pageable memory runs on the legacy
    // stream and may return before its DMA lands, and st_ is non-blocking (not ordered after the legacy stream): the
    // repack could read the reused scratch while it still held the previous tensor (weights loaded wrong at random)
    static const bool lrace = getenv("FLASH_LOAD_RACE") && atoi(getenv("FLASH_LOAD_RACE"));  // debug: the old racy copy
    if (lrace) {
        CUDA_CHECK(cudaMemcpy(dq0, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds0, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
    } else {
    CUDA_CHECK(cudaMemcpyAsync(dq0, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ds0, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice, s));
    }
    W4 out;
    out.n = n;
    out.k = kl;
    out.glu = glu;
    uint4* dq = (uint4*)(w->*alloc)(w4_qbytes(n, kl));
    bf16* ds = (bf16*)(w->*alloc)(w4_sbytes(n, kl));
    if (glu) {
        int32_t* gq = (int32_t*)g_s3.get(hq.size() * 4 + hs.size() * 2);
        bf16* gs = (bf16*)((char*)gq + hq.size() * 4);
        glu_interleave_rows(dq0, ds0, n, kl, gq, gs, s);
        repack_w4(gq, gs, n, kl, dq, ds, s);
    } else {
        repack_w4(dq0, ds0, n, kl, dq, ds, s);
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    out.q = dq;
    out.s = ds;
    return out;
}
// bf16 rows -> W8 (INT8 g128 RTN on the GPU); rows/cols beyond the sources are zero (padding)
struct Bf16Block {
    const bf16* src;   // row-major, ld elements
    int ld, rows, cols;
    int dr, dc;        // destination row / col offset
};
W8 load_w8(int n, int k, const std::vector<Bf16Block>& blocks, Worker* w, void* (Worker::*alloc)(size_t), cudaStream_t s) {
    std::vector<bf16> h((size_t)n * k, __float2bfloat16(0.f));
    for (auto& b : blocks)
        for (int i = 0; i < b.rows; i++) memcpy(&h[(size_t)(b.dr + i) * k + b.dc], b.src + (size_t)i * b.ld, b.cols * 2);
    bf16* d = (bf16*)g_s1.get(h.size() * 2);
    // H2D ordered with the conversion kernel on s (see load_w4: a plain cudaMemcpy raced the kernel on the non-blocking st_)
    static const bool lrace = getenv("FLASH_LOAD_RACE") && atoi(getenv("FLASH_LOAD_RACE"));  // debug: the old racy copy
    if (lrace) CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    else CUDA_CHECK(cudaMemcpyAsync(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice, s));
    W8 out;
    out.n = n;
    out.k = k;
    out.q = (uint4*)(w->*alloc)(w8_qbytes(n, k));
    out.s = (bf16*)(w->*alloc)((size_t)n * (k / 128) * 2);
    quantize_w8(d, n, k, const_cast<uint4*>(out.q), const_cast<bf16*>(out.s), s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    return out;
}
W16 load_w16(int n, int k, const std::vector<Bf16Block>& blocks, Worker* w, void* (Worker::*alloc)(size_t), cudaStream_t s) {
    std::vector<bf16> h((size_t)n * k, __float2bfloat16(0.f));
    for (auto& b : blocks)
        for (int i = 0; i < b.rows; i++) memcpy(&h[(size_t)(b.dr + i) * k + b.dc], b.src + (size_t)i * b.ld, b.cols * 2);
    bf16* d = (bf16*)g_s1.get(h.size() * 2);
    // H2D ordered with the conversion kernel on s (see load_w4: a plain cudaMemcpy raced the kernel on the non-blocking st_)
    static const bool lrace = getenv("FLASH_LOAD_RACE") && atoi(getenv("FLASH_LOAD_RACE"));  // debug: the old racy copy
    if (lrace) CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    else CUDA_CHECK(cudaMemcpyAsync(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice, s));
    W16 out;
    out.n = n;
    out.k = k;
    out.w = (uint4*)(w->*alloc)(w16_bytes(n, k));
    repack_w16(d, n, k, const_cast<uint4*>(out.w), s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    return out;
}
const bf16* B(const SafeTensors& st, const std::string& n) { return (const bf16*)T(st, n).data; }
}  // namespace

void Worker::load(const SafeTensors& st, const HostTables& ht, const SafeTensors* mst) {
    // dynamic expert budget (default with the expert cache): device-resident groups = what fits after everything else
    ec_defer_ = !(getenv("FLASH_EC") && atoi(getenv("FLASH_EC")) == 0) && !(getenv("FLASH_EC_BUDGET") && atoi(getenv("FLASH_EC_BUDGET")) == 0);
    CUDA_CHECK(cudaSetDevice(dev_));
    auto A = &Worker::dalloc;
    embed_ = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&embed_, (void*)ht.embed, 0));
    auto bf_dev = [&](const bf16* src, size_t n) {
        bf16* d = (bf16*)dalloc(n * 2);
        CUDA_CHECK(cudaMemcpy(d, src, n * 2, cudaMemcpyHostToDevice));
        return d;
    };
    auto f32_dev = [&](std::vector<float> v) {
        float* d = (float*)dalloc(v.size() * 4);
        CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return d;
    };
    auto hc_from = [&](const SafeTensors& st, const std::string& pre, bool inject) {
        HcW w;
        w.inject = inject;
        std::vector<Bf16Block> db = {{B(st, pre + "input_mix_weight_down.weight"), RW, LOWRANK, RW, 0, 0}};
        if (inject) db.push_back({B(st, pre + "block_inject_weight.weight"), RW, HC, RW, LOWRANK, 0});
        static const bool fused = !getenv("FLASH_HCF") || atoi(getenv("FLASH_HCF"));
        w.norm = bf_dev(B(st, pre + "hc_norm.weight"), RW);
        if (fused) {
            const bf16* nw = B(st, pre + "hc_norm.weight");
            std::vector<bf16> dh((size_t)HCD_ROWS * RW, __float2bfloat16(0.f));
            const bf16* dsrc = B(st, pre + "input_mix_weight_down.weight");
            const bf16* isrc = inject ? B(st, pre + "block_inject_weight.weight") : nullptr;
            for (int j = 0; j < LOWRANK + (inject ? HC : 0); j++) {
                const bf16* row = j < LOWRANK ? dsrc + (size_t)j * RW : isrc + (size_t)(j - LOWRANK) * RW;
                for (int k = 0; k < RW; k++)
                    dh[(size_t)j * RW + k] = __float2bfloat16(__bfloat162float(row[k]) * (1.f + __bfloat162float(nw[k])));
            }
            w.fd = load_w8(HCD_ROWS, RW, {{dh.data(), RW, HCD_ROWS, RW, 0, 0}}, this, A, st_);
            // up: rows permuted (tile T, row dd * 4 + i <-> U row i * 2560 + 4T + dd), INT8 g64, fragment order
            const bf16* us = B(st, pre + "input_mix_weight_up.weight");
            constexpr int KS = LOWRANK / 16, NG = LOWRANK / 64, TILES = RW / 16;
            std::vector<uint8_t> qb((size_t)TILES * (KS / 2) * 32 * 16);
            std::vector<__half> sb((size_t)TILES * NG * 16);
            for (int T = 0; T < TILES; T++)
                for (int rr = 0; rr < 16; rr++) {
                    const int dd = rr / 4, i = rr % 4;
                    const bf16* row = us + ((size_t)i * H + 4 * T + dd) * LOWRANK;
                    for (int gi = 0; gi < NG; gi++) {
                        float mx = 0.f;
                        for (int k = 0; k < 64; k++) mx = std::max(mx, fabsf(__bfloat162float(row[gi * 64 + k])));
                        const __half sh = __float2half(mx / 127.f);
                        const float sc = __half2float(sh), inv = sc > 0.f ? 1.f / sc : 0.f;
                        sb[((size_t)T * NG + gi) * 16 + rr] = sh;
                        for (int k = gi * 64; k < gi * 64 + 64; k++) {
                            const int qi = std::max(-127, std::min(127, (int)lrintf(__bfloat162float(row[k]) * inv)));
                            const int ks = k >> 4, col = k & 15;
                            const int tq = (col & 7) >> 1, half = col >> 3, odd = col & 1;
                            const int ln = (rr & 7) * 4 + tq;
                            const int ai = (half * 2 + (rr >> 3)) * 2 + odd;
                            qb[(((size_t)T * (KS / 2) + (ks >> 1)) * 32 + ln) * 16 + (ks & 1) * 8 + ai] = (uint8_t)(int8_t)qi;
                        }
                    }
                }
            w.uq = (uint4*)dalloc(qb.size());
            w.us = (__half*)dalloc(sb.size() * 2);
            CUDA_CHECK(cudaMemcpy(w.uq, qb.data(), qb.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(w.us, sb.data(), sb.size() * 2, cudaMemcpyHostToDevice));
            return w;
        }
        static const int hc16 = getenv("FLASH_HC16") ? atoi(getenv("FLASH_HC16")) : 0;  // bit 0: down bf16, bit 1: up bf16
        if (hc16 & 1) w.down16 = load_w16(HCD_ROWS, RW, db, this, A, st_);
        else w.down = load_w8(HCD_ROWS, RW, db, this, A, st_);
        if (hc16 & 2) w.up16 = load_w16(RW, HCU_K, {{B(st, pre + "input_mix_weight_up.weight"), LOWRANK, RW, LOWRANK, 0, 0}}, this, A, st_);
        else w.up = load_w8(RW, HCU_K, {{B(st, pre + "input_mix_weight_up.weight"), LOWRANK, RW, LOWRANK, 0, 0}}, this, A, st_);
        return w;
    };
    const int vl = p_.v1 - p_.v0, kl = p_.k1 - p_.k0, ql = p_.q1 - p_.q0;
    // one decoder layer (trunk layer li, or the MTP block with li = -1) from tensors under `pre`
    auto load_layer = [&](const SafeTensors& st, Layer& L, const std::string& pre, int li) {
        L.full = st.has(pre + "self_attn.q_proj.weight");
        L.hca = hc_from(st, pre + "attn_hyper_connection.", true);
        L.hcm = hc_from(st, pre + "mlp_hyper_connection.", true);
        if (!L.full) {
            const std::string d = pre + "linear_attn.";
            L.dn_in = load_w4(st, {{d + "in_proj_qkv", p_.k0 * DK, p_.k1 * DK},
                                   {d + "in_proj_qkv", LNK * DK + p_.k0 * DK, LNK * DK + p_.k1 * DK},
                                   {d + "in_proj_qkv", 2 * LNK * DK + p_.v0 * DV, 2 * LNK * DK + p_.v1 * DV},
                                   {d + "in_proj_z", p_.v0 * DV, p_.v1 * DV}},
                              0, -1, false, this, A, st_);
            L.dn_out = load_w4(st, {{d + "out_proj", 0, H}}, p_.v0 * DV, p_.v1 * DV, false, this, A, st_);
            L.dn_ab = load_w16((2 * vl + 15) / 16 * 16, H, {{B(st, d + "in_proj_b.weight") + (size_t)p_.v0 * H, H, vl, H, 0, 0},
                                       {B(st, d + "in_proj_a.weight") + (size_t)p_.v0 * H, H, vl, H, vl, 0}},
                               this, A, st_);
            const int ch = 2 * kl * DK + vl * DV;
            std::vector<bf16> cw((size_t)ch * 4);
            const bf16* c = B(st, d + "conv1d.weight");
            int o = 0;
            for (int i = p_.k0 * DK; i < p_.k1 * DK; i++, o++) memcpy(&cw[o * 4], c + (size_t)i * 4, 8);
            for (int i = LNK * DK + p_.k0 * DK; i < LNK * DK + p_.k1 * DK; i++, o++) memcpy(&cw[o * 4], c + (size_t)i * 4, 8);
            for (int i = 2 * LNK * DK + p_.v0 * DV; i < 2 * LNK * DK + p_.v1 * DV; i++, o++) memcpy(&cw[o * 4], c + (size_t)i * 4, 8);
            L.conv_w = bf_dev(cw.data(), cw.size());
            std::vector<float> Av(vl), dt(vl);
            const bf16* al = B(st, d + "A_log");
            const bf16* db = B(st, d + "dt_bias");
            for (int i = 0; i < vl; i++) {
                Av[i] = -expf(__bfloat162float(al[p_.v0 + i]));
                dt[i] = __bfloat162float(db[p_.v0 + i]);
            }
            L.A = f32_dev(Av);
            L.dtb = f32_dev(dt);
            L.dn_norm = bf_dev(B(st, d + "norm.weight"), DV);
        } else {
            const std::string a = pre + "self_attn.";
            const bf16* q = B(st, a + "q_proj.weight");
            // rows [q of nqp heads][gate of the local heads][k 2][v 2]: variant B (default) q only of the local heads
            // (exchanged between the cards), FLASH_QSA_A=1 q of all 24 heads on every card
            const int nqp = qsa_a_ ? NQ : ql, qh0 = qsa_a_ ? 0 : p_.q0;
            std::vector<Bf16Block> qb;
            for (int h = 0; h < nqp; h++) qb.push_back({q + (size_t)(qh0 + h) * 2 * HD * H, H, HD, H, h * HD, 0});
            for (int h = p_.q0; h < p_.q1; h++) qb.push_back({q + ((size_t)h * 2 * HD + HD) * H, H, HD, H, nqp * HD + (h - p_.q0) * HD, 0});
            qb.push_back({B(st, a + "k_proj.weight"), H, NKV * HD, H, nqp * HD + ql * HD, 0});
            qb.push_back({B(st, a + "v_proj.weight"), H, NKV * HD, H, nqp * HD + ql * HD + NKV * HD, 0});
            L.qkv = load_w8(nqp * HD + ql * HD + 2 * NKV * HD, H, qb, this, A, st_);
            L.o = load_w8(H, ql * HD, {{B(st, a + "o_proj.weight") + p_.q0 * HD, NQ * HD, H, ql * HD, 0, 0}}, this, A, st_);
            L.idx = load_w16((IDX_NH + 1) * IDX_HD, H, {{B(st, a + "indexer.index_qk_proj.weight"), H, (IDX_NH + 1) * IDX_HD, H, 0, 0}}, this, A, st_);
            L.q_norm = bf_dev(B(st, a + "q_norm.weight"), HD);
            L.k_norm = bf_dev(B(st, a + "k_norm.weight"), HD);
            L.iq_norm = bf_dev(B(st, a + "indexer.q_layernorm.weight"), IDX_HD);
            L.ik_norm = bf_dev(B(st, a + "indexer.k_layernorm.weight"), IDX_HD);
        }
        // MoE
        static const bool r8 = !getenv("FLASH_ROUTER8") || atoi(getenv("FLASH_ROUTER8"));
        if (r8)
            L.router8 = load_w8(528, H, {{B(st, pre + "mlp.gate.weight"), H, NE, H, 0, 0},
                                         {B(st, pre + "mlp.shared_expert_gate.weight"), H, 1, H, NE, 0}},
                                this, A, st_);
        else
            L.router = load_w16(528, H, {{B(st, pre + "mlp.gate.weight"), H, NE, H, 0, 0},
                                         {B(st, pre + "mlp.shared_expert_gate.weight"), H, 1, H, NE, 0}},
                                this, A, st_);
        if (li >= 0 && ec_defer_) {
            // trunk expert groups are loaded last (Worker::load_deferred_groups, from alloc_state): the groups that fit
            // the memory left after every other allocation stay on the device, the rest is host-backed
            ec_defer_l_.push_back({&st, pre, li});
        } else {
            std::vector<char> isc((NE + 1) * NGRP, 0);
            if (li >= 0 && li < (int)cold_.size())
                for (int e : cold_[li]) for (int g = 0; g < NGRP; g++) isc[e * NGRP + g] = 1;
            load_groups(st, L, pre, isc);
        }
        if (li == PLE_LAYER) {
            const std::string pl = pre + "ple.";
            L.ple_key = load_w4(st, {{pl + "key_proj", 0, RW}}, 0, -1, false, this, A, st_);
            L.ple_val = load_w4(st, {{pl + "value_proj", 0, H}}, 0, -1, false, this, A, st_);
            L.ple_nk = bf_dev(B(st, pl + "norm_key.weight"), RW);
            L.ple_nq = bf_dev(B(st, pl + "norm_query.weight"), RW);
            L.ple_nc = bf_dev(B(st, pl + "norm_conv.weight"), RW);
            L.ple_conv = bf_dev(B(st, pl + "conv1d.weight"), (size_t)RW * 4);
        }
    };
    L_.resize(LAYERS);
    for (int li = 0; li < LAYERS; li++) load_layer(st, L_[li], P + "layers." + std::to_string(li) + ".", li);
    mixer_ = hc_from(st, P + "hyper_connection_mixer.", false);
    if (mst) {  // MTP block: one QSA layer + MoE + hyper-connections, its own head mixer; embed / lm_head shared
        const SafeTensors& ms = *mst;
        load_layer(ms, mtp_, "mtp.layers.0.", -1);
        mtp_mixer_ = hc_from(ms, "mtp.hyper_connection_mixer.", false);
        mtp_fce_ = load_w8(H, H, {{B(ms, "mtp.fc_embedding.weight"), H, H, H, 0, 0}}, this, A, st_);
        mtp_fch_ = load_w8(H, H, {{B(ms, "mtp.fc_hidden.weight"), H, H, H, 0, 0}}, this, A, st_);
        mtp_pre_e_ = bf_dev(B(ms, "mtp.pre_fc_norm_embedding.weight"), H);
        mtp_pre_h_ = bf_dev(B(ms, "mtp.pre_fc_norm_hidden.weight"), RW);
        has_mtp_ = true;
    }
    {
        const bf16* lm = B(st, "lm_head.weight");
        const int nv = p_.voc1 - p_.voc0;
        // rows permuted: the draft vocabulary of this slice first (its first ndraft_ rows are the draft head), map_ =
        // local row -> token id
        std::vector<int> rows;
        std::vector<char> in(nv, 0);
        for (int t : draft_ids_)
            if (t >= p_.voc0 && t < p_.voc1 && !in[t - p_.voc0]) { in[t - p_.voc0] = 1; rows.push_back(t); }
        ndraft_ = ((int)rows.size() + 15) / 16 * 16;
        for (int t = p_.voc0; t < p_.voc1; t++) if (!in[t - p_.voc0]) rows.push_back(t);
        std::vector<Bf16Block> bl;
        for (int i = 0; i < nv; i++) bl.push_back({lm + (size_t)rows[i] * H, H, 1, H, i, 0});
        lm_ = load_w8(nv, H, bl, this, A, st_);
        lmap_ = (int*)dalloc((size_t)nv * 4);
        CUDA_CHECK(cudaMemcpy(lmap_, rows.data(), (size_t)nv * 4, cudaMemcpyHostToDevice));
        if (ndraft_ == 0) ndraft_ = nv;
        // balanced draft head (FLASH_DRAFT_BAL, default on): the draft vocabulary's frequent tokens sit mostly in the
        // first vocabulary slice (GPU0 had 52.9K of 65.5K rows, the others ~6K): every card instead holds every
        // ngpu-th draft token by rank, its own copy of those rows (all cards then read the same bytes)
        static const bool bal = !getenv("FLASH_DRAFT_BAL") || atoi(getenv("FLASH_DRAFT_BAL"));
        if (bal && !draft_ids_.empty()) {
            std::vector<int> mine;
            for (size_t i = p_.rank; i < draft_ids_.size(); i += p_.ngpu) mine.push_back(draft_ids_[i]);
            const int nd = ((int)mine.size() + 15) / 16 * 16;
            while ((int)mine.size() < nd) mine.push_back(mine.back());  // padding rows repeat a token (ties: same id)
            std::vector<Bf16Block> db;
            for (int i = 0; i < nd; i++) db.push_back({lm + (size_t)mine[i] * H, H, 1, H, i, 0});
            dlm_ = load_w8(nd, H, db, this, A, st_);
            dmap_ = (int*)dalloc((size_t)nd * 4);
            CUDA_CHECK(cudaMemcpy(dmap_, mine.data(), (size_t)nd * 4, cudaMemcpyHostToDevice));
        }
    }
    // every load-time copy and conversion done before the scratch goes and inference starts (plain cudaMemcpy from
    // pageable memory may return before its DMA lands; the work streams are non-blocking)
    if (!(getenv("FLASH_LOAD_RACE") && atoi(getenv("FLASH_LOAD_RACE")))) CUDA_CHECK(cudaDeviceSynchronize());
    g_s1.release();
    g_s2.release();
    g_s3.release();
    g_hs.release();
}

// one layer's local expert groups; isc[e * 5 + g]: host-backed (cold) group
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
// the mapped expert tensors of a layer: read ahead (async) or faulted in (sync, sequential reads at the disk rate;
// the per-group row gather alone reads the file in strided 4 KB pieces, ~1.4 GB/s when it is not cached)
static void prefetch_experts(const SafeTensors& st, const std::string& pre, bool sync) {
    for (const char* n : {"mlp.experts.gate_up_proj.weight_packed", "mlp.experts.gate_up_proj.weight_scale",
                          "mlp.experts.down_proj.weight_packed", "mlp.experts.down_proj.weight_scale"}) {
        if (!st.has(pre + n)) continue;
        const TensorInfo& t = st.get(pre + n);
        const uintptr_t a = (uintptr_t)t.data & ~(uintptr_t)4095, e = (uintptr_t)t.data + t.nbytes;
        madvise((void*)a, e - a, sync ? MADV_POPULATE_READ : MADV_WILLNEED);
    }
}

void Worker::load_groups(const SafeTensors& st, Layer& L, const std::string& pre, const std::vector<char>& isc) {
    prefetch_experts(st, pre, true);
    // local expert groups: gate rows [128g, 128g + 128) + up rows [640 + 128g, ...) GLU-interleaved, down
    // columns [128g, 128g + 128); the shared expert (512) from its own tensors (same layout, 1 expert)
    // cold experts (routing statistics, --cold): their groups in pinned host memory, read over PCIe
    std::vector<int> gm((NE + 1) * NGRP, -1);
    int nl = 0, nh = 0;
    for (int k = 0; k < (NE + 1) * NGRP; k++)
        if (p_.gcard[k] == p_.rank && !isc[k]) gm[k] = nl++;
    L.cexp.clear();
    for (int k = 0; k < (NE + 1) * NGRP; k++)
        if (p_.gcard[k] == p_.rank && isc[k]) { gm[k] = nl + nh++; L.cexp.push_back(k / NGRP); }
    L.ncold = nh;
    L.groups = (char*)dalloc((size_t)nl * GRP_BYTES);
    L.ndev = nl;
    if (nh) {
        char* hp;
        hp = pin_host(nullptr, (size_t)nh * GRP_BYTES, 4, false, {gpu_numa_node(dev_)});  // the card's NUMA node
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&L.hgroups, hp, 0));
        L.hgroups_h = hp;
        host_bytes_ += (size_t)nh * GRP_BYTES;
    }
    L.gmap = (int*)dalloc(gm.size() * 4);
    CUDA_CHECK(cudaMemcpy(L.gmap, gm.data(), gm.size() * 4, cudaMemcpyHostToDevice));
    const std::string ep = pre + "mlp.experts.", sp = pre + "mlp.shared_expert.";
    const TensorInfo* q[2][4] = {{&T(st, ep + "gate_up_proj.weight_packed"), &T(st, ep + "gate_up_proj.weight_scale"),
                                  &T(st, ep + "down_proj.weight_packed"), &T(st, ep + "down_proj.weight_scale")},
                                 {&T(st, sp + "gate_up_proj.weight_packed"), &T(st, sp + "gate_up_proj.weight_scale"),
                                  &T(st, sp + "down_proj.weight_packed"), &T(st, sp + "down_proj.weight_scale")}};
    // staging of CH groups per batch, one record per group: gate_up rows [256][320] int32 + [256][20] f16, down
    // [2560][16] int32 + [2560][1] f16. Two pinned host + device buffers: the gather of batch j + 1 (FLASH_LOAD_THREADS
    // threads, the small row copies out of the mapped file are the cost) overlaps the copy and repack of batch j
    constexpr int CH = 64;
    const size_t gqw = (size_t)2 * GRP * (H / 8), gsw = (size_t)2 * GRP * (H / 128);
    const size_t dqw = (size_t)H * (GRP / 8), dsw = (size_t)H;
    const size_t SG = (gqw * 4 + gsw * 2 + dqw * 4 + dsw * 2 + 255) / 256 * 256;
    static const int nth = getenv("FLASH_LOAD_THREADS") ? std::max(1, atoi(getenv("FLASH_LOAD_THREADS"))) : 8;
    std::vector<int> ks;
    for (int k = 0; k < (NE + 1) * NGRP; k++)
        if (gm[k] >= 0) ks.push_back(k);
    if (ks.empty()) return;
    char* hbuf = g_hs.get(2 * CH * SG);
    char* dbuf = (char*)g_s1.get(2 * CH * SG);
    char* tmp = (char*)g_s2.get(gqw * 4 + gsw * 2);
    cudaEvent_t ev[2];
    for (auto& x : ev) CUDA_CHECK(cudaEventCreateWithFlags(&x, cudaEventDisableTiming));
    bool rec[2] = {false, false};
    auto gather = [&](int k, char* o) {  // one group's rows into its staging record
        int32_t* hgq = (int32_t*)o;
        uint16_t* hgs = (uint16_t*)(hgq + gqw);
        int32_t* hdq = (int32_t*)(hgs + gsw);
        uint16_t* hds = (uint16_t*)(hdq + dqw);
        const int e = k / NGRP, g = k % NGRP, sh = e == NE, ee = sh ? 0 : e;
        const char* gq = q[sh][0]->data + (size_t)ee * 2 * INTER * (H / 8) * 4;
        const char* gs = q[sh][1]->data + (size_t)ee * 2 * INTER * (H / 128) * 2;
        for (int half = 0; half < 2; half++)  // gate rows, then up rows
            for (int i = 0; i < GRP; i++) {
                const int row = half * INTER + g * GRP + i;
                memcpy(&hgq[(size_t)(half * GRP + i) * (H / 8)], gq + (size_t)row * (H / 8) * 4, H / 2);
                memcpy(&hgs[(size_t)(half * GRP + i) * (H / 128)], gs + (size_t)row * (H / 128) * 2, H / 128 * 2);
            }
        const char* dq = q[sh][2]->data + (size_t)ee * H * (INTER / 8) * 4;
        const char* ds = q[sh][3]->data + (size_t)ee * H * (INTER / 128) * 2;
        for (int row = 0; row < H; row++) {
            memcpy(&hdq[(size_t)row * (GRP / 8)], dq + ((size_t)row * (INTER / 8) + g * (GRP / 8)) * 4, GRP / 2);
            hds[row] = *(const uint16_t*)(ds + ((size_t)row * (INTER / 128) + g) * 2);
        }
    };
    for (size_t j0 = 0, bi = 0; j0 < ks.size(); j0 += CH, bi ^= 1) {
        const int n = (int)std::min<size_t>(CH, ks.size() - j0);
        double ta = now_s();
        if (rec[bi]) CUDA_CHECK(cudaEventSynchronize(ev[bi]));  // the copy out of this host buffer is done
        g_tw += now_s() - ta, ta = now_s();
        char* hb = hbuf + bi * CH * SG;
        char* db = dbuf + bi * CH * SG;
        {
            std::vector<std::thread> th;
            const int t_n = std::min(nth, n);
            for (int t = 0; t < t_n; t++)
                th.emplace_back([&, t] {
                    for (int b = t; b < n; b += t_n) gather(ks[j0 + b], hb + (size_t)b * SG);
                });
            for (auto& x : th) x.join();
        }
        g_tg += now_s() - ta, ta = now_s();
        CUDA_CHECK(cudaMemcpyAsync(db, hb, (size_t)n * SG, cudaMemcpyHostToDevice, st_));
        for (int b = 0; b < n; b++) {
            const int k = ks[j0 + b];
            char* dst = gm[k] < L.ndev ? L.groups + (size_t)gm[k] * GRP_BYTES : L.hgroups + (size_t)(gm[k] - L.ndev) * GRP_BYTES;
            int32_t* dgq = (int32_t*)(db + (size_t)b * SG);
            uint16_t* dgs = (uint16_t*)(dgq + gqw);
            int32_t* ddq = (int32_t*)(dgs + gsw);
            uint16_t* dds = (uint16_t*)(ddq + dqw);
            glu_interleave_rows(dgq, (const bf16*)dgs, 2 * GRP, H, (int32_t*)tmp, (bf16*)(tmp + gqw * 4), st_);
            repack_w4((const int32_t*)tmp, (const bf16*)(tmp + gqw * 4), 2 * GRP, H, (uint4*)dst, (bf16*)(dst + GG_Q), st_);
            repack_w4(ddq, (const bf16*)dds, H, GRP, (uint4*)(dst + GG_Q + GG_S), (bf16*)(dst + GG_Q + GG_S + GD_Q), st_);
        }
        CUDA_CHECK(cudaEventRecord(ev[bi], st_));
        rec[bi] = true;
        g_tl += now_s() - ta;
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    for (auto& x : ev) CUDA_CHECK(cudaEventDestroy(x));
    g_s1.release();
    g_s2.release();
}

void Worker::alloc_full(Layer& L) {
    L.kc = (uint8_t*)dalloc((size_t)rows_ * NKV * HD);
    L.vc = (uint8_t*)dalloc((size_t)rows_ * NKV * HD);
    L.ks = (float*)dalloc((size_t)rows_ * NKV * 4);
    L.vs = (float*)dalloc((size_t)rows_ * NKV * 4);
    L.ikb = (bf16*)dalloc((size_t)(rows_ / IDX_R) * IDX_HD * 2);
    L.ik_tail = (float*)dalloc(IDX_TAIL * IDX_HD * 4);
}

void Worker::alloc_rows(int n, Bufs& b, int nb) {
    b.rows = n;
    const int n0 = n;
    if (nb > n) n = nb;  // batched rows: R and the MoE-half buffers (the chunk buffers below keep n0 rows)
    // rows past pfbn_base_ of the batch-sized buffers: borrowed from the expert cache (bt_regs_)
    auto bt = [&](void* p, size_t rowb) {
        if (n > pfbn_base_) bt_regs_.push_back({(char*)p + (size_t)pfbn_base_ * rowb, (size_t)(n - pfbn_base_) * rowb});
        return p;
    };
    b.R = (float*)bt(dalloc((size_t)n * RW * 4), (size_t)RW * 4);
    if (n <= MAXM) b.R2 = (float*)dalloc((size_t)n * RW * 4);  // decode: ping-pong for the fused residual update
    b.inj = (float*)dalloc((size_t)n * HC * 4);
    b.hu = (float*)bt(dalloc((size_t)n * HC * HCD_ROWS * 4), (size_t)HC * HCD_ROWS * 4);
    b.hss = (float*)dalloc((size_t)n * HC * 4);
    b.x = (bf16*)bt(dalloc((size_t)n * H * 2), (size_t)H * 2);
    b.xh = (__half*)bt(dalloc((size_t)n * H * 2), (size_t)H * 2);
    {  // in-projection rows (DeltaNet [q k v z] / QSA [q gate k v]) and out-projection inputs of this card's heads
        const int vl = p_.v1 - p_.v0, kl = p_.k1 - p_.k0, ql = p_.q1 - p_.q0;
        projw_ = std::max({9216, 2 * kl * DK + 2 * vl * DV, NQ * HD + ql * HD + 2 * NKV * HD});  // 3 cards: 9216
        outw_ = std::max({2048, vl * DV, ql * HD});                                            // 3 cards: 2048
    }
    {  // the mixer chunk's scratch in one block (gates, proj, proj2, yd, qbuf): the MoE half of a batch runs after every
       // mixer chunk of the layer, so its slot buffer (k_moe_dn_slot) lives in the same block (moe_slots_)
        const size_t sg = (size_t)n0 * RW * 4, sp = (size_t)n0 * projw_ * 4, sq = (size_t)n0 * (NQ + 4) * HD * 4;
        char* blk = (char*)dalloc(3 * sg + sp + sq);
        b.gates = (float*)blk;
        b.proj = (float*)(blk + sg);
        b.proj2 = (float*)(blk + sg + sp);
        b.yd = (float*)(blk + 2 * sg + sp);
        b.qbuf = (float*)(blk + 3 * sg + sp);
        moe_slots_ = (__half*)blk;
        moe_slot_rows_ = (int)((3 * sg + sp + sq) / ((size_t)H * 2));
    }
    b.ah = (__half*)dalloc((size_t)n0 * outw_ * 2);
    b.ab = (bf16*)dalloc((size_t)n0 * outw_ * 2);
    b.part = (float*)bt(dalloc((size_t)n * H * 4), (size_t)H * 4);
    b.y = (float*)bt(dalloc((size_t)n * H * 4), (size_t)H * 4);
    b.rlog = (float*)bt(dalloc((size_t)4 * n * 528 * 4), (size_t)4 * 528 * 4);
    b.pler = (bf16*)bt(dalloc((size_t)n * H * 2), (size_t)H * 2);
    b.pleh = (__half*)dalloc((size_t)n0 * H * 2);
    b.gsel = (int*)dalloc((size_t)n * TOPK * 4);
    b.gw = (float*)dalloc((size_t)n * TOPK * 4);
    b.gsh = (float*)dalloc((size_t)n * 4);
    n = n0;
    if (has_mtp_) {
        // MTP scratch aliases trunk buffers that are free while the MTP block runs
        b.Rm = (float*)dalloc((size_t)n * RW * 4);
        if (n <= MAXM) b.Rm2 = (float*)dalloc((size_t)n * RW * 4);
        b.Rsave = (float*)dalloc((size_t)n * RW * 4);
        b.mtp_en = (bf16*)dalloc((size_t)n * H * 2);
        b.mtp_hn = (bf16*)dalloc((size_t)n * RW * 2);
        b.mtp_fe = (float*)dalloc((size_t)n * H * 4);
        b.mtp_fh = b.yd;  // [n * 4][2560] floats = [n][10240]
    }
}

void Worker::use(const Bufs& b) {
    R_ = b.R; R2_ = b.R2; Rm2_ = b.Rm2; inj_ = b.inj; hu_ = b.hu; hss_ = b.hss; gates_ = b.gates; x_ = b.x; xh_ = b.xh; proj_ = b.proj;
    proj2_ = b.proj2; ah_ = b.ah; ab_ = b.ab; part_ = b.part; y_ = b.y; qbuf_ = b.qbuf; rlog_ = b.rlog; yd_ = b.yd;
    pler_ = b.pler; pleh_ = b.pleh; gsel_ = b.gsel; gw_ = b.gw; gsh_ = b.gsh; Rm_ = b.Rm; Rsave_ = b.Rsave;
    mtp_en_ = b.mtp_en; mtp_hn_ = b.mtp_hn; mtp_fe_ = b.mtp_fe; mtp_fh_ = b.mtp_fh;
    brows_ = b.rows;
}

void Worker::alloc_state() {
    CUDA_CHECK(cudaSetDevice(dev_));
    // owned KV rows for max_ctx positions
    const int periods = (max_ctx_ + p_.period - 1) / p_.period;
    rows_ = periods * p_.own;
    const int vl = p_.v1 - p_.v0, kl = p_.k1 - p_.k0;
    const int ch = 2 * kl * DK + vl * DV;
    for (auto& L : L_) {
        if (L.full) {
            alloc_full(L);
        } else {
            L.S = (float*)dalloc((size_t)vl * DK * DV * 4);
            L.conv_hist = (float*)dalloc((size_t)3 * ch * 4);
            L.conv_pend = (float*)dalloc((size_t)MAXP * ch * 4);
            L.rec_pend = (float*)dalloc((size_t)2 * MAXP * vl * DREC * 4);
        }
        if (L.ple_conv) {
            L.ple_hist = (float*)dalloc((size_t)9 * RW * 4);
            L.ple_pend = (float*)dalloc((size_t)MAXP * RW * 4);
        }
    }
    stepc_ = (unsigned*)dalloc(4);
    coldcnt_ = (unsigned*)dalloc(4);
    rfcnt_ = (unsigned*)dalloc(4);
    rfpart_ = (float*)dalloc((size_t)RF_SPLIT * RF_TILES * 32 * 4 * 4);

    if (getenv("FLASH_STATS")) {
        for (auto& L : L_) L.stats = (int*)dalloc(NE * 4);
        if (has_mtp_) mtp_.stats = (int*)dalloc(NE * 4);
    }
    if (has_mtp_) {
        alloc_full(mtp_);
        Rmin_ = (float*)dalloc((size_t)RW * 4);
    }
    {  // FLASH_PFB: rows of a batched prefill (multiple of PFM; <= PFM: chunk by chunk)
        // FLASH_PFB: the batch whose buffers are always there (2048 = the server chunk); FLASH_PFB_BIG (default 4096,
        // 0 off): larger batches, the extra rows' buffers borrowed from the expert cache while such a batch runs
        const int v = getenv("FLASH_PFB") ? atoi(getenv("FLASH_PFB")) : 2048;
        const int vb = getenv("FLASH_PFB_BIG") ? atoi(getenv("FLASH_PFB_BIG")) : 4096;
        auto rows_of = [](int r) { return std::max(PFM, std::min(32 * PFM, (r + PFM - 1) / PFM * PFM)) + PFM; };  // + one chunk: a short tail joins the last batch
        pfbn_base_ = rows_of(v);
        pfbn_ = vb > v ? rows_of(vb) : pfbn_base_;
    }
    alloc_rows(PFM, pfb_, pfbn_);
    // decode rows alias the prefill buffers (never live at the same time; every forward starts from the embedding),
    // except the decode-only ping-pong residuals
    dec_ = pfb_;
    dec_.rows = MAXM;
    dec_.R2 = (float*)dalloc((size_t)MAXM * RW * 4);
    if (has_mtp_) dec_.Rm2 = (float*)dalloc((size_t)MAXM * RW * 4);
    cposd_ = (int*)dalloc((size_t)2 * (pfbn_ / PFM) * 4);
    {  // prefill collectives: the highest priority, so their few blocks get SMs ahead of the queued blocks of the
       // compute kernels they overlap (default priority: an all-reduce waited for MoE / GEMM blocks to retire)
        static const bool prio = !getenv("FLASH_CST_PRIO") || atoi(getenv("FLASH_CST_PRIO"));
        int lo = 0, hi = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&cst_, cudaStreamNonBlocking, prio ? hi : 0));
    }
    evp_.resize(pfbn_ / PFM);
    evd_.resize(pfbn_ / PFM);
    for (auto& e : evp_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    for (auto& e : evd_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evm_, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreateWithFlags(&pfs_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&pfe0_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&pfe1_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&pfq0_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&pfq1_, cudaEventDisableTiming));
    hcl2_ = getenv("FLASH_HCL2") ? atoi(getenv("FLASH_HCL2")) : 2;
    CUDA_CHECK(cudaHostAlloc((void**)&cposh_, (size_t)2 * (pfbn_ / PFM) * 4, cudaHostAllocDefault));
    use(dec_);
    attn_x_ = (float*)dalloc((size_t)XROWS * NQ * (HD + 2) * 4);
    attn_all_ = (float*)dalloc((size_t)p_.ngpu * MAXM * NQ * (HD + 2) * 4);

    njobs_ = (int*)dalloc(4);
    // job lists: a job is one local group of a routed expert (+ the shared expert's), at most the card's local groups
    // (3 cards: 896, 2 cards: ~1283 -> 1312); prefill jobs add the 64-row chunks of the batch's experts
    maxjob_ = std::max(MAXJOB, (plan_groups(p_) + 8 + 31) / 32 * 32);
    // most local groups of one expert on this card (shared expert included): the prefill job bound below. Jobs =
    // sum_e lg(e) ceil(cnt_e / 64) <= plan_groups + maxlg ceil(m (TOPK + 1) / 64). The former 2 * ... assumed <= 2 local
    // groups per expert, false on 2 GPUs (5 groups over 2 cards: 3 on one) -> jobs past the grid were skipped silently
    // (the shared expert's come last), at 97 % of the bound on a 16K prompt (qfn_prefill_r4)
    for (int e = 0; e <= NE; e++) {
        int c = 0;
        for (int g = 0; g < NGRP; g++) c += (size_t)(e * NGRP + g) < p_.gcard.size() && p_.gcard[e * NGRP + g] == p_.rank;
        maxlg_ = std::max(maxlg_, c);
    }
    if (!maxlg_) maxlg_ = NGRP;
    maxjob_pf_ = std::max(MAXJOB_PF, maxjob_ + maxlg_ * ((pfbn_ * (TOPK + 1) + PF_SHJ - 1) / PF_SHJ));
    jobs_ = (MoeJob*)dalloc((size_t)maxjob_ * sizeof(MoeJob));
    jw_ = (float*)dalloc((size_t)maxjob_ * MAXM * 4);
    acth_ = (__half*)dalloc((size_t)maxjob_ * MAXM * GRP * 2);
    jrow_ = (int*)dalloc((size_t)maxjob_ * (MAXM + 1) * 4);
    toks_ = (int*)dalloc(pfbn_ * 4);
    posd_ = (int*)dalloc(8);  // [pos, rows of the previous forward to commit]
    pfjobs_ = (PfJob*)dalloc((size_t)maxjob_pf_ * sizeof(PfJob));
    pftl_ = (int*)dalloc((size_t)pfbn_ * (TOPK + 1) * 4);
    pfwl_ = (float*)dalloc((size_t)pfbn_ * (TOPK + 1) * 4);
    pfinv_ = (int*)dalloc((size_t)pfbn_ * TOPK * 4);
    pfact_bytes_ = (size_t)pfbn_ * (TOPK + 1) * NGRP * GRP * 2;
    pfact_ = (__half*)dalloc(pfact_bytes_);
    {  // pipelined QSA chunks: slot 1 copies of proj_ and attn_x_ behind the row GEMMs' quantized rows (4 MB)
        const size_t o0 = (size_t)4 << 20, pb = (size_t)PFM * projw_ * 4, ab = (size_t)XROWS * NQ * (HD + 2) * 4 + 256;
        if (o0 + pb + ab <= pfact_bytes_) {
            projB_ = (float*)((char*)pfact_ + o0);
            attnB_ = (float*)((char*)pfact_ + o0 + pb);
        }
    }
    if (pfbn_ > pfbn_base_) {
        const size_t rb = (size_t)(TOPK + 1) * NGRP * GRP * 2;
        bt_regs_.push_back({(char*)pfact_ + (size_t)pfbn_base_ * rb, (size_t)(pfbn_ - pfbn_base_) * rb});
        // slot extension: the big batch's 2048-row MoE parts (k_moe_dn_slot) need 2048 x 11 slot rows
        const int need = 2048 * (TOPK + 1) - moe_slot_rows_;
        if (need > 0) {
            moe_slot_rows2_ = need;
            moe_slots2_ = (__half*)dalloc((size_t)need * H * 2);
            bt_regs_.push_back({(char*)moe_slots2_, (size_t)need * H * 2});
        }
    }
    possub_ = (int*)dalloc(PFM / 32 * 4);
    dnqk_ = (float*)dalloc((size_t)PFM * 2 * LNK * DK * 4);
    dngb_ = (float*)dalloc((size_t)PFM * 2 * LNV * 4);
    zero_d_ = (int*)dalloc(4);
    attn_split_ = getenv("FLASH_ATTN_SPLIT") ? atoi(getenv("FLASH_ATTN_SPLIT")) : 4;  // decode attention key splits (1: old kernel)
    attn_split_ = std::max(1, std::min(8, attn_split_));
    dec_k_ = !getenv("FLASH_ATTN_DEC") || atoi(getenv("FLASH_ATTN_DEC"));  // GQA-tile decode attention (0: tc2 / old)
    dec_split_ = getenv("FLASH_ATTN_DSPLIT") ? std::max(1, std::min(8, atoi(getenv("FLASH_ATTN_DSPLIT")))) : 8;
    if (dec_k_) attn_split_ = std::max(attn_split_, dec_split_);
    CUDA_CHECK(cudaFuncSetAttribute(k_qsa_dec, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)QD_SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(k_idx_score_t, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)IS_SMEM));
    if (attn_split_ > 1) attn_spl_ = (float*)dalloc((size_t)(attn_split_ - 1) * XROWS * NQ * (HD + 2) * 4);
    mtok_ = (int*)dalloc(pfbn_ * 4);
    hh_pf_ = (__half*)dalloc((size_t)pfbn_ * LOWRANK * 2);
    hcs_ = !getenv("FLASH_HCS") || atoi(getenv("FLASH_HCS"));  // decode HC down split by groups (0: k_hcd)
    if (hcs_) hcp_ = (float*)dalloc((size_t)MAXM * HC * HCD_GPS * HCD_ROWS * 4);
    rinv_pf_ = (float*)dalloc((size_t)pfbn_ * HC * 4);
    logits_ = (float*)dalloc((size_t)MAXM * (p_.voc1 - p_.voc0) * 4);
    pairs_ = (float*)dalloc(MAXM * 2 * 4);
    pairs_all_ = (float*)dalloc((size_t)p_.ngpu * MAXM * 2 * 4);
    otok_ = (int*)dalloc(MAXM * 4);
    sp_ = (SampleParams*)dalloc(sizeof(SampleParams));
    CUDA_CHECK(cudaHostAlloc((void**)&tok_h_, MAXM * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&pos_h_, 8, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&out_h_, MAXM * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&ple_h_, (size_t)MAXM * H * 2, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&pfh_tok_, pfbn_ * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&pfh_mt_, pfbn_ * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&pfh_pos_, 8, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&pfh_out_, 8, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&pfh_ple_, (size_t)pfbn_ * H * 2, cudaHostAllocDefault));
    cand_ = (float*)dalloc((size_t)MAXM * kTopK * 2 * 4);
    cand_all_ = (float*)dalloc((size_t)p_.ngpu * MAXM * kTopK * 2 * 4);
    motok_ = (int*)dalloc(MAXM * 4);
    mposd_ = (int*)dalloc(4);
    vtok_ = (int*)dalloc(MAXM * 4);
    acc_ = (int*)dalloc(4);
    dr_ = (int*)dalloc(MAXM * 4);
    hist2_ = (int*)dalloc(8);
    CUDA_CHECK(cudaHostAlloc((void**)&em_h_, (MAXM + 2 + 32) * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(em_h_, 0, (MAXM + 2 + 32) * 4);
    lk_on_ = !getenv("FLASH_LOOKUP") || atoi(getenv("FLASH_LOOKUP"));  // device lookup drafts (0: off)
    if (getenv("FLASH_DETERMINISTIC") && atoi(getenv("FLASH_DETERMINISTIC"))) {  // fixed-order / exact float sums
        detpart_ = (unsigned long long*)dalloc((size_t)pfbn_ * std::max(H, HC * HCD_ROWS + HC) * 8);  // MoE partial / HC down sums
    }
    if (getenv("FLASH_DBGHASH") && atoi(getenv("FLASH_DBGHASH"))) dbgh_ = (unsigned long long*)dalloc((size_t)(LAYERS + 1) * DBGK * 8);
    {  // sequence-parallel HC prefill: row all-gather protocol (2 channels) and the HC scratch of one chunk's own rows
        agflags_ = (unsigned*)dalloc(4 * 4 * 4);  // [2 channels][4 senders] flags, then [2 channels][4 readers] acks (pull)
        agseq_ = (unsigned*)dalloc(2 * 4);
        agscnt_ = (unsigned*)dalloc(2 * 8 * 4);  // [2 channels][push blocks 4 | pull blocks 4]
        CUDA_CHECK(cudaMemset(agflags_, 0, 4 * 4 * 4));
        CUDA_CHECK(cudaMemset(agseq_, 0, 2 * 4));
        CUDA_CHECK(cudaMemset(agscnt_, 0, 2 * 8 * 4));
        // own rows of one collective unit: <= AG_CAP (ARU_MAX / ngpu, 2 cards: 512)
        // [2 sets]: the comm stream's HC reads and the compute stream's (MTP block input) run concurrently
        sp_hu_ = (float*)dalloc((size_t)2 * AG_CAP * HC * HCD_ROWS * 4);
        sp_hss_ = (float*)dalloc((size_t)2 * AG_CAP * HC * 4);
        sp_hh_ = (__half*)dalloc((size_t)2 * AG_CAP * LOWRANK * 2);
        sp_rinv_ = (float*)dalloc((size_t)2 * AG_CAP * HC * 4);
        if (detpart_) {
            sp_det_ = (unsigned long long*)dalloc((size_t)2 * AG_CAP * (HC * HCD_ROWS + HC) * 8);
            CUDA_CHECK(cudaMemset(sp_det_, 0, (size_t)2 * AG_CAP * (HC * HCD_ROWS + HC) * 8));
        }
    }
    // policy (offline sweep over agent / code / edit traces): a match of >= 24 tokens
    // -> 16 drafts, >= 6 -> 8 drafts, shorter -> MTP step
    lk_mmin_ = getenv("FLASH_LK_MIN") ? atoi(getenv("FLASH_LK_MIN")) : 6;
    lk_llo_ = getenv("FLASH_LK_LO") ? atoi(getenv("FLASH_LK_LO")) : 8;
    lk_hi_ = getenv("FLASH_LK_HI") ? atoi(getenv("FLASH_LK_HI")) : 24;
    lk_lmax_ = getenv("FLASH_LK_LMAX") ? std::min(MAXP - 1, atoi(getenv("FLASH_LK_LMAX"))) : 16;
    lk_llo_ = std::min(lk_llo_, lk_lmax_);
    if (lk_on_) {
        hd_ = (int*)dalloc((size_t)(max_ctx_ + 64) * 4);
        hn_d_ = (int*)dalloc(4);
        lkbest_ = (unsigned long long*)dalloc(8);
        lk_ = (int*)dalloc(64 * 4);
        lkrg_ = (int*)dalloc(2 * 4);
        // FLASH_LK_REGION=1: region-aware policy: tool call / code nmin 3, L = clamp(l, 4, 16);
        // text nmin 12, L = clamp(l / 2, 4, 8); FLASH_LK_RP=8 comma values override (beta x4)
        if (getenv("FLASH_LK_REGION") && atoi(getenv("FLASH_LK_REGION"))) {
            int rp[8] = {3, 4, 4, 16, 12, 2, 4, 8};
            if (getenv("FLASH_LK_RP")) sscanf(getenv("FLASH_LK_RP"), "%d,%d,%d,%d,%d,%d,%d,%d", rp, rp + 1, rp + 2, rp + 3, rp + 4, rp + 5, rp + 6, rp + 7);
            rp[3] = std::min(rp[3], MAXP - 1);
            rp[7] = std::min(rp[7], MAXP - 1);
            lkrp_ = (int*)dalloc(8 * 4);
            CUDA_CHECK(cudaMemcpy(lkrp_, rp, 32, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&lkst_, cudaStreamNonBlocking));
        // FLASH_CORPUS=1048576 (tokens) turns it on: server measurements were inconclusive
        ccap_ = getenv("FLASH_CORPUS") ? atoi(getenv("FLASH_CORPUS")) : 0;
        if (ccap_ > 0) {
            cd_ = (int*)dalloc((size_t)(ccap_ + 64) * 4);
            CUDA_CHECK(cudaMemset(cd_, 0xff, (size_t)(ccap_ + 64) * 4));  // -1: separator / empty
            cn_d_ = (int*)dalloc(4);
            CUDA_CHECK(cudaMemset(cn_d_, 0, 4));
            lkbestc_ = (unsigned long long*)dalloc(8);
            CUDA_CHECK(cudaMemset(lkbestc_, 0, 8));
        }
    }
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&em_, em_h_, 0));
    if (!getenv("FLASH_GUARD") || atoi(getenv("FLASH_GUARD"))) {
        CUDA_CHECK(cudaHostAlloc((void**)&nf_h_, 4, cudaHostAllocMapped | cudaHostAllocPortable));
        *nf_h_ = 0;
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&nf_d_, nf_h_, 0));
    }
    xrecv_ = (float*)dalloc((size_t)XSLOT * p_.ngpu * XROWS * xper(p_) * 4);
    xscnt_ = (unsigned*)dalloc(4 * 4);
    xqscnt_ = (unsigned*)dalloc(4 * 4);
    xflags_ = (unsigned*)dalloc(4 * 4);
    xseq_ = (unsigned*)dalloc(4);
    xdone_ = (unsigned*)dalloc(4);
    xqrecv_ = (float*)dalloc((size_t)XSLOT * p_.ngpu * XROWS * qper(p_) * 4);
    xqflags_ = (unsigned*)dalloc(4 * 4);
    xqseq_ = (unsigned*)dalloc(4);
    xqdone_ = (unsigned*)dalloc(4);
    xirecv_ = (float*)dalloc((size_t)XSLOT * p_.ngpu * XROWS * IDX_TOPB * 4);
    xiflags_ = (unsigned*)dalloc(4 * 4);
    xiseq_ = (unsigned*)dalloc(4);
    xidone_ = (unsigned*)dalloc(4);
    alloc_staging();  // staged prefill transfers (FLASH_COMM ce / host)
    // block selection of a whole prefill chunk at once (XROWS rows: one indexer exchange per chunk, not per 32 rows)
    iscore_ = (float*)dalloc((size_t)XROWS * (rows_ / IDX_R) * 4);
    isel_ = (int*)dalloc((size_t)XROWS * (rows_ / IDX_R) * 4);
    insel_ = (int*)dalloc(XROWS * 4);
    {  // block unions of the prefill attention's row groups (k_qpf_union)
        const int ng = (XROWS + QP_R - 1) / QP_R;
        qpu_.U = (int*)dalloc((size_t)ng * 4);
        qpu_.ublk = (int*)dalloc((size_t)ng * QP_UMAX * 4);
        qpu_.um = (unsigned char*)dalloc((size_t)ng * QP_UMAX);
        qpu_.ctr = (unsigned*)dalloc(2 * 4);
        CUDA_CHECK(cudaMemset(qpu_.ctr, 0, 2 * 4));
    }
    iloc_ = (unsigned*)dalloc((size_t)XROWS * IDX_TOPB * 4);
    {  // the local block keys in shared memory when they fit (3 cards at 262144: 87.5 KB), else in place of the scores
        int optin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_));
        const size_t kb = std::max<size_t>((size_t)(rows_ / IDX_R) * 4, 1024);
        idx_gk_ = kb + 2048 > (size_t)optin;
        if (!idx_gk_) CUDA_CHECK(cudaFuncSetAttribute(k_idx_local<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kb));
    }
    alloc_slots();  // recurrent state of the other sequence slots (flash_seq.cu), before the expert budget
    load_deferred_groups();
    ec_init();
    pfs_init();
}

// Trunk expert groups after all other allocations (weights, the full max_ctx KV pool + MTP, buffers): free device
// memory minus a margin (FLASH_EC_MARGIN_MB, default 256: graphs, lazy library allocations) decides how many of this
// card's routed groups stay on the device; the coldest (static cold list first, then by router-trace prior, then by
// index) are host-backed (pinned, full copy) and cached dynamically in the KV tail. The shared expert stays on device.
void Worker::load_deferred_groups() {
    if (ec_defer_l_.empty()) return;
    std::vector<char> incold((size_t)LAYERS * NE, 0);
    for (int li = 0; li < LAYERS && li < (int)cold_.size(); li++)
        for (int e : cold_[li]) incold[(size_t)li * NE + e] = 1;
    struct G { double hot; int li, k; };
    std::vector<G> gs;
    int nshared = 0;
    for (auto& d : ec_defer_l_)
        for (int k = 0; k < (NE + 1) * NGRP; k++) {
            if (p_.gcard[k] != p_.rank) continue;
            const int e = k / NGRP;
            if (e == NE) { nshared++; continue; }
            const double pr = (int)ec_prior_.size() == LAYERS * NE ? ec_prior_[(size_t)d.li * NE + e] : 0.0;
            gs.push_back({(incold[(size_t)d.li * NE + e] ? -1.0 : 0.0) + pr, d.li, k});
        }
    std::stable_sort(gs.begin(), gs.end(), [](const G& a, const G& b) { return a.hot < b.hot; });
    // free VRAM left after the resident groups: "256" for every card or a list per rank,
    // e.g. "256,2048,256" (room for the desktop on GPU1); a short list repeats its last value
    double margin_mb = 256;
    if (const char* e = getenv("FLASH_EC_MARGIN_MB")) {
        std::stringstream ss(e);
        int i = 0;
        for (std::string x; std::getline(ss, x, ',') && i <= p_.rank; i++) margin_mb = atof(x.c_str());
    }
    const long long fr = (long long)free_bytes();
    // + per layer: gmap, allocation rounding (2 MiB), the cache table (8 B per group)
    const long long other = (long long)nshared * GRP_BYTES + (long long)ec_defer_l_.size() * (2 << 20) + (long long)gs.size() * 8;
    long long nres = (fr - (long long)(margin_mb * 1048576.0) - other) / (long long)GRP_BYTES;
    nres = std::max(0LL, std::min<long long>(nres, gs.size()));
    // prefill staging ring (2 x the most cold groups of a layer, capped by FLASH_PF_STAGE_MB): its groups come out of the
    // resident budget (the budget, the per-layer cold counts and the ring size settle in a few rounds)
    pfs_mode_ = getenv("FLASH_PF_STAGE") ? atoi(getenv("FLASH_PF_STAGE")) : 1;  // 2 (lent ring): staging stops when a long context reaches the ring slots
    static const int pfs_mb = getenv("FLASH_PF_STAGE_MB") ? atoi(getenv("FLASH_PF_STAGE_MB")) : 512;
    if (pfs_mode_ == 1 && pfs_mb > 0) {  // own ring: its groups come out of the resident budget
        const long long nres0 = nres;
        long long ring = 0;
        for (int it = 0; it < 4; it++) {
            std::map<int, int> per;
            int mx = 0;
            for (size_t i = 0; i < gs.size() - (size_t)nres; i++) mx = std::max(mx, ++per[gs[i].li]);
            ring = std::min<long long>(2LL * mx, (long long)pfs_mb * 1048576 / GRP_BYTES);
            nres = std::max(0LL, nres0 - ring);
        }
        pfs_cap_ = (int)(ring / 2);
        pfs_resv_ = ring * (long long)GRP_BYTES;
    }
    const size_t ncold = gs.size() - nres;
    std::map<int, std::vector<char>> isc;
    for (auto& d : ec_defer_l_) isc[d.li].assign((NE + 1) * NGRP, 0);
    for (size_t i = 0; i < ncold; i++) isc[gs[i].li][gs[i].k] = 1;
    for (size_t i = 0; i < ec_defer_l_.size(); i++) {
        auto& d = ec_defer_l_[i];
        if (i + 1 < ec_defer_l_.size()) prefetch_experts(*ec_defer_l_[i + 1].st, ec_defer_l_[i + 1].pre, false);
        load_groups(*d.st, L_[d.li], d.pre, isc[d.li]);
    }
    ec_defer_l_.clear();
    g_hs.release();
    if (pfs_mode_ == 1 && pfs_cap_ > 0) pfs_ring_ = (char*)dalloc((size_t)2 * pfs_cap_ * GRP_BYTES);
    fprintf(stderr, "[flash] GPU%d group load: gather %.1f s, buffer wait %.1f s, copy + repack launches %.1f s\n", dev_, g_tg, g_tw, g_tl);
    fprintf(stderr, "[flash] GPU%d expert budget: %.2f GiB free -> %lld of %zu routed groups on the device, %zu host-backed (%.2f GB)\n",
            dev_, fr / 1073741824.0, nres, gs.size(), ncold, ncold * (double)GRP_BYTES / 1e9);
}

unsigned Worker::cold_hits() {
    CUDA_CHECK(cudaSetDevice(dev_));
    unsigned h;
    CUDA_CHECK(cudaMemcpy(&h, coldcnt_, 4, cudaMemcpyDeviceToHost));
    return h;
}

// ---------------------------------------------------------------- dynamic expert cache
// The cold expert groups (static set, --cold) always have their full copy in mapped pinned host memory. Device copies
// live in slots carved from the unused tail of the KV pool: every QSA layer's (trunk + MTP) K and V arrays hold rows
// [0, rows_) and the positions of the context so far use rows [0, rows(ctx)) of each, so the slots are taken from the
// end of each array backward. Before every launch (ec_tick) slots that the coming forward's KV rows (+ margin) would
// overlap are given back to the KV pool (table entry -> host, slot zeroed); slots freed by a shorter context come back.
// A launch that routes to a cold group without a live device copy reads it zero-copy (as the static offload) and the
// group becomes a copy candidate: an async H2D copy on ec_cs_ into a free slot or the slot of the resident group with
// the lowest decayed access count (frequency-aware, LFU with exponential aging, + hysteresis), at most ec_maxcopy_
// per launch; the table entry switches at the first tick after the copy lands. The table is device memory read by the
// job-building kernels, updated only between launches (never inside a running graph).
void Worker::pfs_init() {
    if (!ec_n_ || pfs_mode_ == 0) return;
    std::vector<char*> ring;  // ring position -> device address ([0, cap): half 0, [cap, 2 cap): half 1)
    if (pfs_mode_ == 1) {
        if (!pfs_ring_) return;
        for (int i = 0; i < 2 * pfs_cap_; i++) ring.push_back(pfs_ring_ + (size_t)i * GRP_BYTES);
    } else {
        // ring lent by the expert cache while prefills run: its slots at the highest KV offsets (the last a growing
        // context reclaims), at most FLASH_PF_STAGE_MB and FLASH_PF_STAGE_PCT % of the cache slots
        if (!ec_on_ || ec_slots_.empty()) { pfs_mode_ = 0; return; }
        int mx = 0;
        for (int li = 0; li < LAYERS; li++) mx = std::max(mx, L_[li].ncold);
        static const int mb = getenv("FLASH_PF_STAGE_MB") ? atoi(getenv("FLASH_PF_STAGE_MB")) : 512;
        static const int pct = getenv("FLASH_PF_STAGE_PCT") ? atoi(getenv("FLASH_PF_STAGE_PCT")) : 50;
        pfs_cap_ = std::min({mx, (int)((long long)mb * 1048576 / GRP_BYTES / 2), (int)(ec_slots_.size() * pct / 200)});
        if (pfs_cap_ <= 0) { pfs_mode_ = 0; return; }
        std::vector<int> idx(2 * pfs_cap_);
        for (int i = 0; i < 2 * pfs_cap_; i++) idx[i] = i;  // ec_slots_ is sorted by offset, descending
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return ec_slots_[a].addr < ec_slots_[b].addr; });
        pfs_slots_ = idx;
        for (int i : idx) ring.push_back(ec_slots_[i].addr);
    }
    for (int h = 0; h < 2; h++) {
        pfs_runs_[h].clear();
        for (int p = 0; p < pfs_cap_; p++) {
            char* a = ring[(size_t)h * pfs_cap_ + p];
            auto& rs = pfs_runs_[h];
            if (!rs.empty() && rs.back().addr + (size_t)rs.back().n * GRP_BYTES == a) rs.back().n++;
            else rs.push_back({p, 1, a});
        }
    }
    std::vector<uintptr_t> t(ec_n_, 0);
    for (int li = 0; li < LAYERS; li++) {
        const Layer& L = L_[li];
        for (int j = 0; j < L.ncold; j++)
            t[L.coff + j] = j < pfs_cap_ ? (uintptr_t)ring[(size_t)(li & 1) * pfs_cap_ + j]
                                         : ((uintptr_t)(L.hgroups + (size_t)j * GRP_BYTES) | 1);
    }
    pfs_tab_d_ = (uintptr_t*)dalloc((size_t)ec_n_ * sizeof(uintptr_t));
    CUDA_CHECK(cudaMemcpy(pfs_tab_d_, t.data(), t.size() * sizeof(uintptr_t), cudaMemcpyHostToDevice));
    pfs_touch_d_ = (unsigned char*)dalloc(ec_n_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&dst_, cudaStreamNonBlocking));
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&pfs_cp_[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&pfs_used_[i], cudaEventDisableTiming));
    }
    if (p_.rank == 0 || getenv("FLASH_PF_STAGE_LOG"))
        fprintf(stderr, "[flash] GPU%d prefill staging ring: 2 x %d groups (%.0f MB, %s, %zu + %zu copies per layer pair)\n", dev_,
                pfs_cap_, 2.0 * pfs_cap_ * GRP_BYTES / 1e6, pfs_mode_ == 1 ? "own memory" : "lent by the expert cache",
                pfs_runs_[0].size(), pfs_runs_[1].size());
}

// mode 2: take the ring's cache slots (their groups fall back to host reads) for the prefill batches; false when a
// context reaches them (then the batch reads the cold groups zero-copy)
bool Worker::pfs_borrow() {
    if (pfs_mode_ == 1) return pfs_ring_ != nullptr;
    if (pfs_mode_ != 2) return false;
    for (int s2 : pfs_slots_)
        if (!ec_slots_[s2].live) { pfs_release(); return false; }
    pfs_idle_ = 0;
    if (pfs_borrowed_) return true;
    for (int s2 : pfs_slots_) {
        EcSlot& sl = ec_slots_[s2];
        if (sl.ci >= 0) {
            if (ec_pend_[sl.ci]) { CUDA_CHECK(cudaStreamWaitEvent(st_, sl.ev, 0)); ec_pend_[sl.ci] = 0; }
            ec_tab_h_[sl.ci] = ec_host_[sl.ci] | 1;
            ec_slot_of_[sl.ci] = -1;
            sl.ci = -1;
            ec_dirty_ = true;
        }
        sl.pf = true;
    }
    if (ec_dirty_) {
        CUDA_CHECK(cudaMemcpyAsync(ec_tab_d_, ec_tab_h_, (size_t)ec_n_ * sizeof(uintptr_t), cudaMemcpyHostToDevice, st_));
        ec_dirty_ = false;
    }
    pfs_borrowed_ = true;
    return true;
}

void Worker::pfs_release() {  // the ring's slots back to the cache (empty; prefill batches are synchronous)
    if (!pfs_borrowed_) return;
    for (int s2 : pfs_slots_) ec_slots_[s2].pf = false;
    pfs_borrowed_ = false;
}

// copy engine: layer li's host-backed groups (up to pfs_cap_) into ring half li & 1 (on dst_)
void Worker::pfs_copy(int li) {
    const Layer& L = L_[li];
    const int n = std::min(L.ncold, pfs_cap_);
    for (const PfsRun& r : pfs_runs_[li & 1]) {
        const int k = std::min(r.n, n - r.p0);
        if (k <= 0) break;
        CUDA_CHECK(cudaMemcpyAsync(r.addr, L.hgroups_h + (size_t)r.p0 * GRP_BYTES, (size_t)k * GRP_BYTES, cudaMemcpyHostToDevice, dst_));
    }
    CUDA_CHECK(cudaEventRecord(pfs_cp_[li & 1], dst_));
}

void Worker::ec_init() {
    int n = 0;
    std::vector<Layer*> ls;
    for (auto& L : L_) ls.push_back(&L);
    for (Layer* L : ls) { L->coff = n; n += L->ncold; }
    ec_n_ = n;
    ec_st_.ncold = n;
    ec_on_ = n > 0 && !(getenv("FLASH_EC") && atoi(getenv("FLASH_EC")) == 0);
    if (!n) return;
    ec_tab_d_ = (uintptr_t*)dalloc((size_t)n * sizeof(uintptr_t));
    CUDA_CHECK(cudaHostAlloc((void**)&ec_tab_h_, (size_t)n * sizeof(uintptr_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&ec_touch_h_, n, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(ec_touch_h_, 0, n);
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&ec_touch_d_, ec_touch_h_, 0));
    ec_host_.resize(n);
    ec_src_.resize(n);
    std::vector<double> prior(n, 0.0);
    const double decay = getenv("FLASH_EC_DECAY") ? atof(getenv("FLASH_EC_DECAY")) : 0.999;  // per launch
    ec_logd_ = std::log(decay);
    if (getenv("FLASH_EC_HYST")) ec_hyst_ = atof(getenv("FLASH_EC_HYST"));
    if (getenv("FLASH_EC_COPIES")) ec_maxcopy_ = atoi(getenv("FLASH_EC_COPIES"));
    for (int li = 0; li < LAYERS; li++) {
        const Layer& L = L_[li];
        for (int j = 0; j < L.ncold; j++) {
            const int ci = L.coff + j;
            ec_host_[ci] = (uintptr_t)(L.hgroups + (size_t)j * GRP_BYTES);
            ec_src_[ci] = L.hgroups_h + (size_t)j * GRP_BYTES;
            if ((int)ec_prior_.size() == LAYERS * NE) prior[ci] = ec_prior_[(size_t)li * NE + L.cexp[j]];
        }
    }
    ec_slot_of_.assign(n, -1);
    ec_pend_.assign(n, 0);
    ec_ls_.assign(n, -1e300);
    // prior: steady-state count of a group routed with probability p per token, ~3 tokens per launch
    for (int ci = 0; ci < n; ci++)
        if (prior[ci] > 0) ec_ls_[ci] = std::log(3.0 * prior[ci] / (1.0 - decay));
    for (int ci = 0; ci < n; ci++) ec_tab_h_[ci] = ec_host_[ci] | 1;
    if (ec_on_) {
        std::vector<Layer*> kvl = ls;
        if (has_mtp_) kvl.push_back(&mtp_);
        const size_t asz = (size_t)rows_ * NKV * HD;
        const int per = (int)(asz / GRP_BYTES);
        for (Layer* L : kvl) {
            if (!L->full || !L->kc) continue;
            for (uint8_t* base : {L->kc, L->vc})
                for (int s2 = 0; s2 < per; s2++) {
                    EcSlot sl;
                    sl.off = asz - (size_t)(s2 + 1) * GRP_BYTES;
                    sl.addr = (char*)base + sl.off;
                    sl.ci = -1;
                    sl.live = true;
                    CUDA_CHECK(cudaEventCreateWithFlags(&sl.ev, cudaEventDisableTiming));
                    ec_slots_.push_back(sl);
                }
        }
        // slots ordered by offset descending (the last to be reclaimed by a growing context first)
        std::stable_sort(ec_slots_.begin(), ec_slots_.end(), [](const EcSlot& a, const EcSlot& b) { return a.off > b.off; });
        // the big-batch buffer tails (bt): after the KV slots (filled last), taken back for every big prefill batch
        static const bool bt_on = !getenv("FLASH_PFB_LEND") || atoi(getenv("FLASH_PFB_LEND"));
        for (auto& [base, len] : bt_regs_) {
            if (!bt_on) break;
            const size_t a0 = ((size_t)base + 255) & ~(size_t)255, end = (size_t)base + len;
            for (size_t a = a0; a + GRP_BYTES <= end; a += GRP_BYTES) {
                EcSlot sl;
                sl.off = 0;
                sl.addr = (char*)a;
                sl.ci = -1;
                sl.live = true;
                sl.bt = true;
                CUDA_CHECK(cudaEventCreateWithFlags(&sl.ev, cudaEventDisableTiming));
                ec_slots_.push_back(sl);
                ec_st_.bt_slots++;
            }
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&ec_cs_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ec_evt_, cudaEventDisableTiming));
        // seed: the hottest cold groups by the router-trace prior (ties by index) into the slots, synchronously
        std::vector<int> ord(n);
        for (int i = 0; i < n; i++) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return prior[a] > prior[b]; });
        static const int seed_max = getenv("FLASH_EC_SEED") ? atoi(getenv("FLASH_EC_SEED")) : 1 << 30;
        const int ns = std::min<int>({n, (int)ec_slots_.size(), seed_max});
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
        CUDA_CHECK(cudaEventRecord(a, ec_cs_));
        for (int i = 0; i < ns; i++) {
            const int ci = ord[i];
            CUDA_CHECK(cudaMemcpyAsync(ec_slots_[i].addr, ec_src_[ci], GRP_BYTES, cudaMemcpyHostToDevice, ec_cs_));
            ec_slots_[i].ci = ci;
            ec_slot_of_[ci] = i;
            ec_tab_h_[ci] = (uintptr_t)ec_slots_[i].addr;
        }
        CUDA_CHECK(cudaEventRecord(b, ec_cs_));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        if (ns && ms > 0) ec_st_.seed_gbs = (double)ns * GRP_BYTES / (ms * 1e6);
        CUDA_CHECK(cudaEventDestroy(a));
        CUDA_CHECK(cudaEventDestroy(b));
        ec_st_.slots = ec_slots_.size();
        ec_bound_ = 0;
    }
    CUDA_CHECK(cudaMemcpy(ec_tab_d_, ec_tab_h_, (size_t)n * sizeof(uintptr_t), cudaMemcpyHostToDevice));
}

// touches of the last launch -> statistics, access counts, copy candidates are collected by ec_tick
void Worker::ec_collect() {
    if (!ec_n_ || ec_kind_ < 0) return;
    long long t = 0, ms = 0;
    for (int ci = 0; ci < ec_n_; ci++) {
        if (!ec_touch_h_[ci]) continue;
        ec_touch_h_[ci] = 0;
        t++;
        if (ec_tab_h_[ci] & 1) ms++;
        // decayed count += 1 at the launch's time; decode launches only (a prefill batch touches nearly every group:
        // no frequency information, it would flatten the counts)
        if (ec_kind_ == 0) {
            const double cur = ec_ls_[ci] > -1e299 ? std::exp(ec_ls_[ci] + ec_now_ * ec_logd_) : 0.0;
            ec_ls_[ci] = std::log(cur + 1.0) - ec_now_ * ec_logd_;
        }
        if ((ec_tab_h_[ci] & 1) && ec_on_ && !ec_pend_[ci] && ec_slot_of_[ci] < 0) ec_cand_.push_back(ci);
    }
    if (ec_kind_ == 0) {
        ec_st_.dec_steps++;
        ec_st_.dec_touch += t;
        ec_st_.dec_miss += ms;
        ec_st_.step_miss.push_back((unsigned short)std::min<long long>(ms, 65535));
    } else {
        ec_st_.pf_calls++;
        ec_st_.pf_touch += t;
        ec_st_.pf_miss += ms;
    }
    ec_kind_ = -1;
}

void Worker::ec_tick(int ctx_end, int kind) {
    if (!ec_n_) return;
    ctx_end = (int)std::max<long long>(ctx_end + ctx_off_, kv_top_);  // sequence slots: the region's offset, all regions
    ec_collect();
    ec_kind_ = kind;
    if (!ec_on_) return;
    if (kind == 0) ec_now_++;  // the clock of the decayed counts: decode launches
    static const int pfs_hold = getenv("FLASH_PF_STAGE_HOLD") ? atoi(getenv("FLASH_PF_STAGE_HOLD")) : 64;
    if (kind == 0 && pfs_borrowed_ && ++pfs_idle_ > pfs_hold) pfs_release();  // no prefill for a while: ring back to the cache
    // copies that landed: switch the table entries to the device slots
    for (auto& sl : ec_slots_)
        if (sl.ci >= 0 && ec_pend_[sl.ci] && cudaEventQuery(sl.ev) == cudaSuccess) {
            ec_pend_[sl.ci] = 0;
            ec_tab_h_[sl.ci] = (uintptr_t)sl.addr;
            ec_dirty_ = true;
        }
    // KV pool first: slots overlapping the rows of positions < ctx_end (+ margin) go back to the pool
    static const int margin = getenv("FLASH_EC_MARGIN") ? atoi(getenv("FLASH_EC_MARGIN")) : 256;  // rows
    // FLASH_EC_RESERVE=n: keep the slots of an n-token context reserved (measurement: the cache capacity of a long
    // context with a short prompt)
    static const long long reserve = getenv("FLASH_EC_RESERVE") ? atoll(getenv("FLASH_EC_RESERVE")) : 0;
    const long long c = std::min<long long>((long long)max_ctx_, std::max<long long>(ctx_end, reserve));
    const long long rws = std::min<long long>(rows_, (c + p_.period - 1) / p_.period * p_.own + margin);
    const size_t bound = (size_t)rws * NKV * HD;
    std::vector<int> killed;
    std::vector<std::pair<int, int>> copies;  // (slot, group)
    if (bound != ec_bound_ || bt_want_ != bt_taken_) {
        for (int s2 = 0; s2 < (int)ec_slots_.size(); s2++) {
            EcSlot& sl = ec_slots_[s2];
            const bool live = sl.bt ? !bt_want_ : sl.off >= bound;
            if (sl.live && !live) {
                if (sl.bt) sl.prev = sl.ci;
                if (sl.ci >= 0) {
                    if (ec_pend_[sl.ci]) { CUDA_CHECK(cudaStreamWaitEvent(st_, sl.ev, 0)); ec_pend_[sl.ci] = 0; ec_st_.ev_kv_pend++; }
                    ec_tab_h_[sl.ci] = ec_host_[sl.ci] | 1;
                    ec_slot_of_[sl.ci] = -1;
                    sl.ci = -1;
                    ec_dirty_ = true;
                    ec_st_.ev_kv++;
                }
                if (!sl.bt) killed.push_back(s2);  // (a borrowed tail is overwritten by the batch: no zeroing)
            } else if (!sl.live && live) {
                ec_st_.revived++;
                // a borrowed tail comes back: its groups are copied in again right away (the cache as before the batch)
                if (sl.bt && sl.prev >= 0 && ec_slot_of_[sl.prev] < 0 && !ec_pend_[sl.prev]) {
                    sl.ci = sl.prev;
                    ec_slot_of_[sl.ci] = s2;
                    ec_pend_[sl.ci] = 1;
                    copies.push_back({s2, sl.ci});
                }
                sl.prev = -1;
            }
            sl.live = live;
        }
        ec_bound_ = bound;
        bt_taken_ = bt_want_;
    }
    // admissions: candidates by decayed count (desc), into free live slots, then over the coldest residents
    const size_t nrest = copies.size();
    if (!ec_cand_.empty() && ec_maxcopy_ > 0) {
        std::sort(ec_cand_.begin(), ec_cand_.end(), [&](int a, int b) { return ec_ls_[a] > ec_ls_[b]; });
        std::vector<int> fr, vic;
        for (int s2 = 0; s2 < (int)ec_slots_.size(); s2++) {
            const EcSlot& sl = ec_slots_[s2];
            if (!sl.live || sl.pf) continue;
            if (sl.ci < 0) fr.push_back(s2);
            else if (!ec_pend_[sl.ci]) vic.push_back(s2);
        }
        // FLASH_EC_LOWFIRST=1: stress test, free slots lowest offset first (the next ones a growing context reclaims)
        static const bool lowfirst = getenv("FLASH_EC_LOWFIRST") && atoi(getenv("FLASH_EC_LOWFIRST"));
        if (lowfirst) std::reverse(fr.begin(), fr.end());
        size_t fi = 0, vi = 0;
        bool vsorted = false;
        for (int ci : ec_cand_) {
            if ((int)(copies.size() - nrest) >= ec_maxcopy_) break;
            int s2 = -1;
            if (fi < fr.size()) {
                s2 = fr[fi++];
            } else {
                if (!vsorted) {
                    const size_t k = std::min<size_t>(vic.size(), ec_maxcopy_);
                    std::partial_sort(vic.begin(), vic.begin() + k, vic.end(),
                                      [&](int a, int b) { return ec_ls_[ec_slots_[a].ci] < ec_ls_[ec_slots_[b].ci]; });
                    vsorted = true;
                }
                if (vi >= vic.size()) break;
                const int v = ec_slots_[vic[vi]].ci;
                const double sv = ec_ls_[v] > -1e299 ? std::exp(ec_ls_[v] + ec_now_ * ec_logd_) : 0.0;
                const double sc = std::exp(ec_ls_[ci] + ec_now_ * ec_logd_);
                if (sc <= sv + ec_hyst_) break;  // candidates only get colder, victims only hotter
                s2 = vic[vi++];
                ec_tab_h_[v] = ec_host_[v] | 1;
                ec_slot_of_[v] = -1;
                ec_st_.ev_freq++;
                ec_dirty_ = true;
            }
            EcSlot& sl = ec_slots_[s2];
            sl.ci = ci;
            ec_slot_of_[ci] = s2;
            ec_pend_[ci] = 1;
            copies.push_back({s2, ci});
        }
    }
    ec_cand_.clear();
    if (ec_dirty_) {
        CUDA_CHECK(cudaMemcpyAsync(ec_tab_d_, ec_tab_h_, (size_t)ec_n_ * sizeof(uintptr_t), cudaMemcpyHostToDevice, st_));
        ec_dirty_ = false;
    }
    for (int s2 : killed) CUDA_CHECK(cudaMemsetAsync(ec_slots_[s2].addr, 0, GRP_BYTES, st_));
    if (!copies.empty()) {
        // the copies start after everything queued so far on the work stream (readers of evicted slots, table update)
        CUDA_CHECK(cudaEventRecord(ec_evt_, st_));
        CUDA_CHECK(cudaStreamWaitEvent(ec_cs_, ec_evt_, 0));
        // FLASH_EC_DELAY=ns: stress test, the copies land late (slot reclaim / pointer switch vs in-flight copies)
        static const unsigned delay = getenv("FLASH_EC_DELAY") ? (unsigned)atoi(getenv("FLASH_EC_DELAY")) : 0u;
        if (delay) k_spin<<<1, 1, 0, ec_cs_>>>(delay);
        for (auto& [s2, ci] : copies) {
            CUDA_CHECK(cudaMemcpyAsync(ec_slots_[s2].addr, ec_src_[ci], GRP_BYTES, cudaMemcpyHostToDevice, ec_cs_));
            CUDA_CHECK(cudaEventRecord(ec_slots_[s2].ev, ec_cs_));
        }
        ec_st_.copies += copies.size();
    }
}

const Worker::EcStats& Worker::ec_stats() {
    ec_collect();
    int res = 0, live = 0;
    for (auto& sl : ec_slots_) { live += sl.live; res += sl.live && sl.ci >= 0 && !ec_pend_[sl.ci]; }
    ec_st_.resident = res;
    ec_st_.live_slots = live;
    return ec_st_;
}

// FLASH_JOBCHK: the job builders write every job routing produces, the MoE kernels run `bound` blocks: report a count
// above the launched grid (those jobs would be skipped) or above the job buffer
__global__ void k_jobtrap(const int* njobs, int bound) {
    if (*njobs > bound) { printf("[flash] FATAL: prefill MoE job list %d > grid %d (expert work would be skipped)\n", *njobs, bound); __trap(); }
}
__global__ void k_jobchk(const int* njobs, int bound, int cap, int tag, int m, unsigned* flag) {
    const int n = *njobs;
    if (n > bound || n > cap) {
        if (atomicAdd(flag, 1u) < 8) printf("[jobchk] tag %d m %d: %d jobs > grid %d / buffer %d\n", tag, m, n, bound, cap);
    }
    // the largest count so far (per tag): flag[1 + tag] = max njobs, flag[4 + tag] = the bound of that call
    if (n > (int)flag[1 + tag]) { flag[1 + tag] = (unsigned)n; flag[4 + tag] = (unsigned)bound; }
}
// FLASH_EC_VERIFY: slot / ring contents vs their host groups (k_csum formula on both sides)
static unsigned long long host_csum(const char* p, size_t bytes) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
    unsigned long long s = 0;
    for (size_t i = 0; i < bytes / 4; i++) s += (unsigned long long)w[i] * (1 + (i & 255));
    return s;
}
int Worker::ec_verify(const char* where) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaDeviceSynchronize());
    struct Item { const char* dev; const char* host; int ci; int slot; int li; const char* kind; };
    std::vector<Item> it;
    if (ec_hsum_.size() != ec_src_.size()) ec_hsum_.assign(ec_src_.size(), 0);
    auto layer_of = [&](int ci) { for (int li = 0; li < LAYERS; li++) if (ci >= L_[li].coff && ci < L_[li].coff + L_[li].ncold) return li; return -1; };
    for (int s2 = 0; s2 < (int)ec_slots_.size(); s2++) {
        const EcSlot& sl = ec_slots_[s2];
        if (!sl.live || sl.pf || sl.ci < 0 || ec_pend_[sl.ci]) continue;
        if (ec_tab_h_[sl.ci] != (uintptr_t)sl.addr) continue;  // the table does not point here (yet)
        it.push_back({sl.addr, ec_src_[sl.ci], sl.ci, s2, layer_of(sl.ci), sl.bt ? "cache(bt tail)" : "cache"});
    }
    if (pfs_mode_ == 1 && pfs_ring_)
        for (int li = LAYERS - 2; li < LAYERS; li++) {
            const Layer& L = L_[li];
            for (int j = 0; j < std::min(L.ncold, pfs_cap_); j++)
                it.push_back({pfs_ring_ + ((size_t)(li & 1) * pfs_cap_ + j) * GRP_BYTES, L.hgroups_h + (size_t)j * GRP_BYTES, -1, j, li, "staging ring"});
        }
    if (it.empty()) return 0;
    if (!ec_vsum_d_) ec_vsum_d_ = (unsigned long long*)dalloc((ec_slots_.size() + 2 * (size_t)std::max(1, pfs_cap_)) * 8);
    CUDA_CHECK(cudaMemsetAsync(ec_vsum_d_, 0, it.size() * 8, st_));  // (all on st_: a non-blocking stream)
    for (size_t k = 0; k < it.size(); k++)
        k_csum<<<64, 256, 0, st_>>>(reinterpret_cast<const uint32_t*>(it[k].dev), GRP_BYTES / 4, ec_vsum_d_ + k);
    std::vector<unsigned long long> d(it.size());
    CUDA_CHECK(cudaMemcpyAsync(d.data(), ec_vsum_d_, it.size() * 8, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    int bad = 0, bad_ring = 0, bad_cache = 0;
    for (size_t k = 0; k < it.size(); k++) {
        unsigned long long h;
        if (it[k].ci >= 0) { if (!ec_hsum_[it[k].ci]) ec_hsum_[it[k].ci] = host_csum(it[k].host, GRP_BYTES); h = ec_hsum_[it[k].ci]; }
        else h = host_csum(it[k].host, GRP_BYTES);
        if (h != d[k]) {
            if (it[k].ci < 0) bad_ring++; else bad_cache++;
            if (bad == 0) {  // details of the first mismatch: zero words and the first differing word
                std::vector<uint32_t> dv(GRP_BYTES / 4);
                CUDA_CHECK(cudaMemcpy(dv.data(), it[k].dev, GRP_BYTES, cudaMemcpyDeviceToHost));
                const uint32_t* hv = reinterpret_cast<const uint32_t*>(it[k].host);
                size_t zd = 0, zh = 0, first = (size_t)-1, ndiff = 0;
                for (size_t w = 0; w < dv.size(); w++) {
                    zd += dv[w] == 0; zh += hv[w] == 0;
                    if (dv[w] != hv[w]) { ndiff++; if (first == (size_t)-1) first = w; }
                }
                fprintf(stderr, "[ec_verify] rank %d first mismatch (%s slot %d group %d layer %d): %zu of %zu words differ, first at word %zu (dev %08x host %08x); zero words dev %zu host %zu\n",
                        p_.rank, it[k].kind, it[k].slot, it[k].ci, it[k].li, ndiff, dv.size(), first, dv[first], hv[first], zd, zh);
            }
            if (bad < 4) fprintf(stderr, "[ec_verify] rank %d %s: MISMATCH %s slot %d group %d layer %d\n", p_.rank, where, it[k].kind, it[k].slot, it[k].ci, it[k].li);
            bad++;
        }
    }
    fprintf(stderr, "[ec_verify] rank %d %s: %zu groups checked, %d mismatches (ring %d, cache %d); ring cap %d, layer 46/47 ncold %d/%d\n", p_.rank, where,
            it.size(), bad, bad_ring, bad_cache, pfs_cap_, L_[LAYERS - 2].ncold, L_[LAYERS - 1].ncold);
    return bad;
}
// FLASH_GSUM: per-group sums of nb groups at stride GRP_BYTES (k_csum weights, word index within the group)
__global__ void k_gsum(const uint32_t* __restrict__ g, unsigned long long* __restrict__ out) {
    const uint32_t* p = g + (size_t)blockIdx.x * (GRP_BYTES / 4);
    unsigned long long s = 0;
    for (size_t i = threadIdx.x; i < GRP_BYTES / 4; i += blockDim.x) s += (unsigned long long)p[i] * (1 + (i & 255));
    __shared__ unsigned long long sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if ((int)threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}
static unsigned long long gmix(unsigned long long h, unsigned long long id) {
    unsigned long long x = h ^ (id * 0x9E3779B97F4A7C15ull);
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
    return x;
}
void Worker::gsum(int nw, unsigned long long* experts, unsigned long long* other) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long ex = 0;
    std::vector<Layer*> ls;
    for (auto& L : L_) ls.push_back(&L);
    if (has_mtp_) ls.push_back(&mtp_);
    std::vector<const void*> gbase;
    for (size_t k = 0; k < ls.size(); k++) {
        Layer& L = *ls[k];
        if (!L.gmap) continue;
        std::vector<int> gm((NE + 1) * NGRP);
        CUDA_CHECK(cudaMemcpy(gm.data(), L.gmap, gm.size() * 4, cudaMemcpyDeviceToHost));
        const int nd = std::min(L.ndev, 1 << 20);
        int maxdev = 0;
        for (int v : gm) if (v >= 0 && v < nd) maxdev = std::max(maxdev, v + 1);
        std::vector<unsigned long long> dsum(maxdev);
        if (maxdev) {
            unsigned long long* d;
            CUDA_CHECK(cudaMalloc(&d, (size_t)maxdev * 8));
            k_gsum<<<maxdev, 256, 0, st_>>>(reinterpret_cast<const uint32_t*>(L.groups), d);
            CUDA_CHECK(cudaMemcpyAsync(dsum.data(), d, (size_t)maxdev * 8, cudaMemcpyDeviceToHost, st_));
            CUDA_CHECK(cudaStreamSynchronize(st_));
            CUDA_CHECK(cudaFree(d));
            gbase.push_back(L.groups);
        }
        std::vector<int> cold;
        for (int i = 0; i < (int)gm.size(); i++) if (gm[i] >= nd) cold.push_back(i);
        std::vector<unsigned long long> hs(cold.size());
        std::vector<std::thread> th;
        for (int t = 0; t < 8; t++)
            th.emplace_back([&, t] {
                for (size_t q = t; q < cold.size(); q += 8)
                    hs[q] = host_csum(L.hgroups_h + (size_t)(gm[cold[q]] - nd) * GRP_BYTES, GRP_BYTES);
            });
        for (auto& x : th) x.join();
        size_t ci = 0;
        for (int i = 0; i < (int)gm.size(); i++) {
            if (gm[i] < 0) continue;
            const unsigned long long h = gm[i] < nd ? dsum[gm[i]] : hs[ci++];
            ex += gmix(h, ((unsigned long long)k << 20) | (unsigned)i);
        }
    }
    unsigned long long ot = 0;
    {
        unsigned long long* d;
        CUDA_CHECK(cudaMalloc(&d, 8));
        CUDA_CHECK(cudaMemsetAsync(d, 0, 8, st_));
        static const int gs2 = getenv("FLASH_GSUM") ? atoi(getenv("FLASH_GSUM")) : 0;
        FILE* fa = nullptr;
        if (gs2 >= 2) { char fn[256]; snprintf(fn, sizeof(fn), "%s.rank%d", getenv("FLASH_GSUM_OUT") ? getenv("FLASH_GSUM_OUT") : "gsum", p_.rank); fa = fopen(fn, "w"); }
        for (int i = 0; i < nw && i < (int)allocs_.size(); i++) {
            if (std::find(gbase.begin(), gbase.end(), allocs_[i].first) != gbase.end()) continue;  // the expert group arrays
            if (fa) {  // per allocation (index, bytes, sum) for a diff between processes
                unsigned long long* d1;
                CUDA_CHECK(cudaMalloc(&d1, 8));
                CUDA_CHECK(cudaMemsetAsync(d1, 0, 8, st_));
                k_csum<<<256, 256, 0, st_>>>((const uint32_t*)allocs_[i].first, allocs_[i].second / 4, d1);
                unsigned long long v;
                CUDA_CHECK(cudaMemcpyAsync(&v, d1, 8, cudaMemcpyDeviceToHost, st_));
                CUDA_CHECK(cudaStreamSynchronize(st_));
                CUDA_CHECK(cudaFree(d1));
                fprintf(fa, "%d %zu %016llx\n", i, allocs_[i].second, v);
            }
            k_csum<<<256, 256, 0, st_>>>((const uint32_t*)allocs_[i].first, allocs_[i].second / 4, d);
        }
        if (fa) fclose(fa);
        CUDA_CHECK(cudaMemcpyAsync(&ot, d, 8, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        CUDA_CHECK(cudaFree(d));
    }
    *experts = ex;
    *other = ot;
}
void Worker::ec_reset_stats() {
    ec_collect();
    EcStats z;
    z.ncold = ec_st_.ncold;
    z.slots = ec_st_.slots;
    z.bt_slots = ec_st_.bt_slots;
    z.seed_gbs = ec_st_.seed_gbs;
    ec_st_ = z;
}

void Worker::get_stats(std::vector<int>& out) {
    CUDA_CHECK(cudaSetDevice(dev_));
    out.assign((size_t)(LAYERS + 1) * NE, 0);
    for (int l = 0; l <= LAYERS; l++) {
        const Layer& L = l < LAYERS ? L_[l] : mtp_;
        if (L.stats) CUDA_CHECK(cudaMemcpy(out.data() + (size_t)l * NE, L.stats, NE * 4, cudaMemcpyDeviceToHost));
    }
}

// debug: the pooled indexer keys of the first nb blocks of every QSA layer (+ MTP) of this card -> file
void Worker::dump_ikb(const char* path, int nb) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    nb = std::min(nb, rows_ / IDX_R);
    FILE* f = fopen(path, "wb");
    std::vector<bf16> h((size_t)nb * IDX_HD);
    for (int l = 0; l <= LAYERS; l++) {
        const Layer& L = l < LAYERS ? L_[l] : mtp_;
        if (!L.ikb) continue;
        CUDA_CHECK(cudaMemcpy(h.data(), L.ikb, h.size() * 2, cudaMemcpyDeviceToHost));
        fwrite(h.data(), 2, h.size(), f);
    }
    fclose(f);
}

void Worker::reset() {
    CUDA_CHECK(cudaSetDevice(dev_));
    const int vl = p_.v1 - p_.v0, kl = p_.k1 - p_.k0;
    const int ch = 2 * kl * DK + vl * DV;
    for (auto& L : L_) {
        if (L.S) CUDA_CHECK(cudaMemsetAsync(L.S, 0, (size_t)vl * DK * DV * 4, st_));
        if (L.conv_hist) CUDA_CHECK(cudaMemsetAsync(L.conv_hist, 0, (size_t)3 * ch * 4, st_));
        if (L.ple_hist) CUDA_CHECK(cudaMemsetAsync(L.ple_hist, 0, (size_t)9 * RW * 4, st_));
        if (L.ik_tail) CUDA_CHECK(cudaMemsetAsync(L.ik_tail, 0, IDX_TAIL * IDX_HD * 4, st_));
    }
    if (mtp_.ik_tail) CUDA_CHECK(cudaMemsetAsync(mtp_.ik_tail, 0, IDX_TAIL * IDX_HD * 4, st_));
    prev_m_ = 0;
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

// ---------------------------------------------------------------- activation steering (QE_STEER)
void Worker::set_steer(int nv, const int* flo, const int* fhi, const float* w) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CHECK(!steer_w_, "flash: steering vectors set twice");
    CHECK(nv >= 1, "flash: no steering vectors");
    const int ns = std::max(1, (int)slots_.size());
    steer_nv_ = nv;
    for (int v = 0; v < nv; v++) {
        CHECK(flo[v] >= 0 && fhi[v] < LAYERS && flo[v] <= fhi[v], "flash: steering vector %d: layers %d..%d", v, flo[v], fhi[v]);
        steer_lo_ = std::min(steer_lo_, flo[v]);
        steer_hi_ = std::max(steer_hi_, fhi[v]);
    }
    steer_lo_v_.assign(flo, flo + nv);
    steer_hi_v_.assign(fhi, fhi + nv);
    const size_t wn = (size_t)nv * 2 * LAYERS * H;
    steer_w_ = (float*)dalloc(wn * 4);
    CUDA_CHECK(cudaMemcpy(steer_w_, w, wn * 4, cudaMemcpyHostToDevice));
    steer_sl_d_ = (int*)dalloc((size_t)ns * 4 * 4);
    steer_sc_d_ = (float*)dalloc((size_t)ns * 4);
    steer_csl_d_ = (int*)dalloc(4 * 4);
    steer_csc_d_ = (float*)dalloc(4);
    steer_sl_h_.assign((size_t)ns * 4, 0);
    for (int s = 0; s < ns; s++) steer_sl_h_[(size_t)s * 4] = -1;
    steer_sc_h_.assign(ns, 0.f);
    CUDA_CHECK(cudaMemcpy(steer_sl_d_, steer_sl_h_.data(), (size_t)ns * 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(steer_csl_d_, steer_sl_h_.data(), 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(steer_sc_d_, steer_sc_h_.data(), (size_t)ns * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(steer_csc_d_, steer_sc_h_.data(), 4, cudaMemcpyHostToDevice));
    // fused-path corrections: dw[v][mode][li] = the fused down weights of layer li + 1's attention HC read applied to
    // w[v][mode][li] on each stream, by the decode down kernel itself (row s of a 4-row input = w in stream s)
    steer_dw_ = (float*)dalloc((size_t)nv * 2 * LAYERS * HC * HCD_ROWS * 4);
    float* tmp = (float*)dalloc((size_t)HC * RW * 4);
    for (int vm = 0; vm < 2 * nv; vm++) {
        const int v = vm / 2;
        for (int li = std::max(1, flo[v]); li <= std::min(fhi[v], LAYERS - 2); li++) {
            const HcW& hw = L_[li + 1].hca;
            if (!hw.uq || !hcp_) continue;
            CUDA_CHECK(cudaMemsetAsync(tmp, 0, (size_t)HC * RW * 4, st_));
            for (int s2 = 0; s2 < HC; s2++)
                CUDA_CHECK(cudaMemcpyAsync(tmp + (size_t)s2 * RW + (size_t)s2 * H, steer_w_ + ((size_t)vm * LAYERS + li) * H, (size_t)H * 4,
                                           cudaMemcpyDeviceToDevice, st_));
            NT_LAUNCH(HC, k_hcd_s, dim3(HCD_ROWS / 16, HC * HCD_GPS / HCD_GPW / 4), 128, hw.fd.q, hw.fd.s, tmp, hcp_, HC,
                      (const float*)nullptr, (const float*)nullptr);
            k_steer_dw<<<HC, 384, 0, st_>>>(hcp_, steer_dw_ + (((size_t)vm * LAYERS + li) * HC) * HCD_ROWS);
            KCHK("steer dw");
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void Worker::set_steer_slot(int s, int vec, int mode, int lo, int hi, float scale) {
    if (!steer_w_) return;
    CHECK(vec >= -1 && vec < steer_nv_, "flash: steering vector %d of %d", vec, steer_nv_);
    if (vec >= 0)
        CHECK((mode == 0 || mode == 1) && lo >= steer_lo_v_[vec] && hi <= steer_hi_v_[vec] && lo <= hi,
              "flash: steering vector %d: mode %d layers %d..%d (vector layers %d..%d)", vec, mode, lo, hi, steer_lo_v_[vec], steer_hi_v_[vec]);
    CUDA_CHECK(cudaSetDevice(dev_));
    int* q = &steer_sl_h_.at((size_t)s * 4);
    q[0] = vec;
    q[1] = vec >= 0 ? mode : 0;
    q[2] = vec >= 0 ? lo : 0;
    q[3] = vec >= 0 ? hi : -1;
    steer_sc_h_.at(s) = vec >= 0 ? scale : 0.f;
    CUDA_CHECK(cudaStreamSynchronize(st_));  // no launch in flight reads them
    CUDA_CHECK(cudaMemcpy(steer_sl_d_ + (size_t)s * 4, q, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(steer_sc_d_ + s, &steer_sc_h_[s], 4, cudaMemcpyHostToDevice));
    if (s == cur_slot_) {
        CUDA_CHECK(cudaMemcpy(steer_csl_d_, q, 16, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(steer_csc_d_, &steer_sc_h_[s], 4, cudaMemcpyHostToDevice));
    }
}

SteerArg Worker::steer_arg(int li, const int* rslot) const {
    SteerArg a;
    a.w = steer_w_;
    a.dw = steer_dw_;
    a.sl = rslot ? steer_sl_d_ : steer_csl_d_;
    a.sc = rslot ? steer_sc_d_ : steer_csc_d_;
    a.rslot = rslot;
    a.li = li;
    return a;
}

void Worker::steer(int li, int m, const int* rslot, cudaStream_t s, float* R, const float* inj, float* y) {
    if (!steer_w_ || li < steer_lo_ || li > steer_hi_ || m <= 0) return;
    k_steer<<<m, STEER_T, 0, s>>>(R, inj, y, steer_arg(li, rslot));
    KCHK("k_steer");
}

void Worker::cap_begin() {
    CUDA_CHECK(cudaSetDevice(dev_));
    if (!cap_buf_) cap_buf_ = (float*)dalloc((size_t)LAYERS * H * 4);
    CUDA_CHECK(cudaMemset(cap_buf_, 0, (size_t)LAYERS * H * 4));
    cap_ = cap_buf_;
    cap_skip_ = 0;
}
void Worker::cap_end(float* host) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    if (host) CUDA_CHECK(cudaMemcpy(host, cap_buf_, (size_t)LAYERS * H * 4, cudaMemcpyDeviceToHost));
    cap_ = nullptr;
}

void Worker::steer_flush(int m) {
    if (steer_pend_ < 0) return;
    const int li = steer_pend_;
    steer_pend_ = -1;
    steer(li, m, steer_prs_, st_, R_, inj_, ypend_ ? y_ : nullptr);
}

// ---------------------------------------------------------------- forward pieces
void Worker::flush_y(int m) {  // apply a pending residual update (R_i += inj_i * y) explicitly
    steer_flush(m);  // a pending layer-output steering first (it applies the update of its rows itself)
    if (!ypend_) return;
    k_hc_write<<<m, 512, 0, st_>>>(R_, inj_, y_);
    KCHK("k_hc_write");
    ypend_ = false;
}

void Worker::hc_read(const HcW& w, int m, bool inject) {
    static const bool fuse = getenv("FLASH_HCW") && atoi(getenv("FLASH_HCW"));  // measured slower (extra y loads in k_hcd)
    const bool hcs = hcs_ && w.uq && m <= MAXM;  // split decode HC down: the pending residual update is fused in
    SteerArg sfa;  // a pending layer-output steering fused into the mid kernel, else applied separately first
    if (steer_pend_ >= 0) {
        static const bool fuse = !getenv("FLASH_STEER_FUSE") || atoi(getenv("FLASH_STEER_FUSE"));  // 0: separate kernel (check)
        if (fuse && hcs && steer_pend_ >= 1 && steer_pend_ + 1 < LAYERS && &w == &L_[steer_pend_ + 1].hca) {
            sfa = steer_arg(steer_pend_, steer_prs_);
            steer_pend_ = -1;
        } else {
            steer_flush(m);
        }
    }
    if (!hcs && (m > MAXM || !fuse || !R2_)) flush_y(m);
    if (w.uq && m > MAXM) {
        static const int v2 = getenv("FLASH_HCPF") ? atoi(getenv("FLASH_HCPF")) : 2;
        if (v2 == 2) {  // staged activations + K-split atomics, separate mid step
            if (detpart_) {  // deterministic: exact fixed-point K-split sums (the buffers are zeroed by the conversion)
                unsigned long long* fu = detpart_;
                unsigned long long* fs = detpart_ + (size_t)m * HC * HCD_ROWS;
                k_hcd_pf2<<<dim3(3, HC * 4, (m + 31) / 32), 224, 0, st_>>>(w.fd.q, w.fd.s, R_, hu_, m, hss_, fu, fs);
                k_fx_to_f<<<256, 256, 0, st_>>>(fu, hu_, (size_t)m * HC * HCD_ROWS);
                k_fx_to_f<<<64, 256, 0, st_>>>(fs, hss_, (size_t)m * HC);
            } else {
            CUDA_CHECK(cudaMemsetAsync(hu_, 0, (size_t)m * HC * HCD_ROWS * 4, st_));
            CUDA_CHECK(cudaMemsetAsync(hss_, 0, (size_t)m * HC * 4, st_));
            k_hcd_pf2<<<dim3(3, HC * 4, (m + 31) / 32), 224, 0, st_>>>(w.fd.q, w.fd.s, R_, hu_, m, hss_);  // + sums of squares
            }
            k_hc_mid_pf<<<m, 352, 0, st_>>>(hu_, hss_, hh_pf_, rinv_pf_, inject ? inj_ : nullptr);
            static const int hy = getenv("FLASH_HCU_Y") ? atoi(getenv("FLASH_HCU_Y")) : 1;  // blocks over 32-row groups (1: each block loops, cp.async one group ahead)
            k_hcu_pf2<<<dim3((HCU_TILES + 7) / 8, std::max(1, std::min(hy, (m + 31) / 32))), 256, 0, st_>>>(w.uq, w.us, hh_pf_, rinv_pf_, R_, w.norm, x_, xh_, m);
            KCHK("hc pf2");
            return;
        }
        k_hcd_pf<<<dim3(HCD_ROWS / 16, HC), 256, 0, st_>>>(w.fd.q, w.fd.s, R_, hu_, hss_, m);
        KCHK("k_hcd_pf");
        k_hcu_pf<<<(HCU_TILES + 7) / 8, 256, 0, st_>>>(w.uq, w.us, hu_, hss_, R_, w.norm, inject ? inj_ : nullptr, x_, xh_, m);
        KCHK("k_hcu_pf");
        return;
    }
    if (w.uq) {
        if (hcs) {  // split by groups (no wave tail) + one mid kernel (h, rinv, inj once, not per hcu block)
            const float* yp = ypend_ ? y_ : nullptr;
            static const int hcx = getenv("FLASH_HCDX") ? atoi(getenv("FLASH_HCDX")) : 9;  // staged-x HC down from m rows (0: off)
            if (hcx > 0 && m >= hcx)
                NT_LAUNCH(m, k_hcd_x, dim3(HC * HCD_GPS / HCD_GPW, HCD_ROWS / 16 / HCDX_TW), 32 * HCDX_TW, w.fd.q, w.fd.s, R_, hcp_, m, yp, inj_);
            else
            NT_LAUNCH(m, k_hcd_s, dim3(HCD_ROWS / 16, HC * HCD_GPS / HCD_GPW / 4), 128, w.fd.q, w.fd.s, R_, hcp_, m, yp, inj_);
            const bool l2u = (hcl2_ & 2) && !pf_ && m <= 8;  // measured at m = 4 (MTP k = 3) only
            if (l2u) {  // HC up weights into L2 while the (latency-bound) mid kernel runs
                CUDA_CHECK(cudaEventRecord(pfe0_, st_));
                CUDA_CHECK(cudaStreamWaitEvent(pfs_, pfe0_, 0));
                prefetch_l2(w.uq, (size_t)HCU_TILES * (HCU_KS / 2) * 32 * 16, pfs_);
                CUDA_CHECK(cudaEventRecord(pfe1_, pfs_));
            }
            k_hc_mid_dec<<<m, 512, 0, st_>>>(hcp_, R_, hh_pf_, rinv_pf_, inject ? inj_ : nullptr, yp, inj_, sfa);
            if (l2u) CUDA_CHECK(cudaStreamWaitEvent(st_, pfe1_, 0));
            ypend_ = false;
            KCHK("k_hcd_s");
            NT_LAUNCH(m, k_hcu, (HCU_TILES + 7) / 8, 256, w.uq, w.us, hu_, hss_, R_, w.norm, nullptr, x_, xh_, m,
                      const_cast<float*>(yp), hh_pf_, rinv_pf_);  // y consumed: cleared for the next all-reduce
            KCHK("k_hcu");
            return;
        } else if (ypend_) {  // fused residual update of the previous block (R_ -> R2_), then swap
            NT_LAUNCH(m, k_hcd, dim3(HCD_ROWS / 16, HC), 256, w.fd.q, w.fd.s, R_, hu_, hss_, m, y_, inj_, R2_);
            std::swap(R_, R2_);
        } else {
            NT_LAUNCH(m, k_hcd, dim3(HCD_ROWS / 16, HC), 256, w.fd.q, w.fd.s, R_, hu_, hss_, m);
        }
        KCHK("k_hcd");
        NT_LAUNCH(m, k_hcu, (HCU_TILES + 7) / 8, 256, w.uq, w.us, hu_, hss_, R_, w.norm, inject ? inj_ : nullptr, x_, xh_, m,
                  ypend_ ? y_ : nullptr);
        KCHK("k_hcu");
        ypend_ = false;
        return;
    }
    k_hc_norm<<<dim3(HC, m), 256, 0, st_>>>(R_, w.norm, hn_);
    KCHK("k_hc_norm");
    if (w.down16.w) gemv_w16(w.down16, hn_, RW, z_, HCD_ROWS, m, st_);
    else gemv_w8(w.down, hn_, RW, z_, HCD_ROWS, m, st_);
    k_hc_mid<<<m, 384, 0, st_>>>(z_, zb_, inj_, m);
    KCHK("k_hc_mid");
    if (w.up16.w) gemv_w16(w.up16, zb_, HCU_K, gates_, RW, m, st_);
    else gemv_w8(w.up, zb_, HCU_K, gates_, RW, m, st_);
    k_hc_mix<<<m, 512, 0, st_>>>(gates_, hn_, x_, xh_);
    KCHK("k_hc_mix");
    (void)inject;
}

void Worker::hc_write(int m) {
    // y = sum of the cards' partials, then R_i += inj_i * y
    static const bool ll = !getenv("FLASH_LL") || atoi(getenv("FLASH_LL"));
    KCHK("before AR");
    const bool l2d = (hcl2_ & 1) && !pf_ && next_hc_ && next_hc_->fd.q && m <= MAXM;
    if (l2d) {  // the next HC read's down weights into L2 while the (latency-bound) all-reduce runs
        CUDA_CHECK(cudaEventRecord(pfe0_, st_));
        CUDA_CHECK(cudaStreamWaitEvent(pfs_, pfe0_, 0));
        prefetch_l2(next_hc_->fd.q, (size_t)next_hc_->fd.n * next_hc_->fd.k, pfs_);
        CUDA_CHECK(cudaEventRecord(pfe1_, pfs_));
    }
    allreduce_add(y_, part_, m, H, comm_, p_.rank, ctr_, st_, nullptr, 0.f, nullptr, ll && m <= kMaxBatchRows ? ll_idx_++ : -1);
    if (l2d) CUDA_CHECK(cudaStreamWaitEvent(st_, pfe1_, 0));
    next_hc_ = nullptr;
    KCHK("AR");
    ypend_ = true;  // R_i += inj_i * y: applied by the next HC read (fused) or flush_y
}

void Worker::gdn(Layer& L, int li, int pos, int m) {
    (void)li; (void)pos;
    const int vl = p_.v1 - p_.v0, kl = p_.k1 - p_.k0;
    const int ch = 2 * kl * DK + vl * DV, ld = ch + vl * DV;
    static const int pfx = getenv("FLASH_PFX") ? atoi(getenv("FLASH_PFX")) : 0;  // debug: bit0 GDN, bit1 MoE pf paths off
    if (pf_ && !(pfx & 1)) {  // prefill: tensor-core GEMMs, rows committed at once
        gemm_w4(L.dn_in, x_, H, proj_, ld, m, st_);
        gemv_w16_rows(L.dn_ab, x_, H, proj2_, 2 * vl, m, st_);
        float* cv = qbuf_;
        if (m >= 16) {  // rows in parallel, then the new history
            k_dn_conv_par<<<dim3((ch + 255) / 256, (m + 15) / 16), 256, 0, st_>>>(proj_, ld, ch, L.conv_w, L.conv_hist, L.conv_pend,
                                                                             posd_ + 1, cv, m);
            k_dn_conv_hist<<<(ch + 255) / 256, 256, 0, st_>>>(proj_, ld, ch, L.conv_hist, L.conv_pend, posd_ + 1, m);
        } else
        k_dn_conv_d<<<(ch + 255) / 256, 256, 0, st_>>>(proj_, ld, ch, L.conv_w, L.conv_hist, L.conv_pend, posd_ + 1, cv, m, true);
        float* o = gates_;
        static const int cmin = getenv("FLASH_DN_CHUNK") ? atoi(getenv("FLASH_DN_CHUNK")) : 64;  // 0: sequential
        if (cmin > 0 && m >= cmin) {  // chunked (WY) delta rule on tensor cores (dn_chunk.cu of the 27B engine)
            // rows of a previous forward still pending: commit them first (m = 0: commit + state write only)
            k_dn_rec_d<<<dim3(vl, 16), 128, 0, st_>>>(cv, ch, kl, vl, p_.k0, p_.v0, proj2_, L.A, L.dtb, L.S, L.rec_pend,
                                                       posd_ + 1, stepc_, o, 0, true);
            ::DeltaShape s{};
            s.nk = kl;
            s.nv = vl;
            for (int h = 0; h < vl; h++) s.v2k[h] = (p_.v0 + h) / (LNV / LNK) - p_.k0;
            s.ab_pad = vl;
            s.ab_splits = 1;
            s.ab_rows = m;
            s.lazy_t = 0;
            k_dn_qk<<<dim3(m, kl + 1), 128, 0, st_>>>(cv, ch, kl, vl, p_.k0, p_.v0, proj2_, L.A, L.dtb, dnqk_, dngb_);
            ::dn_core_chunked(cv, ch, s, dnqk_, dngb_, L.S, 0, zero_d_, m, o, st_, 0);
        } else
        k_dn_rec_d<<<dim3(vl, 16), 128, 0, st_>>>(cv, ch, kl, vl, p_.k0, p_.v0, proj2_, L.A, L.dtb, L.S, L.rec_pend,
                                                   posd_ + 1, stepc_, o, m, true);
        k_dn_out<<<dim3(vl, m), 128, 0, st_>>>(o, proj_ + ch, ld, L.dn_norm, vl, nullptr, vl * DV, ab_);
        KCHK("dn pf");
        gemm_w4(L.dn_out, ab_, vl * DV, part_, H, m, st_);
        return;
    }
    static const bool abpe = !getenv("FLASH_DNAB_PAR") || atoi(getenv("FLASH_DNAB_PAR"));
    const bool abp = abpe && m <= 8;  // server agent c=8 (m = 32): 795 -> 642 tok/s with the side branch at every m
    if (abp) {  // the small a/b GEMV (2 blocks) on a side branch next to the in-projection
        CUDA_CHECK(cudaEventRecord(pfe0_, st_));
        CUDA_CHECK(cudaStreamWaitEvent(pfs_, pfe0_, 0));
        gemv_w16(L.dn_ab, x_, H, proj2_, 2 * vl, m, pfs_);
        CUDA_CHECK(cudaEventRecord(pfe1_, pfs_));
    }
    gemv_w4h(L.dn_in, xh_, H, proj_, ld, m, st_);
    if (!abp) gemv_w16(L.dn_ab, x_, H, proj2_, 2 * vl, m, st_);
    float* cv = qbuf_;
    k_dn_conv_d<<<(ch + 255) / 256, 256, 0, st_>>>(proj_, ld, ch, L.conv_w, L.conv_hist, L.conv_pend, posd_ + 1, cv, m);
    KCHK("k_dn_conv_d");
    if (abp) CUDA_CHECK(cudaStreamWaitEvent(st_, pfe1_, 0));
    float* o = gates_;  // [m][vl][128]
    k_dn_rec_d<<<dim3(vl, 16), 128, 0, st_>>>(cv, ch, kl, vl, p_.k0, p_.v0, proj2_, L.A, L.dtb, L.S, L.rec_pend,
                                               posd_ + 1, stepc_, o, m);
    KCHK("k_dn_rec_d");
    k_dn_out<<<dim3(vl, m), 128, 0, st_>>>(o, proj_ + ch, ld, L.dn_norm, vl, ah_, vl * DV);
    KCHK("k_dn_out");
    gemv_w4h(L.dn_out, ah_, vl * DV, part_, H, m, st_);
    KCHK("dn_out gemv");
}

void Worker::qsa(Layer& L, int li, int pos, int m) {
    (void)li; (void)pos;
    float* const proj0 = proj_;
    float* const ax0 = attn_x_;
    if (qsa_split_ && qsa_buf_) { proj_ = projB_; attn_x_ = attnB_; }
    struct Restore { Worker* w; float* p; float* a; ~Restore() { w->proj_ = p; w->attn_x_ = a; } } restore{this, proj0, ax0};
    if (qsa_split_) qpend_[qsa_buf_].L = nullptr;
    const int ql = p_.q1 - p_.q0;
    const int ld = L.qkv.n;
    // decode: the indexer projection + key prep on a graph side branch (only the block selection and later steps use
    // them), joined before the selection or at the end of the layer
    static const bool ipar = !getenv("FLASH_IDX_PAR") || atoi(getenv("FLASH_IDX_PAR"));
    const bool ip = ipar && !pf_ && m <= 8 && !dbgh_;
    cudaStream_t ist = ip ? pfs_ : st_;
    if (ip) {
        CUDA_CHECK(cudaEventRecord(pfe0_, st_));
        CUDA_CHECK(cudaStreamWaitEvent(pfs_, pfe0_, 0));
        gemv_w16_rows(L.idx, x_, H, proj2_, (IDX_NH + 1) * IDX_HD, m, ist);
        k_idx_prep<<<1, IDX_HD, 0, ist>>>(proj2_, (IDX_NH + 1) * IDX_HD, L.iq_norm, L.ik_norm, curpos_, gates_, L.ik_tail, L.ikb, p_, m, rbcur_);
        CUDA_CHECK(cudaEventRecord(pfe1_, pfs_));
    }
    bool ijoined = !ip;
    rows8(L.qkv, x_, H, proj_, ld, m, st_);
    if (!ip) gemv_w16_rows(L.idx, x_, H, proj2_, (IDX_NH + 1) * IDX_HD, m, st_);
    const int nqp = qsa_a_ ? NQ : ql, qh0 = qsa_a_ ? 0 : p_.q0;
    k_qsa_prep<<<dim3(nqp + 4, m), 256, 0, st_>>>(proj_, ld, L.q_norm, L.k_norm, curpos_, qbuf_, L.kc, L.vc, L.ks, L.vs, p_,
                                                   nqp, qh0, nullptr, rbcur_);
    KCHK("k_qsa_prep");
    unsigned long long* dh = (dbgh_ && pf_ && li >= 0 && li < LAYERS) ? dbgh_ + li * DBGK : nullptr;
    if (dh) {
        k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(x_), (size_t)m * H / 2, dh + 4);
        k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(proj_), (size_t)m * ld, dh + 5);
        k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(proj2_), (size_t)m * (IDX_NH + 1) * IDX_HD, dh + 6);
        k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(qbuf_), (size_t)m * (NQ + 4) * HD, dh + 7);
    }
    if (ip) {
    } else if (m > 8) {
        k_idx_prep_rows<<<m, IDX_HD, 0, st_>>>(proj2_, (IDX_NH + 1) * IDX_HD, L.iq_norm, L.ik_norm, curpos_, gates_, L.ik_tail, L.ikb, p_, rbcur_);
        k_idx_tail<<<1, IDX_HD, 0, st_>>>(proj2_, (IDX_NH + 1) * IDX_HD, curpos_, L.ik_tail, m);
    } else
        k_idx_prep<<<1, IDX_HD, 0, st_>>>(proj2_, (IDX_NH + 1) * IDX_HD, L.iq_norm, L.ik_norm, curpos_, gates_, L.ik_tail, L.ikb, p_, m, rbcur_);
    KCHK("k_idx_prep");
    if (m > MAXM) k_possub<<<1, 32, 0, st_>>>(possub_, curpos_, (m + MAXM - 1) / MAXM);
    static const bool p2p = !getenv("FLASH_XP2P") || atoi(getenv("FLASH_XP2P"));
    const int lbmax = rows_ / IDX_R;
    // attention over the chunk at once; with block selection in sub-chunks of MAXM rows (its buffers are MAXM rows)
    static const bool selsub = getenv("FLASH_SEL_SUB") && atoi(getenv("FLASH_SEL_SUB"));  // old: selection per 32 rows
    const int sub = sel_on_ && selsub ? MAXM : XROWS;
    for (int t0 = 0; t0 < m; t0 += sub) {
        const int mr = std::min(sub, m - t0);
        const int* pp = t0 > 0 ? possub_ + t0 / MAXM : curpos_;
        float* qb = qbuf_ + (size_t)t0 * (NQ + 4) * HD;
        static const bool tc_on = !getenv("FLASH_ATTN_TC") || atoi(getenv("FLASH_ATTN_TC")) == 2;
        static const bool x16_on = !getenv("FLASH_X16") || atoi(getenv("FLASH_X16"));  // fp16 exchanges (prefill)
        const bool x16 = tc_on && x16_on && !sel_on_ && mr > 8 && !qsa_a_;
        // the partial-output exchange does not depend on the attention kernel: fp16 also with block selection
        static const bool x16s_on = !getenv("FLASH_X16_SEL") || atoi(getenv("FLASH_X16_SEL"));
        const bool x16o = x16 || (x16s_on && x16_on && mr > 8 && !qsa_a_);
        if (qsa_a_) {
            k_qsa_attn<<<dim3(NQ, mr), 256, 0, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp, attn_x_, p_, rbcur_);
        } else {
            Xchg xq;
            for (int g = 0; g < p_.ngpu; g++) { xq.recv[g] = xqpr_[g]; xq.flags[g] = xqpf_[g]; }
            xq.seq = xqseq_;
            xq.done = xqdone_;
            xq.scnt = xqscnt_;
            {  // debug: FLASH_SPIN_RANK=r,ns delays rank r before each attention exchange (cross-card skew stress)
                static int sr = -1, sns = 0, sph = 0;
                static bool init = false;
                if (!init) { init = true; if (const char* e = getenv("FLASH_SPIN_RANK")) sscanf(e, "%d,%d,%d", &sr, &sns, &sph); }
                if (sr == p_.rank && sns > 0) k_spin<<<1, 1, 0, st_>>>(sns);
            }
            // block selection reads q in k_qsa_dec, which takes the peers' rows as fp16 too (x16o)
            const bool q16 = x16 || (x16o && sel_on_ && dec_k_ && mr <= XROWS && !(pf_ && getenv("FLASH_DEC_NOPF")));
            // staged prefill transfers (FLASH_COMM ce / host): the rows go into the outbox, then by copies
            const int qpar = xstage_ ? (++xq_k_) & (XSLOT - 1) : 0;  // the consumers' slot s & (XSLOT - 1)
            const bool qstg = xstage_ && xout_ && mr > 8;
            const Xchg xqs = qstg ? xchg_outbox(0, xq, qpar, qper(p_)) : xq;
            // prefill with block selection: the q exchange on a side stream, under the indexer's score / selection
            // (joined before the attention, which waits for the peers' q anyway)
            // (opt-in: CE 16K -3 %, P2P neutral, host staging slower; with CE, 2 of 10 det 16K runs differed from the
            // reference, cause not found, 0 of 6 without it)
            static const bool qside_on = getenv("FLASH_QX_SIDE") && atoi(getenv("FLASH_QX_SIDE"));
            const bool qside = qside_on && pf_ && sel_on_ && mr > 8 && !dh;
            cudaStream_t qs = st_;
            if (qside) {
                CUDA_CHECK(cudaEventRecord(pfq0_, st_));
                CUDA_CHECK(cudaStreamWaitEvent(pfs_, pfq0_, 0));
                qs = pfs_;
            }
            if (q16) k_qxsend16<<<dim3(p_.ngpu - 1, std::min(16, (mr + 3) / 4)), 256, 0, qs>>>(qb, mr, xqs, p_);
            else k_qxsend<<<dim3(p_.ngpu - 1, std::min(16, (mr + 3) / 4)), 256, 0, qs>>>(qb, mr, xqs, p_);
            KCHK("k_qxsend");
            if (qstg) xchg_send(0, xq, qpar, mr, (size_t)qper(p_) * (q16 ? 2 : 4), qper(p_), qs);
            if (qside) CUDA_CHECK(cudaEventRecord(pfq1_, pfs_));
            if (sel_on_) {  // context beyond 2051 tokens: indexer top-512 block selection
                if (!ijoined) { CUDA_CHECK(cudaStreamWaitEvent(st_, pfe1_, 0)); ijoined = true; }
                Xchg xi;
                for (int g = 0; g < p_.ngpu; g++) { xi.recv[g] = xipr_[g]; xi.flags[g] = xipf_[g]; }
                xi.seq = xiseq_;
                xi.done = xidone_;
                const float* iq = gates_ + (size_t)t0 * IDX_NH * IDX_HD;
                static const bool ist = !getenv("FLASH_IDX_TILE") || atoi(getenv("FLASH_IDX_TILE"));
                if (ist && mr > 8)  // rows x blocks tiles: keys read once per 16 rows
                    k_idx_score_t<<<dim3((lbmax + IS_BLK - 1) / IS_BLK, (mr + IS_ROWS - 1) / IS_ROWS), 256, IS_SMEM, st_>>>(L.ikb, iq, pp, p_, lbmax, mr, iscore_, rbcur_);
                else
                k_idx_score<<<dim3((lbmax + 255) / 256, mr), 256, 0, st_>>>(L.ikb, iq, pp, p_, lbmax, iscore_, nullptr, rbcur_);
                KCHK("k_idx_score");
                if (idx_gk_) k_idx_local<true><<<mr, 1024, 0, st_>>>(iscore_, pp, p_, lbmax, iloc_);
                else k_idx_local<<<mr, 1024, (size_t)lbmax * 4, st_>>>(iscore_, pp, p_, lbmax, iloc_);
                KCHK("k_idx_local");
                k_idx_send<<<p_.ngpu - 1, 256, 0, st_>>>(iloc_, mr, xi, p_);
                k_idx_select<<<mr, 1024, 0, st_>>>(iscore_, iloc_, pp, p_, lbmax, xi, isel_, insel_, nullptr, idx_gk_);
                KCHK("k_idx_select");
                if (dh) k_dbg_selhash<<<mr, 256, 0, st_>>>(isel_, insel_, lbmax, dh + 8);
                static const bool selstat = getenv("FLASH_SELSTAT") && atoi(getenv("FLASH_SELSTAT"));
                if (selstat && pf_) {
                    if (!selst_) selst_ = (unsigned long long*)dalloc(3 * 8);
                    k_sel_union<<<(mr + QP_R - 1) / QP_R, 256, 0, st_>>>(isel_, insel_, mr, lbmax, selst_);
                }
            }
            if (qside) CUDA_CHECK(cudaStreamWaitEvent(st_, pfq1_, 0));  // join the q exchange
            static const bool rows_k = !getenv("FLASH_ATTN_ROWS") || atoi(getenv("FLASH_ATTN_ROWS"));
            static const bool tc_k = !getenv("FLASH_ATTN_TC") || atoi(getenv("FLASH_ATTN_TC"));
            if (tc_k && !sel_on_ && mr > 8) {  // prefill chunk without block selection: tensor cores
                static const int tcv = getenv("FLASH_ATTN_TC") ? atoi(getenv("FLASH_ATTN_TC")) : 2;
                if (tcv == 2) {  // 2 heads x 32 rows per block, fast fp8 widening, Q in registers
                    AttnDst d{};
                    d.o[0] = attn_x_;
                    if (x16)
                        k_qsa_attn_tc2<2, true><<<dim3(NQ / 2, (mr + 31) / 32, 1), 128, TB_SMEM, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp,
                                                                                                   d, p_, xq, mr, rbcur_);
                    else
                        k_qsa_attn_tc2<2><<<dim3(NQ / 2, (mr + 31) / 32, 1), 128, TB_SMEM, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp, d,
                                                                                              p_, xq, mr, rbcur_);
                } else
                k_qsa_attn_tc<<<dim3(NQ, (mr + TA_ROWS - 1) / TA_ROWS), 128, TA_SMEM, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp,
                                                                                            attn_x_, p_, xq, mr, rbcur_);
            } else if (dec_k_ && mr <= XROWS && !(pf_ && getenv("FLASH_DEC_NOPF"))) {
                // decode / verify rows: GQA tile (12 heads of a kv head x 16-key tiles), keys split over S blocks per
                // (kv head, row), with or without block selection
                AttnDst d{};
                d.o[0] = attn_x_;
                // prefill rows (a chunk with block selection): enough blocks from the rows, keys not split (no merge)
                static const int spf = getenv("FLASH_ATTN_DSPLIT_PF") ? std::max(1, std::min(dec_split_, atoi(getenv("FLASH_ATTN_DSPLIT_PF")))) : 1;
                // prefill rows with block selection: 8 rows per block over the union of their blocks (prefill-r3,
                // FLASH_ATTN_PFSEL=0: k_qsa_dec per row), FLASH_ATTN_PFS key splits (default 2: 2 x 64 x 2 blocks per
                // 512-row chunk, one block per SM)
                static const bool pfsel = !getenv("FLASH_ATTN_PFSEL") || atoi(getenv("FLASH_ATTN_PFSEL"));
                static const int pfs = getenv("FLASH_ATTN_PFS") ? std::max(1, std::min(8, atoi(getenv("FLASH_ATTN_PFS")))) : 2;
                static const int qpf_dbg = getenv("FLASH_QPF_DBG") ? atoi(getenv("FLASH_QPF_DBG")) : 0;
                // (not with copy-engine exchanges: FLASH_COMM=ce + k_qsa_pfsel faults / NaN in batches > 2560 rows,
                // cause open; the kernel's output equals k_qsa_dec's there when run as a shadow)
                static const bool qpf_ce = getenv("FLASH_QPF_CE") && atoi(getenv("FLASH_QPF_CE"));  // debug: pfsel with CE
                if (pfsel && pf_ && sel_on_ && mr > MAXM && std::min(pfs, attn_split_) >= 1 && (qpf_ce || !(xstage_ && comm_.xfer == kXferCE))) {
                    const int S = std::min(pfs, attn_split_);
                    for (int z = 1; z < S; z++) d.o[z] = attn_spl_ + (size_t)(z - 1) * XROWS * NQ * (HD + 2);
                    const size_t smem = qpf_smem(lbmax);
                    static const int hv = getenv("FLASH_QPF_H") ? atoi(getenv("FLASH_QPF_H")) : 3;  // bit 0: fp16 P.V, bit 1: fp16 Q.K
                    auto kf = hv == 3 ? k_qsa_pfsel<true, true> : hv == 2 ? k_qsa_pfsel<false, true> : hv == 1 ? k_qsa_pfsel<true, false> : k_qsa_pfsel<false, false>;
                    if (qpf_attr_ < smem) {
                        for (auto f2 : {k_qsa_pfsel<true, true>, k_qsa_pfsel<false, true>, k_qsa_pfsel<true, false>, k_qsa_pfsel<false, false>})
                            CUDA_CHECK(cudaFuncSetAttribute(f2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
                        qpf_attr_ = smem;
                    }
                    // the row groups' block unions once (prefill-r4, FLASH_QPF_UNION=0: built in every block)
                    static const bool qun = !getenv("FLASH_QPF_UNION") || atoi(getenv("FLASH_QPF_UNION"));
                    QpUnion qu;
                    if (qun && qpu_.U) {
                        qu = qpu_;
                        qu.ctr = nullptr;
                        k_qpf_union<<<(mr + QP_R - 1) / QP_R, 256, (size_t)((lbmax + 31) / 32) * 4, st_>>>(isel_, insel_, mr, lbmax, qu);
                    }
                    // persistent blocks over the (kv head, split, row group) items (prefill-r4, opt-in FLASH_QPF_PERSIST=1;
                    // default: one item per block until the rc2 test round validates it)
                    static const bool qps = getenv("FLASH_QPF_PERSIST") && atoi(getenv("FLASH_QPF_PERSIST"));
                    const int ngrp = (mr + QP_R - 1) / QP_R;
                    if (qps && qpu_.ctr) {
                        qu.ctr = qpu_.ctr;
                        qu.S = S;
                        if (!nsm_) CUDA_CHECK(cudaDeviceGetAttribute(&nsm_, cudaDevAttrMultiProcessorCount, dev_));
                        kf<<<std::min(nsm_, NKV * ngrp * S), QP_W * 32, smem, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp, d, p_, xq, isel_,
                                                                                insel_, lbmax, mr, rbcur_, q16, qpf_dbg, qu, ngrp);
                    } else
                    kf<<<dim3(NKV, ngrp, S), QP_W * 32, smem, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp, d,
                                                                                p_, xq, isel_, insel_, lbmax, mr, rbcur_, q16, qpf_dbg, qu, 0);
                    if (S > 1) k_attn_merge<<<dim3(NQ, mr), HD, 0, st_>>>(d, S);
                } else {
                const int S = mr > MAXM ? spf : dec_split_;
                for (int z = 1; z < S; z++) d.o[z] = attn_spl_ + (size_t)(z - 1) * XROWS * NQ * (HD + 2);
                k_qsa_dec<<<dim3(NKV, mr, S), 128, QD_SMEM, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp, d, p_, xq,
                                                                  sel_on_ ? isel_ : nullptr, insel_, lbmax, nullptr, rbcur_, q16);
                if (S > 1) k_attn_merge<<<dim3(NQ, mr), HD, 0, st_>>>(d, S);
                static const bool qcmp = getenv("FLASH_QPF_CMP") && atoi(getenv("FLASH_QPF_CMP"));
                if (qcmp && pf_ && sel_on_ && mr > MAXM && attnB_ && S == 1) {  // debug: k_qsa_pfsel shadow vs k_qsa_dec (eager)
                    const size_t smem = qpf_smem(lbmax);
                    CUDA_CHECK(cudaFuncSetAttribute(k_qsa_pfsel<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
                    AttnDst d2{};
                    d2.o[0] = attnB_;
                    k_qsa_pfsel<true, true><<<dim3(NKV, (mr + QP_R - 1) / QP_R, 1), QP_W * 32, smem, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp, d2,
                                                                                                    p_, xq, isel_, insel_, lbmax, mr, rbcur_, q16, 3);
                    unsigned* cs = reinterpret_cast<unsigned*>(attnB_ + (size_t)XROWS * NQ * (HD + 2));  // (after the partials: 80 bytes of slack)
                    CUDA_CHECK(cudaMemsetAsync(cs, 0, 80, st_));
                    k_cmp_attn<<<mr, 32, 0, st_>>>(attn_x_, attnB_, mr, cs);
                    unsigned h[20];
                    CUDA_CHECK(cudaMemcpyAsync(h, cs, 80, cudaMemcpyDeviceToHost, st_));
                    CUDA_CHECK(cudaStreamSynchronize(st_));
                    static int nrep = 0;
                    const float e = __builtin_bit_cast(float, h[0]);
                    if (e > 2e-2f && nrep++ < 12) {
                        fprintf(stderr, "[qpf_cmp] rank %d layer %d chunk pos %d: max rel err %.3g, %u (row, head) > 2e-2:", p_.rank, li,
                                cposh_[2 * (int)(posd_ - cposd_) / 2], e, h[1]);
                        for (unsigned k = 0; k < std::min(h[1], 16u); k++) fprintf(stderr, " %u/%u", h[4 + k] / 32, h[4 + k] % 32);
                        fprintf(stderr, "\n");
                    }
                }
                }
            } else if (tc_k && !sel_on_ && attn_split_ > 1 && mr <= MAXM) {
                // decode / verify rows: tensor-core kernel with the keys split over gridDim.z blocks, merged after
                AttnDst d{};
                d.o[0] = attn_x_;
                for (int z = 1; z < attn_split_; z++) d.o[z] = attn_spl_ + (size_t)(z - 1) * XROWS * NQ * (HD + 2);
                k_qsa_attn_tc2<2><<<dim3(NQ / 2, (mr + 31) / 32, attn_split_), 128, TB_SMEM, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp, d,
                                                                                                    p_, xq, mr, rbcur_);
                k_attn_merge<<<dim3(NQ, mr), HD, 0, st_>>>(d, attn_split_);
            } else if (rows_k && mr > 8)  // many rows: one warp per row, rows of a block share K/V
                k_qsa_attn_rows<<<dim3(NQ, (mr + 7) / 8), 256, 0, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp, attn_x_, p_, xq,
                                                                          sel_on_ ? isel_ : nullptr, insel_, lbmax, mr, rbcur_);
            else
                k_qsa_attn_b<<<dim3(NQ, mr), 256, 0, st_>>>(qb, L.kc, L.vc, L.ks, L.vs, pp, attn_x_, p_, xq,
                                                             sel_on_ ? isel_ : nullptr, insel_, lbmax, rbcur_);
        }
        KCHK("k_qsa_attn");
        if (dh) k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(attn_x_), (size_t)mr * NQ * (HD + 2), dh + 9);
        const float* prow = proj_ + (size_t)t0 * ld;
        bf16* abr = ab_ + (size_t)t0 * ql * HD;
        if (p2p && comm_.p2p) {
            Xchg xc;
            for (int g = 0; g < p_.ngpu; g++) { xc.recv[g] = xpr_[g]; xc.flags[g] = xpf_[g]; }
            xc.seq = xseq_;
            xc.done = xdone_;
            xc.scnt = xscnt_;
            const int cpar = xstage_ ? (++xc_k_) & (XSLOT - 1) : 0;
            const bool cstg = xstage_ && xout_ && mr > 8;
            const Xchg xcs = cstg ? xchg_outbox(1, xc, cpar, xper(p_)) : xc;
            if (x16o && qsa_split_ && !cstg && mr == m && !dh) {  // pipelined: the combine + o_proj in qsa_back
                k_xsend16<<<dim3(p_.ngpu - 1, std::min(16, (mr + 3) / 4)), 256, 0, st_>>>(attn_x_, mr, xcs, p_, qsa_ahead_);
                QsaBack& q = qpend_[qsa_buf_];
                q.L = &L; q.prow = prow; q.ax = attn_x_; q.mr = mr; q.ld = ld; q.ql = ql; q.nqp = nqp;
                return;
            } else if (x16o) {
                k_xsend16<<<dim3(p_.ngpu - 1, std::min(16, (mr + 3) / 4)), 256, 0, st_>>>(attn_x_, mr, xcs, p_);
                if (cstg) xchg_send(1, xc, cpar, mr, (size_t)QPC(p_) * XP16 * 4, xper(p_));
                k_qsa_comb16<<<dim3(ql, mr), HD, 0, st_>>>(attn_x_, mr, prow, ld, abr, ql * HD, xc, p_, nqp * HD);
            } else {
            k_xsend<<<dim3(p_.ngpu - 1, std::min(16, (mr + 3) / 4)), 256, 0, st_>>>(attn_x_, mr, xcs, p_);
            KCHK("k_xsend");
            if (cstg) xchg_send(1, xc, cpar, mr, (size_t)xper(p_) * 4, xper(p_));
            k_qsa_comb2<<<dim3(ql, mr), HD, 0, st_>>>(attn_x_, mr, prow, ld, abr, ql * HD, xc, p_, nqp * HD);
            }
            KCHK("k_qsa_comb2");
        } else {
            const int per = NQ * (HD + 2);
            for (int t = 0; t < mr; t++)
                allgather_small(attn_x_ + (size_t)t * per, per, attn_all_ + (size_t)t * p_.ngpu * per, comm_, p_.rank, ctr_, st_);
            k_qsa_comb<<<dim3(ql, mr), HD, 0, st_>>>(attn_all_, p_.ngpu, mr, prow, ld, abr, ql * HD, p_);
            KCHK("k_qsa_comb");
        }
    }
    if (dh) k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(ab_), (size_t)m * ql * HD / 2, dh + 10);
    rows8(L.o, ab_, ql * HD, part_, H, m, st_);
    if (!ijoined) CUDA_CHECK(cudaStreamWaitEvent(st_, pfe1_, 0));
    if (dh) k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(part_), (size_t)m * H, dh + 11);
}

void Worker::qsa_back(int slot) {
    QsaBack& q = qpend_[slot];
    if (!q.L) return;  // the chunk ran unsplit
    Xchg xc;
    for (int g = 0; g < p_.ngpu; g++) { xc.recv[g] = xpr_[g]; xc.flags[g] = xpf_[g]; }
    xc.seq = xseq_;
    xc.done = xdone_;
    xc.scnt = xscnt_;
    k_qsa_comb16<<<dim3(q.ql, q.mr), HD, 0, st_>>>(q.ax, q.mr, q.prow, q.ld, ab_, q.ql * HD, xc, p_, q.nqp * HD);
    KCHK("k_qsa_comb16 (back)");
    rows8(q.L->o, ab_, q.ql * HD, part_, H, q.mr, st_);
    q.L = nullptr;
}

void Worker::moe(Layer& L, int m) {
    static const int pfx = getenv("FLASH_PFX") ? atoi(getenv("FLASH_PFX")) : 0;
    if (pf_ && !(pfx & 2)) {  // prefill: router GEMM, per-expert row lists, each expert group read once for its rows
        int ns = 4;
        if (L.router8.q) { rows8(L.router8, x_, H, rlog_, 528, m, st_); ns = 1; }
        else gemv_w16_rows(L.router, x_, H, rlog_, 528, m, st_, 4, (size_t)m * 528);
        k_topk_rows<<<m, NE, 0, st_>>>(rlog_, ns, gsel_, gw_, gsh_, L.stats, m);
        // down into the slot buffer + ordered sums (FLASH_MOE_SLOT, not in deterministic mode: its fixed-point atomics
        // are exact) when the batch's slots fit the mixer scratch block
        static const bool slot_on = !getenv("FLASH_MOE_SLOT") || atoi(getenv("FLASH_MOE_SLOT"));
        const size_t srows = (size_t)moe_slot_rows_ + (bt_taken_ ? moe_slot_rows2_ : 0);
        const bool slot = slot_on && !detpart_ && moe_slots_ && (size_t)m * (TOPK + 1) <= srows;
        k_jobs_pf<<<1, NE, 0, st_>>>(m, L.gmap, L.groups, pfs_use_ && &L != &mtp_ ? stab(L) : ctab(L), L.ndev, gsel_, gw_, gsh_,
                                     pfjobs_, pftl_, pfwl_, njobs_, slot ? pfinv_ : nullptr);
        KCHK("k_jobs_pf");
        if (!detpart_ && !slot) CUDA_CHECK(cudaMemsetAsync(part_, 0, (size_t)m * H * 4, st_));
        const int mj = std::min(maxjob_pf_, maxjob_ + maxlg_ * ((m * (TOPK + 1) + PF_SHJ - 1) / PF_SHJ));  // bound of the job count
        k_jobtrap<<<1, 1, 0, st_>>>(njobs_, mj);  // a count past the grid would drop expert work silently: abort instead
        static const bool jobchk = getenv("FLASH_JOBCHK") && atoi(getenv("FLASH_JOBCHK"));
        if (jobchk) { if (!jobchk_) { CUDA_CHECK(cudaHostAlloc((void**)&jobchk_, 32, cudaHostAllocMapped)); memset(jobchk_, 0, 32); }
                      k_jobchk<<<1, 1, 0, st_>>>(njobs_, mj, maxjob_pf_, 1, m, jobchk_); }
        static const bool gu2 = !getenv("FLASH_MOE_GU2") || atoi(getenv("FLASH_MOE_GU2"));
        // W4A8 gate_up (prefill-r3, FLASH_MOE_Q=0: fp16 v2): the part's XH rows are quantized in place (each row's
        // XQ bytes + scales fit its 2H bytes; the rows are dead after this MoE, the next layer's input is gathered
        // into them later)
        static const bool moe_q = !getenv("FLASH_MOE_Q") || atoi(getenv("FLASH_MOE_Q"));
        if (moe_q) {
            int8_t* const xq = reinterpret_cast<int8_t*>(xh_);
            float2* const xqs = reinterpret_cast<float2*>(xq + H);
            k_xh_to_q<64><<<m, 160, 0, st_>>>(xh_, xq, xqs);
            // FLASH_MOE_QW=1: half-group blocks (128 registers, 2 blocks / SM, room next to the comm stream's blocks)
            static const int qw = getenv("FLASH_MOE_QW") ? atoi(getenv("FLASH_MOE_QW")) : 2;
            if (qw == 1) k_moe_gu_q<64, 1><<<2 * mj, 256, 0, st_>>>(pfjobs_, njobs_, pftl_, pfwl_, xq, xqs, pfact_);
            else k_moe_gu_q<64><<<mj, 256, 0, st_>>>(pfjobs_, njobs_, pftl_, pfwl_, xq, xqs, pfact_);
        } else if (gu2) k_moe_gu_pf2<<<std::min(mj, 1024), 256, 0, st_>>>(pfjobs_, njobs_, pftl_, pfwl_, xh_, pfact_);
        else k_moe_gu_pf<<<dim3(2 * GRP / 16, mj), 128, 0, st_>>>(pfjobs_, njobs_, pftl_, pfwl_, xh_, pfact_);
        KCHK("k_moe_gu_pf");
        if (slot) {
            // W4A8 down (prefill-r3, FLASH_MOE_DQ=0: fp16): the act rows quantized in place, then s8 MMA
            static const bool dq = !getenv("FLASH_MOE_DQ") || atoi(getenv("FLASH_MOE_DQ"));  // default (user decision): kernel -15 %, e2e = fp16 today (co-run bound)
            if (dq) {
                if (!maxloc_) {  // most local groups of one expert on this card (the act rows: <= m x 11 x maxloc)
                    for (int e = 0; e <= NE; e++) {
                        int c = 0;
                        for (int g = 0; g < NGRP; g++) c += (size_t)(e * NGRP + g) < p_.gcard.size() && p_.gcard[e * NGRP + g] == p_.rank;
                        maxloc_ = std::max(maxloc_, c);
                    }
                    if (!maxloc_) maxloc_ = NGRP;
                }
                const int nact = m * (TOPK + 1) * maxloc_;  // bound of the act rows of this part
                k_act_to_q<<<(nact * 8 + 255) / 256, 256, 0, st_>>>(pfact_, nact);
                static const int dtt = getenv("FLASH_MOE_DQT") ? atoi(getenv("FLASH_MOE_DQT")) : 2;
                if (dtt == 1) k_moe_dn_q<1><<<dim3(H / 16 / 4, mj), 128, 0, st_>>>(pfjobs_, njobs_, pfact_, SlotBuf{moe_slots_, moe_slot_rows_, moe_slots2_});
                else k_moe_dn_q<2><<<dim3(H / 16 / 8, mj), 128, 0, st_>>>(pfjobs_, njobs_, pfact_, SlotBuf{moe_slots_, moe_slot_rows_, moe_slots2_});
            } else
            k_moe_dn_slot<2><<<dim3(H / 16 / 8, mj), 128, 0, st_>>>(pfjobs_, njobs_, pfact_, SlotBuf{moe_slots_, moe_slot_rows_, moe_slots2_});
            k_moe_sum<<<m, 320, 0, st_>>>(SlotBuf{moe_slots_, moe_slot_rows_, moe_slots2_}, pfinv_, m * TOPK, L.gmap, part_);
        } else {
            k_moe_dn_pf<2><<<dim3(H / 16 / 8, mj), 128, 0, st_>>>(pfjobs_, njobs_, pftl_, pfact_, part_, detpart_);
        }
        KCHK("k_moe_dn_pf");
        if (detpart_) k_fx_to_f<<<256, 256, 0, st_>>>(detpart_, part_, (size_t)m * H);
        if (!rdump_.empty() && &L != &mtp_) {  // debug routing dump (eager): this prefill's rows at rd_pos_
            const size_t li = &L - L_.data(), np = rdump_.size() / LAYERS / TOPK;
            if (rd_pos_ + m <= (int)np)
                CUDA_CHECK(cudaMemcpyAsync(rdump_.data() + (li * np + rd_pos_) * TOPK, gsel_, (size_t)m * TOPK * 4,
                                           cudaMemcpyDeviceToHost, st_));
            CUDA_CHECK(cudaStreamSynchronize(st_));
        }
        if (dbgh_) {  // routing and this card's MoE partial of the layer (debug hash)
            const int li = (int)(&L - L_.data());
            const int slot = li >= 0 && li < LAYERS ? li : LAYERS;
            k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(gsel_), (size_t)m * TOPK, dbgh_ + slot * DBGK + 1);
            k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(part_), (size_t)m * H, dbgh_ + slot * DBGK + 2);
        }
        return;
    }
    static const int rv = getenv("FLASH_ROUTE") ? atoi(getenv("FLASH_ROUTE")) : 0;  // 0: v5 if m <= 8, else v4 / route2
    // grouped MoE: from FLASH_MOE_G rows (default 9; 0 off) each job computes only the rows that picked its expert
    static const int gmin = getenv("FLASH_MOE_G") ? atoi(getenv("FLASH_MOE_G")) : 9;
    const bool grouped = gmin > 0 && m >= gmin && m > 1 && !(L.router8.q && m <= 8 && (rv == 0 || rv == 5)) && rv != 2;
    if (L.router8.q && m <= 8 && (rv == 0 || rv == 5)) {  // router v5: GEMV + top-10 + jobs in one kernel
        k_route_f<<<dim3(9, RF_SPLIT), 512, 0, st_>>>(L.router8.q, L.router8.s, x_, m, rfpart_, rfcnt_, rlog_, gsel_, gw_,
                                                     gsh_, L.stats, L.gmap, L.groups, ctab(L), L.ndev, jobs_, jw_, njobs_,
                                                     coldcnt_);
        KCHK("k_route_f");
    } else {
    int RS = 4;  // router K split (33 row tiles alone would leave most SMs idle)
    if (L.router8.q) { gemv_w8(L.router8, x_, H, rlog_, 528, m, st_); RS = 1; }
    else gemv_w16(L.router, x_, H, rlog_, 528, m, st_, RS, (size_t)MAXM * 528);
    if (rv == 4 || (rv != 2 && m > 1)) {  // FLASH_ROUTE=1: the previous default (v4, route2 for m = 1)  // per-row top-10 blocks + one job-building block
        k_topk_rows<<<m, NE, 0, st_>>>(rlog_, RS, gsel_, gw_, gsh_, L.stats);
        KCHK("k_topk_rows");
        k_jobs<<<1, NE, 0, st_>>>(m, L.gmap, L.groups, ctab(L), L.ndev, gsel_, gw_, gsh_, jobs_, jw_, njobs_, coldcnt_,
                                  grouped ? jrow_ : nullptr);
        KCHK("k_jobs");
    } else {
        k_route2<<<1, NE, 0, st_>>>(rlog_, m, L.gmap, L.groups, jobs_, jw_, njobs_, RS, L.stats, ctab(L), L.ndev, coldcnt_);
    }
    }
    // grid over the worst-case job count of this card (blocks past *njobs exit)
    const int maxj = std::min(maxjob_, NGRP * (m * TOPK + 1));  // blocks past *njobs exit
    static const bool jobchk_d = getenv("FLASH_JOBCHK") && atoi(getenv("FLASH_JOBCHK"));
    if (jobchk_d) { if (!jobchk_) { CUDA_CHECK(cudaHostAlloc((void**)&jobchk_, 32, cudaHostAllocMapped)); memset(jobchk_, 0, 32); }
                    k_jobchk<<<1, 1, 0, st_>>>(njobs_, maxj, maxjob_, 2, m, jobchk_); }
    if (grouped) {
        static const int gy = getenv("FLASH_MOE_GY") ? atoi(getenv("FLASH_MOE_GY")) : 320;  // job blocks (loop past it)
        k_moe_gu_g<<<dim3(2 * GRP / 16, std::min(maxj, gy > 0 ? gy : maxj)), 128, 0, st_>>>(jobs_, njobs_, jrow_, jw_, xh_, acth_);
        KCHK("k_moe_gu_g");
        k_moe_dn_g<<<H / 16, 256, 0, st_>>>(jobs_, njobs_, jrow_, acth_, part_, m);
        KCHK("k_moe_dn_g");
        return;
    }
    NT_LAUNCH(m, k_moe_gu, dim3(2 * GRP / 16, maxj), 128, jobs_, njobs_, jw_, xh_, acth_, m);
    KCHK("k_moe_gu");
    NT_LAUNCH(m, k_moe_dn, H / 16, 256, jobs_, njobs_, acth_, part_, m);
    KCHK("k_moe_dn");
}

void Worker::ple(Layer& L, int pos, int m, const bf16* rows) {
    (void)pos; (void)rows;  // rows: staged into pler_ by body()
    flush_y(m);
    float* key = gates_;
    float* val = proj_;
    if (pf_) {
        gemm_w4(L.ple_key, pler_, H, key, RW, m, st_);
        gemm_w4(L.ple_val, pler_, H, val, H, m, st_);
    } else {
        xh_pack(pler_, H, H, m, pleh_, H, st_);
        gemv_w4h(L.ple_key, pleh_, H, key, RW, m, st_);
        gemv_w4h(L.ple_val, pleh_, H, val, H, m, st_);
    }
    float* gv = proj2_;              // [m][10240]
    float* gvn = (float*)hn_;        // hn_ is [MAXM][10240] bf16 = half the bytes: use yd_ instead
    gvn = yd_;
    k_ple_gate<<<dim3(HC, m), 256, 0, st_>>>(key, val, R_, L.ple_nk, L.ple_nq, L.ple_nc, gv, gvn);
    KCHK("k_ple_gate");
    k_ple_conv<<<RW / 256, 256, 0, st_>>>(gv, gvn, L.ple_conv, L.ple_hist, L.ple_pend, posd_ + 1, R_, m, pf_);
    KCHK("k_ple_conv");
}

void Worker::body(int m) {
    CUDA_CHECK(cudaMemcpyAsync(toks_, tok_h_, m * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(posd_, pos_h_, 8, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(pler_, ple_h_, (size_t)m * H * 2, cudaMemcpyHostToDevice, st_));
    comm_begin_unit(ctr_, st_);
    trunk(m, toks_);
    CUDA_CHECK(cudaMemcpyAsync(out_h_, otok_, m * 4, cudaMemcpyDeviceToHost, st_));
}

// trunk forward of m rows (tokens on the device, positions posd_[0].., commit posd_[1], n-gram rows in pler_):
// argmax per row -> otok_, trunk residual rows -> Rsave_
void Worker::trunk(int m, const int* toks) {
    ll_idx_ = 0;
    curpos_ = posd_;
    k_step_bump<<<1, 1, 0, st_>>>(stepc_);
    k_embed<<<m, 256, 0, st_>>>(embed_, toks, R_);
    KCHK("k_embed");
    for (int li = 0; li < LAYERS; li++) {
        Layer& L = L_[li];
        if (li == PLE_LAYER) ple(L, 0, m, nullptr);
        hc_read(L.hca, m, true);
        if (L.full) qsa(L, li, 0, m);
        else gdn(L, li, 0, m);
        next_hc_ = &L.hcm;
        hc_write(m);
        hc_read(L.hcm, m, true);
        moe(L, m);
        next_hc_ = li + 1 < LAYERS ? &L_[li + 1].hca : &mixer_;
        hc_write(m);
        steer_mark(li, nullptr);  // the layer output: steered by the next HC read (fused) or flush_y
        if (cap_) {  // activation capture (eager): the layer output with its steering applied
            steer_flush(m);
            k_capture<<<(H + 255) / 256, 256, 0, st_>>>(R_, inj_, ypend_ ? y_ : nullptr, m, cap_skip_, cap_ + (size_t)li * H);
        }
        if (dump_) {  // debug (eager only): residual after every layer ([48][m][10240])
            flush_y(m);
            CUDA_CHECK(cudaStreamSynchronize(st_));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(dump_ + (size_t)li * m * RW, R_, (size_t)m * RW * 4, cudaMemcpyDeviceToHost));
        }
    }
    flush_y(m);
    if (Rsave_) CUDA_CHECK(cudaMemcpyAsync(Rsave_, R_, (size_t)m * RW * 4, cudaMemcpyDeviceToDevice, st_));
    if (pf_) {  // prefill: only the last row needs the head
        float* Rt = R_;
        R_ = R_ + (size_t)(m - 1) * RW;
        hc_read(mixer_, 1, false);
        R_ = Rt;
        if (sampling_) head_sample(1, otok_);
        else head(1, otok_);
        return;
    }
    hc_read(mixer_, m, false);
    if (sampling_) head_sample(m, otok_);
    else head(m, otok_);
}

__global__ void k_remap_cand(float* __restrict__ pairs, int n, const int* __restrict__ map) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pairs[2 * i + 1] = __int_as_float(map[__float_as_int(pairs[2 * i + 1])]);
}

// grammar mask on a permuted vocabulary slice: logits row r (r < m), local row i = token map[i]; bit clear -> -inf
__global__ void k_mask_perm(float* __restrict__ logits, int n, const int* __restrict__ map, const unsigned* __restrict__ bits,
                            int words) {
    const unsigned* b = bits + (size_t)blockIdx.y * words;
    float* row = logits + (size_t)blockIdx.y * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int t = map[i];
        if (!((b[t >> 5] >> (t & 31)) & 1u)) row[i] = -INFINITY;
    }
}

void Worker::head_sample(int m, int* dst) {  // full lm_head, local top-64 -> gathered -> same sample on every card
    const int nv = p_.voc1 - p_.voc0;
    gemv_w8(lm_, x_, H, logits_, nv, m, st_);
    if (con_) k_mask_perm<<<dim3(32, m), 256, 0, st_>>>(logits_, nv, lmap_, mask_d_, MASK_WORDS);
    topk_local(logits_, nv, m, 0, cand_, st_);
    k_remap_cand<<<(m * kTopK + 255) / 256, 256, 0, st_>>>(cand_, m * kTopK, lmap_);
    allgather_small(cand_, m * kTopK * 2, cand_all_, comm_, p_.rank, ctr_, st_);
    sample_rows(cand_all_, p_.ngpu, m, sp_, dst, st_);
}

// the bound slot's parameters; each slot keeps its own (batched steps sample every row with its slot's, see head_sample_b)
void Worker::set_sampling(float temperature, float top_p, int top_k, float min_p, unsigned long long seed) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    sampling_ = temperature > 0.f;
    SampleParams h{};
    h.temperature = temperature > 0.f ? temperature : 1.f;
    h.top_p = top_p > 0.f && top_p <= 1.f ? top_p : 1.f;
    h.top_k = top_k >= 1 && top_k <= kTopK ? top_k : kTopK;
    h.min_p = min_p;
    h.seed = seed;
    h.counter = 0;
    CUDA_CHECK(cudaMemcpy(sp_, &h, sizeof(h), cudaMemcpyHostToDevice));
    if (spslot_) {
        CUDA_CHECK(cudaMemcpy(spslot_ + cur_slot_, &h, sizeof(h), cudaMemcpyHostToDevice));
        sson_[cur_slot_] = sampling_;
        const int on = sampling_;
        CUDA_CHECK(cudaMemcpy(sson_d_ + cur_slot_, &on, 4, cudaMemcpyHostToDevice));
    }
}

void Worker::head_sample_b(int m, int* dst) {  // as head_sample, every row with its slot's parameters (greedy rows: argmax)
    const int nv = p_.voc1 - p_.voc0;
    gemv_w8(lm_, x_, H, logits_, nv, m, st_);
    topk_local(logits_, nv, m, 0, cand_, st_);
    k_remap_cand<<<(m * kTopK + 255) / 256, 256, 0, st_>>>(cand_, m * kTopK, lmap_);
    allgather_small(cand_, m * kTopK * 2, cand_all_, comm_, p_.rank, ctr_, st_);
    sample_rows_slots(cand_all_, p_.ngpu, m, spslot_, rslot_, sson_d_, dst, st_);
}

void Worker::start_ngram_stage(const HostTables* ht) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CHECK(!ht->ngram8.empty(), "FLASH_NGRAM_PIN=0 needs the INT8 n-gram table");
    const size_t rows = (size_t)MAXM * PLE_HEADS;
    long long* req;
    uint8_t* stage;
    unsigned* flags;
    CUDA_CHECK(cudaHostAlloc((void**)&req, rows * 8, cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&stage, rows * NGRAM_Q8_ROW, cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&flags, 16, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(flags, 0, 16);
    void *dr, *ds, *df;
    CUDA_CHECK(cudaHostGetDevicePointer(&dr, req, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&ds, stage, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&df, flags, 0));
    ngp_.req = (long long*)dr;
    ngp_.stage = (const uint8_t*)ds;
    ngp_.flags = (unsigned*)df;
    // the GPU must never read the (pageable) file mapping in this mode: no table pointers on the device side
    for (int i = 0; i < 128; i++) { ngp_.q8[i] = nullptr; ngp_.shard[i] = nullptr; }
    // the host side: poll the request, copy the rows (page cache / disk through the file mapping), report done
    const bool slog = getenv("FLASH_NGRAM_STAGE_LOG") && atoi(getenv("FLASH_NGRAM_STAGE_LOG"));
    const int rank = p_.rank;
    std::thread([ht, req, stage, flags, slog, rank] {
        unsigned last = 0;
        double tsum = 0, tmax = 0;
        long long nreq = 0, nrows = 0;
        while (true) {
            const unsigned s2 = *(volatile unsigned*)&flags[0];
            if (s2 == last) { __builtin_ia32_pause(); continue; }
            __sync_synchronize();
            const double t0 = slog ? std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() : 0;
            const int n = *(volatile int*)&flags[2];
            for (int i = 0; i < n; i++) {
                const long long r = ((volatile long long*)req)[i];
                memcpy(stage + (size_t)i * NGRAM_Q8_ROW, ht->ngram8[r / NGRAM_ROWS] + (size_t)(r % NGRAM_ROWS) * NGRAM_Q8_ROW,
                       NGRAM_Q8_ROW);
            }
            __sync_synchronize();
            *(volatile unsigned*)&flags[1] = s2;
            last = s2;
            if (slog) {
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
                tsum += dt; tmax = std::max(tmax, dt); nreq++; nrows += n;
                if (nreq % 25 == 0)
                    fprintf(stderr, "[flash] n-gram stage rank %d: %lld requests, %.1f rows each, service mean %.1f us max %.1f us\n",
                            rank, nreq, (double)nrows / nreq, tsum / nreq * 1e6, tmax * 1e6);
            }
        }
    }).detach();
}

void Worker::set_constrained(int slot, bool on) {
    con_slot_.at(slot) = on;
    if (slot == cur_slot_) con_ = on;
}
void Worker::upload_mask(const unsigned* host, int rows) {
    CUDA_CHECK(cudaSetDevice(dev_));
    rows = std::max(1, std::min(rows, (int)MASK_ROWS));
    CUDA_CHECK(cudaMemcpyAsync(mask_d_, host, (size_t)rows * MASK_WORDS * 4, cudaMemcpyHostToDevice, st_));
}
void Worker::upload_drafts(const int* d, int n) {
    CUDA_CHECK(cudaSetDevice(dev_));
    if (n > 0) CUDA_CHECK(cudaMemcpyAsync(vtok_ + 1, d, (size_t)n * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));  // the host buffers are the caller's (it rewrites them for the next step)
}

void Worker::head(int m, int* dst, bool draft) {  // lm_head (vocab slice) + distributed argmax -> dst (device)
    W8 w = lm_;
    const int* map = lmap_;
    if (draft && dlm_.q) { w = dlm_; map = dmap_; }  // balanced draft head
    else if (draft) w.n = ndraft_;  // draft vocabulary: the leading rows of the permuted head
    else if (nf_d_) k_nonfinite<<<8, 256, 0, st_>>>(x_, m * H, nf_d_);  // guard (FLASH_GUARD=0 off), checked by nf_check
    gemv_w8(w, x_, H, logits_, w.n, m, st_);
    if (con_ && !draft) k_mask_perm<<<dim3(32, m), 256, 0, st_>>>(logits_, w.n, map, mask_d_, MASK_WORDS);
    argmax_local(logits_, w.n, m, 0, pairs_, st_, map);
    allgather_small(pairs_, 2 * m, pairs_all_, comm_, p_.rank, ctr_, st_);
    argmax_final(pairs_all_, p_.ngpu, m, dst, st_);
}

void Worker::body_mtp(int m, bool chain) {
    CUDA_CHECK(cudaMemcpyAsync(toks_, tok_h_, m * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(mposd_, pos_h_, 4, cudaMemcpyHostToDevice, st_));
    comm_begin_unit(ctr_, st_);
    ll_idx_ = 0;
    mtp_core(m, chain, toks_, motok_, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(out_h_, motok_, m * 4, cudaMemcpyDeviceToHost, st_));
}

// MTP block over m rows at positions mposd_[0].. (tokens on the device): drafts -> out (device); the residual of
// row *last_row (device; nullptr: m - 1) becomes the next chain input (Rmin_)
void Worker::mtp_core(int m, bool chain, const int* toks, int* out, const int* last_row) {
    curpos_ = mposd_;
    const float* hsrc = chain ? Rmin_ : Rsave_;
    k_mtp_prep<<<dim3(m, HC + 1), 256, 0, st_>>>(embed_, toks, hsrc, mtp_pre_e_, mtp_pre_h_, mtp_en_, mtp_hn_);
    KCHK("k_mtp_prep");
    gemv_w8_rows(mtp_fce_, mtp_en_, H, mtp_fe_, H, m, st_);
    for (int r0 = 0; r0 < m * HC; r0 += kMaxBatchRows)  // gemv_w8: <= 64 rows per call
        gemv_w8(mtp_fch_, mtp_hn_ + (size_t)r0 * H, H, mtp_fh_ + (size_t)r0 * H, H, std::min(kMaxBatchRows, m * HC - r0), st_);
    k_mtp_comb<<<m, 512, 0, st_>>>(mtp_fe_, mtp_fh_, Rm_);
    KCHK("k_mtp_comb");
    float* Rt = R_;
    float* Rt2 = R2_;
    R_ = Rm_;
    R2_ = Rm2_;
    hc_read(mtp_.hca, m, true);
    qsa(mtp_, -1, 0, m);
    hc_write(m);
    hc_read(mtp_.hcm, m, true);
    moe(mtp_, m);
    hc_write(m);
    flush_y(m);
    float* Rcur = R_;
    k_copy_row<<<RW / 512, 512, 0, st_>>>(Rmin_, Rcur, last_row, m - 1);
    KCHK("k_copy_row");
    if (pf_) {  // prefill: draft of the last row only
        R_ = Rcur + (size_t)(m - 1) * RW;
        hc_read(mtp_mixer_, 1, false);
        R_ = Rt;
        R2_ = Rt2;
        head(1, out, true);
        curpos_ = posd_;
        return;
    }
    hc_read(mtp_mixer_, m, false);
    R_ = Rt;
    R2_ = Rt2;
    head(m, out, true);
    curpos_ = posd_;
}

void Worker::prefill_body(int m, bool mtp_fill, int) {
    CUDA_CHECK(cudaMemcpyAsync(posd_, pfh_pos_, 8, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(mposd_, pfh_pos_, 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(toks_, pfh_tok_, (size_t)m * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(pler_, pfh_ple_, (size_t)m * H * 2, cudaMemcpyHostToDevice, st_));
    if (mtp_fill) CUDA_CHECK(cudaMemcpyAsync(mtok_, pfh_mt_, (size_t)m * 4, cudaMemcpyHostToDevice, st_));
    comm_begin_unit(ctr_, st_);
    trunk(m, toks_);
    if (mtp_fill) {
        comm_begin_unit(ctr_, st_);
        ll_idx_ = 0;
        mtp_core(m, false, mtok_, motok_, nullptr);
    }
    CUDA_CHECK(cudaMemcpyAsync(pfh_out_, otok_, 4, cudaMemcpyDeviceToHost, st_));
    if (mtp_fill) CUDA_CHECK(cudaMemcpyAsync(pfh_out_ + 1, motok_, 4, cudaMemcpyDeviceToHost, st_));
}

// runs fn eagerly the first time a key is seen (lazy library allocations), captured into a graph afterwards
template <class F>
void Worker::run_cached(std::map<long long, cudaGraphExec_t>& cache, long long key, F fn) {
    static const bool graphs = !getenv("FLASH_GRAPH") || atoi(getenv("FLASH_GRAPH"));
    auto it = cache.find(key);
    if (!graphs || dump_ || cap_ || it == cache.end()) {
        if (graphs && !dump_ && !cap_) cache[key] = nullptr;  // seen once
        fn();
        return;
    }
    // LRU bound (FLASH_PFGRAPH_CAP, default 4 per cache): every prompt length is its own key, so a server that sees
    // many lengths would otherwise keep instantiating graphs (device memory) until it runs out
    static const int cap = getenv("FLASH_PFGRAPH_CAP") ? std::max(1, atoi(getenv("FLASH_PFGRAPH_CAP"))) : 4;
    auto& use = graph_use_[&cache];
    if (!it->second) {
        int live = 0;
        for (auto& kv : cache) live += kv.second != nullptr;
        if (live >= cap) {  // evict the least recently used graph (nothing of this worker is in flight: callers sync)
            CUDA_CHECK(cudaStreamSynchronize(st_));
            long long victim = 0;
            unsigned long long best = ~0ull;
            for (auto& kv : cache)
                if (kv.second && use[kv.first] < best) { best = use[kv.first]; victim = kv.first; }
            CUDA_CHECK(cudaGraphExecDestroy(cache[victim]));
            cache[victim] = nullptr;
            graphs_evicted_++;
        }
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
        fn();
        CUDA_CHECK(cudaStreamEndCapture(st_, &g));
        cudaError_t e = cudaGraphInstantiate(&it->second, g, 0);
        if (e == cudaErrorMemoryAllocation) {  // little free memory (262K context): drop every cached graph, retry once
            (void)cudaGetLastError();
            CUDA_CHECK(cudaStreamSynchronize(st_));
            for (auto& kv : cache)
                if (kv.second) { CUDA_CHECK(cudaGraphExecDestroy(kv.second)); kv.second = nullptr; graphs_evicted_++; }
            it->second = nullptr;
            e = cudaGraphInstantiate(&it->second, g, 0);
        }
        CUDA_CHECK(cudaGraphDestroy(g));
        if (e == cudaErrorMemoryAllocation) {  // still no room: this key runs eagerly (the captured work never ran)
            (void)cudaGetLastError();
            it->second = nullptr;
            graph_oom_++;
            fn();
            return;
        }
        CUDA_CHECK(e);
    }
    use[key] = ++graph_tick_;
    CUDA_CHECK(cudaGraphLaunch(it->second, st_));
}

void Worker::prefill(const int* tokens, int pos, int m, const bf16* ple_rows, int* out, const int* mtp_toks, int* draft) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CHECK(m >= 1 && m <= PFM, "flash prefill m=%d", m);
    CHECK(pos + m <= max_ctx_, "flash: context %d > %d", pos + m, max_ctx_);
    ec_tick(pos + m, 1);
    const bool mf = mtp_toks && has_mtp_;
    use(pfb_);
    pf_ = true;
    rd_pos_ = pos;
    sel_on_ = pos + m > IDX_BUDGET + IDX_R - 1;
    pfh_pos_[0] = pos;
    pfh_pos_[1] = prev_m_;
    memcpy(pfh_tok_, tokens, (size_t)m * 4);
    memcpy(pfh_ple_, ple_rows, (size_t)m * H * 2);
    if (mf) memcpy(pfh_mt_, mtp_toks, (size_t)m * 4);
    const long long key = ((long long)m << 4) | (mf << 2) | (sel_on_ << 1) | (prev_m_ > 0 ? 1 : 0) | ((long long)sampling_ << 3) |
                          ((long long)con_ << 50);
    run_cached(pgraphs_, key, [&] { prefill_body(m, mf, 0); });
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    if (selst_) {
        unsigned long long h[3];
        CUDA_CHECK(cudaMemcpy(h, selst_, 24, cudaMemcpyDeviceToHost));
        fprintf(stderr, "[selstat] rank %d: %llu groups of 8 rows, mean selected blocks/row %.1f, union per 8 rows %.1f (x%.2f of one row)\n",
                p_.rank, h[2], h[2] ? h[0] / 8.0 / h[2] : 0.0, h[2] ? (double)h[1] / h[2] : 0.0, h[0] ? 8.0 * h[1] / h[0] : 0.0);
    }
    *out = pfh_out_[0];
    if (draft) *draft = pfh_out_[1];
    pf_ = false;
    use(dec_);
    prev_m_ = 0;  // prefill rows are committed already
}

// ---------------------------------------------------------------- batched prefill
unsigned Worker::batch_sel(int pos, int n) {
    unsigned sel = 0;
    const int nch = (n + PFM - 1) / PFM;
    for (int c = 0; c < nch; c++) {
        const int pc = pos + c * PFM, mc = std::min(PFM, n - c * PFM);
        if (pc + mc > IDX_BUDGET + IDX_R - 1) sel |= 1u << c;
        cposh_[2 * c] = pc;
        cposh_[2 * c + 1] = c == 0 ? prev_m_ : 0;
    }
    return sel;
}

// layer by layer over n rows (positions cposd_): per layer the mixer half chunk by chunk (its buffers hold PFM rows;
// the chunk's residual rows are a view into R), then the MoE half over all n rows; R holds the n residual rows
void Worker::set_agpeer(int g, const Worker* w) {
    agdst_[0][g] = (char*)w->pfb_.x;
    agdst_[1][g] = (char*)w->pfb_.xh;
    agdst_[2][g] = (char*)w->pfb_.R;
    agpf_[g] = w->agflags_;
    hag_p_[g] = w->hag_h_;  // staged transfers (host staging: null otherwise)
    hagf_p_[g] = w->hagf_;
    hxq_p_[g] = w->hxq_;
    hxc_p_[g] = w->hxc_;
    hxf_p_[g] = w->hxf_;
}

// Staged prefill transfers (comm_select_path picked the copy engine or host staging): the device outboxes of the QSA
// exchanges (q and partials separately: a receiver may still be copying the q rows when the partials are written), the
// dummy flags and the host-mapped flag words the peers write (both paths); for host staging the pinned staging rows
// (this card's NUMA node: its own D2H copies stay local)
void Worker::alloc_staging() {
    if (comm_.xfer == kXferP2P || p_.ngpu < 2) return;
    xout_ = (float*)dalloc((size_t)p_.ngpu * XROWS * xper(p_) * 4);
    xoutc_ = (float*)dalloc((size_t)p_.ngpu * XROWS * xper(p_) * 4);
    xdummy_ = (unsigned*)dalloc(4 * 4);
    if (comm_.xfer == kXferCE) {  // the sequence-number words the copy engine copies into the peers' flags
        xsw_ = (unsigned*)dalloc(4 * 4);
        agw_ = (unsigned*)dalloc(4 * 4);
    }
    auto mapped = [&](size_t b) {
        unsigned* h = nullptr;
        CUDA_CHECK(cudaHostAlloc((void**)&h, b, cudaHostAllocMapped | cudaHostAllocPortable));
        memset(h, 0, b);
        unsigned* d = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&d, h, 0));
        return d;  // == h (UVA)
    };
    hagf_ = mapped(2 * 2 * 4 * 4);
    hxf_ = mapped(2 * 4 * 4);
    if (comm_.xfer != kXferHost) return;
    auto pinned = [&](size_t b) {
        constexpr size_t PG = (size_t)2 << 20;
        const size_t cap = (b + PG - 1) / PG * PG;
        void* q = nullptr;
        CHECK(posix_memalign(&q, PG, cap) == 0 && q, "staging: out of host memory");
        numa_place(q, cap, {gpu_numa_node(dev_)});
        memset(q, 0, cap);
        CUDA_CHECK(cudaHostRegister(q, cap, cudaHostRegisterPortable));
        return (char*)q;
    };
    hag_h_ = pinned((size_t)2 * AG_CAP * (H * 2 + H * 2 + RW * 4));  // [2 channels]
    hxq_ = (float*)pinned((size_t)p_.ngpu * XROWS * xper(p_) * 4);
    hxc_ = (float*)pinned((size_t)p_.ngpu * XROWS * xper(p_) * 4);
}

// The send kernel's view of a staged exchange: receive buffer g -> outbox region g (the kernel adds the slot offset of
// (parity, rank), subtracted here), flags -> dummy words
Xchg Worker::xchg_outbox(int e, const Xchg& x, int par, size_t per) const {
    Xchg o = x;
    const ptrdiff_t slot = (ptrdiff_t)((size_t)par * p_.ngpu + p_.rank) * XROWS * per;
    float* const ob = e == 0 ? xout_ : xoutc_;
    for (int g = 0; g < 4; g++) {
        o.recv[g] = (float*)((uintptr_t)(ob + (size_t)g * XROWS * xper(p_)) - (uintptr_t)(slot * 4));
        o.flags[g] = xdummy_;
    }
    return o;
}

// after the send kernel of a staged exchange e (0: q, 1: partials): every card PULLS its rows (rowb bytes each) from
// the peers' copies into its own receive slot (parity par), after their flags, then raises its own flag words for the
// consumer. CE: straight from the peers' device outboxes (peer copies issued by the receiver). Host: the sender first
// copies its outbox into its pinned rows (D2H), the receiver copies them in (H2D).
// The receiver copies: a push (the sender's copy into the peer's slot, then its flag by an SM store) has no ordering
// between the copy engine's writes landing in the peer's memory and the flag store; the receiver's own copy is
// complete on its stream before the consumer kernel starts.
void Worker::xchg_send(int e, Xchg& x, int par, int mr, size_t rowb, size_t per, cudaStream_t sx) {
    const int R = p_.rank, ng = p_.ngpu;
    cudaStream_t const st = sx ? sx : st_;
    const size_t bytes = (size_t)mr * rowb;
    float* const ob = e == 0 ? xout_ : xoutc_;
    if (comm_.xfer == kXferCE) {
        // copy engine: push the outbox regions into the peers' receive slots, then the exchange's sequence number into
        // their flag words by copies of the same stream (the flag lands after the rows: an SM flag store after a pushed
        // copy had no ordering with it)
        for (int g = 0; g < ng; g++)
            if (g != R)
                CUDA_CHECK(cudaMemcpyAsync(x.recv[g] + ((size_t)par * ng + R) * XROWS * per, ob + (size_t)g * XROWS * xper(p_), bytes,
                                           cudaMemcpyDeviceToDevice, st));
        k_seq_word<<<1, 1, 0, st>>>(xsw_ + e, x.seq);
        for (int g = 0; g < ng; g++)
            if (g != R) CUDA_CHECK(cudaMemcpyAsync(x.flags[g] + R, xsw_ + e, 4, cudaMemcpyDeviceToDevice, st));
        KCHK("xchg ce");
        return;
    }
    // host staging: D2H of the outbox into this card's pinned rows, flags (host words), wait for the peers', H2D of the
    // peers' rows for this card (the receiver pulls), then the local flag words the consumer waits for
    float* const hb = e == 0 ? hxq_ : hxc_;
    for (int g = 0; g < ng; g++)
        if (g != R)
            CUDA_CHECK(cudaMemcpyAsync(hb + (size_t)g * XROWS * xper(p_), ob + (size_t)g * XROWS * xper(p_), bytes, cudaMemcpyDeviceToHost, st));
    FlagP fp;
    for (int g = 0; g < 4; g++) fp.f[g] = hxf_p_[g] ? hxf_p_[g] + e * 4 : nullptr;
    k_xs_flag<<<1, 32, 0, st>>>(fp, x.seq, R, ng);
    k_xs_wait<<<1, 32, 0, st>>>(hxf_ + e * 4, x.seq, R, ng);
    for (int g = 0; g < ng; g++)
        if (g != R) {
            const float* src = (e == 0 ? hxq_p_[g] : hxc_p_[g]) + (size_t)R * XROWS * xper(p_);
            CUDA_CHECK(cudaMemcpyAsync(x.recv[R] + ((size_t)par * ng + g) * XROWS * per, src, bytes, cudaMemcpyHostToDevice, st));
        }
    k_xs_local<<<1, 32, 0, st>>>(x.flags[R], x.seq, R, ng);
    KCHK("xchg host");
}

// HC read (down, mid, up) of m rows at explicit pointers on stream s (the prefill v2 kernels, own scratch)
void Worker::hc_rows(const HcW& w, float* R, float* inj, bf16* x, __half* xh, int m, cudaStream_t s) {
    if (m <= 0) return;
    const int set = s == st_ ? 1 : 0;  // scratch set of the stream (the comm stream's and st_'s run concurrently)
    float* const sp_hu = sp_hu_ + (size_t)set * AG_CAP * HC * HCD_ROWS;
    float* const sp_hss = sp_hss_ + (size_t)set * AG_CAP * HC;
    __half* const sp_hh = sp_hh_ + (size_t)set * AG_CAP * LOWRANK;
    float* const sp_rinv = sp_rinv_ + (size_t)set * AG_CAP * HC;
    unsigned long long* const sp_det = sp_det_ ? sp_det_ + (size_t)set * AG_CAP * (HC * HCD_ROWS + HC) : nullptr;
    if (sp_det) {
        unsigned long long* fu = sp_det;
        unsigned long long* fs = sp_det + (size_t)m * HC * HCD_ROWS;
        k_hcd_pf2<<<dim3(3, HC * 4, (m + 31) / 32), 224, 0, s>>>(w.fd.q, w.fd.s, R, sp_hu, m, sp_hss, fu, fs);
        k_fx_to_f<<<256, 256, 0, s>>>(fu, sp_hu, (size_t)m * HC * HCD_ROWS);
        k_fx_to_f<<<64, 256, 0, s>>>(fs, sp_hss, (size_t)m * HC);
    } else {
        CUDA_CHECK(cudaMemsetAsync(sp_hu, 0, (size_t)m * HC * HCD_ROWS * 4, s));
        CUDA_CHECK(cudaMemsetAsync(sp_hss, 0, (size_t)m * HC * 4, s));
        k_hcd_pf2<<<dim3(3, HC * 4, (m + 31) / 32), 224, 0, s>>>(w.fd.q, w.fd.s, R, sp_hu, m, sp_hss);
    }
    k_hc_mid_pf<<<m, 352, 0, s>>>(sp_hu, sp_hss, sp_hh, sp_rinv, inj);
    static const int hy = getenv("FLASH_HCU_Y") ? atoi(getenv("FLASH_HCU_Y")) : 1;
    k_hcu_pf2<<<dim3((HCU_TILES + 7) / 8, std::max(1, std::min(hy, (m + 31) / 32))), 256, 0, s>>>(w.uq, w.us, sp_hh, sp_rinv, R, w.norm,
                                                                                                  x, xh, m);
    KCHK("hc_rows");
}

// own rows [r0, r1) of x (bit 0), xh (bit 1), R (bit 2) to every card, then wait for the peers' rows (channel ch)
void Worker::row_ag(unsigned amask, int r0, int r1, cudaStream_t s, int ch, const int* rr) {
    RowAg a;
    for (int k = 0; k < 3; k++)
        for (int g = 0; g < 4; g++) a.dst[k][g] = agdst_[k][g];
    a.rowb[0] = H * 2;
    a.rowb[1] = H * 2;
    a.rowb[2] = RW * 4;
    for (int g = 0; g < 4; g++) a.flags[g] = agpf_[g] ? agpf_[g] + ch * 4 : nullptr;
    for (int g = 0; g < 4; g++) a.acks[g] = agpf_[g] ? agpf_[g] + 8 + ch * 4 : nullptr;
    a.seq = agseq_ + ch;
    a.scnt = agscnt_ + ch * 8;
    a.pm = 0;
    for (int g = 0; g < 8; g++) a.rr[g] = rr ? rr[g] : 0;
    static const bool ce_env = getenv("FLASH_AG_CE") && atoi(getenv("FLASH_AG_CE"));  // 1: rows by the copy engine (26K 6.05 -> 6.13 s: slower)
    const bool ce = ce_env || comm_.xfer == kXferCE;
    if (comm_.xfer == kXferHost && hag_h_ && hagf_ && rr) {
        // host staging: each card PULLS the peers' rows into its own arrays after their flags, then acknowledges, so the
        // sender keeps its pinned rows until every peer has copied them: own rows -> this card's pinned rows (D2H), flag,
        // wait, the peers' rows -> the local arrays (H2D), ack
        size_t koff[3], o = (size_t)ch * AG_CAP * (H * 2 + H * 2 + RW * 4);  // per channel (the streams run concurrently)
        for (int k = 0; k < 3; k++) koff[k] = o, o += (size_t)AG_CAP * a.rowb[k];
        RowAg ah = a;
        AckP ack;
        for (int g = 0; g < 4; g++) {
            ah.flags[g] = hagf_p_[g] ? hagf_p_[g] + ch * 8 : nullptr;
            ack.w[g] = hagf_p_[g] ? hagf_p_[g] + ch * 8 + 4 : nullptr;
        }
        k_rag_ackwait<<<1, 32, 0, s>>>(hagf_ + ch * 8 + 4, a.seq, p_.rank, p_.ngpu);
        for (int k = 0; k < 3; k++)
            if (((amask >> k) & 1) && r1 > r0) {
                CHECK(r1 - r0 <= AG_CAP, "row_ag: %d rows", r1 - r0);
                CUDA_CHECK(cudaMemcpyAsync(hag_h_ + koff[k], a.dst[k][p_.rank] + (size_t)r0 * a.rowb[k], (size_t)(r1 - r0) * a.rowb[k],
                                           cudaMemcpyDeviceToHost, s));
            }
        k_row_ag_flag<<<1, 32, 0, s>>>(ah, p_.rank, p_.ngpu);
        k_row_ag_wait<<<1, 32, 0, s>>>(ah, p_.rank, p_.ngpu);
        for (int g = 0; g < p_.ngpu; g++) {
            const int g0 = rr[2 * g], g1 = rr[2 * g + 1];
            if (g == p_.rank || g1 <= g0) continue;
            for (int k = 0; k < 3; k++)
                if ((amask >> k) & 1)
                    CUDA_CHECK(cudaMemcpyAsync(a.dst[k][p_.rank] + (size_t)g0 * a.rowb[k], hag_p_[g] + koff[k],
                                               (size_t)(g1 - g0) * a.rowb[k], cudaMemcpyHostToDevice, s));
        }
        k_rag_ack<<<1, 32, 0, s>>>(ack, a.seq, p_.rank, p_.ngpu);
        KCHK("row_ag host");
        return;
    }
    if (ce) {
        // copy engine: own rows pushed into every peer's arrays, then the sequence number into their flag words by copies
        // of the same stream (ordered after the rows)
        for (int k = 0; k < 3; k++) {
            if (!((amask >> k) & 1) || r1 <= r0) continue;
            for (int g = 0; g < p_.ngpu; g++)
                if (g != p_.rank)
                    CUDA_CHECK(cudaMemcpyAsync(a.dst[k][g] + (size_t)r0 * a.rowb[k], a.dst[k][p_.rank] + (size_t)r0 * a.rowb[k],
                                               (size_t)(r1 - r0) * a.rowb[k], cudaMemcpyDeviceToDevice, s));
        }
        if (agw_) {
            k_seq_word<<<1, 1, 0, s>>>(agw_ + ch, a.seq);
            for (int g = 0; g < p_.ngpu; g++)
                if (g != p_.rank) CUDA_CHECK(cudaMemcpyAsync(a.flags[g] + p_.rank, agw_ + ch, 4, cudaMemcpyDeviceToDevice, s));
        } else {
            k_row_ag_flag<<<1, 32, 0, s>>>(a, p_.rank, p_.ngpu);  // (FLASH_AG_CE=1 without staging: the old SM flag)
        }
    } else if (rr && comm_.pullm) {  // P2P loads instead of stores on the pulled pairs (prefill-r4)
        a.pm = comm_.pullm;
        k_row_ag_mix<<<dim3(p_.ngpu - 1, 16), 256, 0, s>>>(a, r0, r1, p_.rank, p_.ngpu, amask);
        k_row_ag_pull<<<dim3(p_.ngpu - 1, 16), 256, 0, s>>>(a, p_.rank, p_.ngpu, amask);
        k_row_ag_fin<<<1, 32, 0, s>>>(a, p_.rank, p_.ngpu);
        KCHK("row_ag pull");
        return;
    } else {
        k_row_ag<<<dim3(p_.ngpu - 1, 16), 256, 0, s>>>(a, r0, r1, p_.rank, p_.ngpu, amask);
    }
    k_row_ag_wait<<<1, 32, 0, s>>>(a, p_.rank, p_.ngpu);
    KCHK("row_ag");
}

bool Worker::hcsp_ok(int n) const {
    static const bool on = !getenv("FLASH_HCSP") || atoi(getenv("FLASH_HCSP"));
    static const bool ov = (!getenv("FLASH_AROV") || atoi(getenv("FLASH_AROV"))) && !g_kchk;
    static const bool fused = !getenv("FLASH_ARHC") || atoi(getenv("FLASH_ARHC"));
    if (!on || !ov || !fused || p_.ngpu < 2 || dbgh_ || !agpf_[0] || n <= PFM) return false;
    for (int g = 0; g < p_.ngpu; g++) if (!agpf_[g] || !agdst_[0][g]) return false;
    for (int off = 0; off < n; off += PFM)
        if (!allreduce_hc_ok(comm_, std::min(PFM, n - off), H)) return false;
    int ub[64], cb[64];
    const int nu = pf_units(n, ub, cb);
    for (int u = 0; u < nu; u++)
        if (!allreduce_hc_ok(comm_, ub[u + 1] - ub[u], H) || ub[u + 1] - ub[u] > AG_CAP * p_.ngpu) return false;
    for (const Layer& L : L_) if (!L.hca.uq || !L.hcm.uq) return false;
    if (mtpb_ && (!mtp_.hca.uq || !mtp_.hcm.uq)) return false;
    return true;
}

// Prefill batch with sequence-parallel HC (FLASH_HCSP=1): as the overlapped path of trunk_batch, but from the mixer
// all-reduce of layer 1 on, each card adds y to the residual and runs the next HC read only on the rows it owns in
// the all-reduce (a third of every chunk), then all-gathers x / xh. Layer 0 and layer 1's mixer input (PLE needs the
// residual of every row) run as before. The last row's residual is gathered for the heads.
void Worker::trunk_batch_sp1(int n, unsigned selmask) {
    static const int fastn = getenv("FLASH_AROV_FAST") ? atoi(getenv("FLASH_AROV_FAST")) : 2;
    // overlapped all-reduce blocks: 16 here (the comm stream now also carries the HC reads and gathers: 26K 6.30 -> 6.19 s)
    static const int ovb = getenv("FLASH_HCSP_OVB") ? atoi(getenv("FLASH_HCSP_OVB")) : 16;
    static const bool xh_only = !getenv("FLASH_HCSP_X") || atoi(getenv("FLASH_HCSP_X")) == 0;  // 1: gather x and xh (exact x)
    ll_idx_ = 0;
    pfs_use_ = false;
    k_step_bump<<<1, 1, 0, st_>>>(stepc_);
    float* const Rb = R_;
    bf16* const plerb = pler_;
    int* const posd0 = posd_;
    float* const partb = part_;
    float* const injb = inj_;
    bf16* const xb = x_;
    __half* const xhb = xh_;
    k_embed<<<n, 256, 0, st_>>>(embed_, toks_, Rb);
    KCHK("k_embed");
    const int nch = (n + PFM - 1) / PFM;
    auto own = [&](int c, int& o0, int& o1) {
        const int off = c * PFM, mc = std::min(PFM, n - off);
        o0 = off + (int)((long)mc * p_.rank / p_.ngpu);
        o1 = off + (int)((long)mc * (p_.rank + 1) / p_.ngpu);
    };
    CUDA_CHECK(cudaEventRecord(evm_, st_));
    CUDA_CHECK(cudaStreamWaitEvent(cst_, evm_, 0));  // fork
    pfs_use_ = pfs_now_ && pf_;
    if (pfs_use_) {
        CUDA_CHECK(cudaEventRecord(pfs_used_[0], st_));
        CUDA_CHECK(cudaStreamWaitEvent(dst_, pfs_used_[0], 0));
        pfs_copy(0);
        pfs_copy(1);
    }
    auto ar_full = [&](int c, cudaEvent_t after, bool fast, int sli = -1) {
        const int off = c * PFM, mc = std::min(PFM, n - off);
        CUDA_CHECK(cudaStreamWaitEvent(cst_, after, 0));
        CHECK(allreduce_add_hc(Rb + (size_t)off * RW, injb + (size_t)off * HC, partb + (size_t)off * H, mc, H, comm_, p_.rank, ctr_,
                               cst_, fast ? 64 : ovb), "hcsp: all-reduce");
        steer(sli, mc, nullptr, cst_, Rb + (size_t)off * RW, nullptr, nullptr);  // every row updated on every card
        CUDA_CHECK(cudaEventRecord(evd_[c], cst_));
    };
    auto ar_sp = [&](int c, cudaEvent_t after, bool fast, const HcW* next, int sli = -1) {
        const int off = c * PFM, mc = std::min(PFM, n - off);
        CUDA_CHECK(cudaStreamWaitEvent(cst_, after, 0));
        CHECK(allreduce_add_hc(Rb + (size_t)off * RW, injb + (size_t)off * HC, partb + (size_t)off * H, mc, H, comm_, p_.rank, ctr_,
                               cst_, fast ? 64 : ovb, true), "hcsp: all-reduce");
        if (sli >= 0) {  // layer output: only the rows this card updated (its own), before their next HC read
            int o0, o1;
            own(c, o0, o1);
            steer(sli, o1 - o0, nullptr, cst_, Rb + (size_t)o0 * RW, nullptr, nullptr);
        }
        if (next) {
            int o0, o1;
            own(c, o0, o1);
            int rr[8];
            for (int g = 0; g < p_.ngpu; g++) {
                rr[2 * g] = off + (int)((long)mc * g / p_.ngpu);
                rr[2 * g + 1] = off + (int)((long)mc * (g + 1) / p_.ngpu);
            }
            hc_rows(*next, Rb + (size_t)o0 * RW, injb + (size_t)o0 * HC, xb + (size_t)o0 * H, xhb + (size_t)o0 * H, o1 - o0, cst_);
            if (xh_only) {
                row_ag(2, o0, o1, cst_, 0, rr);
                k_xh_to_x<<<64, 256, 0, cst_>>>(xhb + (size_t)off * H, xb + (size_t)off * H, (size_t)mc * H / 2);
            } else {
                row_ag(3, o0, o1, cst_, 0, rr);
            }
        }
        CUDA_CHECK(cudaEventRecord(evd_[c], cst_));
    };
    auto last_row_R = [&](cudaStream_t s, int ch) {  // the last row's residual (owned by the last rank of the last chunk)
        const int off = (nch - 1) * PFM, mc = n - off;
        int rr[8];
        for (int g = 0; g < p_.ngpu; g++) {
            const int g0 = off + (int)((long)mc * g / p_.ngpu), g1 = off + (int)((long)mc * (g + 1) / p_.ngpu);
            const bool has = n - 1 >= g0 && n - 1 < g1;
            rr[2 * g] = has ? n - 1 : 0;
            rr[2 * g + 1] = has ? n : 0;
        }
        row_ag(4, rr[2 * p_.rank], rr[2 * p_.rank + 1], s, ch, rr);
    };
    for (int li = 0; li < LAYERS; li++) {
        Layer& L = L_[li];
        const bool spm = li >= 1;  // this layer's mixer all-reduce is sequence-parallel (the MoE input HC on own rows)
        const bool spx = li >= 2;  // this layer's mixer input was gathered by the previous layer's MoE all-reduce
        for (int c = 0; c < nch; c++) {
            const int off = c * PFM, mc = std::min(PFM, n - off);
            if (li > 0) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));
            posd_ = cposd_ + 2 * c;
            curpos_ = posd_;
            sel_on_ = (selmask >> c) & 1;
            R_ = Rb + (size_t)off * RW;
            pler_ = plerb + (size_t)off * H;
            inj_ = injb + (size_t)off * HC;
            part_ = partb + (size_t)off * H;
            x_ = xb + (size_t)off * H;  // x / xh rows at their batch positions (the gathers write there)
            xh_ = xhb + (size_t)off * H;
            ypend_ = false;
            if (!spx) {
                if (li == PLE_LAYER) ple(L, 0, mc, nullptr);
                hc_read(L.hca, mc, true);
            }
            if (L.full) qsa(L, li, 0, mc);
            else gdn(L, li, 0, mc);
            CUDA_CHECK(cudaEventRecord(evp_[c], st_));
            const bool fast = fastn > 0 && c == nch - 1;
            if (spm) ar_sp(c, evp_[c], fast, &L.hcm);
            else ar_full(c, evp_[c], fast);
        }
        R_ = Rb; pler_ = plerb; inj_ = injb; part_ = partb; posd_ = posd0; curpos_ = posd_; x_ = xb; xh_ = xhb;
        for (int c = 0; c < nch; c++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));
        if (!spm) hc_read(L.hcm, n, true);
        if (pfs_use_) CUDA_CHECK(cudaStreamWaitEvent(st_, pfs_cp_[li & 1], 0));
        moe(L, n);
        if (pfs_use_ && li + 2 < LAYERS) {
            CUDA_CHECK(cudaEventRecord(pfs_used_[li & 1], st_));
            CUDA_CHECK(cudaStreamWaitEvent(dst_, pfs_used_[li & 1], 0));
            pfs_copy(li + 2);
        }
        CUDA_CHECK(cudaEventRecord(evm_, st_));
        for (int c = 0; c < nch; c++) {
            if (li == 0) ar_full(c, evm_, c < fastn, li);  // layer 1's mixer input (PLE) needs every row's residual
            else ar_sp(c, evm_, c < fastn, li + 1 < LAYERS ? &L_[li + 1].hca : nullptr, li);
        }
    }
    for (int c = 0; c < nch; c++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));  // join
    ypend_ = false;
    last_row_R(st_, 1);
    x_ = xb + (size_t)(n - 1) * H;
    xh_ = xhb + (size_t)(n - 1) * H;
    R_ = Rb + (size_t)(n - 1) * RW;
    hc_read(mixer_, 1, false);
    R_ = Rb;
    if (sampling_) head_sample(1, otok_);
    else head(1, otok_);
    x_ = xb;
    xh_ = xhb;
    if (!mtpb_) return;
    // MTP block as a 49th layer: residual rows from the trunk's own rows (in place), hca on own rows + gather on st_
    // (channel 1), its mixer all-reduces on cst_ (channel 0)
    if (mtpb_last_) k_set_last_tok<<<1, 1, 0, st_>>>(mtok_, n, otok_);
    for (int c = 0; c < nch; c++) {
        const int off = c * PFM, mc = std::min(PFM, n - off);
        int o0, o1;
        own(c, o0, o1);
        const int nr = o1 - o0;
        posd_ = cposd_ + 2 * c;
        curpos_ = posd_;
        sel_on_ = (selmask >> c) & 1;
        if (nr > 0) {
            float* const Ro = Rb + (size_t)o0 * RW;
            k_mtp_prep<<<dim3(nr, HC + 1), 256, 0, st_>>>(embed_, mtok_ + o0, Ro, mtp_pre_e_, mtp_pre_h_, mtp_en_, mtp_hn_);
            rows8(mtp_fce_, mtp_en_, H, mtp_fe_, H, nr, st_);
            rows8(mtp_fch_, mtp_hn_, H, mtp_fh_, H, nr * HC, st_);
            k_mtp_comb<<<nr, 512, 0, st_>>>(mtp_fe_, mtp_fh_, Ro);
            KCHK("mtp prep sp");
            hc_rows(mtp_.hca, Ro, injb + (size_t)o0 * HC, xb + (size_t)o0 * H, xhb + (size_t)o0 * H, nr, st_);
        }
        int rr[8];
        for (int g = 0; g < p_.ngpu; g++) {
            rr[2 * g] = off + (int)((long)mc * g / p_.ngpu);
            rr[2 * g + 1] = off + (int)((long)mc * (g + 1) / p_.ngpu);
        }
        if (xh_only) {
            row_ag(2, o0, o1, st_, 1, rr);
            k_xh_to_x<<<64, 256, 0, st_>>>(xhb + (size_t)off * H, xb + (size_t)off * H, (size_t)mc * H / 2);
        } else {
            row_ag(3, o0, o1, st_, 1, rr);
        }
        R_ = Rb + (size_t)off * RW;
        inj_ = injb + (size_t)off * HC;
        part_ = partb + (size_t)off * H;
        x_ = xb + (size_t)off * H;
        xh_ = xhb + (size_t)off * H;
        ypend_ = false;
        qsa(mtp_, -1, 0, mc);
        CUDA_CHECK(cudaEventRecord(evp_[c], st_));
        ar_sp(c, evp_[c], fastn > 0 && c == nch - 1, &mtp_.hcm);
    }
    R_ = Rb; inj_ = injb; part_ = partb; posd_ = posd0; curpos_ = posd_; x_ = xb; xh_ = xhb;
    for (int c = 0; c < nch; c++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));
    moe(mtp_, n);
    CUDA_CHECK(cudaEventRecord(evm_, st_));
    for (int c = 0; c < nch; c++) ar_sp(c, evm_, false, nullptr);
    for (int c = 0; c < nch; c++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));  // join
    ypend_ = false;
    last_row_R(st_, 1);
    k_copy_row<<<RW / 512, 512, 0, st_>>>(Rmin_, Rb, nullptr, n - 1);  // the draft chain continues from the last row
    x_ = xb + (size_t)(n - 1) * H;
    xh_ = xhb + (size_t)(n - 1) * H;
    R_ = Rb + (size_t)(n - 1) * RW;
    hc_read(mtp_mixer_, 1, false);
    R_ = Rb;
    head(1, motok_, true);
    x_ = xb;
    xh_ = xhb;
}

// Prefill collective units (prefill-r2): a unit is FLASH_ARU rows of whole mixer chunks (default 2 chunks = 1024 rows;
// 1 chunk when the batch has only 2). One all-reduce (+ HC on own rows + row all-gather) per unit instead of per
// chunk, and the MoE half runs unit by unit, so every collective has a compute item after it that does not need it:
//   mixer u0 | AR(u0) under mixer u1 | AR(u1) under MoE u0 | AR(MoE u0) under MoE u1 | AR(MoE u1) under the next
//   layer's mixer u0 (the 27B's two micro-batches, with the MoE weights read once per unit).
// ub: nu + 1 row bounds, cb: nu + 1 chunk bounds; returns nu
int Worker::pf_units(int n, int* ub, int* cb) const {
    static const int aru = getenv("FLASH_ARU") ? atoi(getenv("FLASH_ARU")) : ARU_MAX;
    const int nch = (n + PFM - 1) / PFM;
    int cpu = std::max(1, std::min(ARU_MAX, aru) / PFM);
    if (nch <= 2) cpu = 1;
    int nu = 0;
    for (int c = 0; c < nch; c += cpu, nu++) {
        cb[nu] = c;
        ub[nu] = c * PFM;
    }
    cb[nu] = nch;
    ub[nu] = n;
    return nu;
}

void Worker::trunk_batch_sp(int n, unsigned selmask) {
    static const bool units_on = !getenv("FLASH_PFU") || atoi(getenv("FLASH_PFU"));  // 0: per-chunk collectives (before r2)
    if (!units_on) { trunk_batch_sp1(n, selmask); return; }
    // all-reduce blocks of a unit (every collective is overlapped now)
    static const int ovb = getenv("FLASH_PFU_B") ? atoi(getenv("FLASH_PFU_B")) : 16;
    ll_idx_ = 0;
    pfs_use_ = false;
    k_step_bump<<<1, 1, 0, st_>>>(stepc_);
    float* const Rb = R_;
    bf16* const plerb = pler_;
    int* const posd0 = posd_;
    float* const partb = part_;
    float* const injb = inj_;
    bf16* const xb = x_;
    __half* const xhb = xh_;
    k_embed<<<n, 256, 0, st_>>>(embed_, toks_, Rb);
    KCHK("k_embed");
    int ub[64], cb[64];
    const int nu = pf_units(n, ub, cb);
    CHECK(nu <= (int)evd_.size(), "prefill units %d", nu);
    // MoE parts: whole units of FLASH_MOE_U rows at most (default: one unit; a larger part reads the expert weights
    // once for more rows, its collectives still go unit by unit), at least 2 parts when there are 2 units
    // (big batches, n > pfbn_base_: 2048-row parts, prefill-r3)
    static const int moeu_e = getenv("FLASH_MOE_U") ? atoi(getenv("FLASH_MOE_U")) : -1;
    const int moeu = moeu_e >= 0 ? moeu_e : (n > pfbn_base_ ? 2048 : 0);
    int mb[65], nmp = 0;
    for (int u = 0; u < nu;) {
        mb[nmp++] = u;
        int v = u + 1;
        while (v < nu && moeu > 0 && ub[v + 1] - ub[u] <= moeu && (nmp > 1 || v + 1 < nu)) v++;
        u = v;
    }
    mb[nmp] = nu;
    // a layer whose MoE reads many host-backed groups zero-copy (not staged: a batch below FLASH_PF_STAGE_MIN, or past
    // the ring's capacity, e.g. 2 cards with ~15 GB host-backed) runs its MoE over the whole batch: every part would
    // read them over PCIe again (2 cards 26K: 9.8 -> 10.5 s with 2 parts)
    static const double zc_mb = getenv("FLASH_MOE_ZC_MB") ? atof(getenv("FLASH_MOE_ZC_MB")) : 8.0;
    const int mb1[2] = {0, nu};
    auto parts_of = [&](const Layer& L, const int*& pb) {
        const int zc = pfs_use_ && &L != &mtp_ ? std::max(0, L.ncold - pfs_cap_) : L.ncold;  // (MTP: never staged)
        if (nmp > 1 && (double)zc * GRP_BYTES > zc_mb * 1048576.0) { pb = mb1; return 1; }
        pb = mb;
        return nmp;
    };
    auto rows_of = [&](int u, int g) { return ub[u] + (int)((long)(ub[u + 1] - ub[u]) * g / p_.ngpu); };
    CUDA_CHECK(cudaEventRecord(evm_, st_));
    CUDA_CHECK(cudaStreamWaitEvent(cst_, evm_, 0));  // fork
    pfs_use_ = pfs_now_ && pf_;
    if (pfs_use_) {
        CUDA_CHECK(cudaEventRecord(pfs_used_[0], st_));
        CUDA_CHECK(cudaStreamWaitEvent(dst_, pfs_used_[0], 0));
        pfs_copy(0);
        pfs_copy(1);
    }
    // the collective of unit u on cst_ after event `after`: sp = sequence-parallel (R updated on own rows, then the
    // next HC read on them and the all-gather of xh), else every row updated on every card; sli: steered layer output
    auto ar_unit = [&](int u, cudaEvent_t after, bool sp, const HcW* next, int sli) {
        const int off = ub[u], mc = ub[u + 1] - off;
        CUDA_CHECK(cudaStreamWaitEvent(cst_, after, 0));
        CHECK(allreduce_add_hc(Rb + (size_t)off * RW, injb + (size_t)off * HC, partb + (size_t)off * H, mc, H, comm_, p_.rank,
                               ctr_, cst_, ovb, sp), "prefill unit: all-reduce");
        const int o0 = rows_of(u, p_.rank), o1 = rows_of(u, p_.rank + 1);
        if (!sp) steer(sli, mc, nullptr, cst_, Rb + (size_t)off * RW, nullptr, nullptr);
        else if (sli >= 0) steer(sli, o1 - o0, nullptr, cst_, Rb + (size_t)o0 * RW, nullptr, nullptr);
        if (sp && next) {
            int rr[8];
            for (int g = 0; g < p_.ngpu; g++) rr[2 * g] = rows_of(u, g), rr[2 * g + 1] = rows_of(u, g + 1);
            hc_rows(*next, Rb + (size_t)o0 * RW, injb + (size_t)o0 * HC, xb + (size_t)o0 * H, xhb + (size_t)o0 * H, o1 - o0, cst_);
            row_ag(2, o0, o1, cst_, 0, rr);
            k_xh_to_x<<<64, 256, 0, cst_>>>(xhb + (size_t)off * H, xb + (size_t)off * H, (size_t)mc * H / 2);
        }
        CUDA_CHECK(cudaEventRecord(evd_[u], cst_));
    };
    auto at_rows = [&](int off) {  // batch row views at row off
        R_ = Rb + (size_t)off * RW;
        pler_ = plerb + (size_t)off * H;
        inj_ = injb + (size_t)off * HC;
        part_ = partb + (size_t)off * H;
        x_ = xb + (size_t)off * H;
        xh_ = xhb + (size_t)off * H;
    };
    auto mixer_chunk = [&](Layer& L, int li, int c, bool spx) {
        const int off = c * PFM, mc = std::min(PFM, n - off);
        posd_ = cposd_ + 2 * c;
        curpos_ = posd_;
        sel_on_ = (selmask >> c) & 1;
        at_rows(off);
        ypend_ = false;
        if (!spx) {
            if (li == PLE_LAYER) ple(L, 0, mc, nullptr);
            hc_read(L.hca, mc, true);
        }
        if (L.full) qsa(L, li, 0, mc);
        else gdn(L, li, 0, mc);
    };
    auto last_row_R = [&](cudaStream_t s, int ch) {  // the last row's residual (owned by one rank of the last unit)
        const int u = nu - 1;
        int rr[8];
        for (int g = 0; g < p_.ngpu; g++) {
            const bool has = n - 1 >= rows_of(u, g) && n - 1 < rows_of(u, g + 1);
            rr[2 * g] = has ? n - 1 : 0;
            rr[2 * g + 1] = has ? n : 0;
        }
        row_ag(4, rr[2 * p_.rank], rr[2 * p_.rank + 1], s, ch, rr);
    };
    auto reset_rows = [&]() { at_rows(0); posd_ = posd0; curpos_ = posd_; };
    for (int li = 0; li < LAYERS; li++) {
        Layer& L = L_[li];
        const bool spm = li >= 1;  // this layer's mixer all-reduce is sequence-parallel (the MoE input HC on own rows)
        const bool spx = li >= 2;  // this layer's mixer input was gathered by the previous layer's MoE all-reduce
        // QSA layers, P2P: chunk c's partial combine + o_proj after chunk c + 1's front half (FLASH_QSA_PIPE=1)
        static const bool qpipe = getenv("FLASH_QSA_PIPE") && atoi(getenv("FLASH_QSA_PIPE"));  // opt-in: 26K -0.3 % (noise)
        if (L.full && qpipe && projB_ && !xstage_ && comm_.p2p && !dbgh_ && cb[nu] > 1) {
            int pend = -1, pend_u = -1;
            auto back = [&](int c, int uu) {
                at_rows(c * PFM);
                qsa_back(c & 1);
                if (c == cb[uu + 1] - 1) {  // the unit's last chunk: its collective
                    CUDA_CHECK(cudaEventRecord(evp_[uu], st_));
                    ar_unit(uu, evp_[uu], spm, spm ? &L.hcm : nullptr, -1);
                }
            };
            for (int u = 0; u < nu; u++) {
                if (li > 0) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[u], 0));  // the previous layer's MoE unit u
                for (int c = cb[u]; c < cb[u + 1]; c++) {
                    qsa_split_ = 1;
                    qsa_buf_ = c & 1;
                    qsa_ahead_ = pend >= 0 ? 1 : 0;
                    mixer_chunk(L, li, c, spx);
                    qsa_split_ = 0;
                    if (pend >= 0) back(pend, pend_u);
                    pend = c;
                    pend_u = u;
                }
            }
            back(pend, pend_u);
            qsa_ahead_ = 0;
        } else
        for (int u = 0; u < nu; u++) {
            if (li > 0) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[u], 0));  // the previous layer's MoE unit u
            for (int c = cb[u]; c < cb[u + 1]; c++) mixer_chunk(L, li, c, spx);
            CUDA_CHECK(cudaEventRecord(evp_[u], st_));
            ar_unit(u, evp_[u], spm, spm ? &L.hcm : nullptr, -1);
        }
        const int* pb = mb;
        const int np = parts_of(L, pb);
        for (int p = 0; p < np; p++) {  // MoE parts: units [pb[p], pb[p + 1])
            const int u0 = pb[p], u1 = pb[p + 1], off = ub[u0], mc = ub[u1] - off;
            for (int u = u0; u < u1; u++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[u], 0));
            reset_rows();
            at_rows(off);
            if (!spm) hc_read(L.hcm, mc, true);
            if (p == 0 && pfs_use_) CUDA_CHECK(cudaStreamWaitEvent(st_, pfs_cp_[li & 1], 0));
            moe(L, mc);
            CUDA_CHECK(cudaEventRecord(evp_[u0], st_));
            for (int u = u0; u < u1; u++) {
                if (li == 0) ar_unit(u, evp_[u0], false, nullptr, li);  // layer 1's mixer input (PLE) needs every row's residual
                else ar_unit(u, evp_[u0], true, li + 1 < LAYERS ? &L_[li + 1].hca : nullptr, li);
            }
        }
        reset_rows();
        if (pfs_use_ && li + 2 < LAYERS) {
            CUDA_CHECK(cudaEventRecord(pfs_used_[li & 1], st_));
            CUDA_CHECK(cudaStreamWaitEvent(dst_, pfs_used_[li & 1], 0));
            pfs_copy(li + 2);
        }
    }
    for (int u = 0; u < nu; u++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[u], 0));  // join
    ypend_ = false;
    last_row_R(st_, 1);
    x_ = xb + (size_t)(n - 1) * H;
    xh_ = xhb + (size_t)(n - 1) * H;
    R_ = Rb + (size_t)(n - 1) * RW;
    hc_read(mixer_, 1, false);
    R_ = Rb;
    if (sampling_) head_sample(1, otok_);
    else head(1, otok_);
    x_ = xb;
    xh_ = xhb;
    if (!mtpb_) return;
    // MTP block as a 49th layer: residual rows from the trunk's own rows (in place), hca on own rows + gather on st_
    // (channel 1), its collectives on cst_ (channel 0), unit by unit as the trunk
    if (mtpb_last_) k_set_last_tok<<<1, 1, 0, st_>>>(mtok_, n, otok_);
    for (int u = 0; u < nu; u++) {
        const int off = ub[u], mc = ub[u + 1] - off;
        const int o0 = rows_of(u, p_.rank), o1 = rows_of(u, p_.rank + 1), nr = o1 - o0;
        if (nr > 0) {
            float* const Ro = Rb + (size_t)o0 * RW;
            k_mtp_prep<<<dim3(nr, HC + 1), 256, 0, st_>>>(embed_, mtok_ + o0, Ro, mtp_pre_e_, mtp_pre_h_, mtp_en_, mtp_hn_);
            rows8(mtp_fce_, mtp_en_, H, mtp_fe_, H, nr, st_);
            rows8(mtp_fch_, mtp_hn_, H, mtp_fh_, H, nr * HC, st_);
            k_mtp_comb<<<nr, 512, 0, st_>>>(mtp_fe_, mtp_fh_, Ro);
            KCHK("mtp prep sp");
            hc_rows(mtp_.hca, Ro, injb + (size_t)o0 * HC, xb + (size_t)o0 * H, xhb + (size_t)o0 * H, nr, st_);
        }
        int rr[8];
        for (int g = 0; g < p_.ngpu; g++) rr[2 * g] = rows_of(u, g), rr[2 * g + 1] = rows_of(u, g + 1);
        row_ag(2, o0, o1, st_, 1, rr);
        k_xh_to_x<<<64, 256, 0, st_>>>(xhb + (size_t)off * H, xb + (size_t)off * H, (size_t)mc * H / 2);
        for (int c = cb[u]; c < cb[u + 1]; c++) {
            const int coff = c * PFM, cm = std::min(PFM, n - coff);
            posd_ = cposd_ + 2 * c;
            curpos_ = posd_;
            sel_on_ = (selmask >> c) & 1;
            at_rows(coff);
            ypend_ = false;
            qsa(mtp_, -1, 0, cm);
        }
        CUDA_CHECK(cudaEventRecord(evp_[u], st_));
        ar_unit(u, evp_[u], true, &mtp_.hcm, -1);
    }
    {
        const int* pb = mb;
        const int np = parts_of(mtp_, pb);
        for (int p = 0; p < np; p++) {
            const int u0 = pb[p], u1 = pb[p + 1], off = ub[u0], mc = ub[u1] - off;
            for (int u = u0; u < u1; u++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[u], 0));
            reset_rows();
            at_rows(off);
            moe(mtp_, mc);
            CUDA_CHECK(cudaEventRecord(evp_[u0], st_));
            for (int u = u0; u < u1; u++) ar_unit(u, evp_[u0], true, nullptr, -1);
        }
    }
    reset_rows();
    for (int u = 0; u < nu; u++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[u], 0));  // join
    ypend_ = false;
    last_row_R(st_, 1);
    k_copy_row<<<RW / 512, 512, 0, st_>>>(Rmin_, Rb, nullptr, n - 1);  // the draft chain continues from the last row
    x_ = xb + (size_t)(n - 1) * H;
    xh_ = xhb + (size_t)(n - 1) * H;
    R_ = Rb + (size_t)(n - 1) * RW;
    hc_read(mtp_mixer_, 1, false);
    R_ = Rb;
    head(1, motok_, true);
    x_ = xb;
    xh_ = xhb;
}

void Worker::trunk_batch(int n, unsigned selmask) {
    // staged prefill transfers: the QSA exchanges of this batch address their receive slots on the host (the counters
    // start even, so exchange k of the batch uses slot k & 1, also in a replayed graph)
    struct Stage {
        Worker* w;
        explicit Stage(Worker* x) : w(x) {
            if (!w->xout_) return;
            k_xpar_align<<<1, 1, 0, w->st_>>>(w->xseq_, w->xqseq_);
            w->xstage_ = true;
            w->xq_k_ = w->xc_k_ = 0;
        }
        ~Stage() { w->xstage_ = false; }
    } stage(this);
    if (hcsp_ok(n)) {
        static bool said = false;
        if (!said && p_.rank == 0) { said = true; fprintf(stderr, "[flash] prefill: sequence-parallel HC (FLASH_HCSP)\n"); }
        trunk_batch_sp(n, selmask);
        return;
    }
    static const bool ll = !getenv("FLASH_LL") || atoi(getenv("FLASH_LL"));
    static const bool ov = (!getenv("FLASH_AROV") || atoi(getenv("FLASH_AROV"))) && !g_kchk;  // FLASH_AROV=0: same stream
    ll_idx_ = 0;
    pfs_use_ = false;  // staged cold groups only in the overlapped multi-chunk path below (it issues the copies)
    k_step_bump<<<1, 1, 0, st_>>>(stepc_);
    float* const Rb = R_;
    bf16* const plerb = pler_;
    int* const posd0 = posd_;
    k_embed<<<n, 256, 0, st_>>>(embed_, toks_, Rb);
    KCHK("k_embed");
    const int nch = (n + PFM - 1) / PFM;
    if (ov && nch > 1) {
        // all-reduces + residual updates (R_c += inj_c * y_c) on cst_, chunk by chunk: the mixer of chunk c + 1 does not
        // need chunk c's sum (KV / DeltaNet state are written before the AR), the MoE half waits for all chunks, the
        // next layer's chunk c only for MoE piece c. Rows of part_ / y_ / inj_ are per chunk (batch-sized buffers).
        float* const partb = part_;
        float* const yb = y_;
        float* const injb = inj_;
        CUDA_CHECK(cudaEventRecord(evm_, st_));
        CUDA_CHECK(cudaStreamWaitEvent(cst_, evm_, 0));  // fork (graph capture: cst_ joins here)
        // fast = the all-reduce is on the critical path (the last mixer chunk before the MoE half, the first MoE piece
        // before the next layer): 64 blocks instead of the few overlapped ones
        static const int fastn = getenv("FLASH_AROV_FAST") ? atoi(getenv("FLASH_AROV_FAST")) : 2;
        auto ar_chunk = [&](int c, cudaEvent_t after, bool fast = false, int sli = -1) {
            const int off = c * PFM, mc = std::min(PFM, n - off);
            CUDA_CHECK(cudaStreamWaitEvent(cst_, after, 0));
            static const int sc = getenv("FLASH_AROV_SPINC") ? atoi(getenv("FLASH_AROV_SPINC")) : 0;  // debug: lag the comm stream (ns)
            if (sc) k_spin<<<1, 1, 0, cst_>>>(sc);
            static const bool fused = !getenv("FLASH_ARHC") || atoi(getenv("FLASH_ARHC"));  // residual update in the AR tail
            static const int ovb = getenv("QE_AR_OV_BLOCKS") ? atoi(getenv("QE_AR_OV_BLOCKS")) : 8;  // fused tail: 8 (sweep 4 / 8 / 12 / 16)
            if (!(fused && allreduce_add_hc(Rb + (size_t)off * RW, injb + (size_t)off * HC, partb + (size_t)off * H, mc, H, comm_,
                                            p_.rank, ctr_, cst_, fast ? 64 : ovb))) {
                allreduce_add(yb + (size_t)off * H, partb + (size_t)off * H, mc, H, comm_, p_.rank, ctr_, cst_, nullptr, 0.f,
                              nullptr, -1, !fast);
                k_hc_write<<<mc, 512, 0, cst_>>>(Rb + (size_t)off * RW, injb + (size_t)off * HC, yb + (size_t)off * H);
            }
            steer(sli, mc, nullptr, cst_, Rb + (size_t)off * RW, nullptr, nullptr);  // layer output of the chunk's rows
            CUDA_CHECK(cudaEventRecord(evd_[c], cst_));
        };
        // cold expert groups of layer li + 2 by copy engine while layers li + 1 / li + 2 compute (FLASH_PF_STAGE)
        pfs_use_ = pfs_now_ && pf_;  // decided per batch in prefill_batch (graph key)
        if (pfs_use_) {
            CUDA_CHECK(cudaEventRecord(pfs_used_[0], st_));
            CUDA_CHECK(cudaStreamWaitEvent(dst_, pfs_used_[0], 0));  // fork (graph capture: joined by the MoE waits)
            pfs_copy(0);
            pfs_copy(1);
        }
        for (int li = 0; li < LAYERS; li++) {
            Layer& L = L_[li];
            for (int c = 0; c < nch; c++) {
                const int off = c * PFM, mc = std::min(PFM, n - off);
                if (li > 0) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));  // MoE piece c of the previous layer
                static const int sm = getenv("FLASH_AROV_SPINM") ? atoi(getenv("FLASH_AROV_SPINM")) : 0;  // debug: lag compute
                if (sm && ((li * 7 + c * 3 + p_.rank) % 5 == 0)) k_spin<<<1, 1, 0, st_>>>(sm);
                posd_ = cposd_ + 2 * c;
                curpos_ = posd_;
                sel_on_ = (selmask >> c) & 1;
                R_ = Rb + (size_t)off * RW;
                pler_ = plerb + (size_t)off * H;
                inj_ = injb + (size_t)off * HC;
                part_ = partb + (size_t)off * H;
                ypend_ = false;
                if (li == PLE_LAYER) ple(L, 0, mc, nullptr);
                hc_read(L.hca, mc, true);
                if (L.full) qsa(L, li, 0, mc);
                else gdn(L, li, 0, mc);
                CUDA_CHECK(cudaEventRecord(evp_[c], st_));
                ar_chunk(c, evp_[c], fastn > 0 && c == nch - 1);
            }
            R_ = Rb;
            pler_ = plerb;
            inj_ = injb;
            part_ = partb;
            posd_ = posd0;
            curpos_ = posd_;
            for (int c = 0; c < nch; c++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));
            if (dbgh_) k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(Rb), (size_t)n * RW, dbgh_ + li * DBGK);
            hc_read(L.hcm, n, true);
            if (dbgh_) k_dbg_hash<<<64, 256, 0, st_>>>(reinterpret_cast<const unsigned*>(x_), (size_t)n * H / 2, dbgh_ + li * DBGK + 3);
            if (pfs_use_) CUDA_CHECK(cudaStreamWaitEvent(st_, pfs_cp_[li & 1], 0));
            moe(L, n);
            if (pfs_use_ && li + 2 < LAYERS) {  // ring half li & 1 is free again: layer li + 2 goes there
                CUDA_CHECK(cudaEventRecord(pfs_used_[li & 1], st_));
                CUDA_CHECK(cudaStreamWaitEvent(dst_, pfs_used_[li & 1], 0));
                pfs_copy(li + 2);
            }
            CUDA_CHECK(cudaEventRecord(evm_, st_));
            for (int c = 0; c < nch; c++) ar_chunk(c, evm_, c < fastn, li);
        }
        if (!mtpb_) {
            for (int c = 0; c < nch; c++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));  // join
            ypend_ = false;
            R_ = Rb + (size_t)(n - 1) * RW;
            hc_read(mixer_, 1, false);
            R_ = Rb;
            if (sampling_) head_sample(1, otok_);
            else head(1, otok_);
            return;
        }
        // trunk head of the last row first (its token is the last row's MTP input when this is the prompt's last batch),
        // then the MTP block as a 49th layer in the same pipeline, its residual in place of the trunk rows it consumed
        for (int c = 0; c < nch; c++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));
        ypend_ = false;
        R_ = Rb + (size_t)(n - 1) * RW;
        hc_read(mixer_, 1, false);
        R_ = Rb;
        if (sampling_) head_sample(1, otok_);
        else head(1, otok_);
        if (mtpb_last_) k_set_last_tok<<<1, 1, 0, st_>>>(mtok_, n, otok_);
        for (int c = 0; c < nch; c++) {
            const int off = c * PFM, mc = std::min(PFM, n - off);
            posd_ = cposd_ + 2 * c;
            curpos_ = posd_;
            sel_on_ = (selmask >> c) & 1;
            float* const Rc = Rb + (size_t)off * RW;
            k_mtp_prep<<<dim3(mc, HC + 1), 256, 0, st_>>>(embed_, mtok_ + off, Rc, mtp_pre_e_, mtp_pre_h_, mtp_en_, mtp_hn_);
            rows8(mtp_fce_, mtp_en_, H, mtp_fe_, H, mc, st_);
            rows8(mtp_fch_, mtp_hn_, H, mtp_fh_, H, mc * HC, st_);
            k_mtp_comb<<<mc, 512, 0, st_>>>(mtp_fe_, mtp_fh_, Rc);  // MTP residual rows in place
            KCHK("mtp prep");
            R_ = Rc;
            inj_ = injb + (size_t)off * HC;
            part_ = partb + (size_t)off * H;
            ypend_ = false;
            hc_read(mtp_.hca, mc, true);
            qsa(mtp_, -1, 0, mc);
            CUDA_CHECK(cudaEventRecord(evp_[c], st_));
            ar_chunk(c, evp_[c], fastn > 0 && c == nch - 1);
        }
        R_ = Rb;
        inj_ = injb;
        part_ = partb;
        posd_ = posd0;
        curpos_ = posd_;
        for (int c = 0; c < nch; c++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));
        hc_read(mtp_.hcm, n, true);
        moe(mtp_, n);
        CUDA_CHECK(cudaEventRecord(evm_, st_));
        for (int c = 0; c < nch; c++) ar_chunk(c, evm_, false);
        for (int c = 0; c < nch; c++) CUDA_CHECK(cudaStreamWaitEvent(st_, evd_[c], 0));  // join
        ypend_ = false;
        k_copy_row<<<RW / 512, 512, 0, st_>>>(Rmin_, Rb, nullptr, n - 1);  // the draft chain continues from the last row
        R_ = Rb + (size_t)(n - 1) * RW;
        hc_read(mtp_mixer_, 1, false);
        R_ = Rb;
        head(1, motok_, true);
        return;
    }
    for (int li = 0; li < LAYERS; li++) {
        Layer& L = L_[li];
        for (int c = 0; c < nch; c++) {
            const int off = c * PFM, mc = std::min(PFM, n - off);
            posd_ = cposd_ + 2 * c;
            curpos_ = posd_;
            sel_on_ = (selmask >> c) & 1;
            R_ = Rb + (size_t)off * RW;
            pler_ = plerb + (size_t)off * H;
            if (li == PLE_LAYER) ple(L, 0, mc, nullptr);
            hc_read(L.hca, mc, true);
            if (L.full) qsa(L, li, 0, mc);
            else gdn(L, li, 0, mc);
            hc_write(mc);
            flush_y(mc);
        }
        R_ = Rb;
        pler_ = plerb;
        posd_ = posd0;
        curpos_ = posd_;
        hc_read(L.hcm, n, true);
        moe(L, n);
        for (int off = 0; off < n; off += PFM) {
            const int mc = std::min(PFM, n - off);
            allreduce_add(y_ + (size_t)off * H, part_ + (size_t)off * H, mc, H, comm_, p_.rank, ctr_, st_, nullptr, 0.f, nullptr,
                          ll && mc <= kMaxBatchRows ? ll_idx_++ : -1);
        }
        KCHK("AR batch");
        ypend_ = true;
        flush_y(n);
        steer(li, n, nullptr, st_, R_, nullptr, nullptr);
    }
    R_ = Rb + (size_t)(n - 1) * RW;  // only the last row needs the head
    hc_read(mixer_, 1, false);
    R_ = Rb;
    if (sampling_) head_sample(1, otok_);
    else head(1, otok_);
}

// y (the all-reduced mixer / MoE sums before the residual update) is an accumulator: allreduce_add adds into it and its
// consumer (k_hc_write, the fused HC read) clears the rows it used, so it must be zero wherever a collective adds into
// it. Rows past pfbn_base_ are expert cache slots between big batches (bt_regs_), i.e. arbitrary bytes when a big batch
// takes them: those rows of y are cleared at the start of every big batch (inside its graph). Without this, a chunk whose
// all-reduce takes the generic path (allreduce_add into y + k_hc_write: a last chunk of < 3 x ngpu rows, or FLASH_AROV=0)
// added the sums to expert weights: NaN / garbage residual rows from that layer on (1.3-rc2 server garble: batches end at
// the checkpoints, e.g. [20986, 24576) = 3590 rows, last chunk 6 rows; flash_run never cut there).
void Worker::bt_clear_y(int n) {
    if (n > pfbn_base_) CUDA_CHECK(cudaMemsetAsync(y_ + (size_t)pfbn_base_ * H, 0, (size_t)(n - pfbn_base_) * H * 4, st_));
}

void Worker::prefill_batch(const int* tokens, int pos, int n, const bf16* ple_rows, int* out, const int* mtp_toks, bool last,
                           int* draft) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CHECK(n >= 1 && n <= pfbn_, "flash prefill_batch n=%d", n);
    CHECK(pos + n <= max_ctx_, "flash: context %d > %d", pos + n, max_ctx_);
    bt_want_ = n > pfbn_base_;
    CHECK(!bt_want_ || ec_on_ || !ec_n_, "flash: big batch without the expert cache");
    static const bool pft_host = getenv("FLASH_PFT_HOST") && atoi(getenv("FLASH_PFT_HOST"));  // debug: host phases
    const double th0 = pft_host ? now_s() : 0.0;
    ec_tick(pos + n, 1);
    const double th1 = pft_host ? now_s() : 0.0;
    bt_want_ = false;  // (the next tick gives the tails back to the cache unless it is another big batch)
    // cold groups staged by the copy engine for batches of >= FLASH_PF_STAGE_MIN rows (short ones read only the groups
    // they touch, zero-copy)
    static const int pfs_min = getenv("FLASH_PF_STAGE_MIN") ? atoi(getenv("FLASH_PF_STAGE_MIN")) : 2048;
    pfs_now_ = pfs_mode_ && n >= pfs_min && (n + PFM - 1) / PFM > 1 && pfs_borrow();
    use(pfb_);
    pf_ = true;
    rd_pos_ = pos;
    const unsigned sel = batch_sel(pos, n);
    memcpy(pfh_tok_, tokens, (size_t)n * 4);
    memcpy(pfh_ple_, ple_rows, (size_t)n * H * 2);
    static const bool mtp_in = !getenv("FLASH_MTPB") || atoi(getenv("FLASH_MTPB"));  // MTP fill inside the batch pipeline
    mtpb_ = mtp_toks && has_mtp_ && mtp_in && (n + PFM - 1) / PFM > 1;
    mtpb_last_ = last;
    if (mtpb_) memcpy(pfh_mt_, mtp_toks, (size_t)n * 4);
    const int nch = (n + PFM - 1) / PFM;
    const long long key = ((long long)sel << 24) | ((long long)n << 3) | ((long long)sampling_ << 1) | (prev_m_ > 0 ? 1 : 0) |
                          ((long long)con_ << 50) |
                          ((long long)mtpb_ << 61) | ((long long)(mtpb_ && last) << 62) | ((long long)pfs_now_ << 58);
    const double th2 = pft_host ? now_s() : 0.0;
    run_cached(pbgraphs_, key, [&] {
        CUDA_CHECK(cudaMemcpyAsync(cposd_, cposh_, (size_t)2 * nch * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(toks_, pfh_tok_, (size_t)n * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(pler_, pfh_ple_, (size_t)n * H * 2, cudaMemcpyHostToDevice, st_));
        if (mtpb_) CUDA_CHECK(cudaMemcpyAsync(mtok_, pfh_mt_, (size_t)n * 4, cudaMemcpyHostToDevice, st_));
        bt_clear_y(n);
        comm_begin_unit(ctr_, st_);
        trunk_batch(n, sel);
        CUDA_CHECK(cudaMemcpyAsync(pfh_out_, otok_, 4, cudaMemcpyDeviceToHost, st_));
        if (mtpb_) CUDA_CHECK(cudaMemcpyAsync(pfh_out_ + 1, motok_, 4, cudaMemcpyDeviceToHost, st_));
    });
    const double th3 = pft_host ? now_s() : 0.0;
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    if (pft_host)
        fprintf(stderr, "[pft] rank %d pos %d n %d: tick %.2f ms, prep %.2f, launch %.2f, sync %.2f\n", p_.rank, pos, n, (th1 - th0) * 1e3,
                (th2 - th1) * 1e3, (th3 - th2) * 1e3, (now_s() - th3) * 1e3);
    static const bool ecv = getenv("FLASH_EC_VERIFY") && atoi(getenv("FLASH_EC_VERIFY"));
    if (ecv) { char w[48]; snprintf(w, sizeof(w), "after prefill batch pos %d", pos); ec_verify(w); }
    if (selst_) {
        unsigned long long h[3];
        CUDA_CHECK(cudaMemcpy(h, selst_, 24, cudaMemcpyDeviceToHost));
        fprintf(stderr, "[selstat] rank %d: %llu groups of 8 rows, mean selected blocks/row %.1f, union per 8 rows %.1f (x%.2f of one row)\n",
                p_.rank, h[2], h[2] ? h[0] / 8.0 / h[2] : 0.0, h[2] ? (double)h[1] / h[2] : 0.0, h[0] ? 8.0 * h[1] / h[0] : 0.0);
    }
    *out = pfh_out_[0];
    if (draft) *draft = mtpb_ ? pfh_out_[1] : -1;
    const bool done_mtp = mtpb_;
    mtpb_ = false;
    if (draft && !done_mtp) *draft = -2;  // not handled here: the caller runs prefill_batch_mtp
    pfs_now_ = false;
    pf_ = false;
    use(dec_);
    prev_m_ = 0;  // prefill rows are committed already
}

// MTP block over the rows of the last prefill_batch (its trunk residual rows are still in the batch R), chunk by chunk
void Worker::prefill_batch_mtp(const int* mtp_toks, int pos, int n, int* draft) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CHECK(has_mtp_ && n >= 1 && n <= pfbn_, "flash prefill_batch_mtp n=%d", n);
    bt_want_ = n > pfbn_base_;
    ec_tick(pos + n, 1);
    bt_want_ = false;
    use(pfb_);
    pf_ = true;
    const unsigned sel = batch_sel(pos, n);
    memcpy(pfh_mt_, mtp_toks, (size_t)n * 4);
    const int nch = (n + PFM - 1) / PFM;
    const long long key = ((long long)sel << 24) | ((long long)n << 3) | (1LL << 60);
    run_cached(pbgraphs_, key, [&] {
        CUDA_CHECK(cudaMemcpyAsync(cposd_, cposh_, (size_t)2 * nch * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(mtok_, pfh_mt_, (size_t)n * 4, cudaMemcpyHostToDevice, st_));
        bt_clear_y(n);
        comm_begin_unit(ctr_, st_);
        ll_idx_ = 0;
        int* const mpos0 = mposd_;
        float* const rs0 = Rsave_;
        for (int c = 0; c < nch; c++) {
            const int off = c * PFM, mc = std::min(PFM, n - off);
            mposd_ = cposd_ + 2 * c;
            Rsave_ = R_ + (size_t)off * RW;
            sel_on_ = (sel >> c) & 1;
            mtp_core(mc, false, mtok_ + off, motok_, nullptr);
        }
        mposd_ = mpos0;
        Rsave_ = rs0;
        CUDA_CHECK(cudaMemcpyAsync(pfh_out_ + 1, motok_, 4, cudaMemcpyDeviceToHost, st_));
    });
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    *draft = pfh_out_[1];
    pf_ = false;
    use(dec_);
}

void Worker::prefill_mtp(const int* mtp_toks, int pos, int m, int* draft) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CHECK(has_mtp_ && m >= 1 && m <= PFM, "flash prefill_mtp m=%d", m);
    ec_tick(pos + m, 1);
    use(pfb_);
    pf_ = true;
    sel_on_ = pos + m > IDX_BUDGET + IDX_R - 1;
    pfh_pos_[0] = pos;
    memcpy(pfh_mt_, mtp_toks, (size_t)m * 4);
    const long long key = ((long long)m << 4) | (sel_on_ << 1) | (1LL << 40);
    run_cached(pgraphs_, key, [&] {
        CUDA_CHECK(cudaMemcpyAsync(mposd_, pfh_pos_, 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(mtok_, pfh_mt_, (size_t)m * 4, cudaMemcpyHostToDevice, st_));
        comm_begin_unit(ctr_, st_);
        ll_idx_ = 0;
        mtp_core(m, false, mtok_, motok_, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(pfh_out_ + 1, motok_, 4, cudaMemcpyDeviceToHost, st_));
    });
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    *draft = pfh_out_[1];
    pf_ = false;
    use(dec_);
}

// One speculative step entirely on the device: verify rows vtok_ (last emitted token + k drafts) at posd_[0], accept
// the matching prefix, MTP over the k + 1 verify rows (draft 1 from the last accepted row), the draft chain, and
// the next step's inputs; emitted tokens -> em_ (host-mapped: [count, tokens...]).
void Worker::spec_body(int k, bool lookup, int kd) {
    const int m = k + 1;
    comm_begin_unit(ctr_, st_);
    if (lookup) k_vtok_lk<<<1, 32, 0, st_>>>(vtok_, lk_, k);  // drafts of the previous step's lookup
    if (ngp_.flags) {  // staged rows (table not pinned): ids -> host -> rows
        NgramDev g = ngp_;
        g.phase = 1;
        k_ngram<<<m, 256, 0, st_>>>(vtok_, hist2_, g, pler_);
        k_ngram_handoff<<<1, 32, 0, st_>>>(g, m * PLE_HEADS);
        g.phase = 2;
        k_ngram<<<m, 256, 0, st_>>>(vtok_, hist2_, g, pler_);
    } else
    k_ngram<<<m, 256, 0, st_>>>(vtok_, hist2_, ngp_, pler_);
    KCHK("k_ngram");
    trunk(m, vtok_);
    k_accept<<<1, 32, 0, st_>>>(vtok_, otok_, k, posd_, mposd_, acc_, hist2_, em_);  // otok_: argmax or sampled rows
    KCHK("k_accept");
    mtp_core(m, false, otok_, motok_, acc_);
    k_draft0<<<1, 32, 0, st_>>>(motok_, acc_, dr_);
    // lookup steps: no MTP draft chain (the fill's draft is enough for a k=1 step); constrained: kd drafts
    const int kc = lookup ? 1 : kd >= 0 ? kd : k;
    for (int j = 1; j < kc; j++) {
        k_setpos<<<1, 1, 0, st_>>>(mposd_, posd_, j - 1);
        mtp_core(1, true, dr_ + j - 1, dr_ + j, nullptr);
    }
    k_next<<<1, 32, 0, st_>>>(vtok_, otok_, acc_, dr_, kc, em_);
    KCHK("k_next");
    if (lk_on_ && kd < 0) {  // lookup drafts for the next step (history += this step's tokens)
        k_hist_add<<<1, 32, 0, st_>>>(hd_, hn_d_, em_, lkrg_);
        lk_scan_emit();
        KCHK("lookup");
        if (lookup && lk_kfull_ > 1) lk_chain();
    }
}

// after a lookup step: the MTP chain (drafts 2..kfull) for the next step, only if it is an MTP step. In a captured
// step graph a conditional node decides on the device (k_lk_cond); eager: the host reads the lookup result.
void Worker::lk_chain() {
    auto chain = [&]() {
        for (int j = 1; j < lk_kfull_; j++) {
            k_setpos<<<1, 1, 0, st_>>>(mposd_, posd_, j - 1);
            mtp_core(1, true, dr_ + j - 1, dr_ + j, nullptr);
        }
        k_next<<<1, 32, 0, st_>>>(vtok_, otok_, acc_, dr_, lk_kfull_, em_);
        KCHK("lk chain");
    };
    cudaStreamCaptureStatus cs;
    CUDA_CHECK(cudaStreamIsCapturing(st_, &cs));
    if (cs != cudaStreamCaptureStatusActive) {
        CUDA_CHECK(cudaStreamSynchronize(st_));
        if (em_h_[MAXM + 2] == 0) chain();
        return;
    }
    cudaGraph_t g;
    const cudaGraphNode_t* deps;
    size_t nd;
    CUDA_CHECK(cudaStreamGetCaptureInfo(st_, &cs, nullptr, &g, nullptr, nullptr, nullptr));
    cudaGraphConditionalHandle h;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&h, g, 0, 0));
    k_lk_cond<<<1, 32, 0, st_>>>(h, lk_);
    CUDA_CHECK(cudaStreamGetCaptureInfo(st_, &cs, nullptr, &g, &deps, nullptr, &nd));
    cudaGraphNodeParams cp = {};
    cp.type = cudaGraphNodeTypeConditional;
    cp.conditional.handle = h;
    cp.conditional.type = cudaGraphCondTypeIf;
    cp.conditional.size = 1;
    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddNode(&node, g, deps, nullptr, nd, &cp));
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(st_, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
    cudaGraph_t body = cp.conditional.phGraph_out[0];
    cudaStream_t s0 = st_;
    st_ = lkst_;
    // the side branches (pfs_) of the outer capture stay in it until it ends: the body gets its own side stream
    // (joining pfs_ here failed with cudaErrorStreamCaptureMerge)
    cudaStream_t ps0 = pfs_;
    if (!pfs2_) CUDA_CHECK(cudaStreamCreateWithFlags(&pfs2_, cudaStreamNonBlocking));
    pfs_ = pfs2_;
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(st_, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    chain();
    CUDA_CHECK(cudaStreamEndCapture(st_, &body));
    st_ = s0;
    pfs_ = ps0;
}

void Worker::set_history(const int* ids, int n, int kfull) {
    CUDA_CHECK(cudaSetDevice(dev_));
    if (!lk_on_) return;
    CHECK(lk_kfull_ < 0 || lk_kfull_ == kfull, "flash lookup: MTP k changed (%d -> %d)", lk_kfull_, kfull);
    lk_kfull_ = kfull;
    CUDA_CHECK(cudaMemcpy(hd_, ids, (size_t)n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hn_d_, &n, 4, cudaMemcpyHostToDevice));
    {
        int rg[2] = {0, 0};
        for (int i = 0; i < n; i++) lk_region_tok(ids[i], rg);
        CUDA_CHECK(cudaMemcpy(lkrg_, rg, 8, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemsetAsync(lkbest_, 0, 8, st_));
    if (lkbestc_) CUDA_CHECK(cudaMemsetAsync(lkbestc_, 0, 8, st_));
    lk_scan_emit();
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

// rank 0's lookup result (count + drafts) into lk_ on every rank; counts the calls where a rank's own result differed
__global__ void k_lk_agree(const float* __restrict__ all, int n, int ngpu, int* __restrict__ lk, unsigned* __restrict__ diff) {
    const int i = threadIdx.x;
    bool d = false;
    if (i < n) {
        const unsigned v0 = __float_as_uint(all[i]);
        for (int g = 1; g < ngpu; g++) d |= __float_as_uint(all[(size_t)g * n + i]) != v0;
        lk[i] = (int)v0;
    }
    if (__any_sync(0xffffffffu, d) && i == 0) atomicAdd(diff, 1u);
}
void Worker::lk_scan_emit() {
    k_lk_scan<<<(max_ctx_ + 255) / 256, 256, 0, st_>>>(hd_, hn_d_, lkbest_);
    // FLASH_CORPUS_MIN: shortest corpus match used (default 12)
    static const int cmin = getenv("FLASH_CORPUS_MIN") ? atoi(getenv("FLASH_CORPUS_MIN")) : 12;
    if (cd_) k_lk_scan_c<<<std::min(1024, (ccap_ + 255) / 256), 256, 0, st_>>>(cd_, cn_d_, hd_, hn_d_, lkbestc_);
    k_lk_emit<<<1, 32, 0, st_>>>(hd_, hn_d_, lkbest_, lk_mmin_, lk_llo_, lk_hi_, lk_lmax_, lk_, em_ + MAXM + 2, lkrg_, lkrp_,
                                 cd_, lkbestc_, cmin);
    // one lookup result on every rank (qfn decode hang, rc2): the step graph branches on lk_[0] on each device (k_lk_cond:
    // the MTP draft chain, whose collectives every rank must enter) and k_vtok_lk builds the next verify rows from
    // lk_[1..]; a rank whose lookup differed (e.g. a corpus update landing between ranks) skipped the chain while the
    // others spun on its exchanges. All ranks take rank 0's lk_ (FLASH_LK_UNIFORM=0: off). The em_ copy only rank 0's
    // host reads.
    static const bool uni = !getenv("FLASH_LK_UNIFORM") || atoi(getenv("FLASH_LK_UNIFORM"));
    if (uni && p_.ngpu > 1) {
        if (!lkall_) {
            lkall_ = (float*)dalloc((size_t)p_.ngpu * kLkWords * 4);
            CUDA_CHECK(cudaHostAlloc((void**)&lkdiff_, 4, cudaHostAllocMapped));
            *lkdiff_ = 0;
        }
        allgather_small(reinterpret_cast<const float*>(lk_), kLkWords, lkall_, comm_, p_.rank, ctr_, st_);
        k_lk_agree<<<1, 32, 0, st_>>>(lkall_, kLkWords, p_.ngpu, lk_, lkdiff_);
    }
}

// ring of answers: an answer that does not fit the rest of the buffer starts again at 0 (the oldest are overwritten;
// the separator after it keeps a match from running into what is left of them)
void Worker::corpus_add(const int* ids, int n) {
    if (!cd_ || n <= 0) return;
    CUDA_CHECK(cudaSetDevice(dev_));
    if (n > ccap_ - 1) ids += n - (ccap_ - 1), n = ccap_ - 1;
    if (cpos_ + n + 1 > ccap_) cpos_ = 0;
    // on the work stream (qfn decode hang, rc2): a plain cudaMemcpy ran unordered with a step graph in flight (the
    // lookahead keeps one queued), which reads the corpus in k_lk_scan_c; the ranks saw the answer at different moments
    // and took different branches of the device-side lookup conditional. Pageable sources: the async copy returns once
    // they are staged; the step after it sees the corpus on every rank.
    std::vector<int> v(ids, ids + n);
    v.push_back(-1);
    CUDA_CHECK(cudaMemcpyAsync(cd_ + cpos_, v.data(), v.size() * 4, cudaMemcpyHostToDevice, st_));
    cpos_ += n + 1;
    cfill_ = std::max(cfill_, cpos_);
    const int fill = cfill_;
    CUDA_CHECK(cudaMemcpyAsync(cn_d_, &fill, 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));  // (the size word's host source is a local)
}

void Worker::spec_init(int last_tok, const int* drafts, int k, int pos, int commit, int h1, int h0) {
    CUDA_CHECK(cudaSetDevice(dev_));
    std::vector<int> v = {last_tok};
    v.insert(v.end(), drafts, drafts + k);
    CUDA_CHECK(cudaMemcpy(vtok_, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    const int pp[2] = {pos, commit};
    CUDA_CHECK(cudaMemcpy(posd_, pp, 8, cudaMemcpyHostToDevice));
    spos_ = pos;
    const int hh[2] = {h0, h1};  // [t - 2, t - 1] committed tokens before the first verify row
    CUDA_CHECK(cudaMemcpy(hist2_, hh, 8, cudaMemcpyHostToDevice));
}

int Worker::spec_step(int k, int* emitted, bool lookup) {
    spec_launch(k, lookup);
    return spec_finish(emitted);
}

// a launched step that is not wanted (the reply ended before it): wait for it, keep nothing. It committed the rows
// of the step before it (deferred commit), so nothing is pending afterwards; its own rows are never committed.
void Worker::spec_discard() {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    prev_m_ = 0;
}

void Worker::spec_launch(int k, bool lookup) {
    CUDA_CHECK(cudaSetDevice(dev_));
    ec_tick(spos_ + 2 * k + 2, 0);
    static const bool graphs = !getenv("FLASH_GRAPH") || atoi(getenv("FLASH_GRAPH"));
    if (!graphs) {
        spec_body(k, lookup);
    } else {
        sel_on_ = spos_ + 2 * k + 2 > IDX_BUDGET + IDX_R - 1;  // verify rows and the draft chain of this step
        cudaGraphExec_t& ge = sgraph_[sampling_][sel_on_][lookup][k];
        if (!ge) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            spec_body(k, lookup);
            CUDA_CHECK(cudaStreamEndCapture(st_, &g));
            inst_graph(&ge, g);  // OOM: frees the other slots' graphs and retries (flash_seq.cu)
            CUDA_CHECK(cudaGraphDestroy(g));
        }
        if (!ev0_) { CUDA_CHECK(cudaEventCreate(&ev0_)); CUDA_CHECK(cudaEventCreate(&ev1_)); }
        CUDA_CHECK(cudaEventRecord(ev0_, st_));
        CUDA_CHECK(cudaGraphLaunch(ge, st_));
        CUDA_CHECK(cudaEventRecord(ev1_, st_));
    }
}

void Worker::spec_launch_con(int kv, int kd) {
    CUDA_CHECK(cudaSetDevice(dev_));
    ec_tick(spos_ + 2 * std::max(kv, kd) + 2, 0);
    static const bool graphs = !getenv("FLASH_GRAPH") || atoi(getenv("FLASH_GRAPH"));
    if (!graphs) {
        spec_body(kv, false, kd);
        return;
    }
    sel_on_ = spos_ + 2 * std::max(kv, kd) + 2 > IDX_BUDGET + IDX_R - 1;
    const long long key = ((long long)sampling_ << 20) | ((long long)sel_on_ << 16) | ((long long)kv << 8) | kd;
    cudaGraphExec_t& ge = cgraph_[key];
    if (!ge) {
        cudaGraph_t g;
        CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
        spec_body(kv, false, kd);
        CUDA_CHECK(cudaStreamEndCapture(st_, &g));
        inst_graph(&ge, g);
        CUDA_CHECK(cudaGraphDestroy(g));
    }
    if (!ev0_) { CUDA_CHECK(cudaEventCreate(&ev0_)); CUDA_CHECK(cudaEventCreate(&ev1_)); }
    CUDA_CHECK(cudaEventRecord(ev0_, st_));
    CUDA_CHECK(cudaGraphLaunch(ge, st_));
    CUDA_CHECK(cudaEventRecord(ev1_, st_));
}

int Worker::spec_finish(int* emitted) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    if (ev0_) {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0_, ev1_));
        gpu_ms_.push_back(ms);
    }
    const int n = em_h_[0];
    for (int i = 0; i < n; i++) emitted[i] = em_h_[1 + i];
    nf_check("spec step", spos_);
    spos_ += n;
    return n;
}

// guard: the lm_head input of a launch had non-finite values (the logits would be garbage): log it (stderr), once per
// new count. NaN / Inf only, finite garbage passes.
void Worker::nf_check(const char* what, int pos) {
    if (!nf_h_) return;
    static const bool inject = getenv("FLASH_GUARD_TEST") && atoi(getenv("FLASH_GUARD_TEST"));  // test: as if it fired
    const unsigned c = *(volatile unsigned*)nf_h_ + (inject ? 1u : 0u);
    if (c != nf_seen_) {
        fprintf(stderr, "[flash] GUARD rank %d: non-finite lm_head input in %s at position %d (%u launch blocks so far)\n",
                p_.rank, what, pos, c);
        fflush(stderr);
        nf_seen_ = c;
        // the tokens of this launch are garbage (argmax of NaN logits): stop instead of streaming them. FLASH_GUARD=2
        // only logs (debugging), FLASH_GUARD=0 turns the check off
        static const int mode = getenv("FLASH_GUARD") ? atoi(getenv("FLASH_GUARD")) : 1;
        if (mode == 1) {
            fprintf(stderr, "[flash] FATAL: the forward produced non-finite values (%s, position %d); exiting instead of "
                            "emitting garbage tokens (FLASH_GUARD=2: log only)\n", what, pos);
            fflush(stderr);
            abort();
        }
    }
}

void Worker::forward_mtp(const int* tokens, int pos, int m, bool chain, int* out) {
    CUDA_CHECK(cudaSetDevice(dev_));
    sel_on_ = pos + m > IDX_BUDGET + IDX_R - 1;
    CHECK(has_mtp_ && m >= 1 && m <= MAXM && (!chain || m == 1), "flash mtp m=%d", m);
    ec_tick(pos + m, 1);
    memcpy(tok_h_, tokens, m * 4);
    pos_h_[0] = pos;
    pos_h_[1] = 0;
    static const bool graphs = !getenv("FLASH_GRAPH") || atoi(getenv("FLASH_GRAPH"));
    if (!graphs) {
        body_mtp(m, chain);
    } else {
        cudaGraphExec_t& ge = mgraph_[sel_on_][chain][m];
        if (!ge) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            body_mtp(m, chain);
            CUDA_CHECK(cudaStreamEndCapture(st_, &g));
            inst_graph(&ge, g);  // OOM: frees the other slots' graphs and retries (flash_seq.cu)
            CUDA_CHECK(cudaGraphDestroy(g));
        }
        CUDA_CHECK(cudaGraphLaunch(ge, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    memcpy(out, out_h_, m * 4);
}

void Worker::forward(const int* tokens, int pos, int m, const bf16* ple_rows, int* out, float* logits_host, int commit) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CHECK(m >= 1 && m <= MAXP, "flash forward m=%d (prompts: prefill())", m);
    CHECK(pos + m <= max_ctx_, "flash: context %d > %d", pos + m, max_ctx_);
    ec_tick(pos + m, 0);
    sel_on_ = pos + m > IDX_BUDGET + IDX_R - 1;

    memcpy(tok_h_, tokens, m * 4);
    pos_h_[0] = pos;
    pos_h_[1] = commit < 0 ? prev_m_ : commit;
    prev_m_ = m;
    memcpy(ple_h_, ple_rows, (size_t)m * H * 2);
    static const bool graphs = !getenv("FLASH_GRAPH") || atoi(getenv("FLASH_GRAPH"));
    if (!graphs || dump_) {
        body(m);
    } else {
        cudaGraphExec_t& gm = graph_[sampling_][sel_on_][m];
        if (!gm) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            body(m);
            CUDA_CHECK(cudaStreamEndCapture(st_, &g));
            inst_graph(&gm, g);
            CUDA_CHECK(cudaGraphDestroy(g));
        }
        CUDA_CHECK(cudaGraphLaunch(gm, st_));
    }
    const int nv = p_.voc1 - p_.voc0;
    if (logits_host)
        CUDA_CHECK(cudaMemcpyAsync(logits_host, logits_ + (size_t)(m - 1) * nv, (size_t)nv * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaGetLastError());
    memcpy(out, out_h_, m * 4);
}

void Worker::dump_resid(float* host, int m) {
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaMemcpy(host, R_, (size_t)m * RW * 4, cudaMemcpyDeviceToHost));
}

}  // namespace flash

#include "flash_batch.cuh"  // batched steps over several sequence slots
