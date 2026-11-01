// Per-card calibration for heterogeneous cards (pipeline split, boot-time mode pick) and the slow-card test knob.
// The calibration runs three short synthetic kernels per card: an FP16 tensor-free FMA loop (the compute side of the
// prefill MoE GEMMs / attention), a streaming read (the memory side of the decode GEMVs) and the free VRAM. The numbers
// are relative: pp_plan.h compares the cards with each other and with the 3 x 3090 reference the cost model was fit on.
#include "card_probe.h"
#include "../env.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

namespace flash {

namespace {
__global__ void k_cal_fma(half2* out, int iters) {
    half2 a = __floats2half2_rn(1.0001f, 0.9999f), b = __floats2half2_rn(0.5f, 0.25f);
    half2 c0 = __floats2half2_rn(threadIdx.x * 1e-3f, 0.f), c1 = c0, c2 = c0, c3 = c0;
    for (int i = 0; i < iters; i++) {
        c0 = __hfma2(a, c0, b);
        c1 = __hfma2(a, c1, b);
        c2 = __hfma2(a, c2, b);
        c3 = __hfma2(a, c3, b);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = __hadd2(__hadd2(c0, c1), __hadd2(c2, c3));
}
__global__ void k_cal_read(const uint4* __restrict__ src, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;  // (keeps the reads)
}
// the slow-card test knob: blocks that hold whole SMs (all shared memory) for `ns` nanoseconds; a host thread launches
// them back to back (short launches: an implicit device synchronization, e.g. in cudaFree, waits at most one)
__global__ void k_hog(long long ns) {
    extern __shared__ char sm[];
    if (threadIdx.x == 0) sm[0] = 1;
    long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(2000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}
}  // namespace

CardCal calibrate_card(int dev, int scale) {
    scale = std::max(1, scale);
    CardCal c;
    c.dev = dev;
    int prev = 0;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) != cudaSuccess) { cudaGetLastError(); return c; }
    cudaDeviceProp pr;
    cudaGetDeviceProperties(&pr, dev);
    c.sms = pr.multiProcessorCount;
    c.vram = (double)pr.totalGlobalMem;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    c.free = (double)fr;
    snprintf(c.name, sizeof c.name, "%s", pr.name);
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    // compute: 8 blocks per SM x 256 threads, 4 independent FMA chains of half2 per thread
    // scale > 1 (the re-check of equal hardware that measured apart): scale x longer kernels (~20 ms at 64 on a 3090)
    // after a longer warm-up, best of 5, so a card still in a low P-state or busy for a moment does not count as slower
    const int blocks = c.sms * 8, threads = 256, iters = 4096 * scale, reps = scale > 1 ? 5 : 3;
    half2* out = nullptr;
    const size_t rd = (size_t)256 << 20;  // 256 MB streamed read
    void* src = nullptr;
    unsigned* sink = nullptr;
    if (cudaMalloc(&out, (size_t)blocks * threads * sizeof(half2)) == cudaSuccess && cudaMalloc(&src, rd) == cudaSuccess &&
        cudaMalloc(&sink, 4) == cudaSuccess) {
        cudaMemsetAsync(src, 1, rd, st);
        for (int w = 0; w < (scale > 1 ? 5 : 1); w++) k_cal_fma<<<blocks, threads, 0, st>>>(out, iters);  // (warm-up)
        float best = 1e30f;
        for (int r = 0; r < reps; r++) {
            cudaEventRecord(a, st);
            k_cal_fma<<<blocks, threads, 0, st>>>(out, iters);
            cudaEventRecord(b, st);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            best = std::min(best, ms);
        }
        c.tflops = (double)blocks * threads * iters * 4 * 2 * 2 / (best * 1e-3) / 1e12;  // 4 chains x 2 lanes x FMA = 2 flops
        k_cal_read<<<c.sms * 4, 512, 0, st>>>((const uint4*)src, rd / 16, sink);
        best = 1e30f;
        for (int r = 0; r < reps; r++) {
            cudaEventRecord(a, st);
            for (int x = 0; x < scale; x++) k_cal_read<<<c.sms * 4, 512, 0, st>>>((const uint4*)src, rd / 16, sink);
            cudaEventRecord(b, st);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            best = std::min(best, ms);
        }
        c.gbs = (double)rd * scale / (best * 1e-3) / 1e9;
        c.ok = true;
    }
    cudaGetLastError();
    if (out) cudaFree(out);
    if (src) cudaFree(src);
    if (sink) cudaFree(sink);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaStreamDestroy(st);
    cudaSetDevice(prev);
    return c;
}

// FLASH_TEST_SLOW_CARD=<ordinal>:<fraction>[,...]: blocks holding that fraction of the card's SMs spin for the rest
// of the process (a slower card for the heterogeneous tests; no clock or power change). Returns the cards slowed.
int test_slow_cards() {
    const char* e = env_str("FLASH_TEST_SLOW_CARD");
    if (!e || !*e) return 0;
    int n = 0;
    std::string s(e);
    size_t at = 0;
    while (at < s.size()) {
        size_t c = s.find(',', at);
        if (c == std::string::npos) c = s.size();
        const std::string it = s.substr(at, c - at);
        at = c + 1;
        int dev = -1;
        double frac = 0;
        if (sscanf(it.c_str(), "%d:%lf", &dev, &frac) != 2 || dev < 0 || frac <= 0 || frac >= 1) continue;
        int prev = 0;
        cudaGetDevice(&prev);
        if (cudaSetDevice(dev) != cudaSuccess) { cudaGetLastError(); continue; }
        cudaDeviceProp pr;
        cudaGetDeviceProperties(&pr, dev);
        const int blocks = std::max(1, (int)(pr.multiProcessorCount * frac + 0.5));
        const size_t smem = pr.sharedMemPerBlockOptin;  // one block per SM: no other block fits next to it
        cudaFuncSetAttribute(k_hog, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaStream_t st;
        cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        k_hog<<<blocks, 32, smem, st>>>(1000);
        const cudaError_t er = cudaGetLastError();
        if (er == cudaSuccess)
            std::thread([dev, blocks, smem, st] {
                cudaSetDevice(dev);
                cudaEvent_t ev[2];
                cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming);
                cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming);
                for (unsigned i = 0;; i++) {  // two launches in flight: no gap between them
                    k_hog<<<blocks, 32, smem, st>>>(2000000);  // 2 ms
                    cudaEventRecord(ev[i & 1], st);
                    if (i && cudaEventSynchronize(ev[(i + 1) & 1]) != cudaSuccess) return;
                }
            }).detach();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // (running before the calibration)
        fprintf(stderr, "[flash] FLASH_TEST_SLOW_CARD: GPU%d: %d of %d SMs held by a spinning kernel (test only)%s\n", dev, blocks,
                pr.multiProcessorCount, er == cudaSuccess ? "" : cudaGetErrorString(er));
        cudaSetDevice(prev);
        n++;
    }
    return n;
}

std::string card_line(const std::vector<CardCal>& c) {
    std::string s;
    char b[200];
    for (const CardCal& x : c) {
        snprintf(b, sizeof b, "%sGPU%d %s %.1f TFLOPS fp16-FMA, %.0f GB/s, %.1f / %.1f GB free", s.empty() ? "" : "; ", x.dev, x.name,
                 x.tflops, x.gbs, x.free / 1e9, x.vram / 1e9);
        s += b;
    }
    return s;
}

// ---- box report (remote diagnosis of 4+ card / PP x TP boots): build, cards, links, NUMA, per-pair copy rates
static std::string rd1(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return "";
    char b[256] = "";
    if (!fgets(b, sizeof b, f)) b[0] = 0;
    fclose(f);
    std::string r(b);
    while (!r.empty() && (r.back() == '\n' || r.back() == ' ')) r.pop_back();
    return r;
}
static std::string sysfs_of(int dev) {
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess) { (void)cudaGetLastError(); return ""; }
    std::string id(bus);
    for (char& ch : id) ch = (char)tolower(ch);
    if (id.size() > 12 && id.find(':') == 8) id = id.substr(4);
    return "/sys/bus/pci/devices/" + id;
}
static double copy_gbs(int dst, int src, size_t bytes) {  // one timed copy after a warm-up (peer, or staged by the driver)
    void *a = nullptr, *b = nullptr;
    cudaSetDevice(src);
    if (cudaMalloc(&a, bytes) != cudaSuccess) { (void)cudaGetLastError(); return -1; }
    cudaSetDevice(dst);
    if (cudaMalloc(&b, bytes) != cudaSuccess) { (void)cudaGetLastError(); cudaSetDevice(src); cudaFree(a); return -1; }
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    cudaMemcpyPeerAsync(b, dst, a, src, bytes, st);
    cudaStreamSynchronize(st);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; i++) cudaMemcpyPeerAsync(b, dst, a, src, bytes, st);
    cudaStreamSynchronize(st);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    cudaStreamDestroy(st);
    cudaFree(b);
    cudaSetDevice(src);
    cudaFree(a);
    (void)cudaGetLastError();
    return 3.0 * bytes / s / 1e9;
}
static double h2d_gbs(int dev, size_t bytes) {
    void *h = nullptr, *d = nullptr;
    cudaSetDevice(dev);
    if (cudaMallocHost(&h, bytes) != cudaSuccess) { (void)cudaGetLastError(); return -1; }
    if (cudaMalloc(&d, bytes) != cudaSuccess) { (void)cudaGetLastError(); cudaFreeHost(h); return -1; }
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; i++) cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    cudaFree(d);
    cudaFreeHost(h);
    (void)cudaGetLastError();
    return 3.0 * bytes / s / 1e9;
}
#ifndef QE_GIT_HASH
#define QE_GIT_HASH "unknown"
#endif
#ifndef QE_CUDA_ARCHS
#define QE_CUDA_ARCHS "unknown"
#endif
std::string box_report(const std::vector<int>& devs_in) {
    std::vector<int> devs;
    for (int d : devs_in) if (std::find(devs.begin(), devs.end(), d) == devs.end()) devs.push_back(d);
    int prev = 0;
    cudaGetDevice(&prev);
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::string o;
    char b[512];
    snprintf(b, sizeof b, "[box] build %s, CUDA archs %s, runtime %d, driver API %d, kernel driver: %s\n", QE_GIT_HASH, QE_CUDA_ARCHS, rt,
             drv, rd1("/proc/driver/nvidia/version").c_str());
    o += b;
    for (int d : devs) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess) { (void)cudaGetLastError(); continue; }
        const std::string sf = sysfs_of(d);
        char bus[32] = "";
        cudaDeviceGetPCIBusId(bus, sizeof bus, d);
        snprintf(b, sizeof b, "[box] GPU%d %s sm_%d%d, %d SMs, %.1f GB, bus %s, NUMA %s, PCIe %s x%s (max %s x%s), host->card %.1f GB/s\n", d,
                 p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1e9, bus, rd1(sf + "/numa_node").c_str(),
                 rd1(sf + "/current_link_speed").c_str(), rd1(sf + "/current_link_width").c_str(), rd1(sf + "/max_link_speed").c_str(),
                 rd1(sf + "/max_link_width").c_str(), h2d_gbs(d, 64u << 20));
        o += b;
    }
    for (int a : devs)
        for (int c : devs) {
            if (a == c) continue;
            int can = 0, rank = -1, atom = 0;
            cudaDeviceCanAccessPeer(&can, a, c);
            cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, a, c);
            cudaDeviceGetP2PAttribute(&atom, cudaDevP2PAttrNativeAtomicSupported, a, c);
            (void)cudaGetLastError();
            snprintf(b, sizeof b, "[box] GPU%d -> GPU%d: peer access %s (perf rank %d, atomics %d), cudaMemcpyPeer before peer access is enabled %.1f GB/s\n", a, c, can ? "yes" : "no",
                     rank, atom, copy_gbs(c, a, 64u << 20));
            o += b;
        }
    cudaSetDevice(prev);
    return o;
}

}  // namespace flash
