// Co-scheduling experiment: a stand-in for the prefill all-reduce kernel
// on the comm stream - `blocks` blocks of `threads` threads that occupy their SMs for `us`
// microseconds (polling a clock with __nanosleep, optionally streaming `mb` MB through DRAM, or
// writing them over PCIe into pinned host memory: pcie = 1), so the effect of the resident blocks /
// their traffic on a concurrent persistent GEMM can be measured on one GPU.
#include <cuda_runtime.h>
#include <cstdint>

#include "../common.h"

namespace {
__global__ void spin_kernel(long long ns, float* buf, size_t n4, int pcie) {
    const long long t0 = clock64();
    const long long cyc = ns * 17 / 10;  // ~1.7 GHz
    float4* b4 = reinterpret_cast<float4*>(buf);
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
    while (clock64() - t0 < cyc) {
        if (n4 && pcie) {  // stores only (like the AR's P2P row writes), 16 B per thread
            b4[i % n4] = make_float4(acc, 1.f, 2.f, 3.f);
            acc += 1.f;
            i += (size_t)gridDim.x * blockDim.x;
        } else if (n4) {
            const float4 v = b4[i % n4];
            acc += v.x;
            b4[(i + 7) % n4] = make_float4(acc, v.y, v.z, v.w);
            i += (size_t)gridDim.x * blockDim.x;
        } else {
            __nanosleep(100);
        }
    }
    if (acc == 12345.f) buf[0] = acc;
}
float* g_spin_buf = nullptr;
size_t g_spin_n4 = 0;
float* g_spin_host = nullptr;
size_t g_spin_host_n4 = 0;
}  // namespace

extern "C" void qe_spin(int blocks, int threads, double us, double mb, void* st) {
    const size_t n4 = (size_t)(mb * (1 << 20)) / 16;
    if (n4 > g_spin_n4) {
        if (g_spin_buf) cudaFree(g_spin_buf);
        CUDA_CHECK(cudaMalloc(&g_spin_buf, n4 * 16));
        CUDA_CHECK(cudaMemset(g_spin_buf, 0, n4 * 16));
        g_spin_n4 = n4;
    }
    spin_kernel<<<blocks, threads, 0, (cudaStream_t)st>>>((long long)(us * 1000), g_spin_buf, n4, 0);
}

// SM stores over PCIe into pinned, mapped host memory (mb MB ring) for us microseconds
extern "C" void qe_spin_pcie(int blocks, int threads, double us, double mb, void* st) {
    const size_t n4 = (size_t)(mb * (1 << 20)) / 16;
    if (n4 > g_spin_host_n4) {
        if (g_spin_host) cudaFreeHost(g_spin_host);
        CUDA_CHECK(cudaHostAlloc(&g_spin_host, n4 * 16, cudaHostAllocMapped));
        g_spin_host_n4 = n4;
    }
    float* dp = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dp, g_spin_host, 0));
    spin_kernel<<<blocks, threads, 0, (cudaStream_t)st>>>((long long)(us * 1000), dp, n4, 1);
}

// the same bytes by the copy engine: device -> pinned host, `reps` copies of mb MB
extern "C" void qe_copy_pcie(double mb, int reps, void* st) {
    const size_t n = (size_t)(mb * (1 << 20));
    if (n / 16 > g_spin_n4) {
        if (g_spin_buf) cudaFree(g_spin_buf);
        CUDA_CHECK(cudaMalloc(&g_spin_buf, n));
        g_spin_n4 = n / 16;
    }
    if (n / 16 > g_spin_host_n4) {
        if (g_spin_host) cudaFreeHost(g_spin_host);
        CUDA_CHECK(cudaHostAlloc(&g_spin_host, n, cudaHostAllocMapped));
        g_spin_host_n4 = n / 16;
    }
    for (int r = 0; r < reps; r++)
        CUDA_CHECK(cudaMemcpyAsync(g_spin_host, g_spin_buf, n, cudaMemcpyDeviceToHost, (cudaStream_t)st));
}
