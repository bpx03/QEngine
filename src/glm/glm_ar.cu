// fp32 all-reduce for decode rows over P2P (our own protocol): every rank stores its partial rows straight into each
// peer's receive slot (and its own), then raises a per-block flag word in every peer; each block waits for its chunk's
// flags from all ranks and sums in rank order (bitwise identical on every card). Flag value = step * 256 + index of the
// collective within the step, so a replayed CUDA graph never matches a stale flag.
#include "glm.h"

namespace glm {

constexpr int AR_BLK = 16, AR_SLOTS = 4;

__global__ void __launch_bounds__(256) k_ar_p2p(const float* part, float* sum, int n, const __grid_constant__ P2pAr a, int rank, int ngpu,
                                                const unsigned* step, int idx) {
    const unsigned tag = *step * 256u + (unsigned)idx + 1u;
    const int slot = idx % AR_SLOTS;
    const int per = n / AR_BLK, b = blockIdx.x;
    const int i0 = b * per;
    // push my chunk to every rank (incl. me)
    for (int g = 0; g < ngpu; g++) {
        float* dst = a.rbuf[g] + ((size_t)slot * ngpu + rank) * a.maxn + i0;
        for (int i = threadIdx.x * 4; i < per; i += blockDim.x * 4)
            *reinterpret_cast<float4*>(dst + i) = *reinterpret_cast<const float4*>(part + i0 + i);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x < ngpu)
        *(volatile unsigned*)(a.flag[threadIdx.x] + ((size_t)slot * ngpu + rank) * AR_BLK + b) = tag;
    // wait for all ranks' chunk b in my buffer
    if (threadIdx.x < ngpu) {
        const volatile unsigned* f = a.flag[rank] + ((size_t)slot * ngpu + threadIdx.x) * AR_BLK + b;
        long long it = 0;
        while (*f != tag)
            if (++it > (1ll << 33)) __trap();
    }
    __syncthreads();
    __threadfence();
    const float* rb = a.rbuf[rank] + (size_t)slot * ngpu * a.maxn + i0;
    for (int i = threadIdx.x; i < per; i += blockDim.x) {
        float s = 0.f;
        for (int g = 0; g < ngpu; g++) s += ((const volatile float*)rb)[(size_t)g * a.maxn + i];
        sum[i0 + i] = s;
    }
}

void ar_p2p(const float* part, float* sum, int n, const P2pAr& a, int rank, int ngpu, const unsigned* step, int idx,
            cudaStream_t st) {
    CHECK(n % (AR_BLK * 4) == 0 && n <= a.maxn && idx < 255, "ar_p2p %d %d", n, idx);
    k_ar_p2p<<<AR_BLK, 256, 0, st>>>(part, sum, n, a, rank, ngpu, step, idx);
}

// LL variant: every 8-byte store carries (value, tag), so a reader knows a value has landed without fences or flags
// (8-byte stores are single PCIe transactions). Receive buffer of rank r: [AR_SLOTS][ngpu senders][maxn] uint2.
__global__ void __launch_bounds__(256) k_ar_ll(const float* part, float* sum, int n, const __grid_constant__ P2pAr a, int rank, int ngpu,
                                               const unsigned* step, int idx) {
    const unsigned tag = *step * 256u + (unsigned)idx + 1u;
    const int slot = idx % AR_SLOTS;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = part[i];
    for (int g = 0; g < ngpu; g++) {
        uint2* dst = reinterpret_cast<uint2*>(a.rbuf[g]) + ((size_t)slot * ngpu + rank) * a.maxn + i;
        asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" ::"l"(dst), "r"(__float_as_uint(v)), "r"(tag) : "memory");
    }
    const uint2* rb = reinterpret_cast<const uint2*>(a.rbuf[rank]) + (size_t)slot * ngpu * a.maxn + i;
    float s = 0.f;
    for (int g = 0; g < ngpu; g++) {
        uint2 w;
        long long it = 0;
        do {
            asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(w.x), "=r"(w.y) : "l"(rb + (size_t)g * a.maxn) : "memory");
            if (++it > (1ll << 33)) __trap();
        } while (w.y != tag);
        s += __uint_as_float(w.x);
    }
    sum[i] = s;
}
void ar_ll(const float* part, float* sum, int n, const P2pAr& a, int rank, int ngpu, const unsigned* step, int idx,
           cudaStream_t st) {
    CHECK(n <= a.maxn && idx < 255, "ar_ll %d %d", n, idx);
    k_ar_ll<<<(n + 255) / 256, 256, 0, st>>>(part, sum, n, a, rank, ngpu, step, idx);
}

// LL12 variant: 16-byte stores carry 3 fp32 values + the tag (1.33x the payload instead of LL's 2x; 16-byte stores are
// single PCIe transactions, as the 27B engine's LL16 relies on). Same rank-order sum: bitwise identical on all cards
// and to k_ar_ll. Receive buffer: [AR_SLOTS][ngpu senders][maxn / 2] uint4 (inside the LL allocation).
__global__ void __launch_bounds__(256) k_ar_ll12(const float* part, float* sum, int n, const __grid_constant__ P2pAr a, int rank, int ngpu,
                                                 const unsigned* step, int idx) {
    const unsigned tag = *step * 256u + (unsigned)idx + 1u;
    const int slot = idx % AR_SLOTS;
    const int w = blockIdx.x * blockDim.x + threadIdx.x, i0 = w * 3;
    if (i0 >= n) return;
    const size_t ws = (size_t)a.maxn / 2;
    float v[3];
#pragma unroll
    for (int j = 0; j < 3; j++) v[j] = i0 + j < n ? part[i0 + j] : 0.f;
    for (int g = 0; g < ngpu; g++) {
        uint4* dst = reinterpret_cast<uint4*>(a.rbuf[g]) + ((size_t)slot * ngpu + rank) * ws + w;
        asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "r"(__float_as_uint(v[0])),
                     "r"(__float_as_uint(v[1])), "r"(__float_as_uint(v[2])), "r"(tag) : "memory");
    }
    const uint4* rb = reinterpret_cast<const uint4*>(a.rbuf[rank]) + (size_t)slot * ngpu * ws + w;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f;
    for (int g = 0; g < ngpu; g++) {
        uint4 x;
        long long it = 0;
        do {
            asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(rb + (size_t)g * ws) : "memory");
            if (++it > (1ll << 33)) __trap();
        } while (x.w != tag);
        s0 += __uint_as_float(x.x); s1 += __uint_as_float(x.y); s2 += __uint_as_float(x.z);
    }
    sum[i0] = s0;
    if (i0 + 1 < n) sum[i0 + 1] = s1;
    if (i0 + 2 < n) sum[i0 + 2] = s2;
}
void ar_ll12(const float* part, float* sum, int n, const P2pAr& a, int rank, int ngpu, const unsigned* step, int idx,
             cudaStream_t st) {
    CHECK(n <= a.maxn && idx < 255, "ar_ll12 %d %d", n, idx);
    const int nw = (n + 2) / 3;
    k_ar_ll12<<<(nw + 255) / 256, 256, 0, st>>>(part, sum, n, a, rank, ngpu, step, idx);
}

P2pAr ar_p2p_alloc(int dev, int ngpu, int maxn) {
    P2pAr a;
    a.maxn = maxn;
    CUDA_CHECK(cudaSetDevice(dev));
    float* rb = nullptr;
    unsigned* fl = nullptr;
    CUDA_CHECK(cudaMalloc(&rb, (size_t)AR_SLOTS * ngpu * maxn * 8));  // LL: (value, tag) pairs
    CUDA_CHECK(cudaMemset(rb, 0, (size_t)AR_SLOTS * ngpu * maxn * 8));
    CUDA_CHECK(cudaMalloc(&fl, (size_t)AR_SLOTS * ngpu * AR_BLK * 4));
    CUDA_CHECK(cudaMemset(fl, 0, (size_t)AR_SLOTS * ngpu * AR_BLK * 4));
    a.rbuf[0] = rb;
    a.flag[0] = fl;
    return a;
}

}  // namespace glm
