#pragma once
// Experimental: the LL16 all-reduce of one token row (P2P path) as a device function, so a consumer GEMV can
// run it in the blocks it already has resident (blocks 0 .. m-1) instead of a separate kernel that would
// need SM room beside the waiting GEMV. Same words, tags, slots and arithmetic as allreduce_ll16_kernel
// (comm.cu), including the norm's reduction tree (per word, xor butterfly over 32 consecutive words, then
// the 24 warp sums in order), so xh / resid are bit-identical for any block size.
#include "comm.cuh"
#ifdef __CUDACC__
#include "gemv_h16.cuh"
#include "watchdog.cuh"
#endif

struct FusedAR {
    float* resid;
    const float* partial;
    int h, rank, idx, m;
    Comm c;
    const unsigned* epoch;
    const bf16* norm_w;
    float eps;
    __half* xh;
    unsigned* ready;  // [idx] tag, [1024 + idx] arrival counter, [2048..] guard timeouts
};

#ifdef __CUDACC__
namespace arf {

constexpr int kLL16Threads = 768;  // must match comm.cu (7 * 768 >= hidden; 24 warp sums)

__device__ __forceinline__ void st_ll16(uint4* p, uint4 v) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y),
                 "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 ld_ll16(const uint4* p) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t bf16_bits(float v) {
    const bf16 b = __float2bfloat16(v);
    return *reinterpret_cast<const uint16_t*>(&b);
}
__device__ __forceinline__ float bits_bf16(uint32_t u) { return __uint_as_float(u << 16); }

__device__ __forceinline__ uint4 make_word(const float* pr, int w, int h, uint32_t tag) {
    uint32_t b[7];
#pragma unroll
    for (int e = 0; e < 7; e++) b[e] = 7 * w + e < h ? bf16_bits(pr[7 * w + e]) : 0u;
    return make_uint4(b[0] | (b[1] << 16), b[2] | (b[3] << 16), b[4] | (b[5] << 16), b[6] | tag);
}

// Work item b = blockIdx.x < 24 m: token row b / 24, chunk b % 24 = the 32 LL16 words of warp (b % 24) of the
// 768-thread kernel, handled by warp 0 of the block. Phase 1: send, receive, residual and the chunk's sum of
// squares (same xor butterfly) -> ss; phase 2 (after all 24 chunks of the row): the 24 sums in order -> rs,
// the chunk's normalized values -> xh. The last item publishes ready[idx]. Scratch in ready[3072 ..].
constexpr int kChunks = kLL16Threads / 32;
__device__ __forceinline__ int items(const FusedAR& a) { return a.m * kChunks; }

__device__ void ll16_chunk(const FusedAR& a) {
    const int b = blockIdx.x, row = b / kChunks, chunk = b % kChunks;
    const int lane = threadIdx.x & 31;
    unsigned* rowcnt = a.ready + 3072;                               // [kMaxBatchRows]
    float* ss = reinterpret_cast<float*>(a.ready + 3200);            // [kMaxBatchRows][24]
    unsigned* done = a.ready + 3199;
    const unsigned ep = *a.epoch;
    if (threadIdx.x < 32) {
        const uint32_t tag = ((ep * 1024u + a.idx) % 65535u + 1u) << 16;
        const int slot = ((ep & 1) << 1) | (a.idx & 1);
        const int h = a.h, hw = (h + 6) / 7, hwp = (a.c.ll_hidden + 6) / 7;
        auto pregion = [&](int owner, int sender) {
            return a.c.pll[owner] + (((size_t)slot * a.c.ngpu + sender) * kMaxBatchRows + row) * hwp;
        };
        const float* pr = a.partial + (size_t)row * h;
        float* rr = a.resid + (size_t)row * h;
        const int w = chunk * 32 + lane;
        float q = 0.f;
        float acc[7] = {};
        if (w < hw) {
            const uint4 mine = make_word(pr, w, h, tag);
            for (int g = 0; g < a.c.ngpu; g++)
                if (g != a.rank) st_ll16(pregion(g, a.rank) + w, mine);
            for (int g = 0; g < a.c.ngpu; g++) {
                uint4 u = mine;
                if (g != a.rank) {
                    const uint4* src = pregion(a.rank, g) + w;
                    u = ld_ll16(src);
                    for (unsigned ws = 0; (u.w & 0xFFFF0000u) != tag;) { u = ld_ll16(src); QE_WD_POLL(ws, kWdArFused, g, tag, u.w); }
                }
                acc[0] += bits_bf16(u.x & 0xFFFFu); acc[1] += bits_bf16(u.x >> 16);
                acc[2] += bits_bf16(u.y & 0xFFFFu); acc[3] += bits_bf16(u.y >> 16);
                acc[4] += bits_bf16(u.z & 0xFFFFu); acc[5] += bits_bf16(u.z >> 16);
                acc[6] += bits_bf16(u.w & 0xFFFFu);
            }
#pragma unroll
            for (int e = 0; e < 7; e++) {
                const int i = 7 * w + e;
                if (i >= h) break;
                const float r = rr[i] + acc[e];
                rr[i] = r;
                acc[e] = r;
                q += r * r;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
        if (lane == 0) {
            ss[row * kChunks + chunk] = q;
            __threadfence();
            atomicAdd(rowcnt + row, 1u);
            unsigned v;
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(rowcnt + row) : "memory");
            } while (v < (unsigned)kChunks);
        }
        __syncwarp();
        float tot = 0.f;
#pragma unroll
        for (int x = 0; x < kChunks; x++) tot += __ldcg(ss + row * kChunks + x);
        const float rs = rsqrtf(tot / h + a.eps);
        if (w < hw)
#pragma unroll
            for (int e = 0; e < 7; e++) {
                const int i = 7 * w + e;
                if (i >= h) break;
                store_xh(a.xh + (size_t)row * h, i, acc[e] * rs * (1.f + __bfloat162float(a.norm_w[i])));
            }
        __syncwarp();
        if (lane == 0) {
            __threadfence();
            if (atomicAdd(done, 1u) == (unsigned)items(a) - 1) {  // every item is past its row wait
                for (int r2 = 0; r2 < a.m; r2++) rowcnt[r2] = 0;
                *done = 0;
                __threadfence();
                asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(a.ready + a.idx), "r"(ep * 1024u + a.idx + 1u) : "memory");
            }
        }
    }
}

}  // namespace arf
#endif
