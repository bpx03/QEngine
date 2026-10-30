#include "../env.h"
#include <type_traits>
#include <algorithm>

#include "comm.cuh"
#include "watchdog.cuh"

int gemm_i8_variant();  // gemm.cu: W4A8 kernel variant (record format of the fused quantization)
#include "gemv_h16.cuh"

namespace {

// 16-byte words per row and sender in the P2P receive buffers: the 7-value LL16 words plus spare
// words for the lossless compressed protocol's fallback (allreduce_llc_kernel)
__host__ __device__ __forceinline__ int pll_row_words(int h) { return (h + 6) / 7 + (h + 8) / 9; }

__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_gpu(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint2 pack_bf16x4(float4 v) {
    bf162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    return make_uint2(*reinterpret_cast<uint32_t*>(&a), *reinterpret_cast<uint32_t*>(&b));
}
__device__ __forceinline__ float4 unpack_bf16x4(uint2 u) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<bf162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<bf162*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ float* slot_ptr(const Comm& c, int g, int slot) {
    return c.data + ((size_t)g * kCommSlots + slot) * c.max_elems;
}

// Spin loops sleep briefly between polls: the GPUs are power-bound during decode, and
// busy-waiting SMs would take power (and clock speed) from the kernels doing real work.
// Called by every block after writing its share of this GPU's data to host memory.
// Only one thread per GPU polls the peers over PCIe; the other blocks spin on a device-local
// flag (many concurrent host-memory pollers serialize on the PCIe link).
// p2p: the collective's data went into the peers' memory, so the flags follow the same P2P path
// (written into each peer's pflags after the system fence; polled locally)
__device__ void exchange(const Comm& c, int rank, const CommCounters& ctr, unsigned s, bool p2p = false,
                         unsigned site = kWdArExchange) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();  // cumulative over this block's writes (ordered by bar.sync)
        if (atomicAdd(&ctr.done[0], 1u) == gridDim.x - 1) {
            ctr.done[0] = 0;
            const int slot = s % kCommSlots;
            if (p2p) {
                for (int g = 0; g < c.ngpu; g++)
                    if (g != rank) st_release_sys(c.pflags[g] + rank * kCommSlots + slot, s);
                for (int g = 0; g < c.ngpu; g++)
                    if (g != rank) {
                        unsigned ws = 0;
                        while (ld_acquire_sys(c.pflags[rank] + g * kCommSlots + slot) != s) {
                            __nanosleep(32);
                            QE_WD_POLL(ws, site, g, s, ld_acquire_sys(c.pflags[rank] + g * kCommSlots + slot));
                        }
                    }
            } else {
                st_release_sys(c.flags + rank * kCommSlots + slot, s);
                for (int g = 0; g < c.ngpu; g++)
                    if (g != rank) {
                        unsigned ws = 0;
                        while (ld_acquire_sys(c.flags + g * kCommSlots + slot) != s) {
                            __nanosleep(64);
                            QE_WD_POLL(ws, site, g, s, ld_acquire_sys(c.flags + g * kCommSlots + slot));
                        }
                    }
            }
            st_release_gpu(ctr.go, s);
        } else {
            // the other blocks wait for this GPU's leader (which waits for the peers): watched like the peer waits,
            // so a rank stuck here also records a watchdog wait (peer -1: same GPU)
            unsigned ws = 0;
            while (ld_acquire_gpu(ctr.go) != s) {
                __nanosleep(128);
                QE_WD_POLL(ws, site, -1, s, ld_acquire_gpu(ctr.go));
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void bump_seq(const CommCounters& ctr, unsigned s) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&ctr.done[1], 1u) == gridDim.x - 1) {
            ctr.done[1] = 0;
            *ctr.seq = s;
            __threadfence();
        }
    }
}

// Block b owns columns [b*h/nb, (b+1)*h/nb) of every token row, so the fused RMSNorm only
// needs one cheap device-local barrier to combine per-block sums of squares.
__global__ void allreduce_kernel(float* __restrict__ resid, const float* __restrict__ partial,
                                 int m, int h, Comm c, int rank, CommCounters ctr,
                                 const bf16* __restrict__ norm_w, float eps,
                                 bf16* __restrict__ xn, __half* __restrict__ xh, bool rsum) {
    __shared__ unsigned s_seq;
    __shared__ float s_ss[kMaxBatchRows];
    if (threadIdx.x == 0) s_seq = *ctr.seq + 1;
    if (threadIdx.x < kMaxBatchRows) s_ss[threadIdx.x] = 0.f;
    __syncthreads();
    const unsigned s = s_seq;
    const int slot = s % kCommSlots;
    const int h4 = h >> 2, per = h4 / gridDim.x;  // h4 % gridDim.x == 0 (checked on host)
    const int cb = blockIdx.x * per;
    const float4* p4 = reinterpret_cast<const float4*>(partial);
    float4* r4 = reinterpret_cast<float4*>(resid);

    // Partials cross PCIe as bf16 (half the bytes; HF keeps projection outputs in bf16 too).
    // Our own contribution is rounded the same way so every GPU sums identical values.
    if (c.ngpu > 1) {
        uint2* mine = reinterpret_cast<uint2*>(slot_ptr(c, rank, slot));
        for (int t = threadIdx.x; t < m * per; t += blockDim.x) {
            const int i = (t / per) * h4 + cb + t % per;
            mine[i] = pack_bf16x4(p4[i]);
        }
        exchange(c, rank, ctr, s);
    }
    for (int t = threadIdx.x; t < m * per; t += blockDim.x) {
        const int mi = t / per, i = mi * h4 + cb + t % per;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int g = 0; g < c.ngpu; g++) {  // fixed order -> identical residual on every GPU
            const float4 x =
                c.ngpu == 1 ? p4[i]
                : g == rank ? unpack_bf16x4(pack_bf16x4(p4[i]))
                            : unpack_bf16x4(__ldcv(reinterpret_cast<const uint2*>(slot_ptr(c, g, slot)) + i));
            v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
        }
        if (rsum && c.ngpu > 1) v = unpack_bf16x4(pack_bf16x4(v));
        float4 r = r4[i];
        r.x += v.x; r.y += v.y; r.z += v.z; r.w += v.w;
        r4[i] = r;
    }
    if (norm_w) {
        // deterministic per-row sum of squares of this block's slice (fixed reduction tree)
        __shared__ float s_w[8];
        for (int mi = 0; mi < m; mi++) {
            __syncthreads();
            float q = 0.f;
            for (int t = threadIdx.x; t < per; t += blockDim.x) {
                const float4 r = r4[mi * h4 + cb + t];
                q += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
            if ((threadIdx.x & 31) == 0) s_w[threadIdx.x >> 5] = q;
            __syncthreads();
            if (threadIdx.x == 0) {
                float tot = 0.f;
                for (int w = 0; w < (int)(blockDim.x >> 5); w++) tot += s_w[w];
                s_ss[mi] = tot;
            }
        }
        __syncthreads();
        if (threadIdx.x < m) ctr.ss[blockIdx.x * kMaxBatchRows + threadIdx.x] = s_ss[threadIdx.x];
        __syncthreads();
        if (threadIdx.x == 0) {  // grid barrier (all kCommBlocks blocks are resident)
            __threadfence();
            if (atomicAdd(&ctr.bar[0], 1u) == gridDim.x - 1) {
                ctr.bar[0] = 0;
                st_release_gpu(&ctr.bar[1], s);
            } else {
                while (ld_acquire_gpu(&ctr.bar[1]) != s) __nanosleep(64);
            }
        }
        __syncthreads();
        if (threadIdx.x < m) {
            float tot = 0.f;
            for (int b = 0; b < (int)gridDim.x; b++)
                tot += __ldcg(&ctr.ss[b * kMaxBatchRows + threadIdx.x]);
            s_ss[threadIdx.x] = rsqrtf(tot / h + eps);
        }
        __syncthreads();
        for (int t = threadIdx.x; t < m * per; t += blockDim.x) {
            const int mi = t / per, c4 = cb + t % per;
            const float4 r = r4[mi * h4 + c4];
            const float rs = s_ss[mi];
            const int col = c4 * 4;
            const float v0 = r.x * rs * (1.f + __bfloat162float(norm_w[col]));
            const float v1 = r.y * rs * (1.f + __bfloat162float(norm_w[col + 1]));
            const float v2 = r.z * rs * (1.f + __bfloat162float(norm_w[col + 2]));
            const float v3 = r.w * rs * (1.f + __bfloat162float(norm_w[col + 3]));
            if (xn) {
                bf162* o = reinterpret_cast<bf162*>(xn + (size_t)mi * h + col);
                o[0] = __floats2bfloat162_rn(v0, v1);
                o[1] = __floats2bfloat162_rn(v2, v3);
            }
            if (xh) store_xh4(xh + (size_t)mi * h, col, v0, v1, v2, v3);
        }
    }
    bump_seq(ctr, s);
}

// Large m (prefill): block per row (strided), so the fused norm needs no grid barrier. Only 4
// blocks: the prefill overlaps these with GEMMs, and more blocks would crowd them out of the SMs
// (4 blocks x 512 threads still keep the PCIe link busy).
__global__ void allreduce_rows_kernel(float* __restrict__ resid, const float* __restrict__ partial,
                                      int m, int h, Comm c, int rank, CommCounters ctr,
                                      const bf16* __restrict__ norm_w, float eps,
                                      bf16* __restrict__ xn) {
    __shared__ unsigned s_seq;
    __shared__ float s_w[32];
    if (threadIdx.x == 0) s_seq = *ctr.seq + 1;
    __syncthreads();
    const unsigned s = s_seq;
    const int slot = s % kCommSlots;
    const int h4 = h >> 2;
    const float4* p4 = reinterpret_cast<const float4*>(partial);
    float4* r4 = reinterpret_cast<float4*>(resid);
    if (c.ngpu > 1) {
        uint2* mine = reinterpret_cast<uint2*>(slot_ptr(c, rank, slot));
        for (int r = blockIdx.x; r < m; r += gridDim.x)
            for (int i = threadIdx.x; i < h4; i += blockDim.x)
                mine[r * h4 + i] = pack_bf16x4(p4[r * h4 + i]);
        exchange(c, rank, ctr, s, false, kWdArRowsExchange);
    }
    for (int r = blockIdx.x; r < m; r += gridDim.x) {
        float q = 0.f;
        for (int i = threadIdx.x; i < h4; i += blockDim.x) {
            const int idx = r * h4 + i;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int g = 0; g < c.ngpu; g++) {
                const float4 x =
                    c.ngpu == 1 ? p4[idx]
                    : g == rank ? unpack_bf16x4(pack_bf16x4(p4[idx]))
                                : unpack_bf16x4(__ldcv(reinterpret_cast<const uint2*>(slot_ptr(c, g, slot)) + idx));
                v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
            }
            float4 rr = r4[idx];
            rr.x += v.x; rr.y += v.y; rr.z += v.z; rr.w += v.w;
            r4[idx] = rr;
            q += rr.x * rr.x + rr.y * rr.y + rr.z * rr.z + rr.w * rr.w;
        }
        if (!norm_w) continue;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
        __syncthreads();
        if ((threadIdx.x & 31) == 0) s_w[threadIdx.x >> 5] = q;
        __syncthreads();
        float tot = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) tot += s_w[w];
        const float rs = rsqrtf(tot / h + eps);
        for (int i = threadIdx.x; i < h4; i += blockDim.x) {
            const float4 rr = r4[r * h4 + i];
            const int col = i * 4;
            bf162* o = reinterpret_cast<bf162*>(xn + (size_t)r * h + col);
            o[0] = __floats2bfloat162_rn(rr.x * rs * (1.f + __bfloat162float(norm_w[col])),
                                         rr.y * rs * (1.f + __bfloat162float(norm_w[col + 1])));
            o[1] = __floats2bfloat162_rn(rr.z * rs * (1.f + __bfloat162float(norm_w[col + 2])),
                                         rr.w * rs * (1.f + __bfloat162float(norm_w[col + 3])));
        }
    }
    bump_seq(ctr, s);
}

// Prefill all-reduce as reduce-scatter + all-gather over host memory (less host traffic than
// every GPU reading every peer's rows: 8/3 instead of 3 row-volumes per GPU). Phase 1: every GPU
// publishes its bf16 partial rows; the owner of a third of the rows sums them (fixed GPU order),
// rounds to bf16 and publishes the sums (phase 2), which every GPU (owner included) adds to the
// residual, so residuals stay identical. Uses two sequence numbers (slots s, s + 1).
// rs_off: row offset of this call's rows in the P2P sums buffer prs2 (so that the sums of two
// micro-batches stay apart); finish = false: leave the bf16 sums in prs2[rank] for
// allreduce_finish (the residual update + norm then run on the compute stream with all SMs)
// PB: the partial rows are already bf16 (the GEMM epilogue rounded them, RN as pack_bf16x4):
// half the local reads, the same packed words, so bit-identical sums
// OFG >= 0: owner-finished (OwnFin, G = OFG; wxn also writes the bf16 rows); finish must be false
template <bool PB = false, int OFG = -1, bool HCW = false>
__global__ void allreduce_rs_rows_kernel(float* __restrict__ resid, const float* __restrict__ partial,
                                         int m, int h, Comm c, int rank, CommCounters ctr,
                                         const bf16* __restrict__ norm_w, float eps,
                                         bf16* __restrict__ xn, int rs_off, bool finish, bool wxn = false,
                                         bool v2 = false, int dbg = 0, const float* __restrict__ hc_inj = nullptr,
                                         bool hc_own = false) {
    // HCW (Flash-Next hyper-connections): resid is the 4-stream residual R [m][4][h] and the tail adds
    // hc_inj[r][i] * sum to every stream i (= allreduce into y + k_hc_write, same arithmetic)
    // dbg (QE_DBG_AR, timing ablations only, results wrong): 1 = no phase-1 row transfers, 2 = no
    // phase-2 sums / publication, 4 = no cross-GPU exchange (flags), 8 = partial reads replaced by 0
    __shared__ unsigned s_seq;
    __shared__ float s_w[32];
    if (threadIdx.x == 0) s_seq = *ctr.seq + 1;
    __syncthreads();
    const unsigned s1 = s_seq, s2 = s_seq + 1;
    const int slot1 = s1 % kCommSlots, slot2 = s2 % kCommSlots;
    const int h4 = h >> 2;
    const float4* p4 = reinterpret_cast<const float4*>(partial);
    const uint2* pb4 = reinterpret_cast<const uint2*>(partial);
    auto pk = [&](size_t idx) {
        if (dbg & 8) return make_uint2(0u, 0u);
        if constexpr (PB) return pb4[idx]; else return pack_bf16x4(p4[idx]);
    };
    float4* r4 = reinterpret_cast<float4*>(resid);
    auto owner_begin = [&](int g) { return (int)((long)m * g / c.ngpu); };
    const int ob = owner_begin(rank), oe = owner_begin(rank + 1);
    // P2P: rows go straight into their owner's memory, the sums into every GPU's memory (local
    // reads only); same packing and summation order as the host path
    const bool pp = c.p2p && c.prs1[0] != nullptr && c.pflags[0] != nullptr && m <= c.rs_rows &&
                    h == c.ll_hidden;
    auto owner_of = [&](int r) {
        int og = 0;
        while (og + 1 < c.ngpu && r >= owner_begin(og + 1)) og++;
        return og;
    };
    uint2* mine1 = reinterpret_cast<uint2*>(slot_ptr(c, rank, slot1));
    for (int r = blockIdx.x; r < m && !(dbg & 1); r += gridDim.x) {
        if (r >= ob && r < oe) continue;  // own rows are reduced locally
        if (pp) {
            const int og = owner_of(r);
            uint2* dst = c.prs1[og] + ((size_t)rank * c.rs_own + (r - owner_begin(og))) * h4;
            if (dbg & 16) {  // 16-byte P2P stores (8 bf16 per thread): half the write transactions
                uint4* d4 = reinterpret_cast<uint4*>(dst);
                for (int i = threadIdx.x; i < h4 / 2; i += blockDim.x) {
                    const uint2 a = pk((size_t)r * h4 + 2 * i), b = pk((size_t)r * h4 + 2 * i + 1);
                    d4[i] = make_uint4(a.x, a.y, b.x, b.y);
                }
            } else {
                for (int i = threadIdx.x; i < h4; i += blockDim.x) dst[i] = pk((size_t)r * h4 + i);
            }
        } else {
            for (int i = threadIdx.x; i < h4; i += blockDim.x) mine1[r * h4 + i] = pk((size_t)r * h4 + i);
        }
        if (dbg & 32) __nanosleep(2000);  // (QE_AR_PACE: pace the P2P row writes)
    }
    if (!(dbg & 4)) exchange(c, rank, ctr, s1, pp, kWdArRowsExchange);
    if constexpr (OFG >= 0) {
        // owner finish: resid += bf16(sum) on the owned rows, norm (512 threads: ar_finish_kernel's
        // summation order), normed rows -> int8 / bf16 receive rows of every GPU
        constexpr int G = OFG;
        constexpr int L = G > 0 ? G / 4 : 1;
        constexpr int NREC = G > 0 ? (128 / G + 1) / 2 : 1;
        __shared__ float s_w2[16];
        for (int r = ob + blockIdx.x; r < oe; r += gridDim.x) {
            float4 keep[4];
            float q = 0.f;
#pragma unroll
            for (int it = 0; it < 4; it++) {
                const int i = threadIdx.x + it * blockDim.x;
                if (i < h4) {
                    const int idx = r * h4 + i;
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    for (int g = 0; g < c.ngpu; g++) {
                        const uint2* src = c.prs1[rank] + ((size_t)g * c.rs_own + (r - ob)) * h4 + i;
                        const float4 x = g == rank ? unpack_bf16x4(pk(idx)) : unpack_bf16x4(__ldcv(src));
                        v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
                    }
                    const float4 vb = unpack_bf16x4(pack_bf16x4(v));
                    float4 rr = r4[idx];
                    rr.x += vb.x; rr.y += vb.y; rr.z += vb.z; rr.w += vb.w;
                    r4[idx] = rr;
                    keep[it] = rr;
                    q += rr.x * rr.x + rr.y * rr.y + rr.z * rr.z + rr.w * rr.w;
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
            if ((threadIdx.x & 31) == 0) s_w2[threadIdx.x >> 5] = q;
            __syncthreads();
            float tot = 0.f;
            for (int w = 0; w < (int)(blockDim.x >> 5); w++) tot += s_w2[w];
            const float rs = rsqrtf(tot / h + eps);
            const size_t orow = (size_t)rs_off + r;
#pragma unroll
            for (int it = 0; it < 4; it++) {
                const int i = threadIdx.x + it * blockDim.x;
                if (i >= h4) continue;
                const float4 rr = keep[it];
                const int col = i * 4;
                const bf162 o0 = __floats2bfloat162_rn(rr.x * rs * (1.f + __bfloat162float(norm_w[col])),
                                                       rr.y * rs * (1.f + __bfloat162float(norm_w[col + 1])));
                const bf162 o1 = __floats2bfloat162_rn(rr.z * rs * (1.f + __bfloat162float(norm_w[col + 2])),
                                                       rr.w * rs * (1.f + __bfloat162float(norm_w[col + 3])));
                if (G == 0 || wxn) {
                    uint2 w2;
                    w2.x = *reinterpret_cast<const uint32_t*>(&o0);
                    w2.y = *reinterpret_cast<const uint32_t*>(&o1);
                    for (int g = 0; g < c.ngpu; g++) c.pxn[g][orow * h4 + i] = w2;
                }
                if constexpr (G > 0) {
                    const float2 f0 = __bfloat1622float2(o0), f1 = __bfloat1622float2(o1);
                    float mx = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
#pragma unroll
                    for (int d = 1; d < L; d <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
                    const float sc = mx > 0.f ? __uint_as_float(__float_as_uint(mx / 127.f) & ~3u) : 1.f;
                    const float inv = mx > 0.f ? __frcp_rn(sc) : 0.f;
                    const int q0 = __float2int_rn(f0.x * inv), q1 = __float2int_rn(f0.y * inv);
                    const int q2 = __float2int_rn(f1.x * inv), q3 = __float2int_rn(f1.y * inv);
                    int sum = q0 + q1 + q2 + q3;
#pragma unroll
                    for (int d = 1; d < L; d <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
                    if ((i % L) == 0 && v2) {  // gemm_i8v_kernel records (gemm.cu i8v_rec_store<G, false>)
                        constexpr int SUB = 128 / G;
                        const size_t base = (size_t)(rs_off + 15) / 16 * 16 * ((size_t)(h / 128) * SUB * 2 * 2);
                        const size_t f4 = ((size_t)(r >> 4) * (h / 128) + col / 128) * (SUB * 2 * 8) + ((col % 128) / G) * 2 * 8 + (r & 7);
                        const float cz = __int_as_float((int)0x4B400000 - 8 * sum);
                        for (int g = 0; g < c.ngpu; g++) {
                            float2* p = reinterpret_cast<float2*>(reinterpret_cast<float4*>(c.pxsc[g] + base) + f4) + ((r >> 3) & 1);
                            p[0] = make_float2(cz, cz);
                            p[16] = make_float2(sc, -12582912.f * sc);
                        }
                    } else if ((i % L) == 0) {
                        const int sub = (col % 128) / G;
                        const float2 rec = make_float2(sc, __int_as_float((int)0x4B400000 - 8 * sum));
                        const size_t recf = (size_t)(h / 128) * NREC * 4;  // x_to_i8 record floats per row
                        const size_t off = orow * recf + ((size_t)(col / 128) * NREC + sub / 2) * 4 + (sub & 1) * 2;
                        for (int g = 0; g < c.ngpu; g++) *reinterpret_cast<float2*>(c.pxsc[g] + off) = rec;
                    }
                    const uint32_t a = (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) |
                                       ((uint32_t)(q3 & 0xFF) << 24);
                    const uint32_t b = __shfl_xor_sync(0xffffffffu, a, 2);
                    const int ip = i & 3;
                    uint32_t out;
                    if ((ip & 2) == 0) out = (a & 0xFF) | ((b & 0xFF) << 8) | (((a >> 8) & 0xFF) << 16) | (((b >> 8) & 0xFF) << 24);
                    else out = ((b >> 16) & 0xFF) | (((a >> 16) & 0xFF) << 8) | (((b >> 24) & 0xFF) << 16) | ((a >> 24) << 24);
                    const int j = 2 * (ip & 1) + (ip >> 1);
                    for (int g = 0; g < c.ngpu; g++)
                        reinterpret_cast<uint32_t*>(c.pxq[g] + orow * h + (col & ~15))[j] = out;
                }
            }
            __syncthreads();  // s_w2 of the next row
        }
        exchange(c, rank, ctr, s2, pp, kWdArRowsExchange);
        bump_seq(ctr, s2);
        return;
    }
    if constexpr (HCW) {
        if (hc_own && pp) {
            // sequence-parallel HC (Flash prefill): the owner adds hc_inj * bf16(sum) to its own rows only (the same
            // rounded sums and arithmetic as the HCW tail below); nothing is published. The second exchange stays as
            // the barrier that keeps the next call's phase-1 writes off these receive rows until they are read.
            for (int r = ob + blockIdx.x; r < oe; r += gridDim.x) {
                const float j0 = hc_inj[r * 4], j1 = hc_inj[r * 4 + 1], j2 = hc_inj[r * 4 + 2], j3 = hc_inj[r * 4 + 3];
                const float jj[4] = {j0, j1, j2, j3};
                for (int i = threadIdx.x; i < h4; i += blockDim.x) {
                    const int idx = r * h4 + i;
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    for (int g = 0; g < c.ngpu; g++) {
                        const uint2* src = c.prs1[rank] + ((size_t)g * c.rs_own + (r - ob)) * h4 + i;
                        const float4 x = g == rank ? unpack_bf16x4(pk(idx)) : unpack_bf16x4(__ldcv(src));
                        v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
                    }
                    const float4 vb = unpack_bf16x4(pack_bf16x4(v));
                    float4* rr = r4 + (size_t)r * 4 * h4 + i;
#pragma unroll
                    for (int s4 = 0; s4 < 4; s4++) {
                        float4 x = rr[s4 * h4];
                        x.x += jj[s4] * vb.x; x.y += jj[s4] * vb.y; x.z += jj[s4] * vb.z; x.w += jj[s4] * vb.w;
                        rr[s4 * h4] = x;
                    }
                }
            }
            if (!(dbg & 4)) exchange(c, rank, ctr, s2, pp, kWdArRowsExchange);
            bump_seq(ctr, s2);
            return;
        }
    }
    // reduce the owned rows, publish the bf16 sums
    uint2* mine2 = reinterpret_cast<uint2*>(slot_ptr(c, rank, slot2));
    if (pp && (dbg & 16) && !(dbg & 2)) {  // 16-byte stores: two float4 of sums per thread
        for (int r = ob + blockIdx.x; r < oe; r += gridDim.x)
            for (int i2 = threadIdx.x; i2 < h4 / 2; i2 += blockDim.x) {
                uint2 pkk[2];
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    const int i = 2 * i2 + u, idx = r * h4 + i;
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    for (int g = 0; g < c.ngpu; g++) {
                        const uint2* src = c.prs1[rank] + ((size_t)g * c.rs_own + (r - ob)) * h4 + i;
                        const float4 x = g == rank ? unpack_bf16x4(pk(idx)) : unpack_bf16x4(__ldcv(src));
                        v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
                    }
                    pkk[u] = pack_bf16x4(v);
                }
                const uint4 w4 = make_uint4(pkk[0].x, pkk[0].y, pkk[1].x, pkk[1].y);
                for (int g = 0; g < c.ngpu; g++)
                    reinterpret_cast<uint4*>(c.prs2[g] + (size_t)rs_off * h4 + (size_t)r * h4)[i2] = w4;
            }
        dbg |= 2;  // done
    }
    for (int r = ob + blockIdx.x; r < oe && !(dbg & 2); r += gridDim.x)
        for (int i = threadIdx.x; i < h4; i += blockDim.x) {
            const int idx = r * h4 + i;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int g = 0; g < c.ngpu; g++) {
                const uint2* src = pp ? c.prs1[rank] + ((size_t)g * c.rs_own + (r - ob)) * h4 + i
                                      : reinterpret_cast<const uint2*>(slot_ptr(c, g, slot1)) + idx;
                const float4 x = g == rank ? unpack_bf16x4(pk(idx)) : unpack_bf16x4(__ldcv(src));
                v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
            }
            const uint2 pk = pack_bf16x4(v);
            if (pp) {
                for (int g = 0; g < c.ngpu; g++) c.prs2[g][(size_t)rs_off * h4 + idx] = pk;
            } else {
                mine2[idx] = pk;
            }
        }
    if (!(dbg & 4)) exchange(c, rank, ctr, s2, pp, kWdArRowsExchange);
    if (!finish) {
        bump_seq(ctr, s2);
        return;
    }
    if constexpr (HCW) {
        for (int r = blockIdx.x; r < m; r += gridDim.x) {
            int og = 0;
            while (og + 1 < c.ngpu && r >= owner_begin(og + 1)) og++;
            const uint2* src = pp ? c.prs2[rank] + (size_t)rs_off * h4 : reinterpret_cast<const uint2*>(slot_ptr(c, og, slot2));
            const float j0 = hc_inj[r * 4], j1 = hc_inj[r * 4 + 1], j2 = hc_inj[r * 4 + 2], j3 = hc_inj[r * 4 + 3];
            for (int i = threadIdx.x; i < h4; i += blockDim.x) {
                const int idx = r * h4 + i;
                const float4 v = unpack_bf16x4((og == rank && !pp) ? src[idx] : __ldcv(src + idx));
                float4* rr = r4 + (size_t)r * 4 * h4 + i;
                const float jj[4] = {j0, j1, j2, j3};
#pragma unroll
                for (int s4 = 0; s4 < 4; s4++) {
                    float4 x = rr[s4 * h4];
                    x.x += jj[s4] * v.x; x.y += jj[s4] * v.y; x.z += jj[s4] * v.z; x.w += jj[s4] * v.w;
                    rr[s4 * h4] = x;
                }
            }
        }
        bump_seq(ctr, s2);
        return;
    }
    for (int r = blockIdx.x; r < m; r += gridDim.x) {
        int og = 0;
        while (og + 1 < c.ngpu && r >= owner_begin(og + 1)) og++;
        const uint2* src = pp ? c.prs2[rank] + (size_t)rs_off * h4 : reinterpret_cast<const uint2*>(slot_ptr(c, og, slot2));
        float q = 0.f;
        for (int i = threadIdx.x; i < h4; i += blockDim.x) {
            const int idx = r * h4 + i;
            const float4 v = unpack_bf16x4((og == rank && !pp) ? src[idx] : __ldcv(src + idx));
            float4 rr = r4[idx];
            rr.x += v.x; rr.y += v.y; rr.z += v.z; rr.w += v.w;
            r4[idx] = rr;
            q += rr.x * rr.x + rr.y * rr.y + rr.z * rr.z + rr.w * rr.w;
        }
        if (!norm_w) continue;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
        __syncthreads();
        if ((threadIdx.x & 31) == 0) s_w[threadIdx.x >> 5] = q;
        __syncthreads();
        float tot = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) tot += s_w[w];
        const float rs = rsqrtf(tot / h + eps);
        for (int i = threadIdx.x; i < h4; i += blockDim.x) {
            const float4 rr = r4[r * h4 + i];
            const int col = i * 4;
            bf162* o = reinterpret_cast<bf162*>(xn + (size_t)r * h + col);
            o[0] = __floats2bfloat162_rn(rr.x * rs * (1.f + __bfloat162float(norm_w[col])),
                                         rr.y * rs * (1.f + __bfloat162float(norm_w[col + 1])));
            o[1] = __floats2bfloat162_rn(rr.z * rs * (1.f + __bfloat162float(norm_w[col + 2])),
                                         rr.w * rs * (1.f + __bfloat162float(norm_w[col + 3])));
        }
    }
    bump_seq(ctr, s2);
}

// Deferred tail of allreduce_rs_rows_kernel (P2P path): one block per row, resid += sums,
// xn = bf16(rmsnorm(resid) * (1 + norm_w)); same arithmetic and summation order as the fused tail.
// G > 0: also the W4A8 input of the next GEMM (x_to_i8_kernel's layouts: int8 rows in the MMA
// k permutation, (sx, -M sx) per group of G), quantized from the bf16-rounded xn values.
// FMT: 1 = gemm_i8t_kernel records, 0 = gemm_i8_kernel pairs, 2 = gemm_i8v_kernel (16-token tiles, MMA-operand order)
template <int G, int FMT = 1>
__global__ void __launch_bounds__(512) ar_finish_kernel(float* __restrict__ resid, const uint2* __restrict__ sums,
                                                        int h, const bf16* __restrict__ norm_w, float eps,
                                                        bf16* __restrict__ xn, int8_t* __restrict__ xq,
                                                        float* __restrict__ xsc, int* __restrict__ ctr, bool wxn,
                                                        const SteerArg sa) {
    __shared__ float s_w[16];  // 512 threads as the fused tail: identical summation order
    const int r = blockIdx.x, h4 = h >> 2;
    // stream-K work counter of the W4A8 GEMM that consumes these rows next on this stream (replaces a
    // cudaMemsetAsync: a memset node between two kernels cost ~25-80 us of idle SMs per GEMM)
    if (ctr && r == 0 && threadIdx.x == 0) *ctr = 0;
    float4* r4 = reinterpret_cast<float4*>(resid) + (size_t)r * h4;
    const uint2* src = sums + (size_t)r * h4;
    float q = 0.f;
    constexpr int KEEP = 4;  // updated residual float4s kept in registers for the norm pass (h <= 8192)
    float4 keep[KEEP];
#pragma unroll
    for (int it = 0; it < KEEP; it++) {
        const int i = threadIdx.x + it * blockDim.x;
        if (i < h4) {
            const float4 v = unpack_bf16x4(__ldcv(src + i));  // written by peers over P2P
            float4 rr = r4[i];
            rr.x += v.x; rr.y += v.y; rr.z += v.z; rr.w += v.w;
            r4[i] = rr;
            keep[it] = rr;
            q += rr.x * rr.x + rr.y * rr.y + rr.z * rr.z + rr.w * rr.w;
        }
    }
    for (int i = threadIdx.x + KEEP * blockDim.x; i < h4; i += blockDim.x) {
        const float4 v = unpack_bf16x4(__ldcv(src + i));
        float4 rr = r4[i];
        rr.x += v.x; rr.y += v.y; rr.z += v.z; rr.w += v.w;
        r4[i] = rr;
        q += rr.x * rr.x + rr.y * rr.y + rr.z * rr.z + rr.w * rr.w;
    }
    {  // layer-output steering of this row (steer.cuh) before the norm: h += c w, c = scale or -scale (h . w)
        float ssc = 0.f;
        int smode = 0;
        const float* sdir = steer_dir(sa, r, h, ssc, smode);
        if (sdir) {  // block-uniform
            const float4* w4 = reinterpret_cast<const float4*>(sdir);
            float c = ssc;
            if (smode == 1) {
                float d = 0.f;
#pragma unroll
                for (int it = 0; it < KEEP; it++) {
                    const int i = threadIdx.x + it * blockDim.x;
                    if (i < h4) {
                        const float4 v = w4[i];
                        d += keep[it].x * v.x + keep[it].y * v.y + keep[it].z * v.z + keep[it].w * v.w;
                    }
                }
                for (int i = threadIdx.x + KEEP * blockDim.x; i < h4; i += blockDim.x) {
                    const float4 v = w4[i], x = r4[i];
                    d += x.x * v.x + x.y * v.y + x.z * v.z + x.w * v.w;
                }
                c = -ssc * steer_block_sum<512>(d);
            }
            q = 0.f;
#pragma unroll
            for (int it = 0; it < KEEP; it++) {
                const int i = threadIdx.x + it * blockDim.x;
                if (i < h4) {
                    const float4 v = w4[i];
                    float4& x = keep[it];
                    x.x += c * v.x; x.y += c * v.y; x.z += c * v.z; x.w += c * v.w;
                    r4[i] = x;
                    q += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
                }
            }
            for (int i = threadIdx.x + KEEP * blockDim.x; i < h4; i += blockDim.x) {
                const float4 v = w4[i];
                float4 x = r4[i];
                x.x += c * v.x; x.y += c * v.y; x.z += c * v.z; x.w += c * v.w;
                r4[i] = x;
                q += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
            }
        }
    }
    if (!norm_w) return;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if ((threadIdx.x & 31) == 0) s_w[threadIdx.x >> 5] = q;
    __syncthreads();
    float tot = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) tot += s_w[w];
    const float rs = rsqrtf(tot / h + eps);
    constexpr int L = G > 0 ? G / 4 : 1;  // float4s (lanes) per activation group
    // wxn = false: only the int8 copy is consumed (the next GEMM is W4A8), the bf16 rows are not written
    if (G == 0) wxn = true;
    auto body = [&](const float4 rr, const int i) {
        const int col = i * 4;
        bf162* o = reinterpret_cast<bf162*>(xn + (size_t)r * h + col);
        const bf162 o0 = __floats2bfloat162_rn(rr.x * rs * (1.f + __bfloat162float(norm_w[col])),
                                               rr.y * rs * (1.f + __bfloat162float(norm_w[col + 1])));
        const bf162 o1 = __floats2bfloat162_rn(rr.z * rs * (1.f + __bfloat162float(norm_w[col + 2])),
                                               rr.w * rs * (1.f + __bfloat162float(norm_w[col + 3])));
        if (wxn) {
            o[0] = o0;
            o[1] = o1;
        }
        if constexpr (G > 0) {
            // sweeps of 512 float4 (2048 k): a group's L float4s sit in L consecutive lanes of one
            // warp (L <= 32) and partial sweeps consist of whole warps (h4 % 32 == 0)
            const float2 f0 = __bfloat1622float2(o0), f1 = __bfloat1622float2(o1);
            float mx = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
#pragma unroll
            for (int d = 1; d < L; d <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
            const float sc = mx > 0.f ? __uint_as_float(__float_as_uint(mx / 127.f) & ~3u) : 1.f;
            const float inv = mx > 0.f ? __frcp_rn(sc) : 0.f;
            const int q0 = __float2int_rn(f0.x * inv), q1 = __float2int_rn(f0.y * inv);
            const int q2 = __float2int_rn(f1.x * inv), q3 = __float2int_rn(f1.y * inv);
            if constexpr (FMT == 2) {
                int sum = q0 + q1 + q2 + q3;
#pragma unroll
                for (int d = 1; d < L; d <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
                if ((i % L) == 0) {  // as gemm.cu i8v_rec_store<G, false>
                    constexpr int SUB = 128 / G;
                    float2* p = reinterpret_cast<float2*>(reinterpret_cast<float4*>(xsc) +
                                                          ((size_t)(r >> 4) * (h / 128) + col / 128) * (SUB * 2 * 8) +
                                                          ((col % 128) / G) * 2 * 8 + (r & 7)) + ((r >> 3) & 1);
                    const float cz = __int_as_float((int)0x4B400000 - 8 * sum);
                    p[0] = make_float2(cz, cz);
                    p[16] = make_float2(sc, -12582912.f * sc);
                }
            } else if constexpr (FMT == 1) {
                int sum = q0 + q1 + q2 + q3;  // record (sx, M_bits - 8 sum_g q) as x_to_i8_kernel
#pragma unroll
                for (int d = 1; d < L; d <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
                constexpr int NREC = (128 / G + 1) / 2;  // {sx0, C0, sx1, C1} per 128-k group
                if ((i % L) == 0) {
                    const int sub = (col % 128) / G;
                    *reinterpret_cast<float2*>(xsc + (((size_t)r * (h / 128) + col / 128) * NREC + sub / 2) * 4 + (sub & 1) * 2) =
                        make_float2(sc, __int_as_float((int)0x4B400000 - 8 * sum));
                }
            } else {  // master's format for gemm_i8_kernel: {sx, -M sx} pairs, SP = max(2, 128 / G) per 128-k group
                constexpr int SP = G == 128 ? 2 : 128 / G;
                if ((i % L) == 0)
                    *reinterpret_cast<float2*>(xsc + (((size_t)r * (h / 128) + col / 128) * SP + (col % 128) / G) * 2) =
                        make_float2(sc, -12582912.f * sc);
            }
            const uint32_t a = (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) |
                               ((uint32_t)(q3 & 0xFF) << 24);
            // MMA permutation within each 16-k half: u32 j holds k {2j, 2j+8, 2j+1, 2j+9}; this
            // thread's 4 k (4 i' .. 4 i' + 3, i' = i & 3 within the half) pair with lane ^ 2
            const uint32_t b = __shfl_xor_sync(0xffffffffu, a, 2);
            const int ip = i & 3;
            uint32_t out;
            if ((ip & 2) == 0) out = (a & 0xFF) | ((b & 0xFF) << 8) | (((a >> 8) & 0xFF) << 16) | (((b >> 8) & 0xFF) << 24);
            else out = ((b >> 16) & 0xFF) | (((a >> 16) & 0xFF) << 8) | (((b >> 24) & 0xFF) << 16) | ((a >> 24) << 24);
            const int j = 2 * (ip & 1) + (ip >> 1);
            reinterpret_cast<uint32_t*>(xq + (size_t)r * h + (col & ~15))[j] = out;
        }
    };
#pragma unroll
    for (int it = 0; it < KEEP; it++) {
        const int i = threadIdx.x + it * blockDim.x;
        if (i < h4) body(keep[it], i);
    }
    for (int i = threadIdx.x + KEEP * blockDim.x; i < h4; i += blockDim.x) body(r4[i], i);
}

__device__ __forceinline__ void st_ll(uint2* p, uint32_t data, uint32_t seq) {
    asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" ::"l"(p), "r"(data), "r"(seq) : "memory");
}
__device__ __forceinline__ uint2 ld_ll(const uint2* p) {
    uint2 v;
    asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
    return v;
}

constexpr int kLLThreads = 1024, kLLMaxWords = 4;  // hidden <= 3 * 1024 * 4; 1024 threads keep
                                                    // enough PCIe reads in flight per row

__global__ void bump_epoch_kernel(unsigned* epoch) { *epoch += 1; }

__device__ __forceinline__ uint32_t bf16_bits(float v) {
    const bf16 b = __float2bfloat16(v);
    return *reinterpret_cast<const uint16_t*>(&b);
}
__device__ __forceinline__ float bits_bf16(uint32_t u) {
    return __uint_as_float(u << 16);
}
__device__ __forceinline__ float round_bf16(float v) { return bits_bf16(bf16_bits(v)); }

// LL all-reduce for decode: one block per token row. An 8-byte word carries three bf16 values
// and a 16-bit tag (64-bit stores are atomic, so a reader that sees the tag sees the data; a
// slot is rewritten every few collectives, so 16 bits cannot alias stale data). Each thread
// publishes its words, issues all peer loads before checking tags (re-polling only words that
// are not there yet), sums in fixed GPU order, updates the residual and applies the fused norm.
__global__ void __launch_bounds__(kLLThreads) allreduce_ll_kernel(
    float* __restrict__ resid, const float* __restrict__ partial, int h, Comm c, int rank,
    CommCounters ctr, int idx, const bf16* __restrict__ norm_w, float eps,
    bf16* __restrict__ xn, __half* __restrict__ xh, bool rsum) {
    __shared__ float s_w[kLLThreads / 32];
    const int row = blockIdx.x, tid = threadIdx.x;
    const unsigned ep = *ctr.epoch;
    const uint32_t tag = ((ep * 1024u + idx) % 65535u + 1u) << 16;  // never 0
    const int slot = ((ep & 1) << 1) | (idx & 1);  // differs from the previous collective's
    const int hw = (h + 2) / 3;
    const size_t slot_words = (size_t)kMaxBatchRows * ((c.ll_hidden + 2) / 3);
    auto region = [&](int g) { return c.ll + ((size_t)g * 4 + slot) * slot_words + (size_t)row * hw; };
    const float* pr = partial + (size_t)row * h;
    float* rr = resid + (size_t)row * h;

    uint2 mine[kLLMaxWords];
    uint2* my = region(rank);
#pragma unroll
    for (int j = 0; j < kLLMaxWords; j++) {
        const int w = tid + j * kLLThreads;
        if (w < hw) {
            const int i = 3 * w;
            const uint32_t a = bf16_bits(pr[i]);
            const uint32_t b = i + 1 < h ? bf16_bits(pr[i + 1]) : 0u;
            const uint32_t d = i + 2 < h ? bf16_bits(pr[i + 2]) : 0u;
            mine[j] = make_uint2(a | (b << 16), d | tag);
            st_ll(my + w, mine[j].x, mine[j].y);
        }
    }
    float acc[kLLMaxWords][3];
#pragma unroll
    for (int j = 0; j < kLLMaxWords; j++) acc[j][0] = acc[j][1] = acc[j][2] = 0.f;
    for (int g = 0; g < c.ngpu; g++) {  // fixed order -> identical residual on every GPU
        const uint2* src = region(g);
        uint2 u[kLLMaxWords];
#pragma unroll
        for (int j = 0; j < kLLMaxWords; j++) {
            const int w = tid + j * kLLThreads;
            if (w < hw) u[j] = g == rank ? mine[j] : ld_ll(src + w);
        }
#pragma unroll
        for (int j = 0; j < kLLMaxWords; j++) {
            const int w = tid + j * kLLThreads;
            if (w >= hw) continue;
            if (g != rank)
                for (unsigned ws = 0; (u[j].y & 0xFFFF0000u) != tag;) { u[j] = ld_ll(src + w); QE_WD_POLL(ws, kWdLL, g, tag, u[j].y); }
            acc[j][0] += bits_bf16(u[j].x & 0xFFFFu);
            acc[j][1] += bits_bf16(u[j].x >> 16);
            acc[j][2] += bits_bf16(u[j].y & 0xFFFFu);
        }
    }
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < kLLMaxWords; j++) {
        const int w = tid + j * kLLThreads;
        if (w >= hw) continue;
#pragma unroll
        for (int e = 0; e < 3; e++) {
            const int i = 3 * w + e;
            if (i >= h) break;
            const float r = rr[i] + (rsum ? round_bf16(acc[j][e]) : acc[j][e]);
            rr[i] = r;
            acc[j][e] = r;
            q += r * r;
        }
    }
    if (!norm_w) return;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if ((tid & 31) == 0) s_w[tid >> 5] = q;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < kLLThreads / 32; w++) tot += s_w[w];
    const float rs = rsqrtf(tot / h + eps);
    bf16* o = xn + (size_t)row * h;
#pragma unroll
    for (int j = 0; j < kLLMaxWords; j++) {
        const int w = tid + j * kLLThreads;
        if (w >= hw) continue;
#pragma unroll
        for (int e = 0; e < 3; e++) {
            const int i = 3 * w + e;
            if (i >= h) break;
            const float v = acc[j][e] * rs * (1.f + __bfloat162float(norm_w[i]));
            if (xn) o[i] = __float2bfloat16(v);
            if (xh) store_xh(xh + (size_t)row * h, i, v);
        }
    }
}

// LL with 16-byte words: 7 bf16 + a 16-bit tag (14 of 16 bytes payload instead of 6 of 8), for
// the bandwidth-bound multi-row case. Relies on 16-byte aligned stores / loads to host memory
// being single PCIe transactions (checked by stress runs).
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
constexpr int kLL16Threads = 768;  // 7 * 768 = 5376 >= hidden

// P2P receive words are read by exactly one thread of one GPU (the buffer's owner): once consumed,
// the word is cleared (tag 0 never matches), so a later collective can only ever see words written
// for it. Without this a word that went unwritten for a multiple of the tag period (rows or
// protocol not used for a while: m changes, all-to-all <-> reduce-scatter) could alias.
__device__ __forceinline__ void ll16_consume(const uint4* p) {
    st_ll16(const_cast<uint4*>(p), make_uint4(0u, 0u, 0u, 0u));
}

// MODE 0: all-to-all, the fp32 sum of the three bf16 partials is added to the residual.
// MODE 1: all-to-all, the sum is rounded to bf16 before it is added (the arithmetic of MODE 2).
// MODE 2 (P2P only): reduce-scatter + all-gather over the 16-byte words. Word w of every row is
//   owned by GPU w / ceil(hw / ngpu); the others send it only to its owner, which sums the three
//   partials (same fixed GPU order as MODE 1), rounds the sum to bf16 and pushes it to the peers.
//   Every GPU sends 2/3 of the bytes of the all-to-all (one extra P2P hop). MODE 1 and MODE 2 give
//   bit-identical results, so the protocol may be picked per row count (MTP == greedy holds).
//   Buffers: partials for my words land in pregion(me, sender) at word w; the owner's sum of word
//   w lands in pregion(peer, owner) at word w (a location the reduce-scatter never writes).
// Layer-output steering in the 7-value tail of the LL16 / LLC all-reduces (steer.cuh): the row's values v (already
// residual + sum, stored to rr) get + c w with c = scale (add) or -scale (v . w) (project, a block reduction), are
// stored again and the norm's sum of squares q is recomputed in the same order. Called block-uniformly (a steered row).
template <class IDX>
__device__ __forceinline__ void steer_vals7(const float* __restrict__ w, float sc, int mode, float (&v)[7], IDX cidx,
                                            bool on, int h, float* __restrict__ rr, float& q) {
    float wv[7];
    float d = 0.f;
#pragma unroll
    for (int e = 0; e < 7; e++) {
        const int i = cidx(e);
        const bool ok = on && i < h;
        wv[e] = ok ? w[i] : 0.f;
        if (ok) d += v[e] * wv[e];
    }
    const float c = mode == 1 ? -sc * steer_block_sum<kLL16Threads>(d) : sc;
    q = 0.f;
    if (on)
#pragma unroll
        for (int e = 0; e < 7; e++) {
            const int i = cidx(e);
            if (i >= h) break;
            const float r = v[e] + c * wv[e];
            rr[i] = r;
            v[e] = r;
            q += r * r;
        }
}

template <int MODE>
__global__ void __launch_bounds__(kLL16Threads) allreduce_ll16_kernel(
    float* __restrict__ resid, const float* __restrict__ partial, int h, Comm c, int rank,
    CommCounters ctr, int idx, const bf16* __restrict__ norm_w, float eps,
    bf16* __restrict__ xn, __half* __restrict__ xh, const SteerArg sa) {
    __shared__ float s_w[kLL16Threads / 32];
    const int row = blockIdx.x, tid = threadIdx.x;
    float ssc = 0.f;
    int smode = 0;
    const float* sdir = steer_dir(sa, row, h, ssc, smode);  // layer-output steering of this row (steer.cuh), or null
    const unsigned ep = *ctr.epoch;
    const uint32_t tag = ((ep * 1024u + idx) % 65535u + 1u) << 16;  // never 0
    const int slot = ((ep & 1) << 1) | (idx & 1);
    const int hw = (h + 6) / 7;
    const size_t slot_words = (size_t)kMaxBatchRows * ((c.ll_hidden + 2) / 3);  // uint2 units
    auto region = [&](int g) {  // host path: sender g's words
        return reinterpret_cast<uint4*>(c.ll + ((size_t)g * 4 + slot) * slot_words) + (size_t)row * hw;
    };
    const int hwp = pll_row_words(c.ll_hidden);
    auto pregion = [&](int owner, int sender) {  // P2P path: in `owner`'s memory, from `sender`
        return c.pll[owner] + (((size_t)slot * c.ngpu + sender) * kMaxBatchRows + row) * hwp;
    };
    const float* pr = partial + (size_t)row * h;
    float* rr = resid + (size_t)row * h;
    const int w = tid;
    const bool on = w < hw;
    // value e of word w: 7 w + e (master, MODE 0) or w + hw e (QE_AR_COAL: coalesced across the warp)
    const bool coal = MODE != 0 && c.coal;
    auto cidx = [&](int e) { return coal ? w + hw * e : 7 * w + e; };
    float rv[7], nwv[7];  // residual and norm weights, loaded before the polls (latency overlaps the wait)
#pragma unroll
    for (int e = 0; e < 7; e++) {
        const int i = cidx(e);
        const bool ok = on && i < h;
        rv[e] = ok ? rr[i] : 0.f;
        nwv[e] = ok && norm_w ? __bfloat162float(norm_w[i]) : 0.f;
    }
    uint4 mine = make_uint4(0, 0, 0, 0);
    if (on) {
        uint32_t b[7];
#pragma unroll
        for (int e = 0; e < 7; e++) b[e] = cidx(e) < h ? bf16_bits(pr[cidx(e)]) : 0u;
        mine = make_uint4(b[0] | (b[1] << 16), b[2] | (b[3] << 16), b[4] | (b[5] << 16), b[6] | tag);
        if (MODE == 2) {
            const int own = w / ((hw + c.ngpu - 1) / c.ngpu);
            if (own != rank) st_ll16(pregion(own, rank) + w, mine);
        } else if (c.p2p) {
            for (int g = 0; g < c.ngpu; g++)
                if (g != rank) st_ll16(pregion(g, rank) + w, mine);
        } else {
            st_ll16(region(rank) + w, mine);
        }
    }
    float acc[7] = {};
    const int own = MODE == 2 ? w / ((hw + c.ngpu - 1) / c.ngpu) : rank;
    if (MODE != 2 || own == rank)
        for (int g = 0; g < c.ngpu; g++) {  // fixed order -> identical residual on every GPU
            if (!on) continue;
            uint4 u = mine;
            if (g != rank) {
                const uint4* src = c.p2p ? pregion(rank, g) + w : region(g) + w;
                u = ld_ll16(src);
#ifndef QE_LL16_NOWAIT
                for (unsigned ws = 0; (u.w & 0xFFFF0000u) != tag;) { u = ld_ll16(src); QE_WD_POLL(ws, kWdLL16, g, tag, u.w); }
#endif
                if (c.p2p && c.clear) ll16_consume(src);
            }
            acc[0] += bits_bf16(u.x & 0xFFFFu); acc[1] += bits_bf16(u.x >> 16);
            acc[2] += bits_bf16(u.y & 0xFFFFu); acc[3] += bits_bf16(u.y >> 16);
            acc[4] += bits_bf16(u.z & 0xFFFFu); acc[5] += bits_bf16(u.z >> 16);
            acc[6] += bits_bf16(u.w & 0xFFFFu);
        }
    if (MODE >= 1 && on) {
        if (MODE == 1 || own == rank) {
            uint32_t b[7];
#pragma unroll
            for (int e = 0; e < 7; e++) {
                b[e] = bf16_bits(acc[e]);
                acc[e] = bits_bf16(b[e]);
            }
            if (MODE == 2) {  // all-gather: the owner pushes the rounded sums
                const uint4 sw = make_uint4(b[0] | (b[1] << 16), b[2] | (b[3] << 16), b[4] | (b[5] << 16), b[6] | tag);
                for (int g = 0; g < c.ngpu; g++)
                    if (g != rank) st_ll16(pregion(g, rank) + w, sw);
            }
        } else {  // MODE 2, a word owned by a peer: wait for its sum
            const uint4* src = pregion(rank, own) + w;
            uint4 u = ld_ll16(src);
            for (unsigned ws = 0; (u.w & 0xFFFF0000u) != tag;) { u = ld_ll16(src); QE_WD_POLL(ws, kWdLL16Sum, own, tag, u.w); }
            if (c.clear) ll16_consume(src);
            acc[0] = bits_bf16(u.x & 0xFFFFu); acc[1] = bits_bf16(u.x >> 16);
            acc[2] = bits_bf16(u.y & 0xFFFFu); acc[3] = bits_bf16(u.y >> 16);
            acc[4] = bits_bf16(u.z & 0xFFFFu); acc[5] = bits_bf16(u.z >> 16);
            acc[6] = bits_bf16(u.w & 0xFFFFu);
        }
    }
    float q = 0.f;
    if (on)
#pragma unroll
        for (int e = 0; e < 7; e++) {
            const int i = cidx(e);
            if (i >= h) break;
            const float r = rv[e] + acc[e];
            rr[i] = r;
            acc[e] = r;
            q += r * r;
        }
    if (sdir) steer_vals7(sdir, ssc, smode, acc, cidx, on, h, rr, q);
    if (!norm_w) return;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if ((tid & 31) == 0) s_w[tid >> 5] = q;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int x = 0; x < kLL16Threads / 32; x++) tot += s_w[x];
    const float rs = rsqrtf(tot / h + eps);
    bf16* o = xn + (size_t)row * h;
    if (on)
#pragma unroll
        for (int e = 0; e < 7; e++) {
            const int i = cidx(e);
            if (i >= h) break;
            const float v = acc[e] * rs * (1.f + nwv[e]);
            if (xn) o[i] = __float2bfloat16(v);
            if (xh) store_xh(xh + (size_t)row * h, i, v);
        }
}


// MODE 3 (P2P, QE_AR_LOSSLESS): the reduce-scatter + all-gather of MODE 2 with a lossless packing of
// the bf16 values: a thread owns 9 consecutive values; when their exponents span <= 8 binades the
// word carries the max exponent (8 b) and per value sign + 3-b exponent delta + 7-b mantissa
// (107 b), otherwise two raw words (7 + 2 bf16). The tag shrinks to 15 bits, bit 16 of .w flags
// the raw form. Sums are staged in shared memory and the residual / norm tail runs in the 7-value
// thread mapping of allreduce_ll16_kernel, so the result is bit-identical to MODE 1 / MODE 2.
constexpr int kCE = 9;
__device__ __forceinline__ bool enc9(const uint32_t (&b)[kCE], uint32_t tagc, uint4& a, uint4& bw) {
    int mx = 0, mn = 255;
#pragma unroll
    for (int e = 0; e < kCE; e++) {
        const int ex = (b[e] >> 7) & 0xFF;
        mx = max(mx, ex);
        mn = min(mn, ex);
    }
    if (mx - mn > 7) {
        a = make_uint4(b[0] | (b[1] << 16), b[2] | (b[3] << 16), b[4] | (b[5] << 16), b[6] | tagc | 0x10000u);
        bw = make_uint4(b[7] | (b[8] << 16), 0u, 0u, tagc | 0x10000u);
        return false;
    }
    unsigned long long lo = (unsigned long long)mx, hi = 0ull;
#pragma unroll
    for (int e = 0; e < kCE; e++) {
        const int ex = (b[e] >> 7) & 0xFF;
        const unsigned long long v = ((b[e] >> 15) & 1u) | ((unsigned)(mx - ex) << 1) | ((b[e] & 0x7Fu) << 4);
        const int pos = 8 + 11 * e;
        if (pos + 11 <= 64) lo |= v << pos;
        else if (pos >= 64) hi |= v << (pos - 64);
        else {
            lo |= v << pos;
            hi |= v >> (64 - pos);
        }
    }
    a = make_uint4((uint32_t)lo, (uint32_t)(lo >> 32), (uint32_t)hi, ((uint32_t)(hi >> 32) & 0xFFFFu) | tagc);
    return true;
}
__device__ __forceinline__ void dec9(uint4 a, uint32_t (&b)[kCE]) {
    const unsigned long long lo = a.x | ((unsigned long long)a.y << 32);
    const unsigned long long hi = a.z | ((unsigned long long)(a.w & 0xFFFFu) << 32);
    const int mx = (int)(lo & 0xFF);
#pragma unroll
    for (int e = 0; e < kCE; e++) {
        const int pos = 8 + 11 * e;
        unsigned long long v;
        if (pos + 11 <= 64) v = lo >> pos;
        else if (pos >= 64) v = hi >> (pos - 64);
        else v = (lo >> pos) | (hi << (64 - pos));
        const uint32_t vv = (uint32_t)v & 0x7FFu;
        b[e] = ((vv & 1u) << 15) | ((uint32_t)(mx - (int)((vv >> 1) & 7u)) << 7) | ((vv >> 4) & 0x7Fu);
    }
}

__global__ void __launch_bounds__(kLL16Threads) allreduce_llc_kernel(
    float* __restrict__ resid, const float* __restrict__ partial, int h, Comm c, int rank,
    CommCounters ctr, int idx, const bf16* __restrict__ norm_w, float eps,
    bf16* __restrict__ xn, __half* __restrict__ xh, const SteerArg sa) {
    extern __shared__ float s_sum[];  // [h]
    __shared__ float s_w[kLL16Threads / 32];
    const int row = blockIdx.x, tid = threadIdx.x;
    float ssc = 0.f;
    int smode = 0;
    const float* sdir = steer_dir(sa, row, h, ssc, smode);  // as allreduce_ll16_kernel
    const unsigned ep = *ctr.epoch;
    const uint32_t tagc = ((ep * 1024u + idx) % 32767u + 1u) << 17;
    const int slot = ((ep & 1) << 1) | (idx & 1);
    const int hw9 = (h + kCE - 1) / kCE, per = (hw9 + c.ngpu - 1) / c.ngpu;
    const int hwp = pll_row_words(c.ll_hidden);
    auto pregion = [&](int owner, int sender) {
        return c.pll[owner] + (((size_t)slot * c.ngpu + sender) * kMaxBatchRows + row) * hwp;
    };
    auto poll = [&](const uint4* p, int peer) {
        uint4 u = ld_ll16(p);
        for (unsigned ws = 0; (u.w & 0xFFFE0000u) != tagc;) { u = ld_ll16(p); QE_WD_POLL(ws, kWdLLC, peer, tagc, u.w); }
        if (c.clear) ll16_consume(p);
        return u;
    };
    const float* pr = partial + (size_t)row * h;
    float* rr = resid + (size_t)row * h;
    const int hw = (h + 6) / 7, w = tid;
    const bool on = w < hw;
    auto cidx = [&](int e) { return c.coal ? w + hw * e : 7 * w + e; };  // as allreduce_ll16_kernel
    float rv[7], nwv[7];  // residual and norm weights, loaded before the polls (latency overlaps the wait)
#pragma unroll
    for (int e = 0; e < 7; e++) {
        const int i = cidx(e);
        const bool ok = on && i < h;
        rv[e] = ok ? rr[i] : 0.f;
        nwv[e] = ok && norm_w ? __bfloat162float(norm_w[i]) : 0.f;
    }
    const int t = tid;
    if (t < hw9) {
        const int own = t / per, i0 = kCE * t;
        uint32_t b[kCE];
#pragma unroll
        for (int e = 0; e < kCE; e++) b[e] = i0 + e < h ? bf16_bits(pr[i0 + e]) : 0u;
        uint32_t sb[kCE];
        if (own != rank) {
            uint4 wa, wb;
            uint4* dst = pregion(own, rank);
            if (!enc9(b, tagc, wa, wb)) st_ll16(dst + hw9 + t, wb);
            st_ll16(dst + t, wa);
            const uint4* src = pregion(rank, own);
            const uint4 u = poll(src + t, own);
            if (u.w & 0x10000u) {
                const uint4 u2 = poll(src + hw9 + t, own);
                sb[0] = u.x & 0xFFFFu; sb[1] = u.x >> 16; sb[2] = u.y & 0xFFFFu; sb[3] = u.y >> 16;
                sb[4] = u.z & 0xFFFFu; sb[5] = u.z >> 16; sb[6] = u.w & 0xFFFFu;
                sb[7] = u2.x & 0xFFFFu; sb[8] = u2.x >> 16;
            } else {
                dec9(u, sb);
            }
        } else {
            float acc[kCE] = {};
            for (int g = 0; g < c.ngpu; g++) {  // fixed order -> identical residual on every GPU
                uint32_t v[kCE];
                if (g == rank) {
#pragma unroll
                    for (int e = 0; e < kCE; e++) v[e] = b[e];
                } else {
                    const uint4* src = pregion(rank, g);
                    const uint4 u = poll(src + t, g);
                    if (u.w & 0x10000u) {
                        const uint4 u2 = poll(src + hw9 + t, g);
                        v[0] = u.x & 0xFFFFu; v[1] = u.x >> 16; v[2] = u.y & 0xFFFFu; v[3] = u.y >> 16;
                        v[4] = u.z & 0xFFFFu; v[5] = u.z >> 16; v[6] = u.w & 0xFFFFu;
                        v[7] = u2.x & 0xFFFFu; v[8] = u2.x >> 16;
                    } else {
                        dec9(u, v);
                    }
                }
#pragma unroll
                for (int e = 0; e < kCE; e++) acc[e] += bits_bf16(v[e]);
            }
#pragma unroll
            for (int e = 0; e < kCE; e++) sb[e] = bf16_bits(acc[e]);
            uint4 wa, wb;
            const bool packed = enc9(sb, tagc, wa, wb);
            for (int g = 0; g < c.ngpu; g++)
                if (g != rank) {
                    uint4* dst = pregion(g, rank);
                    if (!packed) st_ll16(dst + hw9 + t, wb);
                    st_ll16(dst + t, wa);
                }
        }
#pragma unroll
        for (int e = 0; e < kCE; e++)
            if (i0 + e < h) s_sum[i0 + e] = bits_bf16(sb[e]);
    }
    __syncthreads();
    // the tail of allreduce_ll16_kernel (7 values per thread, same order of the norm's sums)
    float acc[7];
    float q = 0.f;
    if (on)
#pragma unroll
        for (int e = 0; e < 7; e++) {
            const int i = cidx(e);
            if (i >= h) break;
            const float r = rv[e] + s_sum[i];
            rr[i] = r;
            acc[e] = r;
            q += r * r;
        }
    if (sdir) steer_vals7(sdir, ssc, smode, acc, cidx, on, h, rr, q);
    if (!norm_w) return;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if ((tid & 31) == 0) s_w[tid >> 5] = q;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int x = 0; x < kLL16Threads / 32; x++) tot += s_w[x];
    const float rs = rsqrtf(tot / h + eps);
    bf16* o = xn + (size_t)row * h;
    if (on)
#pragma unroll
        for (int e = 0; e < 7; e++) {
            const int i = cidx(e);
            if (i >= h) break;
            const float v = acc[e] * rs * (1.f + nwv[e]);
            if (xn) o[i] = __float2bfloat16(v);
            if (xh) store_xh(xh + (size_t)row * h, i, v);
        }
}

// Experimental compressed payload (QE_AR_Q = 1: reduce-scatter words in INT8, all-gather in bf16;
// QE_AR_Q = 2: both phases INT8). A 16-byte word carries 12 int8 values, their fp16 scale
// (amax / 127) and the tag; a thread owns 12 consecutive elements (427 words per row). Every GPU
// adds dequant(q(partial_g)) in fixed GPU order; the sum is rounded to bf16 (AGQ = 0) or quantized
// again (AGQ = 1) by the owner. RS = false: all-to-all of the quantized words with the identical
// arithmetic (the AG rounding is emulated locally), so both protocols give the same bits for every m.
constexpr int kQE = 12;
__device__ __forceinline__ uint4 q_word(const float (&v)[kQE], uint32_t tag, float (&dq)[kQE]) {
    float mx = 0.f;
#pragma unroll
    for (int e = 0; e < kQE; e++) mx = fmaxf(mx, fabsf(v[e]));
    const __half sh = __float2half_rn(mx / 127.f);
    const float sc = __half2float(sh);
    const float inv = sc > 0.f ? 1.f / sc : 0.f;
    uint32_t b[3] = {0, 0, 0};
#pragma unroll
    for (int e = 0; e < kQE; e++) {
        const int q = max(-127, min(127, __float2int_rn(v[e] * inv)));
        dq[e] = (float)q * sc;
        b[e >> 2] |= (uint32_t)(q & 0xFF) << (8 * (e & 3));
    }
    return make_uint4(b[0], b[1], b[2], (uint32_t)__half_as_ushort(sh) | tag);
}
__device__ __forceinline__ void dq_word(uint4 u, float (&dq)[kQE]) {
    const float sc = __half2float(__ushort_as_half((unsigned short)(u.w & 0xFFFFu)));
    const uint32_t b[3] = {u.x, u.y, u.z};
#pragma unroll
    for (int e = 0; e < kQE; e++) dq[e] = (float)(int8_t)((b[e >> 2] >> (8 * (e & 3))) & 0xFF) * sc;
}

template <bool RS, bool AGQ>
__global__ void __launch_bounds__(kLL16Threads) allreduce_llq_kernel(
    float* __restrict__ resid, const float* __restrict__ partial, int h, Comm c, int rank,
    CommCounters ctr, int idx, const bf16* __restrict__ norm_w, float eps,
    bf16* __restrict__ xn, __half* __restrict__ xh) {
    __shared__ float s_w[kLL16Threads / 32];
    const int row = blockIdx.x, tid = threadIdx.x;
    const unsigned ep = *ctr.epoch;
    const uint32_t tag = ((ep * 1024u + idx) % 65535u + 1u) << 16;
    const int slot = ((ep & 1) << 1) | (idx & 1);
    const int hw = (h + kQE - 1) / kQE, per = (hw + c.ngpu - 1) / c.ngpu;
    const int hwp = pll_row_words(c.ll_hidden);
    auto pregion = [&](int owner, int sender) {
        return c.pll[owner] + (((size_t)slot * c.ngpu + sender) * kMaxBatchRows + row) * hwp;
    };
    const float* pr = partial + (size_t)row * h;
    float* rr = resid + (size_t)row * h;
    const int w = tid, i0 = kQE * w;
    const bool on = w < hw;
    const int own = w / per;
    float acc[kQE] = {};
    float v[kQE], dq[kQE];
    uint4 mine = make_uint4(0, 0, 0, 0);
    if (on) {
#pragma unroll
        for (int e = 0; e < kQE; e++) v[e] = i0 + e < h ? pr[i0 + e] : 0.f;
        mine = q_word(v, tag, dq);
        if (RS) {
            if (own != rank) st_ll16(pregion(own, rank) + w, mine);
        } else {
            for (int g = 0; g < c.ngpu; g++)
                if (g != rank) st_ll16(pregion(g, rank) + w, mine);
        }
    }
    if (on && (!RS || own == rank)) {
        for (int g = 0; g < c.ngpu; g++) {
            float d[kQE];
            if (g == rank) {
#pragma unroll
                for (int e = 0; e < kQE; e++) d[e] = dq[e];
            } else {
                const uint4* src = pregion(rank, g) + w;
                uint4 u = ld_ll16(src);
                for (unsigned ws = 0; (u.w & 0xFFFF0000u) != tag;) { u = ld_ll16(src); QE_WD_POLL(ws, kWdLLQ, g, tag, u.w); }
                if (c.clear) ll16_consume(src);
                dq_word(u, d);
            }
#pragma unroll
            for (int e = 0; e < kQE; e++) acc[e] += d[e];
        }
        // the all-gather rounding (applied by the owner in RS, emulated by everyone in all-to-all)
        if (AGQ) {
            const uint4 sw = q_word(acc, tag, acc);
            if (RS)
                for (int g = 0; g < c.ngpu; g++)
                    if (g != rank) st_ll16(pregion(g, rank) + w, sw);
        } else {
            uint32_t b[kQE];
#pragma unroll
            for (int e = 0; e < kQE; e++) {
                b[e] = bf16_bits(acc[e]);
                acc[e] = bits_bf16(b[e]);
            }
            if (RS) {
                const uint4 wa = make_uint4(b[0] | (b[1] << 16), b[2] | (b[3] << 16), b[4] | (b[5] << 16), b[6] | tag);
                const uint4 wb = make_uint4(b[7] | (b[8] << 16), b[9] | (b[10] << 16), b[11], tag);
                for (int g = 0; g < c.ngpu; g++)
                    if (g != rank) {
                        st_ll16(pregion(g, rank) + w, wa);
                        st_ll16(pregion(g, rank) + hw + (w - rank * per), wb);
                    }
            }
        }
    } else if (on) {  // RS: wait for the owner's sum
        const uint4* src = pregion(rank, own) + w;
        uint4 u = ld_ll16(src);
        for (unsigned ws = 0; (u.w & 0xFFFF0000u) != tag;) { u = ld_ll16(src); QE_WD_POLL(ws, kWdLLQ, own, tag, u.w); }
        if (c.clear) ll16_consume(src);
        if (AGQ) {
            dq_word(u, acc);
        } else {
            const uint4* src2 = pregion(rank, own) + hw + (w - own * per);
            uint4 u2 = ld_ll16(src2);
            for (unsigned ws = 0; (u2.w & 0xFFFF0000u) != tag;) { u2 = ld_ll16(src2); QE_WD_POLL(ws, kWdLLQ, own, tag, u2.w); }
            if (c.clear) ll16_consume(src2);
            acc[0] = bits_bf16(u.x & 0xFFFFu); acc[1] = bits_bf16(u.x >> 16);
            acc[2] = bits_bf16(u.y & 0xFFFFu); acc[3] = bits_bf16(u.y >> 16);
            acc[4] = bits_bf16(u.z & 0xFFFFu); acc[5] = bits_bf16(u.z >> 16);
            acc[6] = bits_bf16(u.w & 0xFFFFu);
            acc[7] = bits_bf16(u2.x & 0xFFFFu); acc[8] = bits_bf16(u2.x >> 16);
            acc[9] = bits_bf16(u2.y & 0xFFFFu); acc[10] = bits_bf16(u2.y >> 16);
            acc[11] = bits_bf16(u2.z & 0xFFFFu);
        }
    }
    float q = 0.f;
    if (on)
#pragma unroll
        for (int e = 0; e < kQE; e++) {
            const int i = i0 + e;
            if (i >= h) break;
            const float r = rr[i] + acc[e];
            rr[i] = r;
            acc[e] = r;
            q += r * r;
        }
    if (!norm_w) return;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if ((tid & 31) == 0) s_w[tid >> 5] = q;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int x = 0; x < kLL16Threads / 32; x++) tot += s_w[x];
    const float rs = rsqrtf(tot / h + eps);
    bf16* o = xn + (size_t)row * h;
    if (on)
#pragma unroll
        for (int e = 0; e < kQE; e++) {
            const int i = i0 + e;
            if (i >= h) break;
            const float vv = acc[e] * rs * (1.f + __bfloat162float(norm_w[i]));
            if (xn) o[i] = __float2bfloat16(vv);
            if (xh) store_xh(xh + (size_t)row * h, i, vv);
        }
    if (ctr.ready) {  // early consumer: the last row block publishes the tag
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            if (atomicAdd(ctr.ready + 1024 + idx, 1u) == gridDim.x - 1) {
                ctr.ready[1024 + idx] = 0;
                __threadfence();
                st_release_gpu(ctr.ready + idx, ep * 1024u + idx + 1u);
            }
        }
    }
}

__global__ void allgather_kernel(const float* __restrict__ src, int n, float* __restrict__ out,
                                 Comm c, int rank, CommCounters ctr) {
    __shared__ unsigned s_seq;
    if (threadIdx.x == 0) s_seq = *ctr.seq + 1;
    __syncthreads();
    const unsigned s = s_seq;
    const int slot = s % kCommSlots;
    if (c.ngpu > 1) {
        float* mine = slot_ptr(c, rank, slot);
        for (int i = threadIdx.x; i < n; i += blockDim.x) mine[i] = src[i];
        exchange(c, rank, ctr, s, false, kWdAllgather);
    }
    for (int g = 0; g < c.ngpu; g++)
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            out[g * n + i] = g == rank ? src[i] : __ldcv(slot_ptr(c, g, slot) + i);
    bump_seq(ctr, s);
}

}  // namespace

Comm comm_create(int ngpu, int max_elems, int ll_hidden) {
    Comm c;
    c.ngpu = ngpu;
    c.max_elems = max_elems;
    c.ll_hidden = ll_hidden;
    c.clear = !(env_int("QE_LL_NOCLEAR", 0));
    c.coal = env_set("QE_AR_COAL") ? env_int("QE_AR_COAL", true) != 0 : true;
    const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
    const size_t llb = (size_t)ngpu * 4 * kMaxBatchRows * ((ll_hidden + 2) / 3) * sizeof(uint2);
    CUDA_CHECK(cudaHostAlloc(&c.ll, llb, fl));
    memset(c.ll, 0, llb);
    CUDA_CHECK(cudaHostAlloc(&c.data, (size_t)ngpu * kCommSlots * max_elems * sizeof(float), fl));
    CUDA_CHECK(cudaHostAlloc(&c.flags, (size_t)ngpu * kCommSlots * sizeof(unsigned), fl));
    memset(c.flags, 0, (size_t)ngpu * kCommSlots * sizeof(unsigned));
    return c;
}

bool comm_enable_p2p(Comm& c, const int* devices) {
    if (!env_int("QE_P2P", 1) || c.ngpu < 2 || c.ngpu > 4) return false;
    for (int i = 0; i < c.ngpu; i++)
        for (int j = 0; j < c.ngpu; j++) {
            int can = 0;
            if (i != j && (cudaDeviceCanAccessPeer(&can, devices[i], devices[j]) != cudaSuccess || !can))
                return false;
        }
    const size_t bytes = (size_t)4 * c.ngpu * kMaxBatchRows * pll_row_words(c.ll_hidden) * sizeof(uint4);
    for (int i = 0; i < c.ngpu; i++) {
        CUDA_CHECK(cudaSetDevice(devices[i]));
        for (int j = 0; j < c.ngpu; j++) {
            if (i == j) continue;
            const cudaError_t r = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (r != cudaSuccess && r != cudaErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr, "P2P: enabling peer access %d->%d failed (%s), host path\n",
                        devices[i], devices[j], cudaGetErrorString(r));
                cudaGetLastError();
                comm_free_p2p(c, devices);
                return false;
            }
            cudaGetLastError();
        }
        CUDA_CHECK(cudaMalloc(&c.pll[i], bytes));
        CUDA_CHECK(cudaMemset(c.pll[i], 0, bytes));
        c.rs_rows = c.max_elems / c.ll_hidden;
        c.rs_own = (c.rs_rows + c.ngpu - 1) / c.ngpu;
        CUDA_CHECK(cudaMalloc(&c.prs1[i], (size_t)c.ngpu * c.rs_own * c.ll_hidden * 2));
        CUDA_CHECK(cudaMalloc(&c.prs2[i], (size_t)c.rs_rows * c.ll_hidden * 2));
        CUDA_CHECK(cudaMalloc(&c.pflags[i], (size_t)c.ngpu * kCommSlots * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(c.pflags[i], 0, (size_t)c.ngpu * kCommSlots * sizeof(unsigned)));
        if (env_int("QE_AR_CE", 1)) {  // copy-engine prefill all-reduce
            CUDA_CHECK(cudaMalloc(&c.cfl[i], (size_t)2 * c.ngpu * kCeSlots * sizeof(unsigned)));
            CUDA_CHECK(cudaMemset(c.cfl[i], 0, (size_t)2 * c.ngpu * kCeSlots * sizeof(unsigned)));
            CUDA_CHECK(cudaMalloc(&c.pstage[i], (size_t)c.rs_rows * c.ll_hidden * 2));
        }
        if ((env_int("QE_AR_OWNFIN", 0)) ||
            ((env_int("QE_AR_CE", 1)) && env_int("QE_AR_CEOF", 0))) {
            // owner-finished prefill AR: receive rows (SM path off by default: slower; CE path QE_AR_CEOF=1)
            c.pxsc_row = (c.ll_hidden / 128) * 4 * 2 * 2;  // the largest record set: gemm_i8v G = 32 (>= x_to_i8 layout)
            CUDA_CHECK(cudaMalloc(&c.pxn[i], (size_t)c.rs_rows * c.ll_hidden * 2));
            CUDA_CHECK(cudaMalloc(&c.pxq[i], (size_t)c.rs_rows * c.ll_hidden));
            CUDA_CHECK(cudaMalloc(&c.pxsc[i], (size_t)(c.rs_rows + 32) * c.pxsc_row * 4));
        }
    }
    CUDA_CHECK(cudaSetDevice(devices[0]));
    for (int i = 0; i < c.ngpu; i++) c.dev[i] = devices[i];
    c.p2p = true;
    return true;
}

void comm_free_p2p(Comm& c, const int* devices) {
    for (int i = 0; i < c.ngpu; i++) {
        cudaSetDevice(devices[i]);
        for (void* q : {(void*)c.pll[i], (void*)c.prs1[i], (void*)c.prs2[i], (void*)c.pflags[i], (void*)c.pxn[i],
                        (void*)c.pxq[i], (void*)c.pxsc[i], (void*)c.cfl[i], (void*)c.pstage[i]})
            if (q) cudaFree(q);
        c.cfl[i] = nullptr;
        c.pstage[i] = nullptr;
        c.pxn[i] = nullptr;
        c.pxq[i] = nullptr;
        c.pxsc[i] = nullptr;
        c.pll[i] = nullptr;
        c.prs1[i] = c.prs2[i] = nullptr;
        c.pflags[i] = nullptr;
    }
    c.p2p = false;
}

void comm_begin_unit(const CommCounters& ctr, cudaStream_t st) {
    bump_epoch_kernel<<<1, 1, 0, st>>>(ctr.epoch);
}

static bool rs_rows_p2p(const Comm& c, int m, int h) {
    static const bool rs = env_set("QE_AR_RS_ROWS") ? env_int("QE_AR_RS_ROWS", true) != 0 : true;
    return rs && c.ngpu > 1 && m >= 3 * c.ngpu && m > kMaxGemvW4Tokens && c.p2p && c.prs1[0] != nullptr &&
           c.pflags[0] != nullptr && m <= c.rs_rows && h == c.ll_hidden;
}
bool allreduce_deferrable(const Comm& c, int m, int h, int rs_off) {
    static const bool on = env_int("QE_AR_DEFER", 1);
    return on && rs_rows_p2p(c, m, h) && rs_off + m <= c.rs_rows;
}
void allreduce_finish(float* resid, int m, int h, const Comm& c, int rank, const bf16* norm_w, float eps,
                      bf16* xn, int rs_off, cudaStream_t st, int8_t* xq, float* xsc, int G, int* ctr, bool wxn,
                      const SteerArg* sa) {
    const uint2* sums = c.prs2[rank] + (size_t)rs_off * (h >> 2);
    const SteerArg S = sa ? *sa : SteerArg();  // layer-output steering fused before the norm
    if (xq && G) CHECK((h >> 2) % 32 == 0, "fused quantization needs h %% 128 == 0 (%d)", h);
    if (!xq || !G) ar_finish_kernel<0><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, nullptr, nullptr, ctr, true, S);
    else if (gemm_i8_variant() >= 2) {
        if (G == 128) ar_finish_kernel<128, 2><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, xq, xsc, ctr, wxn, S);
        else if (G == 64) ar_finish_kernel<64, 2><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, xq, xsc, ctr, wxn, S);
        else ar_finish_kernel<32, 2><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, xq, xsc, ctr, wxn, S);
    } else if (gemm_i8_variant()) {
        if (G == 128) ar_finish_kernel<128><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, xq, xsc, ctr, wxn, S);
        else if (G == 64) ar_finish_kernel<64><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, xq, xsc, ctr, wxn, S);
        else ar_finish_kernel<32><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, xq, xsc, ctr, wxn, S);
    } else {
        if (G == 128) ar_finish_kernel<128, 0><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, xq, xsc, ctr, wxn, S);
        else if (G == 64) ar_finish_kernel<64, 0><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, xq, xsc, ctr, wxn, S);
        else ar_finish_kernel<32, 0><<<m, 512, 0, st>>>(resid, sums, h, norm_w, eps, xn, xq, xsc, ctr, wxn, S);
    }
}

static bool ar_rsum() {
    static const bool on = env_int("QE_AR_RSUM", 1);
    return on;
}
static int ar_rs_min() {  // min rows of the decode reduce-scatter + all-gather (MODE 2)
    static const int v = env_int("QE_AR_RS_MIN", 3);  // m >= 3: fewer bytes beat the extra hop (ar_bench)
    return v;
}

void allreduce_add(float* resid, const float* partial, int m, int h, const Comm& c, int rank,
                   const CommCounters& ctr, cudaStream_t st, const bf16* norm_w, float eps,
                   bf16* xn, int ll_idx, bool overlapped, __half* xh, bool defer, int rs_off, bool part_bf16,
                   const OwnFin* of, const SteerArg* sa) {
    if (qe_fault_skip(rank, "ar")) return;  // test only (QE_FAULT_SKIP_AR_RANK): this rank skips the collective
    // layer-output steering (steer.cuh): fused into the LL16 / LLC kernels below; after the others, the separate
    // steer_fix (a deferred all-reduce gets it in allreduce_finish instead)
    CHECK(!sa || sa->li < 0 || (!defer && !of), "steering: a deferred all-reduce is steered by allreduce_finish");
    const SteerArg S = sa && !defer ? *sa : SteerArg();
    bool fused = false;
    CHECK(!defer || allreduce_deferrable(c, m, h, rs_off), "allreduce defer %d", m);
    CHECK(!part_bf16 || (defer && m > kMaxGemvW4Tokens), "bf16 partials: deferred prefill rows only");
    CHECK(!of || (defer && allreduce_ownfin_ok(c, h) && norm_w), "owner-finished AR: deferred P2P rows only");
    CHECK(m * h <= c.max_elems && (h / 4) % kCommBlocks == 0, "allreduce %dx%d", m, h);
    CHECK(!xh || (m <= kMaxBatchRows && h % 64 == 0), "allreduce xh %d %d", m, h);
    if (ll_idx >= 0 && m <= kMaxBatchRows) {
        CHECK(ll_idx < 1023 && h == c.ll_hidden && h <= 3 * kLLThreads * kLLMaxWords &&
                  c.ngpu <= 4, "ll");
        static const int ll16 = env_int("QE_LL16", 1);  // min rows (0: off); one protocol for every row count keeps the norm bit-identical (MTP == greedy)
        // QE_AR_RSUM (default 1): the reduced sum is rounded to bf16 before the residual update (as
        // the prefill reduce-scatter does); this lets rows >= QE_AR_RS_MIN use the reduce-scatter +
        // all-gather protocol (MODE 2, P2P) with results bit-identical to the all-to-all (MODE 1).
        // QE_AR_RSUM=0: the old fp32 sum, all-to-all only.
        const int mode = !ar_rsum() ? 0 : (c.p2p && c.ngpu > 2 && m >= ar_rs_min()) ? 2 : 1;
        static const int arq = env_int("QE_AR_Q", 0);  // experimental, lossy
        if (arq && c.p2p && c.ngpu > 2 && h <= kQE * kLL16Threads) {
            const bool rs = m >= ar_rs_min();
            if (arq == 2) {
                if (rs) allreduce_llq_kernel<true, true><<<m, kLL16Threads, 0, st>>>(resid, partial, h, c, rank, ctr, ll_idx, norm_w, eps, xn, xh);
                else allreduce_llq_kernel<false, true><<<m, kLL16Threads, 0, st>>>(resid, partial, h, c, rank, ctr, ll_idx, norm_w, eps, xn, xh);
            } else {
                if (rs) allreduce_llq_kernel<true, false><<<m, kLL16Threads, 0, st>>>(resid, partial, h, c, rank, ctr, ll_idx, norm_w, eps, xn, xh);
                else allreduce_llq_kernel<false, false><<<m, kLL16Threads, 0, st>>>(resid, partial, h, c, rank, ctr, ll_idx, norm_w, eps, xn, xh);
            }
        } else if (ll16 > 0 && m >= ll16 && h <= 7 * kLL16Threads) {
            // QE_AR_LOSSLESS = min rows of the packed protocol (default 8: ar_bench m8 11,6 -> 10,5 us, m4 slower; 0: off)
            static const int lossless = env_int("QE_AR_LOSSLESS", 8);
            if (mode == 2 && lossless > 0 && m >= lossless)
                allreduce_llc_kernel<<<m, kLL16Threads, h * sizeof(float), st>>>(resid, partial, h, c, rank, ctr, ll_idx,
                                                                             norm_w, eps, xn, xh, S), fused = true;
            else if (mode == 2)
                allreduce_ll16_kernel<2><<<m, kLL16Threads, 0, st>>>(resid, partial, h, c, rank, ctr, ll_idx,
                                                                     norm_w, eps, xn, xh, S), fused = true;
            else if (mode == 1)
                allreduce_ll16_kernel<1><<<m, kLL16Threads, 0, st>>>(resid, partial, h, c, rank, ctr, ll_idx,
                                                                     norm_w, eps, xn, xh, S), fused = true;
            else
                allreduce_ll16_kernel<0><<<m, kLL16Threads, 0, st>>>(resid, partial, h, c, rank, ctr, ll_idx,
                                                                     norm_w, eps, xn, xh, S), fused = true;
        } else {
            allreduce_ll_kernel<<<m, kLLThreads, 0, st>>>(resid, partial, h, c, rank, ctr, ll_idx,
                                                          norm_w, eps, xn, xh, mode != 0);
        }
    } else if (m <= kMaxGemvW4Tokens)  // without LL (prefill, LL off): 33+ rows take the RS / overlapped path
        allreduce_kernel<<<kCommBlocks, 256, 0, st>>>(resid, partial, m, h, c, rank, ctr, norm_w,
                                                      eps, xn, xh, ar_rsum());
    else {
        const int ovb = env_int("QE_AR_OV_BLOCKS", 4);  // (read per call: in-process A/B)
        static const bool rs = env_set("QE_AR_RS_ROWS") ? env_int("QE_AR_RS_ROWS", true) != 0 : true;
        static const int allb = env_int("QE_AR_BLOCKS_ALL", 0);  // experiments
        const int nb = std::min(m, allb ? allb : overlapped ? ovb : 64);
        if (of) {
            auto go = [&](auto pbt, auto gt) {
                constexpr bool P = decltype(pbt)::value;
                constexpr int GG = decltype(gt)::value;
                allreduce_rs_rows_kernel<P, GG><<<nb, 512, 0, st>>>(resid, partial, m, h, c, rank, ctr, norm_w,
                                                                    eps, xn, rs_off, false, of->wxn, of->v2);
            };
            using T = std::true_type;
            using F = std::false_type;
            auto gsel = [&](auto pbt) {
                if (of->G == 0) go(pbt, std::integral_constant<int, 0>{});
                else if (of->G == 32) go(pbt, std::integral_constant<int, 32>{});
                else if (of->G == 64) go(pbt, std::integral_constant<int, 64>{});
                else if (of->G == 128) go(pbt, std::integral_constant<int, 128>{});
                else CHECK(false, "OwnFin G %d", of->G);
            };
            if (part_bf16) gsel(T{});
            else gsel(F{});
        } else if (rs && c.ngpu > 1 && m >= 3 * c.ngpu && part_bf16)
            allreduce_rs_rows_kernel<true><<<nb, 512, 0, st>>>(resid, partial, m, h, c, rank, ctr, norm_w,
                                                               eps, xn, rs_off, !defer, false, false, defer ? (env_int("QE_DBG_AR", 0) | (env_int("QE_AR_W16", 0) ? 16 : 0) | (env_int("QE_AR_PACE", 0) ? 32 : 0)) : 0);
        else if (rs && c.ngpu > 1 && m >= 3 * c.ngpu)
            allreduce_rs_rows_kernel<<<nb, 512, 0, st>>>(resid, partial, m, h, c, rank, ctr, norm_w,
                                                         eps, xn, rs_off, !defer, false, false, defer ? (env_int("QE_DBG_AR", 0) | (env_int("QE_AR_W16", 0) ? 16 : 0) | (env_int("QE_AR_PACE", 0) ? 32 : 0)) : 0);
        else
            allreduce_rows_kernel<<<nb, 512, 0, st>>>(resid, partial, m, h, c, rank, ctr, norm_w,
                                                      eps, xn);
    }
    if (!fused && S.li >= 0) steer_fix(resid, m, h, S, norm_w, eps, xn, xh, st);
}

// Flash-Next: R[i] += inj_i * allreduce(partial) for the 4 hyper-connection streams (rows >= 3 * ngpu, P2P);
// returns false when the shape needs the generic path (allreduce_add into y + the residual update kernel)
static void allreduce_hc_staged(float* R, const float* inj, const float* partial, int m, int h, const Comm& c, int rank,
                                cudaStream_t st, int blocks, bool own_rows);  // c.xfer != kXferP2P (below)
// Pull variant of allreduce_rs_rows_kernel<HCW> (prefill-r4): SM stores into a peer's memory (P2P over PCIe) stall the
// memory system of the sending card for every kernel next to them (W4A8 GEMM next to a 16-block store stream: 4.3x
// slower; next to SM loads of the same bytes from the peer: +12 %; tools/mb/mb_gdn.cu), so the rows move by loads where
// the probe allows it (c.pullm bit 4 i + j: rank i reads rank j's rows; else j stores them into i's memory, as before).
// Every card packs its rows (bf16) into its staging rows c.pstage (pulled pairs, local stores) or the owner's prs1
// (pushed pairs), raises the RS flags; the owner of a row range reads the peers' rows of its range and sums them in the
// fixed GPU order (the same values and order as the push path: bit-identical); the second exchange is the barrier that
// keeps the next call's writes off rows a peer may still read. Full mode (!own_rows): the owner writes its sums into
// its own prs2 rows (and stores them into pushed peers' prs2), every card reads the owners' sums after the second
// exchange (the next call's sums are written only after its first exchange, i.e. after every peer finished this tail).
__device__ __forceinline__ bool pull_pair(const Comm& c, int reader, int owner) { return (c.pullm >> (4 * reader + owner)) & 1u; }
__global__ void __launch_bounds__(512) allreduce_hc_pull_kernel(float* __restrict__ R, const float* __restrict__ partial,
                                                                 int m, int h, Comm c, int rank, CommCounters ctr,
                                                                 const float* __restrict__ inj, bool own_rows, bool flat) {
    __shared__ unsigned s_seq;
    if (threadIdx.x == 0) s_seq = *ctr.seq + 1;
    __syncthreads();
    const unsigned s1 = s_seq, s2 = s_seq + 1;
    const int h4 = h >> 2;
    const float4* p4 = reinterpret_cast<const float4*>(partial);
    float4* r4 = reinterpret_cast<float4*>(R);
    auto owner_begin = [&](int g) { return (int)((long)m * g / c.ngpu); };
    const int ob = owner_begin(rank), oe = owner_begin(rank + 1);
    uint2* const stage = c.pstage[rank];
    for (int r = blockIdx.x; r < m; r += gridDim.x) {
        if (r >= ob && r < oe) continue;
        int og = 0;
        while (og + 1 < c.ngpu && r >= owner_begin(og + 1)) og++;
        uint2* const dst = pull_pair(c, og, rank) ? stage + (size_t)r * h4
                                                  : c.prs1[og] + ((size_t)rank * c.rs_own + (r - owner_begin(og))) * h4;
        for (int i = threadIdx.x; i < h4; i += blockDim.x) dst[i] = pack_bf16x4(p4[(size_t)r * h4 + i]);
    }
    exchange(c, rank, ctr, s1, true, kWdArRowsExchange);
    uint2* const sums = c.prs2[rank];
    // own rows (sequence-parallel HC, perf F.3): (row, column) items over all threads of the grid, 2 items per thread
    // with every peer's load issued before the sums (one row per block left 128 of 512 threads busy in its second pass
    // and one load pair in flight per thread); the same values and summation order per element (bit-identical)
    static constexpr int kU = 2;
    if (flat && own_rows && c.ngpu <= 4 && (h & 3) == 0) {
        const long ne = (long)(oe - ob) * h4, stride = (long)gridDim.x * blockDim.x;
        for (long e0 = (long)blockIdx.x * blockDim.x + threadIdx.x; e0 < ne; e0 += stride * kU) {
            uint2 pv[kU][4];
#pragma unroll
            for (int u = 0; u < kU; u++) {
                const long e = e0 + u * stride;
                if (e >= ne) continue;
                const int r = ob + (int)(e / h4), i = (int)(e % h4);
                const size_t idx = (size_t)r * h4 + i;
#pragma unroll
                for (int g = 0; g < 4; g++)
                    if (g < c.ngpu)
                        pv[u][g] = g == rank ? pack_bf16x4(p4[idx])
                                             : __ldcv(pull_pair(c, rank, g) ? c.pstage[g] + idx
                                                                            : c.prs1[rank] + ((size_t)g * c.rs_own + (r - ob)) * h4 + i);
            }
#pragma unroll
            for (int u = 0; u < kU; u++) {
                const long e = e0 + u * stride;
                if (e >= ne) continue;
                const int r = ob + (int)(e / h4), i = (int)(e % h4);
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
                for (int g = 0; g < 4; g++)
                    if (g < c.ngpu) {
                        const float4 x = unpack_bf16x4(pv[u][g]);
                        v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
                    }
                const float4 vb = unpack_bf16x4(pack_bf16x4(v));
                float4* rr = r4 + (size_t)r * 4 * h4 + i;
#pragma unroll
                for (int s4 = 0; s4 < 4; s4++) {
                    const float jj = inj[r * 4 + s4];
                    float4 x = rr[s4 * h4];
                    x.x += jj * vb.x; x.y += jj * vb.y; x.z += jj * vb.z; x.w += jj * vb.w;
                    rr[s4 * h4] = x;
                }
            }
        }
    } else
    for (int r = ob + blockIdx.x; r < oe; r += gridDim.x) {
        const float jj[4] = {inj[r * 4], inj[r * 4 + 1], inj[r * 4 + 2], inj[r * 4 + 3]};
        for (int i = threadIdx.x; i < h4; i += blockDim.x) {
            const size_t idx = (size_t)r * h4 + i;
            uint2 pv[4];
#pragma unroll
            for (int g = 0; g < 4; g++)
                if (g < c.ngpu)
                    pv[g] = g == rank ? pack_bf16x4(p4[idx])
                                      : __ldcv(pull_pair(c, rank, g) ? c.pstage[g] + idx
                                                                     : c.prs1[rank] + ((size_t)g * c.rs_own + (r - ob)) * h4 + i);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int g = 0; g < 4; g++)
                if (g < c.ngpu) {
                    const float4 x = unpack_bf16x4(pv[g]);
                    v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
                }
            if (!own_rows) {
                const uint2 pk = pack_bf16x4(v);
                sums[idx] = pk;
                for (int g = 0; g < c.ngpu; g++)
                    if (g != rank && !pull_pair(c, g, rank)) c.prs2[g][idx] = pk;
                continue;
            }
            const float4 vb = unpack_bf16x4(pack_bf16x4(v));
            float4* rr = r4 + (size_t)r * 4 * h4 + i;
#pragma unroll
            for (int s4 = 0; s4 < 4; s4++) {
                float4 x = rr[s4 * h4];
                x.x += jj[s4] * vb.x; x.y += jj[s4] * vb.y; x.z += jj[s4] * vb.z; x.w += jj[s4] * vb.w;
                rr[s4 * h4] = x;
            }
        }
    }
    exchange(c, rank, ctr, s2, true, kWdArRowsExchange);
    if (!own_rows) {
        for (int r = blockIdx.x; r < m; r += gridDim.x) {
            int og = 0;
            while (og + 1 < c.ngpu && r >= owner_begin(og + 1)) og++;
            const uint2* const src = (og != rank && pull_pair(c, rank, og)) ? c.prs2[og] : c.prs2[rank];
            const float jj[4] = {inj[r * 4], inj[r * 4 + 1], inj[r * 4 + 2], inj[r * 4 + 3]};
            for (int i = threadIdx.x; i < h4; i += blockDim.x) {
                const size_t idx = (size_t)r * h4 + i;
                const float4 v = unpack_bf16x4(__ldcv(src + idx));
                float4* rr = r4 + (size_t)r * 4 * h4 + i;
#pragma unroll
                for (int s4 = 0; s4 < 4; s4++) {
                    float4 x = rr[s4 * h4];
                    x.x += jj[s4] * v.x; x.y += jj[s4] * v.y; x.z += jj[s4] * v.z; x.w += jj[s4] * v.w;
                    rr[s4 * h4] = x;
                }
            }
        }
    }
    bump_seq(ctr, s2);
}

bool allreduce_add_hc(float* R, const float* inj, const float* partial, int m, int h, const Comm& c, int rank,
                      const CommCounters& ctr, cudaStream_t st, int blocks, bool own_rows) {
    if (!allreduce_hc_ok(c, m, h)) return false;
    if (c.xfer != kXferP2P && c.xcnt[rank]) {
        allreduce_hc_staged(R, inj, partial, m, h, c, rank, st, blocks, own_rows);
        return true;
    }
    bool stg = c.pullm != 0 && c.ngpu <= 4;
    for (int g = 0; g < c.ngpu; g++) stg = stg && c.pstage[g] && c.prs2[g];
    if (stg) {
        static const bool flat = env_int("FLASH_AR_FLAT", 1);  // perf F.3 (0: one row per block in the pull phase)
        allreduce_hc_pull_kernel<<<std::min(m, blocks), 512, 0, st>>>(R, partial, m, h, c, rank, ctr, inj, own_rows, flat);
        return true;
    }
    const int nb = std::min(m, blocks);
    static const int w16 = env_int("FLASH_AR_W16", 0) ? 16 : 0;  // 16-byte P2P stores in phase 1 (and the sums)
    allreduce_rs_rows_kernel<false, -1, true><<<nb, 512, 0, st>>>(R, partial, m, h, c, rank, ctr, nullptr, 0.f, nullptr, 0,
                                                                  true, false, false, w16, inj, own_rows);
    return true;
}
bool allreduce_hc_ok(const Comm& c, int m, int h) {
    return c.p2p && c.ngpu > 1 && m >= 3 * c.ngpu && m > kMaxGemvW4Tokens && m <= c.rs_rows && h == c.ll_hidden &&
           c.prs1[0] != nullptr && c.pflags[0] != nullptr && m * h <= c.max_elems;
}

// ---------------------------------------------------------------- copy-engine prefill all-reduce
__global__ void __launch_bounds__(512) ce_pack_kernel(const float* __restrict__ partial, int m, int h4,
                                                      uint2* __restrict__ stage) {
    for (int r = blockIdx.x; r < m; r += gridDim.x)
        for (int i = threadIdx.x; i < h4; i += blockDim.x)
            stage[(size_t)r * h4 + i] = pack_bf16x4(reinterpret_cast<const float4*>(partial)[(size_t)r * h4 + i]);
}
__device__ __forceinline__ void ce_wait_flags(const unsigned* fl, int ngpu, int rank, unsigned seq) {
    for (int g = 0; g < ngpu; g++)
        if (g != rank)
            // 16-bit sequence numbers (the copy engine writes values of a ring v[i] = i)
            for (unsigned ws = 0; (short)(unsigned short)((*reinterpret_cast<const volatile unsigned*>(fl + g * kCeSlots + seq % kCeSlots) - seq) & 0xFFFFu) < 0;) {
                __nanosleep(64);
                QE_WD_POLL(ws, kWdCeFlags, g, seq, ld_acquire_sys(fl + g * kCeSlots + seq % kCeSlots));
            }
    __threadfence_system();  // acquire: the rows the flags cover are read after this
}
__global__ void __launch_bounds__(512) ce_sum_kernel(const float* __restrict__ partial, bool pb, int m, int h,
                                                     Comm c, int rank, int rs_off, unsigned seq) {
    const int h4 = h >> 2;  // (the RS rows of the peers landed: ce_wait(kind 0) before this kernel)
    auto owner_begin = [&](int g) { return (int)((long)m * g / c.ngpu); };
    const int ob = owner_begin(rank), oe = owner_begin(rank + 1);
    const float4* p4 = reinterpret_cast<const float4*>(partial);
    const uint2* pb4 = reinterpret_cast<const uint2*>(partial);
    for (int r = ob + blockIdx.x; r < oe; r += gridDim.x)
        for (int i = threadIdx.x; i < h4; i += blockDim.x) {
            const size_t idx = (size_t)r * h4 + i;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int g = 0; g < c.ngpu; g++) {  // fixed GPU order, own term rounded to bf16 (as the SM path)
                const float4 x = g == rank ? unpack_bf16x4(pb ? pb4[idx] : pack_bf16x4(p4[idx]))
                                           : unpack_bf16x4(__ldcv(c.prs1[rank] + ((size_t)g * c.rs_own + (r - ob)) * h4 + i));
                v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
            }
            c.prs2[rank][(size_t)rs_off * h4 + idx] = pack_bf16x4(v);
        }
}
// one warp spins (it fits next to a GEMM block on an SM: no SM is taken from the GEMM's grid)
__global__ void ce_wait_kernel(Comm c, int rank, unsigned seq, int kind) {
    if (threadIdx.x == 0) ce_wait_flags(c.cfl[rank] + kind * c.ngpu * kCeSlots, c.ngpu, rank, seq);
}
__global__ void ce_flag_kernel(unsigned* p, unsigned v) {
    __threadfence_system();
    st_release_sys(p, v);
}

// copy-engine all-reduce, owner-finished (CE + OwnFin): after the RS copies, the owner sums its rows,
// updates ITS residual rows, applies the norm and writes the int8 (G > 0) / bf16 (wxn or G == 0) rows
// into its own receive buffers; the AG copies then send exactly those rows to the peers (int8 is 1.1 B per
// element instead of the 2 B of the bf16 sums) and no GPU runs ar_finish over all rows.
template <bool PB, int GG>
__global__ void __launch_bounds__(512) ce_finish_kernel(float* __restrict__ resid, const float* __restrict__ partial,
                                                        int m, int h, Comm c, int rank, const bf16* __restrict__ norm_w,
                                                        float eps, int rs_off, bool wxn) {
    const int h4 = h >> 2;
    const float4* p4 = reinterpret_cast<const float4*>(partial);
    const uint2* pb4 = reinterpret_cast<const uint2*>(partial);
    auto pk = [&](size_t idx) { if constexpr (PB) return pb4[idx]; else return pack_bf16x4(p4[idx]); };
    float4* r4 = reinterpret_cast<float4*>(resid);
    auto owner_begin = [&](int g) { return (int)((long)m * g / c.ngpu); };
    const int ob = owner_begin(rank), oe = owner_begin(rank + 1);
        constexpr int G = GG;
        constexpr int L = G > 0 ? G / 4 : 1;
        constexpr int NREC = G > 0 ? (128 / G + 1) / 2 : 1;
        __shared__ float s_w2[16];
        for (int r = ob + blockIdx.x; r < oe; r += gridDim.x) {
            float4 keep[4];
            float q = 0.f;
#pragma unroll
            for (int it = 0; it < 4; it++) {
                const int i = threadIdx.x + it * blockDim.x;
                if (i < h4) {
                    const int idx = r * h4 + i;
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    for (int g = 0; g < c.ngpu; g++) {
                        const uint2* src = c.prs1[rank] + ((size_t)g * c.rs_own + (r - ob)) * h4 + i;
                        const float4 x = g == rank ? unpack_bf16x4(pk(idx)) : unpack_bf16x4(__ldcv(src));
                        v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
                    }
                    const float4 vb = unpack_bf16x4(pack_bf16x4(v));
                    float4 rr = r4[idx];
                    rr.x += vb.x; rr.y += vb.y; rr.z += vb.z; rr.w += vb.w;
                    r4[idx] = rr;
                    keep[it] = rr;
                    q += rr.x * rr.x + rr.y * rr.y + rr.z * rr.z + rr.w * rr.w;
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
            if ((threadIdx.x & 31) == 0) s_w2[threadIdx.x >> 5] = q;
            __syncthreads();
            float tot = 0.f;
            for (int w = 0; w < (int)(blockDim.x >> 5); w++) tot += s_w2[w];
            const float rs = rsqrtf(tot / h + eps);
            const size_t orow = (size_t)rs_off + r;
#pragma unroll
            for (int it = 0; it < 4; it++) {
                const int i = threadIdx.x + it * blockDim.x;
                if (i >= h4) continue;
                const float4 rr = keep[it];
                const int col = i * 4;
                const bf162 o0 = __floats2bfloat162_rn(rr.x * rs * (1.f + __bfloat162float(norm_w[col])),
                                                       rr.y * rs * (1.f + __bfloat162float(norm_w[col + 1])));
                const bf162 o1 = __floats2bfloat162_rn(rr.z * rs * (1.f + __bfloat162float(norm_w[col + 2])),
                                                       rr.w * rs * (1.f + __bfloat162float(norm_w[col + 3])));
                if (G == 0 || wxn) {
                    uint2 w2;
                    w2.x = *reinterpret_cast<const uint32_t*>(&o0);
                    w2.y = *reinterpret_cast<const uint32_t*>(&o1);
                    c.pxn[rank][orow * h4 + i] = w2;
                }
                if constexpr (G > 0) {
                    const float2 f0 = __bfloat1622float2(o0), f1 = __bfloat1622float2(o1);
                    float mx = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
#pragma unroll
                    for (int d = 1; d < L; d <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
                    const float sc = mx > 0.f ? __uint_as_float(__float_as_uint(mx / 127.f) & ~3u) : 1.f;
                    const float inv = mx > 0.f ? __frcp_rn(sc) : 0.f;
                    const int q0 = __float2int_rn(f0.x * inv), q1 = __float2int_rn(f0.y * inv);
                    const int q2 = __float2int_rn(f1.x * inv), q3 = __float2int_rn(f1.y * inv);
                    int sum = q0 + q1 + q2 + q3;
#pragma unroll
                    for (int d = 1; d < L; d <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
                    if ((i % L) == 0) {
                        const int sub = (col % 128) / G;
                        const float2 rec = make_float2(sc, __int_as_float((int)0x4B400000 - 8 * sum));
                        const size_t recf = (size_t)(h / 128) * NREC * 4;  // x_to_i8 record floats per row
                        const size_t off = orow * recf + ((size_t)(col / 128) * NREC + sub / 2) * 4 + (sub & 1) * 2;
                        *reinterpret_cast<float2*>(c.pxsc[rank] + off) = rec;
                    }
                    const uint32_t a = (uint32_t)(q0 & 0xFF) | ((uint32_t)(q1 & 0xFF) << 8) | ((uint32_t)(q2 & 0xFF) << 16) |
                                       ((uint32_t)(q3 & 0xFF) << 24);
                    const uint32_t b = __shfl_xor_sync(0xffffffffu, a, 2);
                    const int ip = i & 3;
                    uint32_t out;
                    if ((ip & 2) == 0) out = (a & 0xFF) | ((b & 0xFF) << 8) | (((a >> 8) & 0xFF) << 16) | (((b >> 8) & 0xFF) << 24);
                    else out = ((b >> 16) & 0xFF) | (((a >> 16) & 0xFF) << 8) | (((b >> 24) & 0xFF) << 16) | ((a >> 24) << 24);
                    const int j = 2 * (ip & 1) + (ip >> 1);
                    reinterpret_cast<uint32_t*>(c.pxq[rank] + orow * h + (col & ~15))[j] = out;
                }
            }
            __syncthreads();  // s_w2 of the next row
        }
}

bool ce_ar_ok(const Comm& c, int m, int h, int rs_off) {
    return c.p2p && c.cfl[0] && c.pstage[0] && h == c.ll_hidden && m <= c.rs_rows && rs_off + m <= c.rs_rows &&
           m >= 3 * c.ngpu && m > kMaxGemvW4Tokens;
}
void ce_pack(const float* partial, int m, int h, const Comm& c, int rank, int rs_off, cudaStream_t st, int blocks) {
    ce_pack_kernel<<<blocks > 0 ? std::min(m, blocks) : m, 512, 0, st>>>(
        partial, m, h >> 2, c.pstage[rank] + (size_t)rs_off * (h >> 2));
}
void ce_sum(const float* partial, bool part_bf16, int m, int h, const Comm& c, int rank, int rs_off,
            unsigned seq, cudaStream_t st) {
    ce_sum_kernel<<<std::max(1, std::min(m / c.ngpu, env_int("QE_CE_SUM_BLOCKS", 64))), 512, 0, st>>>(
        partial, part_bf16, m, h, c, rank, rs_off, seq);
}
void ce_wait(const Comm& c, int rank, unsigned seq, cudaStream_t st, int kind) {
    if (env_int("QE_CE_NOWAIT", 0)) return;  // (debug: no device-side wait)
    ce_wait_kernel<<<1, 32, 0, st>>>(c, rank, seq, kind);
}
void ce_flag(unsigned* p, unsigned v, cudaStream_t st) { ce_flag_kernel<<<1, 1, 0, st>>>(p, v); }
CeCopy ce_rs_copy(const Comm& c, int rank, int peer, int m, int h, int rs_off, const void* src_rows, unsigned seq) {
    (void)rs_off;
    auto owner_begin = [&](int g) { return (int)((long)m * g / c.ngpu); };
    const int ob = owner_begin(peer), oe = owner_begin(peer + 1);
    const size_t row = (size_t)h * 2;
    CeCopy x;
    x.dst = c.prs1[peer] + (size_t)rank * c.rs_own * (h >> 2);
    x.src = static_cast<const char*>(src_rows) + (size_t)ob * row;
    x.bytes = (size_t)(oe - ob) * row;
    x.flag = c.cfl[peer] + rank * kCeSlots + seq % kCeSlots;
    x.dev = c.dev[peer];
    return x;
}
void ce_finish(float* resid, const float* partial, bool pb, int m, int h, const Comm& c, int rank, const bf16* norm_w,
               float eps, int rs_off, int G, bool wxn, cudaStream_t st) {
    const int nb = std::max(1, std::min(m / c.ngpu, env_int("QE_CE_SUM_BLOCKS", 64)));
    auto go = [&](auto pbt, auto gt) {
        ce_finish_kernel<decltype(pbt)::value, decltype(gt)::value><<<nb, 512, 0, st>>>(resid, partial, m, h, c, rank,
                                                                                       norm_w, eps, rs_off, wxn);
    };
    auto gsel = [&](auto pbt) {
        if (G == 0) go(pbt, std::integral_constant<int, 0>{});
        else if (G == 32) go(pbt, std::integral_constant<int, 32>{});
        else if (G == 64) go(pbt, std::integral_constant<int, 64>{});
        else if (G == 128) go(pbt, std::integral_constant<int, 128>{});
        else CHECK(false, "ce_finish G %d", G);
    };
    if (pb) gsel(std::true_type{});
    else gsel(std::false_type{});
}
int ce_of_copies(const Comm& c, int rank, int peer, int m, int h, int rs_off, int G, bool wxn, CeCopy* out) {
    auto owner_begin = [&](int g) { return (int)((long)m * g / c.ngpu); };
    const int ob = owner_begin(rank), oe = owner_begin(rank + 1), rows = oe - ob;
    const size_t r0 = (size_t)rs_off + ob;
    int n = 0;
    if (G > 0) {
        const size_t recf = (size_t)(h / 128) * ((128 / G + 1) / 2) * 4;
        out[n++] = CeCopy{c.pxq[peer] + r0 * h, c.pxq[rank] + r0 * h, (size_t)rows * h, nullptr, c.dev[peer]};
        out[n++] = CeCopy{c.pxsc[peer] + r0 * recf, c.pxsc[rank] + r0 * recf, (size_t)rows * recf * 4, nullptr, c.dev[peer]};
    }
    if (G == 0 || wxn)
        out[n++] = CeCopy{c.pxn[peer] + r0 * (h >> 2), c.pxn[rank] + r0 * (h >> 2), (size_t)rows * h * 2, nullptr, c.dev[peer]};
    return n;
}
CeCopy ce_ag_copy(const Comm& c, int rank, int peer, int m, int h, int rs_off, unsigned seq) {
    auto owner_begin = [&](int g) { return (int)((long)m * g / c.ngpu); };
    const int ob = owner_begin(rank), oe = owner_begin(rank + 1);
    const size_t off = ((size_t)rs_off + ob) * (h >> 2);
    CeCopy x;
    x.dst = c.prs2[peer] + off;
    x.src = c.prs2[rank] + off;
    x.bytes = (size_t)(oe - ob) * h * 2;
    x.flag = c.cfl[peer] + (c.ngpu + rank) * kCeSlots + seq % kCeSlots;
    x.dev = c.dev[peer];
    return x;
}

// ---- staged HC all-reduce (c.xfer kXferCE / kXferHost): the rows move by copies, only local work runs on SMs.
// Same arithmetic as allreduce_rs_rows_kernel<HCW> (owner sums in the fixed GPU order, own term rounded to bf16 as the
// packed rows, bf16 sums), so the results are bit-identical to the SM-store path.
//   pack (partial -> bf16 rows; host: -> own pinned rows), flag RS | wait RS, pull the owner rows of every peer into the
//   local prs1 (CE: from the peers' staging rows, host: from their pinned rows) | sum (own rows: R += inj * sum, or bf16
//   sums -> prs2; host: -> own pinned rows), flag AG | wait AG (full: pull the owners' sum rows into the local prs2) |
//   tail (full): R += inj * sums of every row. The AG flags are also the barrier that keeps the next call off rows a
//   peer is still copying (own-rows mode: the flag only).
namespace {
__global__ void xs_seqw_kernel(unsigned* w, const unsigned* cnt) { *w = *cnt + 1; }  // CE: the call's flag value
__global__ void xs_flag_kernel(Comm c, int rank, int kind) {
    if (threadIdx.x != 0) return;
    const unsigned s = *c.xcnt[rank] + 1;
    __threadfence_system();
    for (int g = 0; g < c.ngpu; g++)
        if (g != rank) st_release_sys(c.xfl[g] + kind * c.ngpu + rank, s);
}
__global__ void xs_wait_kernel(Comm c, int rank, int kind, bool bump) {
    if (threadIdx.x != 0) return;
    const unsigned s = *c.xcnt[rank] + 1;
    for (int g = 0; g < c.ngpu; g++)
        if (g != rank) {
            const unsigned* f = c.xfl[rank] + kind * c.ngpu + g;
            for (unsigned ws = 0; (int)(ld_acquire_sys(f) - s) < 0;) {
                __nanosleep(64);
                QE_WD_POLL(ws, kWdCeFlags, g, s, ld_acquire_sys(f));
            }
        }
    __threadfence_system();
    if (bump) *c.xcnt[rank] = s;
}
// owner rows: sum of every GPU's bf16 rows (own term from the fp32 partial); own_rows: R += inj * bf16(sum), else the
// bf16 sums into the local prs2 rows
__global__ void __launch_bounds__(512) xs_sum_kernel(float* __restrict__ R, const float* __restrict__ inj,
                                                     const float* __restrict__ partial, int m, int h, Comm c, int rank,
                                                     bool own_rows) {
    const int h4 = h >> 2;
    auto owner_begin = [&](int g) { return (int)((long)m * g / c.ngpu); };
    const int ob = owner_begin(rank), oe = owner_begin(rank + 1);
    const float4* p4 = reinterpret_cast<const float4*>(partial);
    float4* r4 = reinterpret_cast<float4*>(R);
    for (int r = ob + blockIdx.x; r < oe; r += gridDim.x) {
        const float j0 = inj[r * 4], j1 = inj[r * 4 + 1], j2 = inj[r * 4 + 2], j3 = inj[r * 4 + 3];
        const float jj[4] = {j0, j1, j2, j3};
        for (int i = threadIdx.x; i < h4; i += blockDim.x) {
            const size_t idx = (size_t)r * h4 + i;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int g = 0; g < c.ngpu; g++) {
                const float4 x = g == rank ? unpack_bf16x4(pack_bf16x4(p4[idx]))
                                           : unpack_bf16x4(__ldcv(c.prs1[rank] + ((size_t)g * c.rs_own + (r - ob)) * h4 + i));
                v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
            }
            if (!own_rows) {
                c.prs2[rank][idx] = pack_bf16x4(v);
                continue;
            }
            const float4 vb = unpack_bf16x4(pack_bf16x4(v));
            float4* rr = r4 + (size_t)r * 4 * h4 + i;
#pragma unroll
            for (int s4 = 0; s4 < 4; s4++) {
                float4 x = rr[s4 * h4];
                x.x += jj[s4] * vb.x; x.y += jj[s4] * vb.y; x.z += jj[s4] * vb.z; x.w += jj[s4] * vb.w;
                rr[s4 * h4] = x;
            }
        }
    }
}
__global__ void __launch_bounds__(512) xs_tail_kernel(float* __restrict__ R, const float* __restrict__ inj, int m, int h,
                                                      const uint2* __restrict__ sums) {
    const int h4 = h >> 2;
    float4* r4 = reinterpret_cast<float4*>(R);
    for (int r = blockIdx.x; r < m; r += gridDim.x) {
        const float j0 = inj[r * 4], j1 = inj[r * 4 + 1], j2 = inj[r * 4 + 2], j3 = inj[r * 4 + 3];
        const float jj[4] = {j0, j1, j2, j3};
        for (int i = threadIdx.x; i < h4; i += blockDim.x) {
            const float4 v = unpack_bf16x4(__ldcv(sums + (size_t)r * h4 + i));
            float4* rr = r4 + (size_t)r * 4 * h4 + i;
#pragma unroll
            for (int s4 = 0; s4 < 4; s4++) {
                float4 x = rr[s4 * h4];
                x.x += jj[s4] * v.x; x.y += jj[s4] * v.y; x.z += jj[s4] * v.z; x.w += jj[s4] * v.w;
                rr[s4 * h4] = x;
            }
        }
    }
}
}  // namespace

__global__ void k_link_emul(unsigned long long* busy, unsigned long long bytes, float emu, float real) {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    const unsigned long long d = (unsigned long long)((double)bytes * emu), r = (unsigned long long)((double)bytes * real);
    const unsigned long long s0 = now > r ? now - r : 0;  // the real copy started about r ns ago
    unsigned long long old = *(volatile unsigned long long*)busy, end;
    for (;;) {  // FIFO on the card's egress link: start after the previous transfer of any stream
        end = (old > s0 ? old : s0) + d;
        const unsigned long long seen = atomicCAS(busy, old, end);
        if (seen == old) break;
        old = seen;
    }
    for (;;) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (now >= end) break;
        __nanosleep(end - now > 20000 ? 10000 : 500);
    }
}
void comm_link_emul(const Comm& c, int rank, size_t bytes, cudaStream_t st, int kind) {
    // FLASH_TEST_LINK_SKIP (bits, attribution test): kinds whose transfers are not slowed (1 HC all-reduce, 2 row
    // all-gather, 4 QSA q, 8 QSA partials, 16 indexer keys)
    static const int skip = env_int("FLASH_TEST_LINK_SKIP", 0);
    if (c.lemu <= 0.f || !c.lbusy[rank] || !bytes || ((skip >> kind) & 1)) return;
    k_link_emul<<<1, 1, 0, st>>>(c.lbusy[rank], bytes, c.lemu, c.lreal);
}

static void allreduce_hc_staged(float* R, const float* inj, const float* partial, int m, int h, const Comm& c, int rank,
                                cudaStream_t st, int blocks, bool own_rows) {
    // Host staging: every card PULLS the rows it needs (owner: the peers' partial rows of its range; full mode: the
    // owners' sums) from the peers' pinned rows after their flags (SM stores into host-mapped words).
    // Copy engine: every card PUSHES its rows into the peers' receive rows by peer copies and then its flag word by a
    // copy of the same stream (xs_ce_flags): the flag lands after the rows it covers (an SM flag store after a pushed
    // copy had no ordering with it; the receiver-pull variant of c0e8ac3 hung the first prefill batch).
    const int h4 = h >> 2, ng = c.ngpu;
    const size_t row = (size_t)h * 2;
    auto ob = [&](int g) { return (int)((long)m * g / ng); };
    const int o0 = ob(rank), o1 = ob(rank + 1);
    const bool host = c.xfer == kXferHost;
    uint2* stage = c.pstage[rank];
    ce_pack_kernel<<<std::max(1, std::min(m, blocks)), 512, 0, st>>>(partial, m, h4, stage);
    auto ce_flags = [&](int kind) {  // the call's sequence number -> every peer's flag word [kind][rank]
        for (int g = 0; g < ng; g++)
            if (g != rank)
                CUDA_CHECK(cudaMemcpyAsync(c.xfl[g] + kind * ng + rank, c.xsw[rank], 4, cudaMemcpyDeviceToDevice, st));
    };
    if (host) {
        CUDA_CHECK(cudaMemcpyAsync(c.hrs[rank], stage, (size_t)m * row, cudaMemcpyDeviceToHost, st));
        xs_flag_kernel<<<1, 32, 0, st>>>(c, rank, 0);
    } else {
        for (int g = 0; g < ng; g++)
            if (g != rank && ob(g + 1) > ob(g))
                CUDA_CHECK(cudaMemcpyAsync(c.prs1[g] + (size_t)rank * c.rs_own * h4, stage + (size_t)ob(g) * h4,
                                           (size_t)(ob(g + 1) - ob(g)) * row, cudaMemcpyDeviceToDevice, st));
        comm_link_emul(c, rank, (size_t)(m - (o1 - o0)) * row, st, 0);
        xs_seqw_kernel<<<1, 1, 0, st>>>(c.xsw[rank], c.xcnt[rank]);
        ce_flags(0);
    }
    xs_wait_kernel<<<1, 32, 0, st>>>(c, rank, 0, false);
    if (host && o1 > o0)
        for (int g = 0; g < ng; g++)
            if (g != rank)
                CUDA_CHECK(cudaMemcpyAsync(c.prs1[rank] + (size_t)g * c.rs_own * h4, c.hrs[g] + (size_t)o0 * h4,
                                           (size_t)(o1 - o0) * row, cudaMemcpyHostToDevice, st));
    if (o1 > o0) xs_sum_kernel<<<std::max(1, std::min(o1 - o0, blocks)), 512, 0, st>>>(R, inj, partial, m, h, c, rank, own_rows);
    // AG flag: the sums are ready (full mode) and this card is done with its receive rows (both modes: the barrier that
    // keeps the next call's rows off receive rows a peer is still reading / copying)
    if (host) {
        if (!own_rows && o1 > o0)
            CUDA_CHECK(cudaMemcpyAsync(c.hag[rank] + (size_t)o0 * h4, c.prs2[rank] + (size_t)o0 * h4, (size_t)(o1 - o0) * row,
                                       cudaMemcpyDeviceToHost, st));
        xs_flag_kernel<<<1, 32, 0, st>>>(c, rank, 1);
    } else {
        if (!own_rows && o1 > o0)
            for (int g = 0; g < ng; g++)
                if (g != rank)
                    CUDA_CHECK(cudaMemcpyAsync(c.prs2[g] + (size_t)o0 * h4, c.prs2[rank] + (size_t)o0 * h4, (size_t)(o1 - o0) * row,
                                               cudaMemcpyDeviceToDevice, st));
        if (!own_rows) comm_link_emul(c, rank, (size_t)(o1 - o0) * row * (ng - 1), st, 0);
        ce_flags(1);
    }
    xs_wait_kernel<<<1, 32, 0, st>>>(c, rank, 1, true);
    if (own_rows) return;
    if (host)
        for (int g = 0; g < ng; g++)
            if (g != rank && ob(g + 1) > ob(g))
                CUDA_CHECK(cudaMemcpyAsync(c.prs2[rank] + (size_t)ob(g) * h4, c.hag[g] + (size_t)ob(g) * h4,
                                           (size_t)(ob(g + 1) - ob(g)) * row, cudaMemcpyHostToDevice, st));
    xs_tail_kernel<<<std::max(1, std::min(m, blocks)), 512, 0, st>>>(R, inj, m, h, c.prs2[rank]);
}

bool allreduce_ownfin_ok(const Comm& c, int h) {
    return c.p2p && c.pxq[0] && h == c.ll_hidden && (h >> 2) <= 4 * 512 && (h >> 2) % 32 == 0 && h % 128 == 0;
}

__global__ void zero_int_kernel(int* p) { *p = 0; }
void zero_int(int* p, cudaStream_t st) { zero_int_kernel<<<1, 1, 0, st>>>(p); }

void allgather_small(const float* src, int n, float* out, const Comm& c, int rank,
                     const CommCounters& ctr, cudaStream_t st) {
    CHECK(n <= 8192 && n <= c.max_elems, "allgather too large");
    if (qe_fault_skip(rank, "ar")) return;  // test only (QE_FAULT_SKIP_AR_RANK)
    allgather_kernel<<<1, 256, 0, st>>>(src, n, out, c, rank, ctr);
}

#include "comm_probe.cuh"
