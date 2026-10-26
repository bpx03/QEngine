#pragma once
// Activation steering of the 27B engine (QE_STEER / QE_STEER_DIR, src/steer.h): at the output of layer L (the
// residual after the layer's MLP all-reduce, before the next RMSNorm; llama.cpp's "l_out"), per row:
//   add:     h += scale w           (w = alpha v, the sidecar strength folded in)
//   project: h -= scale (h . w) w   (w = sqrt(alpha) unit v)
// The loaded vectors share one buffer w[vec][mode][layer][H]; every row picks {vector, mode, lo, hi} and the scale
// of its sequence slot (sl[slot * 4], vector < 0: off). These live at fixed device addresses, so toggling a
// request's steering changes no graph. The steering is fused into the all-reduce kernels that finish the layer
// output (the LL16 / LLC decode all-reduces and the deferred prefill tail ar_finish_kernel) and runs before their
// fused norm; other all-reduce variants take the separate k_steer_fix after them. An unsteered row takes the
// unchanged code path (bit-identical); li < 0 (no vectors loaded, or another layer) skips everything.
#include "../common.h"

struct SteerArg {
    const float* w = nullptr;   // [vectors][2 modes][layers][h]
    const int* sl = nullptr;    // per slot {vector (-1 off), mode, lo, hi} (16-byte aligned)
    const float* sc = nullptr;  // per slot scale
    int li = -1;                // layer whose output is steered (-1: none)
    int layers = 0;
    int slot = 0;               // the slot of every row (nrs == 0)
    int nrs = 0;                // > 0: row r < nrs belongs to slot rs[r] (batched steps)
    unsigned char rs[kMaxBatchRows] = {};
};

#ifdef __CUDACC__
// the row's direction (nullptr: not steered) with its scale and mode; block-uniform
__device__ __forceinline__ const float* steer_dir(const SteerArg& a, int row, int h, float& sc, int& mode) {
    if (a.li < 0) return nullptr;
    const int slot = a.nrs ? a.rs[row] : a.slot;
    const int4 q = *reinterpret_cast<const int4*>(a.sl + slot * 4);
    if (q.x < 0 || a.li < q.z || a.li > q.w) return nullptr;
    sc = a.sc[slot];
    mode = q.y;
    return a.w + (((size_t)q.x * 2 + mode) * a.layers + a.li) * h;
}

// sum of v over the block (NT threads, fixed order); every thread gets it. Called block-uniformly.
template <int NT>
__device__ __forceinline__ float steer_block_sum(float v) {
    __shared__ float red[NT / 32];
    __shared__ float tot;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < NT / 32; i++) s += red[i];
        tot = s;
    }
    __syncthreads();
    return tot;
}
#endif

// Separate steering of rows [0, m) (one block of 512 threads per row) after an all-reduce without the fused hook:
// the steered rows' residual updated and their norm outputs (xn bf16 and / or the XH copy, when given) rewritten
// with the fused norm's formula. Unsteered rows exit before touching memory.
void steer_fix(float* resid, int m, int h, const SteerArg& a, const bf16* norm_w, float eps, bf16* xn, __half* xh,
               cudaStream_t st);
// one slot's {vector, mode, lo, hi} and scale, written in stream order (a kernel: no host staging buffer to race)
void steer_set_slot(int* sl, float* sc, int vec, int mode, int lo, int hi, float scale, cudaStream_t st);
// activation capture: acc[d] += sum over rows [skip, m) of resid[row][d]
void steer_capture(const float* resid, int m, int skip, int h, float* acc, cudaStream_t st);
