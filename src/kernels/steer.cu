// Activation steering: the separate kernel after all-reduce variants without the fused hook, and the activation
// capture of steering/make_vector.py (qe_capture). See steer.cuh.
#include "steer.cuh"
#include "gemv_h16.cuh"

namespace {

constexpr int kFixT = 512;

__global__ void __launch_bounds__(kFixT) steer_fix_kernel(float* __restrict__ resid, int h, SteerArg a,
                                                          const bf16* __restrict__ norm_w, float eps,
                                                          bf16* __restrict__ xn, __half* __restrict__ xh) {
    const int row = blockIdx.x, h4 = h >> 2;
    float sc = 0.f;
    int mode = 0;
    const float* w = steer_dir(a, row, h, sc, mode);
    if (!w) return;
    float4* r4 = reinterpret_cast<float4*>(resid + (size_t)row * h);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float c = sc;
    if (mode == 1) {
        float d = 0.f;
        for (int i = threadIdx.x; i < h4; i += kFixT) {
            const float4 r = r4[i], v = w4[i];
            d += r.x * v.x + r.y * v.y + r.z * v.z + r.w * v.w;
        }
        c = -sc * steer_block_sum<kFixT>(d);
    }
    float q = 0.f;
    for (int i = threadIdx.x; i < h4; i += kFixT) {
        float4 r = r4[i];
        const float4 v = w4[i];
        r.x += c * v.x; r.y += c * v.y; r.z += c * v.z; r.w += c * v.w;
        r4[i] = r;
        q += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
    }
    if (!norm_w || (!xn && !xh)) return;
    const float rs = rsqrtf(steer_block_sum<kFixT>(q) / h + eps);
    for (int i = threadIdx.x; i < h4; i += kFixT) {
        const float4 r = r4[i];
        const int col = i * 4;
        const float v0 = r.x * rs * (1.f + __bfloat162float(norm_w[col]));
        const float v1 = r.y * rs * (1.f + __bfloat162float(norm_w[col + 1]));
        const float v2 = r.z * rs * (1.f + __bfloat162float(norm_w[col + 2]));
        const float v3 = r.w * rs * (1.f + __bfloat162float(norm_w[col + 3]));
        if (xn) {
            bf162* o = reinterpret_cast<bf162*>(xn + (size_t)row * h + col);
            o[0] = __floats2bfloat162_rn(v0, v1);
            o[1] = __floats2bfloat162_rn(v2, v3);
        }
        if (xh) store_xh4(xh + (size_t)row * h, col, v0, v1, v2, v3);
    }
}

__global__ void steer_capture_kernel(const float* __restrict__ resid, int m, int skip, int h, float* __restrict__ acc) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= h) return;
    float s = 0.f;
    for (int t = skip; t < m; t++) s += resid[(size_t)t * h + d];
    acc[d] += s;
}

__global__ void steer_set_slot_kernel(int* sl, float* sc, int4 v, float scale) {
    *reinterpret_cast<int4*>(sl) = v;
    *sc = scale;
}

}  // namespace

void steer_set_slot(int* sl, float* sc, int vec, int mode, int lo, int hi, float scale, cudaStream_t st) {
    steer_set_slot_kernel<<<1, 1, 0, st>>>(sl, sc, make_int4(vec, mode, lo, hi), scale);
}

void steer_fix(float* resid, int m, int h, const SteerArg& a, const bf16* norm_w, float eps, bf16* xn, __half* xh,
               cudaStream_t st) {
    if (a.li < 0 || m <= 0) return;
    CHECK(h % 4 == 0 && (a.nrs == 0 || m <= a.nrs), "steer_fix %d rows (%d mapped)", m, a.nrs);
    steer_fix_kernel<<<m, kFixT, 0, st>>>(resid, h, a, norm_w, eps, xn, xh);
}

void steer_capture(const float* resid, int m, int skip, int h, float* acc, cudaStream_t st) {
    if (skip >= m) return;
    steer_capture_kernel<<<(h + 255) / 256, 256, 0, st>>>(resid, m, skip, h, acc);
}
