// DeltaNet gate projection (a / b logits, 2 * ab_pad = 32 output rows) over a prefill
// micro-batch: ab[t][c] = sum_k W[c][k] x[t][k] with W in the W16 fragment layout.
// gemm_w16 spends ~43 us per 1024 tokens on it (tile of 128+ rows for 32 outputs, 16 K-splits);
// here a block of 8 warps owns 16 tokens and all 32 rows, the warps split K, and the partial
// sums are reduced in shared memory in a fixed warp order (one ab slice, no split-K buffer).
#include "gemv.cuh"
#include "ops.cuh"

namespace {

__device__ __forceinline__ void mma_bf16_ab(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int kAbWarps = 8, kAbRT = 2, kAbNT = 2;  // 32 rows, 16 tokens per block

__global__ void __launch_bounds__(kAbWarps * 32) dn_ab_kernel(const uint4* __restrict__ w, int K,
                                                              const bf16* __restrict__ x, int ldx, int m,
                                                              float* __restrict__ out, int ldo) {
    __shared__ float red[kAbWarps][kAbRT * 16][kAbNT * 8 + 1];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tok0 = blockIdx.x * kAbNT * 8;
    const int steps = K >> 4;
    const int sb = warp * steps / kAbWarps, se = (warp + 1) * steps / kAbWarps;
    const bf16* xr[kAbNT];
    bool act[kAbNT];
#pragma unroll
    for (int j = 0; j < kAbNT; j++) {
        const int tok = tok0 + 8 * j + g;
        act[j] = tok < m;
        xr[j] = x + (size_t)(act[j] ? tok : 0) * ldx + 2 * t;
    }
    float acc[kAbRT][kAbNT][4] = {};
#pragma unroll 4
    for (int s = sb; s < se; s++) {
        uint32_t b[kAbNT][2];
#pragma unroll
        for (int j = 0; j < kAbNT; j++) {
            b[j][0] = act[j] ? *reinterpret_cast<const uint32_t*>(xr[j] + s * 16) : 0u;
            b[j][1] = act[j] ? *reinterpret_cast<const uint32_t*>(xr[j] + s * 16 + 8) : 0u;
        }
#pragma unroll
        for (int rt = 0; rt < kAbRT; rt++) {
            const uint4 v = __ldg(w + ((size_t)rt * steps + s) * 32 + lane);
            const uint32_t a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < kAbNT; j++) mma_bf16_ab(acc[rt][j], a, b[j][0], b[j][1]);
        }
    }
    // C fragment: rows (output channels) g / g + 8, columns (tokens) 2t / 2t + 1
#pragma unroll
    for (int rt = 0; rt < kAbRT; rt++)
#pragma unroll
        for (int j = 0; j < kAbNT; j++) {
            red[warp][rt * 16 + g][j * 8 + 2 * t] = acc[rt][j][0];
            red[warp][rt * 16 + g][j * 8 + 2 * t + 1] = acc[rt][j][1];
            red[warp][rt * 16 + g + 8][j * 8 + 2 * t] = acc[rt][j][2];
            red[warp][rt * 16 + g + 8][j * 8 + 2 * t + 1] = acc[rt][j][3];
        }
    __syncthreads();
    for (int e = threadIdx.x; e < kAbRT * 16 * kAbNT * 8; e += blockDim.x) {
        const int c = e % (kAbRT * 16), tk = e / (kAbRT * 16);
        float s = 0.f;
#pragma unroll
        for (int wv = 0; wv < kAbWarps; wv++) s += red[wv][c][tk];
        if (tok0 + tk < m) out[(size_t)(tok0 + tk) * ldo + c] = s;
    }
}

}  // namespace

bool dn_ab_prefill_ok(const W16& w) { return w.n == kAbRT * 16 && w.k % (16 * kAbWarps) == 0; }

void dn_ab_prefill(const W16& w, const bf16* x, int ldx, float* out, int ldo, int m, cudaStream_t st) {
    dn_ab_kernel<<<(m + kAbNT * 8 - 1) / (kAbNT * 8), kAbWarps * 32, 0, st>>>(w.w, w.k, x, ldx, m, out, ldo);
}
