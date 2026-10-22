#include "spec.cuh"

namespace {

__device__ float block_sum512(float v) {
    __shared__ float sh[32];
    __shared__ float total;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float x = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
        if (threadIdx.x == 0) total = x;
    }
    __syncthreads();
    return total;
}

// grid (m, 2): y = 0 embedding half, y = 1 hidden half
// One block per (row, half): the source row (embedding from host memory over PCIe, or h) is read
// once with 16-byte loads into registers (hidden <= 8 * 8 * 128), then RMS-normalized.
constexpr int kPrepThreads = 128, kPrepVec = 8;  // 8 x uint4 = 64 bf16 per thread
__device__ const bf16* g_img_table_spec = nullptr;  // see set_image_table

__global__ void __launch_bounds__(kPrepThreads) mtp_prep_kernel(
    const bf16* __restrict__ table, const int* __restrict__ tok, const bf16* __restrict__ h,
    const int* __restrict__ hrow, const bf16* __restrict__ we, const bf16* __restrict__ wh,
    float eps, int hidden, bf16* __restrict__ out) {
    const int mi = blockIdx.x, half = blockIdx.y;
    const int tk = half == 0 ? tok[mi] : 0;
    const bf16* src = half == 0 ? (tk >= kImgToken ? g_img_table_spec + (size_t)(tk - kImgToken) * hidden
                                                   : table + (size_t)tk * hidden)
                                : h + (size_t)((hrow ? *hrow : 0) + mi) * hidden;
    const bf16* w = half == 0 ? we : wh;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    const int n4 = hidden / 8;
    uint4 v[kPrepVec];
#pragma unroll
    for (int j = 0; j < kPrepVec; j++) {
        const int i = threadIdx.x + j * kPrepThreads;
        if (i < n4) v[j] = s4[i];
    }
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < kPrepVec; j++) {
        const int i = threadIdx.x + j * kPrepThreads;
        if (i >= n4) continue;
        const bf16* e = reinterpret_cast<const bf16*>(&v[j]);
#pragma unroll
        for (int q = 0; q < 8; q++) {
            const float x = __bfloat162float(e[q]);
            ss += x * x;
        }
    }
    __shared__ float red[kPrepThreads / 32];
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < kPrepThreads / 32; i++) tot += red[i];
    const float r = rsqrtf(tot / hidden + eps);
    uint4* o4 = reinterpret_cast<uint4*>(out + (size_t)mi * 2 * hidden + half * hidden);
    const uint4* w4 = reinterpret_cast<const uint4*>(w);
#pragma unroll
    for (int j = 0; j < kPrepVec; j++) {
        const int i = threadIdx.x + j * kPrepThreads;
        if (i >= n4) continue;
        const uint4 wv = w4[i];
        const bf16* e = reinterpret_cast<const bf16*>(&v[j]);
        const bf16* ww = reinterpret_cast<const bf16*>(&wv);
        uint4 res;
        bf16* rr = reinterpret_cast<bf16*>(&res);
#pragma unroll
        for (int q = 0; q < 8; q++)
            rr[q] = __float2bfloat16(__bfloat162float(e[q]) * r * (1.f + __bfloat162float(ww[q])));
        o4[i] = res;
    }
}

__global__ void accept_kernel(SpecState s, int k) {
    const int n = *s.pos;
    int a = 0;
    while (a < k && s.tokens[a + 1] == s.out[a] && s.out[a] != s.eos0 && s.out[a] != s.eos1) a++;
    s.emit[0] = a + 1;
    for (int j = 0; j <= a; j++) s.emit[1 + j] = s.out[j];
    for (int j = 0; j <= k; j++) s.mtp_tok[j] = s.out[j];
    *s.acc = a;
    *s.mtp_pos = n;
    *s.hrow = 0;
}

__global__ void select_kernel(SpecState s) {
    const int a = *s.acc;
    s.drafts[0] = s.mtp_out[a];
    s.mtp_tok[0] = s.mtp_out[a];
    *s.hrow = a;
    *s.mtp_pos = *s.pos + a + 1;
}

__global__ void append_kernel(SpecState s, int r) {
    s.drafts[r] = s.mtp_out[0];
    s.mtp_tok[0] = s.mtp_out[0];
    *s.hrow = 0;
    *s.mtp_pos += 1;
}

__global__ void finish_kernel(SpecState s, int k) {
    const int a = *s.acc;
    s.tokens[0] = s.out[a];
    for (int j = 0; j < k; j++) s.tokens[1 + j] = s.drafts[j];
    *s.pos += a + 1;
}

__global__ void select_batch_kernel(SpecBatch b) {
    const int z = threadIdx.x;
    if (z >= b.n) return;
    const SpecState& s = b.s[z];
    const int a = *s.acc;
    s.drafts[0] = s.mtp_out[a];
    s.mtp_tok[0] = s.mtp_out[a];
    *s.hrow = a;
    *s.mtp_pos = *s.pos + a + 1;
}

__global__ void finish_batch_kernel(SpecBatch b, int k) {
    const int z = threadIdx.x;
    if (z >= b.n) return;
    const SpecState& s = b.s[z];
    const int a = *s.acc;
    s.tokens[0] = s.out[a];
    for (int j = 0; j < k; j++) s.tokens[1 + j] = s.drafts[j];
    *s.pos += a + 1;
}

__global__ void prefill_select_kernel(SpecState s, int m, int n) {
    s.drafts[0] = s.mtp_out[m - 1];
    s.mtp_tok[0] = s.mtp_out[m - 1];
    s.tokens[0] = s.out[m - 1];
    *s.hrow = m - 1;
    *s.mtp_pos = n;
}

__global__ void copy_last_kernel(SpecState s, int m) { s.mtp_tok[m - 1] = s.out[m - 1]; }

__global__ void finish_prefill_kernel(SpecState s, int k) {
    for (int j = 0; j < k; j++) s.tokens[1 + j] = s.drafts[j];
}

}  // namespace

void set_image_table_spec(const bf16* table) {
    CUDA_CHECK(cudaMemcpyToSymbol(g_img_table_spec, &table, sizeof(table)));
}

void mtp_prep(const bf16* table, const int* tok, const bf16* h, const int* hrow, const bf16* we,
              const bf16* wh, float eps, int m, int hidden, bf16* out, cudaStream_t st) {
    CHECK(hidden % 8 == 0 && hidden <= 8 * kPrepVec * kPrepThreads, "mtp_prep hidden %d", hidden);
    mtp_prep_kernel<<<dim3(m, 2), kPrepThreads, 0, st>>>(table, tok, h, hrow, we, wh, eps, hidden, out);
}
void spec_accept(const SpecState& s, int k, cudaStream_t st) { accept_kernel<<<1, 1, 0, st>>>(s, k); }
void spec_select(const SpecState& s, cudaStream_t st) { select_kernel<<<1, 1, 0, st>>>(s); }
void spec_select_batch(const SpecBatch& b, cudaStream_t st) { if (b.n) select_batch_kernel<<<1, 32, 0, st>>>(b); }
void spec_finish_batch(const SpecBatch& b, int k, cudaStream_t st) { if (b.n) finish_batch_kernel<<<1, 32, 0, st>>>(b, k); }
void spec_append(const SpecState& s, int r, cudaStream_t st) {
    append_kernel<<<1, 1, 0, st>>>(s, r);
}
void spec_finish(const SpecState& s, int k, cudaStream_t st) { finish_kernel<<<1, 1, 0, st>>>(s, k); }
void spec_prefill_select(const SpecState& s, int m, int n, cudaStream_t st) {
    prefill_select_kernel<<<1, 1, 0, st>>>(s, m, n);
}
void spec_copy_last(const SpecState& s, int m, cudaStream_t st) {
    copy_last_kernel<<<1, 1, 0, st>>>(s, m);
}
void spec_finish_prefill(const SpecState& s, int k, cudaStream_t st) {
    finish_prefill_kernel<<<1, 1, 0, st>>>(s, k);
}
