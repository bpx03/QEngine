// qe_w8q IN OUT_Q OUT_S N K: INT8 g128 quantization of a row-major bf16 matrix [N][K] (raw file IN) with the engine's
// own quantize_w8 kernel (the values load_w8 computes at load, bit for bit when this tool and the engine are built
// with the same flags: under --use_fast_math the approximate reciprocal decides some .5 ties), written back row-major: OUT_Q int8 [N][K],
// OUT_S bf16 scales [N][K/128]. Used by tools/convert_flash_next.py --int8 so a checkpoint can store exactly the
// INT8 weights the engine runs (load_w8i -> tile_w8 gives the identical W8 layout).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../kernels/gemv.cuh"

int main(int argc, char** argv) {
    if (argc != 6) { fprintf(stderr, "usage: qe_w8q IN OUT_Q OUT_S N K\n"); return 2; }
    const int N = atoi(argv[4]), K = atoi(argv[5]);
    if (N % 16 || K % 128) { fprintf(stderr, "N %% 16 and K %% 128 required\n"); return 2; }
    const size_t e = (size_t)N * K, G = K / 128;
    std::vector<bf16> h(e);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(h.data(), 2, e, f) != e) { fprintf(stderr, "read %s\n", argv[1]); return 1; }
    fclose(f);
    bf16* d; uint4* q; bf16* s;
    CUDA_CHECK(cudaMalloc(&d, e * 2));
    CUDA_CHECK(cudaMalloc(&q, w8_qbytes(N, K)));
    CUDA_CHECK(cudaMalloc(&s, (size_t)N * G * 2));
    CUDA_CHECK(cudaMemcpy(d, h.data(), e * 2, cudaMemcpyHostToDevice));
    quantize_w8(d, N, K, q, s, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> tq(w8_qbytes(N, K));
    std::vector<bf16> ts((size_t)N * G);
    CUDA_CHECK(cudaMemcpy(tq.data(), q, tq.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ts.data(), s, ts.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<int8_t> oq(e);
    std::vector<bf16> os((size_t)N * G);
    for (int row = 0; row < N; row++) {  // inverse of the fragment index math of quantize_w8_kernel / tile_w8_kernel
        const int rt = row >> 4, rr = row & 15;
        for (int k = 0; k < K; k++) {
            const int ks = k >> 4, col = k & 15;
            const int tq4 = (col & 7) >> 1, half = col >> 3, odd = col & 1;
            const int ln = (rr & 7) * 4 + tq4;
            const int ai = (half * 2 + (rr >> 3)) * 2 + odd;
            const size_t base = (((size_t)rt * (K >> 5) + (ks >> 1)) * 32 + ln) * 16 + (ks & 1) * 8;
            oq[(size_t)row * K + k] = (int8_t)tq[base + ai];
        }
        for (size_t g = 0; g < G; g++) os[(size_t)row * G + g] = ts[((size_t)rt * G + g) * 16 + rr];
    }
    f = fopen(argv[2], "wb");
    if (!f || fwrite(oq.data(), 1, e, f) != e || fclose(f) != 0) { fprintf(stderr, "write %s\n", argv[2]); return 1; }
    f = fopen(argv[3], "wb");
    if (!f || fwrite(os.data(), 2, os.size(), f) != os.size() || fclose(f) != 0) { fprintf(stderr, "write %s\n", argv[3]); return 1; }
    return 0;
}
