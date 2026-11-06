#pragma once
// The random numbers of the sampling kernels (sample.cu; tests/sample_rng_unit.cpp). Every draw has its own hash input:
// the sampling kernel's step counter above bit 20, the row and the draw kind (salt 0 or 1) below it. Row and step must
// not share bits: with counter * 64 a row >= 16 (17 verify rows of a 16-draft lookup, the global rows of a batched step)
// reused the numbers of the next step. Rows stay below 2^18, far above kMaxBatchRows (static_assert in sample.cu); the
// counter would need 2^44 steps to overflow.
#ifdef __CUDACC__
#define QE_HD __host__ __device__ __forceinline__
#else
#define QE_HD inline
#endif
constexpr int kSampleRowShift = 2;      // salt 0 / 1, stored + 1: bits 0-1
constexpr int kSampleCounterShift = 20;
constexpr int kSampleMaxRows = 1 << (kSampleCounterShift - kSampleRowShift);

QE_HD unsigned long long sample_splitmix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
QE_HD unsigned long long sample_rng_input(unsigned long long counter, int row, int salt) {
    return (counter << kSampleCounterShift) + (unsigned long long)row * (1u << kSampleRowShift) + salt + 1;
}
QE_HD float sample_uniform(unsigned long long seed, unsigned long long counter, int row, int salt) {
    const unsigned long long h = sample_splitmix(seed ^ sample_splitmix(sample_rng_input(counter, row, salt)));
    return (h >> 40) * (1.0f / 16777216.0f);  // 24-bit uniform in [0, 1)
}
#undef QE_HD
