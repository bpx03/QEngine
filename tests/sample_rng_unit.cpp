// CPU unit test of the sampling random numbers (src/kernels/sample_rng.h, the code the kernels run): no two draws share a
// hash input across steps, rows and salts, including the 17 verify rows of a 16-draft lookup and the global rows of a batched
// step, up to the top of the counter range; the values are uniform and adjacent steps are uncorrelated. Also shows the old
// counter * 64 input failing (the reason for the change).
// Built by CMake (ctest) or: g++ -std=c++17 -O2 -Isrc tests/sample_rng_unit.cpp -o /tmp/sru && /tmp/sru
#include "kernels/sample_rng.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static int fails = 0;
#define EXPECT(c)                                                                     \
    do {                                                                              \
        if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

using u64 = unsigned long long;
constexpr int kMaxBatchRows = 64;  // common.h (which needs the CUDA headers); sample.cu has the static_assert on the real one
static u64 old_input(u64 counter, int row, int salt) { return counter * 64 + row * 4 + salt + 1; }

// repeated inputs among steps [c0, c0 + steps), rows [0, rows), salts 0 and 1
template <class F>
static size_t repeats(F input, u64 c0, u64 steps, int rows) {
    std::vector<u64> v;
    v.reserve(steps * rows * 2);
    for (u64 c = c0; c < c0 + steps; c++)
        for (int r = 0; r < rows; r++)
            for (int s = 0; s < 2; s++) v.push_back(input(c, r, s));
    std::sort(v.begin(), v.end());
    size_t n = 0;
    for (size_t i = 1; i < v.size(); i++) n += v[i] == v[i - 1];
    return n;
}

int main() {
    // the flaw: row 16 of step c is row 0 of step c + 1, for both salts
    EXPECT(old_input(5, 16, 0) == old_input(6, 0, 0));
    EXPECT(old_input(5, 16, 1) == old_input(6, 0, 1));
    EXPECT(sample_rng_input(5, 16, 0) != sample_rng_input(6, 0, 0));
    EXPECT(sample_uniform(1234, 5, 16, 0) != sample_uniform(1234, 6, 0, 0));

    printf("%-34s %14s %14s %8s\n", "range", "draws", "old repeats", "new");
    const struct { const char* name; u64 c0, steps; int rows; } R[] = {
        {"16 rows (27B lookup 12 fits)", 0, 100000, 16},
        {"17 rows (QFN lookup 16)", 0, 100000, 17},
        {"24 rows (27B batch)", 0, 100000, 24},
        {"64 rows (kMaxBatchRows)", 0, 20000, kMaxBatchRows},
        {"4096 rows", 0, 200, 4096},
        {"top of the counter, 64 rows", (1ull << 44) - 20000, 20000, kMaxBatchRows},
    };
    for (const auto& r : R) {
        const size_t o = repeats(old_input, r.c0, r.steps, r.rows), n = repeats(sample_rng_input, r.c0, r.steps, r.rows);
        printf("%-34s %14llu %14zu %8zu\n", r.name, r.steps * r.rows * 2ull, o, n);
        EXPECT(n == 0);
        EXPECT((r.rows <= 16) == (o == 0));  // the old input repeats exactly from 17 rows on
    }
    // the last row and salt stay below the next step
    EXPECT(sample_rng_input(7, kSampleMaxRows - 1, 1) < sample_rng_input(8, 0, 0));
    EXPECT(kMaxBatchRows <= kSampleMaxRows);

    // uniformity: 256 bins (chi-square, 255 dof, 5 sigma about 255 + 5 * sqrt(510) = 368), mean, range; per kind of walk
    const struct { const char* name; int rows; } U[] = {{"c=1 row 0", 1}, {"17 rows", 17}, {"64 rows", 64}};
    for (u64 seed : {1234ull, 0ull, 1780000000000000ull})
        for (const auto& u : U)
            for (int salt : {0, 1}) {
                const int N = 2000000, B = 256;
                std::vector<long> h(B);
                double sum = 0, sq = 0, lo = 1, hi = 0;
                for (int i = 0; i < N; i++) {
                    const float x = sample_uniform(seed, (u64)(i / u.rows), i % u.rows, salt);
                    EXPECT(x >= 0.f && x < 1.f);
                    h[std::min(B - 1, (int)(x * B))]++;
                    sum += x; sq += x * x; lo = std::min(lo, (double)x); hi = std::max(hi, (double)x);
                }
                double chi = 0;
                for (long c : h) chi += (c - (double)N / B) * (c - (double)N / B) / ((double)N / B);
                const double mean = sum / N, var = sq / N - mean * mean;
                if (seed == 1234ull)
                    printf("seed %llu %-10s salt %d: chi2 %.1f (255 dof), mean %.5f, var %.5f (1/12 = 0.08333), range %.6f..%.6f\n", seed,
                           u.name, salt, chi, mean, var, lo, hi);
                EXPECT(chi < 368.0);
                EXPECT(std::fabs(mean - 0.5) < 5 * sqrt(1.0 / 12 / N));
                EXPECT(std::fabs(var - 1.0 / 12) < 0.001);
            }

    // correlation between neighbours in the (step, row) table: next row, next step, salt 0 vs salt 1 (5 sigma = 5 / sqrt(n))
    {
        const int C = 20000, M = 17;
        double xy[3] = {}, x2 = 0, sx = 0;
        const u64 seed = 1234;
        int n = 0;
        for (int c = 0; c < C; c++)
            for (int r = 0; r < M - 1; r++, n++) {
                const double x = sample_uniform(seed, c, r, 0) - 0.5;
                xy[0] += x * (sample_uniform(seed, c, r + 1, 0) - 0.5);
                xy[1] += x * (sample_uniform(seed, c + 1, r, 0) - 0.5);
                xy[2] += x * (sample_uniform(seed, c, r, 1) - 0.5);
                x2 += x * x; sx += x;
            }
        printf("correlation over %d pairs (5 sigma = %.4f): next row %.4f, next step %.4f, other salt %.4f\n", n, 5 / sqrt((double)n),
               xy[0] / x2, xy[1] / x2, xy[2] / x2);
        for (double v : xy) EXPECT(std::fabs(v / x2) < 5 / sqrt((double)n));
    }
    printf(fails ? "FAILED %d\n" : "ok\n", fails);
    return fails != 0;
}
