#pragma once
// [q27-head4r] INT4 screen + exact rescore of the verify head (head4r.cu)
#include <cstddef>

#include "../common.h"

constexpr int kH4Cap = 256;     // candidates per row before the sort (superset of the top-R)
constexpr int kH4MaxR = 64;     // rescored candidates per row
constexpr int kH4MaxRows = 64;  // rows per call (kMaxBatchRows)

struct H4Ws {  // device workspace (h4_ws)
    unsigned *hist, *pfx, *t16;  // hist [m][4][256]; pfx: the key prefix found so far (level 3: the full key T)
    int *above, *cnt, *ovf;      // ovf: the 16-bit threshold took more than kH4Cap (levels 2, 3 refine it)
    float* cv;
    int* ci;
    int* cand;  // [m][R]
    float* ex;  // [m][R] exact values
    float* cq;  // [m][R] the candidates' INT4 values (h4_scatter back = true)
    unsigned long long* key;  // [m * R] (id, row slot) sorted by id: the deduplicated rescore
};
size_t h4_ws_bytes(int m);
H4Ws h4_ws(void* base, int m);
// candidates of each row: exactly the top-R by (INT4 logit desc, id asc), -inf (masked) logits excluded -> ws.cand
// (sorted; -1 past the finite ones)
void h4_select(const float* logits, int v, int m, int R, const H4Ws& ws, cudaStream_t st);
// exact values of the candidates from bf16 rows w [v][H] (device or mapped host; row = local id) -> ws.ex. Rows in
// the bit mask `full` rescore all R, the others the first rmin (the greedy pick's; review M6)
void h4_rescore_bf16(const bf16* w, const bf16* x, int H, int m, int R, const H4Ws& ws, cudaStream_t st,
                     unsigned long long full = ~0ull, int rmin = 0);
// argmax of the exact values over the first `use` candidates (ties: lowest id) -> pairs [m][2] (value, id + off),
// for argmax_final
void h4_pick(const H4Ws& ws, int m, int R, int use, int off, float* pairs, cudaStream_t st);
// sampling candidates (R = 64 = kTopK): the exact values sorted -> pairs [m][64][2] in topk_local's format
void h4_topk(const H4Ws& ws, int m, int R, unsigned long long full, int rmin, int off, float* pairs, cudaStream_t st);
// exact values (back: the INT4 values again) written over the logits [m][v] at the candidates (rows outside `full`:
// the first rmin)
void h4_scatter(const H4Ws& ws, int m, int R, float* logits, int v, cudaStream_t st, unsigned long long full = ~0ull,
                int rmin = 0, bool back = false);
