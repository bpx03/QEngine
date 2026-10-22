#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Fatal errors (kernels/watchdog.cu): the shared abort word is raised first, so the peers' waits on
// this GPU trap within milliseconds instead of spinning while the process ends; a collective timeout
// (watchdog.cuh) is reported as such and exits with code 75 (restartable), other errors abort().
[[noreturn]] void qe_fatal_cuda(int err, const char* file, int line);
[[noreturn]] void qe_fatal_check();
void qe_fatal_begin();

#define CUDA_CHECK(x)                                                \
    do {                                                             \
        cudaError_t err__ = (x);                                     \
        if (err__ != cudaSuccess) qe_fatal_cuda(err__, __FILE__, __LINE__); \
    } while (0)

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "check failed %s at %s:%d: ", #cond, __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
            qe_fatal_check();                                                  \
        }                                                                      \
    } while (0)

typedef __nv_bfloat16 bf16;

// Runtime switch read on every call (not cached): lets one process A/B prefill variants by
// changing the environment between prompts. Only for host code outside
// the hot decode path (prefill launches: a getenv per call is negligible).
inline int env_int(const char* name, int def) {
    const char* e = getenv(name);
    return e && *e ? atoi(e) : def;
}
typedef __nv_bfloat162 bf162;

// Max tokens processed by the tensor-core GEMV path (decode + MTP verification).
constexpr int kMaxGemvTokens = 8;
constexpr int kMaxGemvW4Tokens = 32;
// Token ids >= kImgToken are image rows (see set_image_table); the host-side ids of a prompt
// keep their content-derived values for prefix matching.
constexpr int kImgToken = 1 << 30;
// rows of a batched decode step (several sequences): the INT4 GEMV handles up to 32 tokens itself, 33..64 go
// to the wide kernel (gemv_w4hx); the step policy caps the rows at QE_BATCH_MAXROWS (engine.cpp)
constexpr int kMaxBatchRows = 64;
// Prompt tokens per prefill forward pass (GEMM + flash-attention path).
constexpr int kChunk = 2048;
// Prompt chunks of at least this many tokens overlap their all-reduces with compute.
constexpr int kOverlapMin = 128;
