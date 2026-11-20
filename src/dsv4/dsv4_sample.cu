// Release prep [dsv4-rel-sample]: sampling for the DSV4 engine through the shared sampler (src/kernels/sample.cu: top-64
// candidates per row, temperature / top-k / top-p / min-p, the sample_rng.h hash with a device counter; speculative
// acceptance that is lossless for deterministic drafts: draft d accepted with probability p(d), else a token drawn from p
// without d, stop after an end-of-turn token). A separate translation unit: the shared headers stay out of dsv4.cu.
#include "dsv4_sample.h"

#include <cstdio>
#include <cstdlib>

#include "../kernels/sample.cuh"

namespace dsv4 {

struct SampDev {
    SampleParams p;
    int tokens[DSS_MAXR], pos, out[DSS_MAXR], emit[DSS_MAXR + 1], acc, mtp_tok[DSS_MAXR], mtp_pos, hrow, drafts[DSS_MAXR];
};

size_t samp_bytes() { return sizeof(SampDev) + (size_t)DSS_MAXR * kTopK * 2 * 4; }

void samp_set(void* dev, float temperature, float top_p, int top_k, float min_p, unsigned long long seed, cudaStream_t st) {
    SampleParams p{};
    p.temperature = temperature;
    p.top_p = top_p > 0.f && top_p <= 1.f ? top_p : 1.f;
    p.top_k = top_k >= 1 && top_k <= kTopK ? top_k : kTopK;
    p.min_p = min_p;
    p.seed = seed;
    p.counter = 0;
    cudaMemcpyAsync(dev, &p, sizeof p, cudaMemcpyHostToDevice, st);
}

static float* pairs_of(void* dev) { return (float*)((char*)dev + sizeof(SampDev)); }

void samp_rows(void* dev, const float* logits, int v, int m, int* tokens_dev, cudaStream_t st) {
    SampDev* d = (SampDev*)dev;
    topk_local(logits, v, m, 0, pairs_of(dev), st);
    sample_rows(pairs_of(dev), 1, m, &d->p, tokens_dev, st, true);
}

void samp_accept(void* dev, const float* logits, int v, const int* in_tokens_host, int k, int pos, int eos0, int eos1,
                 int* emit_host, cudaStream_t st) {
    SampDev* d = (SampDev*)dev;
    topk_local(logits, v, k + 1, 0, pairs_of(dev), st);
    int hdr[DSS_MAXR + 1];
    for (int j = 0; j <= k; j++) hdr[j] = in_tokens_host[j];
    hdr[k + 1] = pos;   // tokens[0..k] then pos: contiguous in SampDev
    cudaMemcpyAsync(d->tokens, hdr, (size_t)(k + 1) * 4, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(&d->pos, &hdr[k + 1], 4, cudaMemcpyHostToDevice, st);
    SpecState s;
    s.tokens = d->tokens;
    s.pos = &d->pos;
    s.out = d->out;
    s.emit = d->emit;
    s.acc = &d->acc;
    s.mtp_tok = d->mtp_tok;
    s.mtp_pos = &d->mtp_pos;
    s.hrow = &d->hrow;
    s.drafts = d->drafts;
    s.eos0 = eos0;
    s.eos1 = eos1;
    spec_accept_sampled(pairs_of(dev), 1, k, &d->p, s, st);
    cudaMemcpyAsync(emit_host, d->emit, (size_t)(k + 2) * 4, cudaMemcpyDeviceToHost, st);
}

}  // namespace dsv4

// the shared library's CUDA error exit (src/common.h; defined in the main engine's capi.cpp, not linked here)
[[noreturn]] void qe_fatal_cuda(int err, const char* file, int line) {
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString((cudaError_t)err), file, line);
    abort();
}
