#pragma once
#include <string>
#include <vector>

#include "kernels/ops.cuh"  // PosMap

// Qwen3.5-architecture text model (Qwen3.8-27B). The engine is specialised for this
// family; load() validates that config.json matches what the kernels assume.
struct Config {
    int hidden = 0, inter = 0, vocab = 0, layers = 0;
    int n_q = 0, n_kv = 0, head_dim = 0, rot_dim = 0;
    double rope_theta = 0;
    int native_ctx = 262144;  // max_position_embeddings: the trained context
    // set by the Engine before the Workers are made: positions the KV structures hold (native_ctx, or the YaRN
    // context) and the YaRN factor (0: native RoPE; QE_YARN / --max-ctx above native_ctx)
    int max_pos = 262144;
    double yarn = 0;
    double yarn_cfg = 0;  // config.json rope_parameters rope_type yarn: its factor (0: rope_type default)
    int lin_nk = 0, lin_nv = 0, lin_dk = 0, lin_dv = 0, conv_k = 0;
    float eps = 1e-6f;
    int mtp_layers = 0;
    std::vector<bool> is_full;  // per layer: full attention (true) or gated delta net
    int eos = 0;
    int eos2 = -1;  // second end-of-turn id (generation_config.json), e.g. <|im_end|>
    // <|image_pad|> (config.json image_token_id; eos without one): what the lookup history holds for an image row, as
    // Flash-Next's dev_tok. The host keeps the rows' content-derived ids (>= kImgToken) for prefix matching; a lookup
    // draft copied from them read the image table at that id (Xid 13)
    int image_pad = -1;

    int n_full() const;
    int n_lin() const { return layers - n_full(); }
    static Config load(const std::string& model_dir);
};

// RoPE table of the attention layers (model.cpp set_rope, unit test qe_rope_table): out[0..31] the inverse frequencies
// of the 32 rotary pairs, out[32] the factor on the rotary dims of q (1 native; YaRN: mscale^2, so the scores equal
// mscale on q and k while the cached K keeps its calibrated range). factor <= 1: native RoPE (the double pow as
// before). YaRN (vLLM / HF definition): NTK-by-parts blend over the rotary dim, interpolated 1 / (factor base^(2i/d))
// for the low frequencies, extrapolated for the high ones, a linear ramp between the correction dims of beta_fast 32 /
// beta_slow 1 (floor / ceil), mscale 0.1 ln(factor) + 1, base^(2i/d) in fp32 as the references. lo / hi: correction dims
void rope_table_host(double factor, double theta, int rot_dim, int native_ctx, float* out, double* lo = nullptr, double* hi = nullptr);

struct Range {
    int b = 0, e = 0;
    int size() const { return e - b; }
};

// Per-GPU slice of every tensor-parallel dimension. Splits may be uneven; when a group
// boundary (GQA kv head, delta-net key head) falls inside a card, that head is duplicated
// on both neighbours so compute stays balanced.
struct CardPlan {
    Range q_heads, kv_heads;    // full attention: q heads this card computes, kv heads it writes
    Range o_heads;              // q heads whose o_proj columns live here (= q_heads unless split)
    Range v_heads, k_heads;     // gated delta net
    Range inter;                // MLP intermediate (elements, multiple of 128)
    Range vocab;                // lm_head rows (multiple of 128)
    Range fc_k;                 // MTP fc input columns (of 2*hidden, multiple of 128)
    // kv-pos-split: per local kv head its position map (keys of a head split between two cards by
    // absolute position), its exchange role (0 none, 1 this card sends the partial attention of the
    // head's q heads to kv_peer, 2 it receives the peer's) and the peer's rank
    PosMap kv_pm[kMaxLocalKv];
    int kv_x[kMaxLocalKv] = {};
    int kv_peer[kMaxLocalKv] = {-1, -1, -1, -1};
    bool pos_split = false;
};

// Position split of the kv heads (3 GPUs, 4 kv heads): kv0 on GPU0, kv3 on GPU2, kv1 split by
// position between GPU0 / GPU1 and kv2 between GPU1 / GPU2; GPU1 owns the first a01 (a12) keys of
// every period of `period` keys. The two cards of a pair exchange only the softmax stats of the
// split head's q heads; each holds the o_proj columns of all its 12 q heads and pushes its own share
// of the attention through them (the all-reduce sums). a01 = a12 = 0: no split (the old plan).
struct KvSplit {
    int period = 1024, a01 = 0, a12 = 0;
    bool on() const { return a01 > 0 || a12 > 0; }
    // GPU1's keys per period from the other card's share of the head (multiple of 32)
    int to_a(double share_other) const {
        const int a = (int)((1.0 - share_other) * period / 32 + 0.5) * 32;
        return a < 32 ? 32 : (a > period - 32 ? period - 32 : a);
    }
};

std::vector<CardPlan> make_plan(const Config& c, int ngpu, const std::vector<double>& weights,
                                const KvSplit& ks = {});
// (re)assign the position maps of a split plan from the shares in ks (the heads stay as planned)
void kv_split_maps(std::vector<CardPlan>& plan, const KvSplit& ks);
void print_plan(const Config& c, const std::vector<CardPlan>& plan);
