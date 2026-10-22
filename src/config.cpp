#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.h"

int Config::n_full() const {
    int n = 0;
    for (bool f : is_full) n += f;
    return n;
}

Config Config::load(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f) throw std::runtime_error("cannot read config.json in " + model_dir);
    std::stringstream ss;
    ss << f.rdbuf();
    Json root = Json::parse(ss.str());
    const Json& t = root.has("text_config") ? root["text_config"] : root;
    Config c;
    c.hidden = t["hidden_size"].i();
    c.inter = t["intermediate_size"].i();
    c.vocab = t["vocab_size"].i();
    c.layers = t["num_hidden_layers"].i();
    c.n_q = t["num_attention_heads"].i();
    c.n_kv = t["num_key_value_heads"].i();
    c.head_dim = t["head_dim"].i();
    const Json& rp = t["rope_parameters"];
    c.rope_theta = rp["rope_theta"].f();
    c.rot_dim = (int)std::lround(c.head_dim * rp["partial_rotary_factor"].f());
    c.lin_nk = t["linear_num_key_heads"].i();
    c.lin_nv = t["linear_num_value_heads"].i();
    c.lin_dk = t["linear_key_head_dim"].i();
    c.lin_dv = t["linear_value_head_dim"].i();
    c.conv_k = t["linear_conv_kernel_dim"].i();
    c.eps = (float)t["rms_norm_eps"].f();
    c.mtp_layers = t.has("mtp_num_hidden_layers") ? (int)t["mtp_num_hidden_layers"].i() : 0;
    c.eos = t["eos_token_id"].i();
    {  // chat models end a turn with <|im_end|>, listed only in generation_config.json
        std::ifstream g(model_dir + "/generation_config.json");
        if (g) {
            std::stringstream gs;
            gs << g.rdbuf();
            const Json gc = Json::parse(gs.str());
            if (gc.has("eos_token_id"))
                for (auto& e : gc["eos_token_id"].arr)
                    if (e.i() != c.eos) c.eos2 = (int)e.i();
        }
    }
    for (auto& lt : t["layer_types"].arr) c.is_full.push_back(lt.str == "full_attention");

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported config: ") + what);
    };
    need(c.head_dim == 256 && c.rot_dim == 64, "head_dim 256 / rotary 64");
    need(c.lin_dk == 128 && c.lin_dv == 128, "linear head dims 128");
    need(c.conv_k == 4, "conv kernel 4");
    need(c.hidden % 128 == 0 && c.inter % 128 == 0 && c.vocab % 128 == 0, "dims % 128");
    need(c.n_q % c.n_kv == 0 && c.lin_nv % c.lin_nk == 0, "head grouping");
    need(t["attn_output_gate"].b, "attn_output_gate");
    need((int)c.is_full.size() == c.layers, "layer_types");
    return c;
}

namespace {

// Split `units` into contiguous parts proportional to `w`, largest remainders first.
std::vector<Range> split_units(int units, const std::vector<double>& w) {
    double tot = 0;
    for (double x : w) tot += x;
    int n = w.size();
    std::vector<int> cnt(n);
    std::vector<std::pair<double, int>> rem;
    int used = 0;
    for (int i = 0; i < n; i++) {
        double exact = units * w[i] / tot;
        cnt[i] = (int)std::floor(exact);
        used += cnt[i];
        rem.push_back({exact - cnt[i], i});
    }
    std::stable_sort(rem.begin(), rem.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (int i = 0; used < units; i++, used++) cnt[rem[i % n].second]++;
    std::vector<Range> r(n);
    int b = 0;
    for (int i = 0; i < n; i++) {
        r[i] = {b, b + cnt[i]};
        b += cnt[i];
    }
    return r;
}

}  // namespace

std::vector<CardPlan> make_plan(const Config& c, int ngpu, const std::vector<double>& weights,
                                const KvSplit& ks) {
    std::vector<double> w = weights;
    if ((int)w.size() != ngpu) w.assign(ngpu, 1.0);
    auto q = split_units(c.n_q, w);
    auto v = split_units(c.lin_nv, w);
    auto in = split_units(c.inter / 128, w);
    auto vo = split_units(c.vocab / 128, w);
    auto fk = split_units(2 * c.hidden / 128, w);
    const int qpk = c.n_q / c.n_kv, vpk = c.lin_nv / c.lin_nk;
    std::vector<CardPlan> plan(ngpu);
    for (int i = 0; i < ngpu; i++) {
        CardPlan& p = plan[i];
        p.q_heads = q[i];
        p.kv_heads = {q[i].b / qpk, (q[i].e - 1) / qpk + 1};
        p.o_heads = p.q_heads;
        p.v_heads = v[i];
        p.k_heads = {v[i].b / vpk, (v[i].e - 1) / vpk + 1};
        p.inter = {in[i].b * 128, in[i].e * 128};
        p.vocab = {vo[i].b * 128, vo[i].e * 128};
        p.fc_k = {fk[i].b * 128, fk[i].e * 128};
        for (int j = 0; j < kMaxLocalKv; j++) p.kv_pm[j] = PosMap{ks.period, ks.period, 0};
        if (p.q_heads.size() == 0 || p.v_heads.size() == 0)
            throw std::runtime_error("too many GPUs for this model's head count");
    }
    if (!ks.on()) return plan;
    if (ngpu != 3 || c.n_kv != 4)
        throw std::runtime_error("kv position split needs 3 GPUs and 4 kv heads");
    // q heads computed: GPU0 kv0 + kv1 (q0-11), GPU1 kv1 + kv2 (q6-17), GPU2 kv2 + kv3 (q12-23);
    // o_proj: the columns of all 12 q heads on every card (the shared heads' outputs are summed by the all-reduce)
    plan[0].q_heads = {0, 2 * qpk};
    plan[1].q_heads = {qpk, 3 * qpk};
    plan[2].q_heads = {2 * qpk, 4 * qpk};
    // o_proj columns of every q head computed here (the shared heads' columns live on both cards
    // of a pair: each card pushes its own share of the attention through them, the all-reduce sums)
    for (int i = 0; i < 3; i++) {
        plan[i].o_heads = plan[i].q_heads;
        plan[i].kv_heads = {i, i + 2};
        plan[i].pos_split = true;
    }
    plan[1].kv_x[0] = plan[1].kv_x[1] = 3;
    plan[1].kv_peer[0] = 0;
    plan[1].kv_peer[1] = 2;
    plan[0].kv_x[1] = 3;
    plan[0].kv_peer[1] = 1;
    plan[2].kv_x[0] = 3;
    plan[2].kv_peer[0] = 1;
    kv_split_maps(plan, ks);
    return plan;
}

void kv_split_maps(std::vector<CardPlan>& plan, const KvSplit& ks) {
    const int P = ks.period;
    if (ks.a01 <= 0 || ks.a01 >= P || ks.a12 <= 0 || ks.a12 >= P || P % 32 || ks.a01 % 32 || ks.a12 % 32)
        throw std::runtime_error("kv position split: period and shares must be multiples of 32 within (0, period)");
    for (auto& p : plan)
        for (int j = 0; j < kMaxLocalKv; j++) p.kv_pm[j] = PosMap{P, P, 0};
    // GPU1 owns the first a01 keys of every period of kv1 (a12 of kv2); the complement map of
    // the other owner: owned <=> ((k + P - a) mod P) < P - a
    plan[1].kv_pm[0] = PosMap{P, ks.a01, 0};
    plan[1].kv_pm[1] = PosMap{P, ks.a12, 0};
    plan[0].kv_pm[1] = PosMap{P, P - ks.a01, P - ks.a01};
    plan[2].kv_pm[0] = PosMap{P, P - ks.a12, P - ks.a12};
}

void print_plan(const Config& c, const std::vector<CardPlan>& plan) {
    for (size_t i = 0; i < plan.size(); i++) {
        const CardPlan& p = plan[i];
        printf("GPU%zu: attn q[%d,%d) o[%d,%d) kv[%d,%d) | deltanet v[%d,%d) k[%d,%d) | mlp [%d,%d) | "
               "vocab [%d,%d)\n",
               i, p.q_heads.b, p.q_heads.e, p.o_heads.b, p.o_heads.e, p.kv_heads.b, p.kv_heads.e, p.v_heads.b,
               p.v_heads.e, p.k_heads.b, p.k_heads.e, p.inter.b, p.inter.e, p.vocab.b, p.vocab.e);
        if (p.pos_split)
            for (int j = 0; j < p.kv_heads.size(); j++) {
                const PosMap& m = p.kv_pm[j];
                if (m.A == m.P) printf("      kv%d: all positions\n", p.kv_heads.b + j);
                else
                    printf("      kv%d: %d of every %d positions (offset %d), softmax stats exchanged with GPU%d\n", p.kv_heads.b + j, m.A, m.P,
                           m.O, p.kv_peer[j]);
            }
    }
}
