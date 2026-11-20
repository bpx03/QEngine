// [srv-gptoss-capi] CPU unit test of the gpt-oss server C API (gptoss_capi.cpp + capi_srv.cpp) over a fake Engine (no
// GPU): start codes, greedy = the engine's argmax, seeded sampling (reproducible, follows the logits), top_k 1 = greedy,
// the limit, trim inside / beyond the ring window, faults.
//   g++ -O2 -std=c++20 -I/usr/local/cuda/include -Isrc/gptoss tests/gptoss_capi_unit.cpp src/gptoss/gptoss_capi.cpp \
//       src/gptoss/capi_srv.cpp -o /tmp/capi_unit && /tmp/capi_unit
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

#include "gptoss.h"
#include "capi_srv.h"

namespace gptoss {
struct Impl {
    Config cfg;
    int pos = 0;
    Dump d;
    bool fail = false;
    // sampled decode (Engine::set_sampling): one card holding the whole vocabulary
    float temp = 0;
    unsigned long long seed = 0;
    std::vector<float> lg, out;
    Plan plan;
    // slots: the bound slot's base / limit; rows[r] = the token written at absolute row r of the full layers
    int slot = 0, base = 0, lim = 0;
    long steps = 0, joint = 0;   // engine steps; step_rows calls with rows of more than one slot
    std::vector<int> rows;
    long copies = 0;
    // [gptoss-rel2-penalty] per slot: token -> penalty; the one-row steps' setting (set_sampling_penalty)
    std::vector<std::unordered_map<int, float>> pen;
    int pslot = -1;
    bool draw = true;
    long pen_resets = 0;
};
static Impl* g_impl = nullptr;
// the fake model: the logits of row r peak at token (r * 7) % 1000 + 10; a second peak at +1 (logit 1 lower)
static int fake_next(int row, float* logits) {
    const int t = (row * 7) % 1000 + 10;
    if (logits) {
        for (int i = 0; i < VOCAB; i++) logits[i] = -5.f;
        logits[t] = 3.f;
        logits[t + 1] = 2.f;
    }
    return t;
}
Engine::Engine(const Config& cfg) : p_(new Impl) {
    p_->cfg = cfg;
    p_->lim = cfg.max_ctx;
    p_->rows.assign(cfg.max_ctx, -1);
    g_impl = p_;
}
int Engine::num_slots() const { return p_->cfg.slots; }
bool Engine::broken() const { return false; }
void Engine::set_soft_rows(bool) {}
void Engine::bind(int slot, int base, int pos, int lim) {
    if (slot < 0 || slot >= p_->cfg.slots || base < 0 || base + lim > p_->cfg.max_ctx || pos > lim) throw std::runtime_error("bind");
    p_->slot = slot;
    p_->base = base;
    p_->pos = pos;
    p_->lim = lim;
}
void Engine::copy_rows(int src, int dst, int n) {
    if (!(src + n <= dst || dst + n <= src) || src + n > p_->cfg.max_ctx || dst + n > p_->cfg.max_ctx) throw std::runtime_error("copy_rows overlap");
    for (int i = 0; i < n; i++) p_->rows[dst + i] = p_->rows[src + i];
    p_->copies++;
}
Engine::~Engine() { delete p_; }
int Engine::max_ctx() const { return p_->cfg.max_ctx; }
int Engine::chunk() const { return p_->cfg.chunk; }
int Engine::pos() const { return p_->pos; }
void Engine::reset() { p_->pos = 0; }
void Engine::truncate(int n) {
    if (!(n >= 0 && n <= p_->pos && (n == 0 || p_->pos - n <= 128))) throw std::runtime_error("truncate");
    p_->pos = n;
}
int Engine::prefill(const int* ids, int n) {
    if (p_->fail) throw std::runtime_error("injected prefill failure");
    if (p_->pos + n > p_->lim) throw std::runtime_error("prefill past the limit");
    for (int i = 0; i < n; i++) p_->rows[p_->base + p_->pos + i] = ids[i];
    p_->pos += n;
    return fake_next(p_->pos - 1, p_->d.logits);
}
// the token after the row at position q: greedy, or the draw (temperature > 0) as gptoss_sample.cuh makes it (exact
// softmax(l / T), seeded by (seed, position)); cands fills the card's result for the host filters
static int fake_row(Impl& m, int q, float temp, unsigned long long seed, bool draw = true, const std::unordered_map<int, float>* pen = nullptr) {
    const int g = fake_next(q, m.d.logits);
    if (temp <= 0) return g;
    m.lg.assign(VOCAB, -INFINITY);
    fake_next(q, m.lg.data());
    for (int i = 200019; i < VOCAB; i++) m.lg[i] = -INFINITY;
    if (pen)
        for (const auto& kv : *pen) m.lg[kv.first] -= kv.second;
    if (!draw) {   // the argmax of the (penalised) logits, lowest id of equal values
        int b = 0;
        for (int i = 1; i < 200019; i++)
            if (m.lg[i] > m.lg[b]) b = i;
        return b;
    }
    m.out.assign(4 + 128, 0.f);
    std::vector<int> o(200019);
    for (int i = 0; i < 200019; i++) o[i] = i;
    std::partial_sort(o.begin(), o.begin() + 64, o.end(), [&](int a, int b) { return m.lg[a] > m.lg[b] || (m.lg[a] == m.lg[b] && a < b); });
    const float M = m.lg[o[0]];
    double S = 0;
    for (int i = 0; i < 200019; i++) S += std::exp((m.lg[i] - M) / temp);
    m.out[0] = M;
    m.out[1] = (float)S;
    m.out[2] = 64;
    for (int j = 0; j < 64; j++) {
        m.out[4 + j] = m.lg[o[j]];
        memcpy(&m.out[4 + 64 + j], &o[j], 4);
    }
    std::mt19937_64 r(seed * 1000003ull + q);
    double u = std::uniform_real_distribution<double>(0, S)(r);
    for (int i = 0; i < 200019; i++)
        if ((u -= std::exp((m.lg[i] - M) / temp)) <= 0) return i;
    return g;
}
int Engine::decode(int token) {
    if (p_->fail) throw std::runtime_error("injected decode failure");
    if (p_->pos >= p_->lim) throw std::runtime_error("decode past the limit");
    p_->rows[p_->base + p_->pos] = token;
    p_->pos += 1;
    p_->steps++;
    return fake_row(*p_, p_->pos - 1, p_->temp, p_->seed, p_->draw, p_->pslot >= 0 ? &p_->pen[p_->pslot] : nullptr);
}
void Engine::pen_reset(int slot, const int* ids, const float* vals, int n) {
    Impl& m = *p_;
    if (slot < 0 || slot >= m.cfg.slots) throw std::runtime_error("pen_reset slot");
    m.pen.resize(m.cfg.slots);
    m.pen[slot].clear();
    for (int i = 0; i < n; i++) m.pen[slot][ids[i]] = vals[i];
    m.pen_resets++;
}
void Engine::set_sampling_penalty(int slot, int id, float val) {
    Impl& m = *p_;
    m.pslot = slot;
    if (slot < 0) return;
    if (slot != m.slot) throw std::runtime_error("penalty row of another slot than the bound one");
    m.pen.resize(m.cfg.slots);
    if (id >= 0) m.pen[slot][id] = val;
}
void Engine::step_rows(const StepRow* rows, int n, int* out) { step_rows_sampled(rows, nullptr, n, out); }
void Engine::step_rows_sampled(const StepRow* rows, const RowSamp* rs, int n, int* out) {
    Impl& m = *p_;
    if (m.fail) throw std::runtime_error("injected step failure");
    if (n < 1 || n > 16) throw std::runtime_error("step_rows n");
    for (int i = 0; i < n; i++)
        if (rows[i].slot < 0 || rows[i].slot >= m.cfg.slots || rows[i].base + rows[i].pos >= m.cfg.max_ctx) throw std::runtime_error("step_rows row");
    for (int i = 0; i < n; i++) m.rows[rows[i].base + rows[i].pos] = rows[i].token;
    m.pen.resize(m.cfg.slots);
    for (int i = 0; i < n; i++) {
        const bool pn = rs && rs[i].pen;
        if (pn && rs[i].pen_id >= 0) m.pen[rows[i].slot][rs[i].pen_id] = rs[i].pen_val;
        // (a row with draw = 0 inside a sampled step: the argmax; the fake's greedy result is the same without a penalty)
        out[i] = pn ? fake_row(m, rows[i].pos, 1.f, rs[i].seed, rs[i].temperature > 0.f, &m.pen[rows[i].slot])
                    : fake_row(m, rows[i].pos, rs ? rs[i].temperature : 0.f, rs ? rs[i].seed : 0);
    }
    for (int i = 1; i < n; i++) m.joint += rows[i].slot != rows[0].slot;
    m.steps++;
}
void Engine::set_sampling(float temperature, unsigned long long seed, bool, bool draw) { p_->temp = temperature; p_->seed = seed; p_->draw = draw; p_->pslot = -1; }
SampleCard Engine::sample_card(int, int) const { return SampleCard{p_->out.data(), p_->lg.data(), 0, VOCAB}; }
const Plan& Engine::plan() const { p_->plan.ngpu = 1; return p_->plan; }
void Engine::set_dump(const Dump& d) { p_->d = d; }
void Engine::decode_rows(const int* tokens, int R, int* out) {
    for (int i = 0; i < R; i++) out[i] = decode(tokens[i]);
    p_->steps -= R - 1;
}
// the fake rings: the saved position and the tokens of the 256 rows before it (the real rings' rows); a load at n must
// be 0..128 rows behind the saved point, and the bound slot's rows [n - 128, n) must hold those tokens (rows first)
static int g_saves = 0, g_loads = 0;
void Engine::ring_save(std::vector<unsigned char>& out) const {
    std::vector<int> v{p_->pos};
    for (int q = std::max(0, p_->pos - 256); q < p_->pos; q++) v.push_back(p_->rows[p_->base + q]);
    out.assign((unsigned char*)v.data(), (unsigned char*)(v.data() + v.size()));
    g_saves++;
}
void Engine::ring_load(const std::vector<unsigned char>& in, int n) {
    std::vector<int> v(in.size() / 4);
    memcpy(v.data(), in.data(), v.size() * 4);
    if (v.empty() || v[0] - n < 0 || v[0] - n > 128) throw std::runtime_error("ring_load of another position");
    const int s0 = std::max(0, v[0] - 256);
    for (int q = std::max(0, n - 128); q < n; q++)
        if (p_->rows[p_->base + q] != v[1 + q - s0]) throw std::runtime_error("ring_load over other rows");
    p_->pos = n;
    g_loads++;
}
size_t Engine::row_bytes() const { return 4; }
static long g_rows_saved = 0;
void Engine::rows_save(int src, int n, std::vector<unsigned char>& out) const {
    g_rows_saved += n;
    out.assign((unsigned char*)(p_->rows.data() + src), (unsigned char*)(p_->rows.data() + src + n));
}
void Engine::rows_load(const std::vector<unsigned char>& in, int dst, int n) {
    if (in.size() < (size_t)n * 4) throw std::runtime_error("rows_load");
    memcpy(p_->rows.data() + dst, in.data(), (size_t)n * 4);
}
double Engine::load_seconds() const { return 0; }
}  // namespace gptoss

extern "C" {
int qe_start_begin(void*, const int*, int, int, unsigned long long, int, const void*, int, const int*, int);
int qe_prefill_more_slot(void*, int, int);
int qe_start_finish_slot(void*, int);
int qe_prefill_pos(void*, int);
int qe_num_prefills(void*);
void qe_step_batch(void*, const int*, int, int*, int*);
void qe_release(void*, int);
void qe_resume(void*, int);
void qe_set_concurrent(void*, int);
unsigned* qe_mask_buffer(void*, int);
void qe_constrain(void*, int, int);
int qe_can_admit(void*, int);
int qe_num_slots(void*);
int qe_last_slot(void*);
void qe_destroy(void*);
void qe_reset(void*);
void* qe_create(const char*, const char*, int, const char*);
int qe_start3(void*, const int*, int, int, unsigned long long, int);
int qe_spec_step(void*, int*);
void qe_set_sampling(void*, float, float, int, float, unsigned long long);
void qe_set_penalty(void*, float, float, const int*, int);
void qe_set_limit(void*, int, int);
int qe_trim(void*, int, int);
long long qe_fault_count(void*);
int qe_last_reused(void*);
int qe_slot_fault(void*, int, char*, int);
int qe_last_copied(void*);
int qe_last_from_archive(void*);
int qe_num_archives(void*);
int qe_archive_more(void*, long long);
int qe_probe_cached(void*, const int*, int);
void qe_mem_guard(void*);
}

static int fails = 0;
static bool consistent_rows(gptoss::Handle& H) {
    for (size_t i = 0; i < H.s.size(); i++) {
        const auto& a = H.s[i];
        for (size_t j = 0; j < a.toks.size(); j++)
            if (gptoss::g_impl->rows[a.base + j] != a.toks[j]) return false;
    }
    return true;
}
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s: ", #c); printf(__VA_ARGS__); printf("\n"); } else printf("PASS %s\n", #c); } while (0)

int main() {
    setenv("GPTOSS_MAX_CTX", "4096", 1);
    setenv("GPTOSS_SLOTS", "1", 1);   // (the single-sequence checks; the slot pool below)
    void* h = qe_create("/nonexistent", "0", 4096, "");
    std::vector<int> ids(100, 5);
    CHECK(qe_start3(h, ids.data(), 0, 1, 0, 0) == -2, "n 0");
    std::vector<int> big(5000, 5);
    CHECK(qe_start3(h, big.data(), 5000, 1, 0, 0) == -3, "too long");
    ids[3] = gptoss::VOCAB;
    CHECK(qe_start3(h, ids.data(), 100, 1, 0, 0) == -2, "bad id");
    ids[3] = 5;
    // greedy
    qe_set_sampling(h, 0, 1, 0, 0, 1);
    int t = qe_start3(h, ids.data(), 100, 1, 0, 0);
    CHECK(t == (99 * 7) % 1000 + 10, "greedy first %d", t);
    int out[64];
    std::vector<int> g;
    for (int i = 0; i < 20; i++) { CHECK(qe_spec_step(h, out) == 1, "step"); g.push_back(out[0]); }
    // sampled, seeded: reproducible; top_k 1 = greedy; temperature small = greedy
    auto run = [&](float temp, int top_k, unsigned long long seed) {
        std::vector<int> r;
        qe_set_sampling(h, temp, 1, top_k, 0, seed);
        std::vector<int> other(50, 9);
        r.push_back(qe_start3(h, other.data(), 50, 1, 0, 0));   // (another prompt: a fresh prefill)
        r.push_back(qe_start3(h, ids.data(), 100, 1, 0, 0));
        for (int i = 0; i < 20; i++) { qe_spec_step(h, out); r.push_back(out[0]); }
        return r;
    };
    auto a = run(1.0f, 0, 42), b = run(1.0f, 0, 42), c = run(1.0f, 0, 43);
    CHECK(a == b, "seeded runs differ");
    int off = 0;
    for (size_t i = 1; i < a.size(); i++) off += a[i] != c[i];
    CHECK(off > 0, "another seed gave the same 21 tokens");
    // every sampled token is one of the two peaks (the rest sits 8 logits lower: p < 1e-3 each... over 200K tokens the
    // tail mass is not small, so temperature 0.25 makes it negligible)
    auto s = run(0.25f, 0, 7);
    int peak = 0;
    for (size_t i = 1; i < s.size(); i++) {
        int row = i == 1 ? 99 : 99 + (int)i - 1;
        int t0 = (row * 7) % 1000 + 10;
        peak += s[i] == t0 || s[i] == t0 + 1;
    }
    CHECK(peak == (int)s.size() - 1, "sampled tokens off the peaks: %d of %zu", peak, s.size() - 1);
    auto k1 = run(1.0f, 1, 3);
    std::vector<int> k1g(k1.begin() + 2, k1.end());
    CHECK(k1g == g, "top_k 1 != greedy");
    // filters on the cards' candidates: top_k 2 / min_p 0.3 keep only the two peaks (both drawn over 20 steps);
    // top_p 0.5 needs the tail (the peaks hold ~2 % of the mass): the full-logit fallback, still a valid token
    auto peaks = [&](const std::vector<int>& v, int& second) {
        int on = 0;
        second = 0;
        for (size_t i = 2; i < v.size(); i++) {
            const int t0 = ((99 + (int)i - 1) * 7) % 1000 + 10;
            on += v[i] == t0 || v[i] == t0 + 1;
            second += v[i] == t0 + 1;
        }
        return on;
    };
    int sec = 0;
    auto k2 = run(1.0f, 2, 5);
    CHECK(peaks(k2, sec) == 20 && sec > 0 && sec < 20, "top_k 2: %d on the peaks, %d second", peaks(k2, sec), sec);
    auto mp = [&](float min_p, float top_p) {
        std::vector<int> r;
        qe_set_sampling(h, 1.0f, top_p, 0, min_p, 11);
        std::vector<int> other(50, 9);
        r.push_back(qe_start3(h, other.data(), 50, 1, 0, 0));
        r.push_back(qe_start3(h, ids.data(), 100, 1, 0, 0));
        for (int i = 0; i < 20; i++) { qe_spec_step(h, out); r.push_back(out[0]); }
        return r;
    };
    auto m3 = mp(0.3f, 1.0f);
    CHECK(peaks(m3, sec) == 20 && sec > 0, "min_p 0.3: %d on the peaks, %d second", peaks(m3, sec), sec);
    auto m5 = mp(0.5f, 1.0f);
    CHECK(peaks(m5, sec) == 20 && sec == 0, "min_p 0.5: %d on the peaks, %d second", peaks(m5, sec), sec);
    auto tp = mp(0.0f, 0.5f);
    bool valid = true;
    for (int v : tp) valid = valid && v >= 0 && v < 200019;
    CHECK(valid && peaks(tp, sec) < 20, "top_p 0.5 (fallback): valid tokens, tail drawn");
    // limit
    qe_set_sampling(h, 0, 1, 0, 0, 1);
    qe_start3(h, ids.data(), 100, 1, 0, 0);
    qe_set_limit(h, 0, 3);
    int n = 0;
    for (int i = 0; i < 10; i++) n += qe_spec_step(h, out);
    CHECK(n == 3, "limit 3 gave %d", n);
    // trim: prompt 100 + 10 reply tokens; keep 105 (the reply's first 5, the 5th pending): the next start reuses the 104 rows
    qe_set_sampling(h, 0, 1, 0, 0, 1);
    int f0 = qe_start3(h, ids.data(), 100, 1, 0, 0);
    std::vector<int> seq = ids;
    seq.push_back(f0);
    for (int i = 0; i < 10; i++) { qe_spec_step(h, out); seq.push_back(out[0]); }
    CHECK(qe_trim(h, 0, 105) == 1, "trim");
    CHECK(gptoss::g_impl->pos == 104, "pos after trim %d", gptoss::g_impl->pos);
    std::vector<int> p2(seq.begin(), seq.begin() + 105);
    p2.push_back(77);
    qe_start3(h, p2.data(), (int)p2.size(), 1, 0, 0);
    CHECK(qe_last_reused(h) == 104, "reused after trim %d", qe_last_reused(h));
    // trim beyond the 128-row window: refused, the hold dropped
    std::vector<int> longp(400, 6);
    qe_start3(h, longp.data(), 400, 1, 0, 0);
    CHECK(qe_trim(h, 0, 200) == 0, "trim 200 rows back");
    // prompt-end snapshot: turn 1 = 100 prompt + 200 reply tokens (beyond the 128-row truncate window); turn 2 extends
    // the prompt only (the reply re-rendered differently): back to the snapshot, 100 rows reused
    qe_set_sampling(h, 0, 1, 0, 0, 1);
    std::vector<int> t1(100, 11);
    qe_start3(h, t1.data(), 100, 1, 0, 0);
    for (int i = 0; i < 200; i++) qe_spec_step(h, out);
    std::vector<int> t2 = t1;
    for (int i = 0; i < 30; i++) t2.push_back(12);
    int l0 = gptoss::g_loads;
    qe_start3(h, t2.data(), (int)t2.size(), 1, 0, 0);
    CHECK(qe_last_reused(h) == 100 && gptoss::g_loads == l0 + 1, "snapshot reuse %d loads %d", qe_last_reused(h), gptoss::g_loads - l0);
    // another conversation in between: the slot's snapshot is its own now (never used for t2); t2's sequence went to the
    // host store when the other conversation took the slot, so t2 comes back from there (all but the last row)
    std::vector<int> o(60, 13);
    qe_start3(h, o.data(), 60, 1, 0, 0);
    for (int i = 0; i < 200; i++) qe_spec_step(h, out);
    qe_start3(h, t2.data(), (int)t2.size(), 1, 0, 0);
    CHECK(qe_last_reused(h) == (int)t2.size() - 1 && qe_last_from_archive(h) == 1, "t2 from the store, not the stale snapshot: reused %d",
          qe_last_reused(h));
    // faults
    gptoss::g_impl->fail = true;
    std::vector<int> q(30, 8);
    CHECK(qe_start3(h, q.data(), 30, 1, 0, 0) == -2, "failed start");
    char buf[256];
    CHECK(qe_fault_count(h) == 1 && qe_slot_fault(h, 0, buf, 256) == 1 && strstr(buf, "injected"), "fault %lld", qe_fault_count(h));
    CHECK(qe_slot_fault(h, 0, buf, 256) == 0, "fault read twice");
    gptoss::g_impl->fail = false;
    qe_destroy(h);

    // ---- slot pool: 3 slots over 1024 rows, growth in steps of 64 (moves and evictions happen)
    setenv("GPTOSS_SLOTS", "3", 1);
    setenv("GPTOSS_MAX_CTX", "1024", 1);
    setenv("GPTOSS_SLOT_GROW", "64", 1);
    void* hp = qe_create("/nonexistent", "0", 1024, "");
    qe_set_concurrent(hp, 1);
    gptoss::Handle& H = *(gptoss::Handle*)hp;
    auto f = [](int row) { return (row * 7) % 1000 + 10; };   // the fake's greedy token after row `row`
    auto consistent = [&]() {   // every slot's held tokens sit in its rows; no two intervals overlap
        for (size_t i = 0; i < H.s.size(); i++) {
            const auto& a = H.s[i];
            for (size_t j = 0; j < a.toks.size(); j++)
                if (gptoss::g_impl->rows[a.base + j] != a.toks[j]) return false;
            for (size_t k = i + 1; k < H.s.size(); k++) {
                const auto& b = H.s[k];
                if (a.cap && b.cap && a.base < b.base + b.cap && b.base < a.base + a.cap) return false;
            }
        }
        return true;
    };
    CHECK(qe_num_slots(hp) == 3, "slots %d", qe_num_slots(hp));
    qe_set_sampling(hp, 0, 1, 0, 0, 1);
    std::vector<int> A(300), B(200), C(150);
    for (int i = 0; i < 300; i++) A[i] = 1000 + i;
    for (int i = 0; i < 200; i++) B[i] = 2000 + i;
    for (int i = 0; i < 150; i++) C[i] = 3000 + i;
    const int sa = qe_start_begin(hp, A.data(), 300, 1, 0, 0, nullptr, 0, nullptr, 0);
    CHECK(sa >= 0 && qe_prefill_more_slot(hp, sa, 128) == 0 && qe_prefill_pos(hp, sa) == 128, "chunk 1 of A");
    const int sb = qe_start_begin(hp, B.data(), 200, 1, 0, 0, nullptr, 0, nullptr, 0);
    CHECK(sb >= 0 && sb != sa && qe_num_prefills(hp) == 2, "B begins next to A (slot %d)", sb);
    CHECK(qe_start_begin(hp, C.data(), 150, 1, 0, 0, nullptr, 0, nullptr, 0) == -4, "a third prompt in flight: busy");
    CHECK(qe_prefill_more_slot(hp, sb, 1000) == 1, "B in one chunk");
    CHECK(qe_prefill_more_slot(hp, sa, 128) == 0 && qe_prefill_more_slot(hp, sa, 128) == 1, "A chunks 2, 3");
    CHECK(qe_start_finish_slot(hp, sa) == f(299) && qe_start_finish_slot(hp, sb) == f(199), "first tokens");
    int sl[2] = {sa, sb}, bo[128], bc[2];
    bool ok = true;
    for (int k = 0; k < 100; k++) {   // A grows 300 -> 400, B 200 -> 300: past their first intervals
        qe_step_batch(hp, sl, 2, bo, bc);
        if (ok && !(bc[0] == 1 && bc[1] == 1 && bo[0] == f(300 + k) && bo[64] == f(200 + k)))
            printf("step %d: cnt %d %d tok %d %d want %d %d (faults %lld)\n", k, bc[0], bc[1], bo[0], bo[64], f(300 + k), f(200 + k), qe_fault_count(hp));
        ok = ok && bc[0] == 1 && bc[1] == 1 && bo[0] == f(300 + k) && bo[64] == f(200 + k);
    }
    CHECK(ok, "batch steps follow each slot's own position");
    CHECK(consistent(), "rows consistent after growth (moves %ld)", H.moves);
    // C while A and B generate; then A released and a longer A' that extends A: back on A's slot, all rows reused
    CHECK(qe_start3(hp, C.data(), 150, 1, 0, 0) == f(149), "C first token");
    const int sc = qe_last_slot(hp);
    CHECK(sc != sa && sc != sb, "C on the third slot (%d)", sc);
    CHECK(qe_start3(hp, A.data(), 300, 1, 0, 0) == -1, "no free slot: -1 (retried)");
    qe_release(hp, sa);
    std::vector<int> A2(H.s[sa].toks);
    A2.push_back(H.s[sa].next);
    for (int i = 0; i < 20; i++) A2.push_back(4000 + i);
    const int r2 = qe_start3(hp, A2.data(), (int)A2.size(), 1, 0, 0);
    CHECK(r2 == f((int)A2.size() - 1) && qe_last_slot(hp) == sa && qe_last_reused(hp) == 400, "A' reuses A's 400 rows on its slot (%d, %d)",
          qe_last_slot(hp), qe_last_reused(hp));
    CHECK(consistent(), "rows consistent");
    // pressure: B and A' released, a 700-row prompt: room only by evicting the idle slots
    qe_release(hp, sb);
    qe_release(hp, sa);
    std::vector<int> D(700, 7);
    CHECK(qe_can_admit(hp, 700) == 1, "admit 700 next to C");
    const int td = qe_start3(hp, D.data(), 700, 1, 0, 0);
    const int sd = qe_last_slot(hp);
    CHECK(td == f(699) && sd != sc && H.evictions >= 1 && consistent(), "D placed (slot %d, evictions %ld)", sd, H.evictions);
    ok = true;
    int sl2[2] = {sc, sd};
    int grown = 0;
    for (int k = 0; k < 400; k++) {   // C and D grow until the pool is full: D or C fails with a fault, no overlap
        qe_step_batch(hp, sl2, 2, bo, bc);
        grown += bc[0] + bc[1];
        if (!bc[0] || !bc[1]) {
            break;
        }
    }
    CHECK(consistent() && qe_fault_count(hp) >= 1, "pool full: a fault, rows consistent (%d steps)", grown);
    {   // [gptoss-rel1-full] the slot that could not grow: "KV pool full" (the server ends it with length), rows kept
        const int fs = !bc[0] ? sc : sd;
        char fb[256];
        const size_t held = H.s[fs].toks.size();
        CHECK(qe_slot_fault(hp, fs, fb, 256) == 1 && !strncmp(fb, "KV pool full", 12) && !H.s[fs].gen && held > 0 && H.full_ends >= 1,
              "pool full: length end of slot %d keeps its %zu rows (%s)", fs, held, fb);
    }
    // joint steps: 3 slots (greedy, greedy with a repeating prompt for lookup drafts, sampled at temperature 1) in
    // qe_step_batch give each slot exactly its serial run's tokens
    {
        qe_reset(hp);
        std::vector<std::vector<int>> P(3);
        for (int i = 0; i < 120; i++) P[0].push_back(5000 + i);
        for (int i = 0; i < 120; i++) P[1].push_back(6000 + (i % 40));
        for (int i = 0; i < 90; i++) P[2].push_back(7000 + i);
        const float temps[3] = {0.f, 0.f, 1.f};
        auto serial = [&](int j, int want) {
            qe_set_sampling(hp, temps[j], 1, 0, 0, 21);
            std::vector<int> r{qe_start3(hp, P[j].data(), (int)P[j].size(), 1, 0, 0)};
            const int sl = qe_last_slot(hp);
            qe_resume(hp, sl);
            while ((int)r.size() < want) {
                const int c = qe_spec_step(hp, out);
                if (c <= 0) break;
                r.insert(r.end(), out, out + c);
            }
            r.resize(std::min<size_t>(r.size(), want));
            qe_release(hp, sl);
            qe_reset(hp);
            return r;
        };
        std::vector<std::vector<int>> ser;
        for (int j = 0; j < 3; j++) ser.push_back(serial(j, 40));
        std::vector<std::vector<int>> got(3);
        int sj[3];
        for (int j = 0; j < 3; j++) {
            qe_set_sampling(hp, temps[j], 1, 0, 0, 21);
            sj[j] = qe_start_begin(hp, P[j].data(), (int)P[j].size(), 1, 0, 0, nullptr, 0, nullptr, 0);
            got[j].push_back(qe_start_finish_slot(hp, sj[j]));
        }
        const long j0 = gptoss::g_impl->joint;
        int bo3[192], bc3[3];
        for (int k = 0; k < 40; k++) {
            qe_step_batch(hp, sj, 3, bo3, bc3);
            for (int j = 0; j < 3; j++) got[j].insert(got[j].end(), bo3 + j * 64, bo3 + j * 64 + bc3[j]);
        }
        bool same = true;
        for (int j = 0; j < 3; j++) {
            got[j].resize(std::min<size_t>(got[j].size(), 40));
            same = same && got[j] == ser[j];
        }
        CHECK(same && gptoss::g_impl->joint > j0 && H.batch_steps > 0, "joint steps == serial runs (joint rows %ld, lookup steps %ld)",
              gptoss::g_impl->joint - j0, H.lk_steps);
        CHECK(consistent(), "rows consistent after joint steps");
        for (int j = 0; j < 3; j++) qe_release(hp, sj[j]);
    }
    {   // review srv L6: a greedy prompt's lookup index grows with its prefill chunks (not inside its first decode step);
        // a sampled prompt keeps none
        qe_reset(hp);
        qe_set_sampling(hp, 0.f, 1, 0, 0, 3);
        std::vector<int> Q;
        for (int i = 0; i < 300; i++) Q.push_back(6000 + (i % 40));
        const int sq = qe_start_begin(hp, Q.data(), 300, 1, 0, 0, nullptr, 0, nullptr, 0);
        qe_prefill_more_slot(hp, sq, 100);
        const int k1 = H.s[sq].lk_known;
        qe_start_finish_slot(hp, sq);
        const int k2 = H.s[sq].lk_known;
        const size_t n2 = H.s[sq].lk.size();
        CHECK(k1 == 101 && k2 == 301 && n2 == 40, "lookup index built by the prefill chunks (%d, %d rows, %zu keys)", k1, k2, n2);
        // the same entries as a rebuild over the held prompt (the index the first decode step built before; review lows L4)
        auto rebuilt = [&](const gptoss::Slot& sl) {
            std::unordered_map<unsigned long long, int> m;
            for (int p = 6; p <= (int)sl.toks.size(); p++) {
                unsigned long long x = 1469598103934665603ull;
                for (int k = p - 6; k < p; k++) x = (x ^ (unsigned long long)sl.toks[k]) * 1099511628211ull;
                m[x] = p;
            }
            return m;
        };
        CHECK(H.s[sq].lk == rebuilt(H.s[sq]), "lookup index == a rebuild over the prompt");
        qe_release(hp, sq);
        {   // a prompt that extends the held one (rows reused): the index covers the reused rows too
            std::vector<int> Q2 = Q;
            for (int i = 0; i < 50; i++) Q2.push_back(7000 + i);
            const int s2 = qe_start_begin(hp, Q2.data(), (int)Q2.size(), 1, 0, 0, nullptr, 0, nullptr, 0);
            qe_start_finish_slot(hp, s2);
            CHECK(H.reused > 0 && H.s[s2].lk_known == (int)Q2.size() + 1 && H.s[s2].lk == rebuilt(H.s[s2]),
                  "reused prefix: lookup index == a rebuild (reused %d)", H.reused);
            qe_release(hp, s2);
        }
        qe_reset(hp);
        qe_set_sampling(hp, 1.f, 1, 0, 0, 3);
        const int ss = qe_start_begin(hp, Q.data(), 300, 1, 0, 0, nullptr, 0, nullptr, 0);
        qe_start_finish_slot(hp, ss);
        CHECK(H.s[ss].lk_known == 6 && H.s[ss].lk.empty(), "no lookup index for a sampled prompt");
        qe_release(hp, ss);
    }
    // grammar: a constrained slot picks inside mask row 0 (first token and steps; greedy and sampled)
    qe_reset(hp);
    for (int temp = 0; temp < 2; temp++) {
        qe_set_sampling(hp, temp ? 1.0f : 0.f, 1, 0, 0, 5);
        std::vector<int> G(40, 9);
        const int sg = qe_start_begin(hp, G.data(), 40, 1, 0, 0, nullptr, 0, nullptr, 0);
        unsigned* mb = qe_mask_buffer(hp, sg);
        auto allow_only = [&](int a, int b) {
            for (int w = 0; w < gptoss::MASK_WORDS; w++) mb[w] = 0;
            mb[a >> 5] |= 1u << (a & 31);
            mb[b >> 5] |= 1u << (b & 31);
        };
        allow_only(500, 501);
        qe_constrain(hp, sg, 1);
        const int t0 = qe_start_finish_slot(hp, sg);
        bool in = t0 == 500 || t0 == 501;
        qe_resume(hp, sg);
        for (int k = 0; k < 10; k++) {
            allow_only(600 + k, 700 + k);
            qe_spec_step(hp, out);
            in = in && (out[0] == 600 + k || out[0] == 700 + k);
        }
        CHECK(in, "constrained %s: every token inside the mask", temp ? "sampled" : "greedy");
        qe_release(hp, sg);
    }
    qe_destroy(hp);

    // ---- [srv-gptoss-cache] prefix sources: another slot, the system-part snapshot, the archive
    setenv("GPTOSS_SLOTS", "2", 1);
    setenv("GPTOSS_MAX_CTX", "4096", 1);
    setenv("GPTOSS_SLOT_GROW", "64", 1);
    void* hc = qe_create("/nonexistent", "0", 4096, "");
    qe_set_concurrent(hc, 1);
    gptoss::Handle& HC = *(gptoss::Handle*)hc;
    qe_set_sampling(hc, 0, 1, 0, 0, 1);
    std::vector<int> sys(300), u1, u2;
    for (int i = 0; i < 300; i++) sys[i] = 8000 + i;
    u1 = sys;
    for (int i = 0; i < 100; i++) u1.push_back(9000 + i);
    u2 = sys;
    for (int i = 0; i < 80; i++) u2.push_back(9500 + i);
    // 1. conversation 1 with hint 300: its prefill stops there once and keeps the system part in the store
    const int c1 = qe_start_begin(hc, u1.data(), (int)u1.size(), 1, 0, 300, nullptr, 0, nullptr, 0);
    qe_start_finish_slot(hc, c1);
    CHECK(HC.hint_saves == 1 && qe_num_archives(hc) == 1, "system-part snapshot saved (%ld, %d entries)", HC.hint_saves, qe_num_archives(hc));
    int sl1[1] = {c1}, b1[64], n1[1];
    for (int k = 0; k < 150; k++) qe_step_batch(hc, sl1, 1, b1, n1);   // c1 runs 150 tokens past its prompt (rings beyond 128)
    // 2. conversation 2 (same system part, another user turn) on the other slot while c1 generates: the 300 rows come
    // from c1's slot (copy in VRAM), the rings from the snapshot
    const int c2 = qe_start_begin(hc, u2.data(), (int)u2.size(), 1, 0, 300, nullptr, 0, nullptr, 0);
    const int f2 = qe_start_finish_slot(hc, c2);
    CHECK(c2 != c1 && qe_last_reused(hc) == 300 && qe_last_copied(hc) == 300 && f2 == f((int)u2.size() - 1),
          "conversation 2 starts from the system part (slot %d, reused %d, copied %d)", c2, qe_last_reused(hc), qe_last_copied(hc));
    CHECK(HC.hint_saves == 1, "no second snapshot of the same system part (%ld)", HC.hint_saves);
    // 3. a third prompt extending c1's live sequence (c1 still generating): the rows and the live rings of slot c1
    qe_release(hc, c2);
    std::vector<int> u3(HC.s[c1].toks);
    for (int i = 0; i < 30; i++) u3.push_back(9900 + i);
    const int probe = qe_probe_cached(hc, u3.data(), (int)u3.size());
    const int c3 = qe_start_begin(hc, u3.data(), (int)u3.size(), 1, 0, 300, nullptr, 0, nullptr, 0);
    qe_start_finish_slot(hc, c3);
    CHECK(c3 == c2 && qe_last_reused(hc) == (int)u3.size() - 30 && probe == qe_last_reused(hc) && qe_last_copied(hc) > 0,
          "a fork of a generating conversation: its rows + live rings (reused %d of %zu, probe %d)", qe_last_reused(hc), u3.size(), probe);
    // 4. archive: both released, idle archiving stores them; a long unrelated prompt evicts both slots; then c1's
    // conversation comes back (its sequence + a new turn): everything from the store
    qe_release(hc, c1);
    qe_release(hc, c3);
    std::vector<int> back(HC.s[c1].toks);
    for (int i = 0; i < 20; i++) back.push_back(9700 + i);
    int am = 0, guard = 0;
    const long rs0 = gptoss::g_rows_saved;
    const long held2 = (long)HC.s[c1].toks.size() + (long)HC.s[c3].toks.size();
    while ((am = qe_archive_more(hc, 400)) == 0 && guard++ < 32) {}   // (256 rows per call: in parts)
    CHECK(am == 1 && HC.archived >= 2 && guard >= 2, "idle archiving in parts (%d, %ld archived, %d calls)", am, HC.archived, guard);
    CHECK(gptoss::g_rows_saved - rs0 == held2 - 2 * 300, "archives copy only the rows past the stored system part (%ld of %ld)",
          gptoss::g_rows_saved - rs0, held2);
    std::vector<int> big2(3900, 77);
    qe_start3(hc, big2.data(), (int)big2.size(), 1, 0, 0);
    CHECK(HC.evictions >= 1, "evicted for the long prompt (%ld)", HC.evictions);
    qe_release(hc, qe_last_slot(hc));
    std::vector<int> big3(3900, 78);   // the other slot too
    qe_start3(hc, big3.data(), (int)big3.size(), 1, 0, 0);
    qe_release(hc, qe_last_slot(hc));
    const int fb = qe_start3(hc, back.data(), (int)back.size(), 1, 0, 0);
    CHECK(qe_last_reused(hc) == (int)back.size() - 20 && qe_last_from_archive(hc) == 1 && fb == f((int)back.size() - 1),
          "conversation back from the store (reused %d of %zu, from archive %d)", qe_last_reused(hc), back.size(), qe_last_from_archive(hc));
    CHECK(consistent_rows(HC), "rows consistent after the cache paths");
    qe_destroy(hc);

    // ---- review fixes
    {   // H2: the store counts each blob once, before and after a slot lets go of the parts it shares
        setenv("GPTOSS_MAX_CTX", "4096", 1);
        setenv("GPTOSS_SLOTS", "1", 1);
        void* hr = qe_create("/x", "0", 4096, "");
        qe_set_concurrent(hr, 1);
        gptoss::Handle& R = *(gptoss::Handle*)hr;
        std::vector<int> a(600), b(300);
        for (int i = 0; i < 600; i++) a[i] = 1000 + i;
        for (int i = 0; i < 300; i++) b[i] = 5000 + i;
        int sr = qe_start_begin(hr, a.data(), 600, 1, 0, 0, nullptr, 0, nullptr, 0);
        qe_start_finish_slot(hr, sr);
        for (int i = 0; i < 5; i++) qe_spec_step(hr, out);
        qe_release(hr, sr);
        qe_archive_more(hr, 1 << 30);
        const size_t b1 = R.store_bytes;
        sr = qe_start_begin(hr, b.data(), 300, 1, 0, 0, nullptr, 0, nullptr, 0);
        qe_start_finish_slot(hr, sr);
        qe_release(hr, sr);
        CHECK(R.store_bytes == b1 && b1 == R.store[0].bytes(), "H2 store bytes stable when the slot leaves the shared parts (%zu, %zu)", b1,
              R.store_bytes);
        setenv("QE_KV_ARCHIVE_GB", "0.000000001", 1);
        qe_mem_guard(hr);
        unsetenv("QE_KV_ARCHIVE_GB");
        CHECK(R.store.empty() && R.store_bytes == 0, "H2 store empty after the drop (%zu entries, %lld bytes)", R.store.size(),
              (long long)R.store_bytes);
        qe_destroy(hr);
    }
    {   // H1: compaction inside a joint step keeps the room of the stepping slots (no write into a neighbour)
        setenv("GPTOSS_MAX_CTX", "1024", 1);
        setenv("GPTOSS_SLOTS", "3", 1);
        setenv("GPTOSS_SLOT_GROW", "64", 1);
        void* hr = qe_create("/x", "0", 1024, "");
        qe_set_concurrent(hr, 1);
        gptoss::Handle& R = *(gptoss::Handle*)hr;
        int sk[3];
        for (int k = 0; k < 3; k++) {
            std::vector<int> pr(199);
            for (int i = 0; i < 199; i++) pr[i] = 100 * k + i + 20;
            sk[k] = qe_start_begin(hr, pr.data(), 199, 1, 0, 0, nullptr, 0, nullptr, 0);
            qe_start_finish_slot(hr, sk[k]);
        }
        for (int k = 0; k < 3; k++) {
            qe_resume(hr, sk[k]);
            qe_spec_step(hr, out);
        }
        auto place = [&](int k, int base, int cap) {   // slot k's rows to [base, ..) in the fake
            gptoss::Slot& o = R.s[k];
            o.base = base;
            o.cap = cap;
            for (size_t j = 0; j < o.toks.size(); j++) gptoss::g_impl->rows[base + j] = o.toks[j];
        };
        std::fill(gptoss::g_impl->rows.begin(), gptoss::g_impl->rows.end(), -1);
        place(1, 0, 200);     // B full
        place(0, 200, 210);   // A with room
        place(2, 410, 614);   // C idle after it
        qe_release(hr, sk[2]);
        int both[2] = {sk[0], sk[1]}, bo2[128], bc2[2];
        qe_step_batch(hr, both, 2, bo2, bc2);
        CHECK(consistent_rows(R) && bc2[0] == 1 && bc2[1] == 1, "H1 joint step after a compaction: rows consistent (compactions %ld, cnt %d %d)",
              R.compactions, bc2[0], bc2[1]);
        qe_destroy(hr);
    }
    {   // H5: rings written past the held end (drafts / a longer earlier sequence) block a reuse > 128 rows behind them
        setenv("GPTOSS_MAX_CTX", "4096", 1);
        setenv("GPTOSS_SLOTS", "1", 1);
        void* hr = qe_create("/x", "0", 4096, "");
        gptoss::Handle& R = *(gptoss::Handle*)hr;
        std::vector<int> pr(300, 3);
        qe_start3(hr, pr.data(), 300, 1, 0, 0);
        for (int i = 0; i < 100; i++) qe_spec_step(hr, out);   // held 399
        R.s[0].hi = (int)R.s[0].toks.size() + 15;               // as after a step with 15 rejected drafts
        std::vector<int> p2(R.s[0].toks.begin(), R.s[0].toks.begin() + 280);
        p2.push_back(4);
        bool snapx = false;
        const int ru = gptoss::srv_reuse(R, R.s[0], p2.data(), (int)p2.size(), &snapx);
        CHECK(ru == 0 || snapx, "H5 no live-ring reuse 134 rows behind the rings' end (reuse %d, snapshot %d)", ru, snapx);
        std::vector<int> p3(R.s[0].toks.begin(), R.s[0].toks.begin() + 300);
        p3.push_back(4);
        CHECK(gptoss::srv_reuse(R, R.s[0], p3.data(), (int)p3.size(), &snapx) == 300, "H5 reuse 114 rows behind the rings' end");
        qe_destroy(hr);
    }
    {   // [gptoss-rel2-penalty] presence / frequency penalties (qe_set_penalty). The fake model's row r peaks at
        // t(r) = (7 r) % 1000 + 10 (logit 3) and t(r) + 1 (logit 2).
        setenv("GPTOSS_MAX_CTX", "4096", 1);
        setenv("GPTOSS_SLOTS", "2", 1);
        void* hq = qe_create("/x", "0", 4096, "");
        gptoss::Handle& Q = *(gptoss::Handle*)hq;
        auto peak = [](int row) { return (row * 7) % 1000 + 10; };
        std::vector<int> pr(100, 5), seedv;
        for (int r = 99; r < 99 + 30; r++) seedv.push_back(peak(r));
        // greedy with a presence penalty seeded with the peaks: every token (the first one from the prefill's logits
        // on the host too) is the second peak
        qe_set_sampling(hq, 1.0f, 1, 1, 0, 3);
        qe_set_penalty(hq, 5.0f, 0.f, seedv.data(), (int)seedv.size());
        std::vector<int> r1{qe_start3(hq, pr.data(), 100, 1, 0, 0)};
        const int s1 = qe_last_slot(hq);
        for (int i = 0; i < 20; i++) { qe_spec_step(hq, out); r1.push_back(out[0]); }
        bool second = true;
        for (size_t i = 0; i < r1.size(); i++) second = second && r1[i] == peak(99 + (int)i) + 1;
        CHECK(second, "penalty: presence 5 on the seeded peaks picks the second peak at all %zu steps (first %d)", r1.size(), r1[0]);
        // the counts: every token of the reply once more than its seed count; the cards' row = frequency * c + presence
        qe_set_sampling(hq, 1.0f, 1, 1, 0, 3);
        const int sd[3] = {peak(99), peak(99), 77};   // (a seed token counts once)
        qe_set_penalty(hq, 0.5f, 0.25f, sd, 3);
        std::vector<int> r2{qe_start3(hq, pr.data(), 100, 1, 0, 0)};
        const int s2 = qe_last_slot(hq);
        for (int i = 0; i < 20; i++) { qe_spec_step(hq, out); r2.push_back(out[0]); }
        // peak(99) carries 0.5 + 0.25 = 0.75 < 1: still the first peak; then counted twice
        bool cnt_ok = r2[0] == peak(99) && Q.s[s2].sp.cnt[peak(99)] == 2 && Q.s[s2].sp.cnt[77] == 1;
        for (size_t i = 1; i < r2.size(); i++) cnt_ok = cnt_ok && Q.s[s2].sp.cnt[r2[i]] == 1;
        CHECK(cnt_ok, "penalty: counts = seed (once each) + the reply's tokens");
        bool row_ok = gptoss::g_impl->pen[s2][peak(99)] == 0.25f * 2 + 0.5f && gptoss::g_impl->pen[s2][77] == 0.25f + 0.5f;
        for (size_t i = 1; i + 1 < r2.size(); i++) row_ok = row_ok && gptoss::g_impl->pen[s2][r2[i]] == 0.25f + 0.5f;
        CHECK(row_ok, "penalty: the cards' row holds frequency * c + presence for every token fed");
        CHECK(gptoss::g_impl->pen[s2].count(r2.back()) == 0, "penalty: the newest token goes to the cards with its step");
        // qe_set_sampling turns the penalty off; the plain greedy reply is back
        const long resets = gptoss::g_impl->pen_resets;
        qe_set_sampling(hq, 1.0f, 1, 1, 0, 3);
        std::vector<int> r3{qe_start3(hq, pr.data(), 100, 1, 0, 0)};
        for (int i = 0; i < 20; i++) { qe_spec_step(hq, out); r3.push_back(out[0]); }
        bool first = true;
        for (size_t i = 0; i < r3.size(); i++) first = first && r3[i] == peak(99 + (int)i);
        CHECK(first && gptoss::g_impl->pen_resets == resets && gptoss::g_impl->pslot < 0, "penalty: off after qe_set_sampling");
        // a greedy reply (temperature 0) ignores a penalty (sampled replies only: the server sends temperature 1, top_k 1)
        qe_set_sampling(hq, 0.f, 1, 0, 0, 3);
        qe_set_penalty(hq, 5.0f, 0.f, seedv.data(), (int)seedv.size());
        std::vector<int> r4{qe_start3(hq, pr.data(), 100, 1, 0, 0)};
        for (int i = 0; i < 20; i++) { qe_spec_step(hq, out); r4.push_back(out[0]); }
        CHECK(r4 == r3, "penalty: temperature 0 stays the plain greedy reply");
        // two penalised replies stepped together: each slot its own row and counts, tokens as alone
        qe_reset(hq);
        qe_set_concurrent(hq, 1);
        std::vector<int> pa(100, 5), pb(140, 6), sb;
        for (int r = 139; r < 139 + 30; r++) sb.push_back(peak(r));
        int sj[2];
        std::vector<std::vector<int>> got(2);
        qe_set_sampling(hq, 1.0f, 1, 1, 0, 3);
        sj[0] = qe_start_begin(hq, pa.data(), 100, 1, 0, 0, nullptr, 0, nullptr, 0);
        qe_set_sampling(hq, 1.0f, 1, 1, 0, 3);
        qe_set_penalty(hq, 5.0f, 0.f, seedv.data(), (int)seedv.size());
        got[0].push_back(qe_start_finish_slot(hq, sj[0]));
        qe_set_sampling(hq, 1.0f, 1, 1, 0, 4);
        sj[1] = qe_start_begin(hq, pb.data(), 140, 1, 0, 0, nullptr, 0, nullptr, 0);
        qe_set_sampling(hq, 1.0f, 1, 1, 0, 4);
        qe_set_penalty(hq, 5.0f, 0.f, sb.data(), (int)sb.size());
        got[1].push_back(qe_start_finish_slot(hq, sj[1]));
        const long j0 = gptoss::g_impl->joint;
        int bo[128], bc[2];
        for (int k = 0; k < 20; k++) {
            qe_step_batch(hq, sj, 2, bo, bc);
            for (int j = 0; j < 2; j++) got[j].insert(got[j].end(), bo + j * 64, bo + j * 64 + bc[j]);
        }
        bool both = got[0] == r1;
        for (size_t i = 0; i < got[1].size(); i++) both = both && got[1][i] == peak(139 + (int)i) + 1;
        CHECK(both && got[1].size() == 21 && gptoss::g_impl->joint - j0 == 20, "penalty: two penalised slots in joint steps, each its own second peaks (%ld joint steps)",
              gptoss::g_impl->joint - j0);
        CHECK(Q.s[sj[0]].sp.cnt.count(got[1][3]) == 0 && Q.s[sj[1]].sp.cnt.count(got[0][3]) == 0, "penalty: counts are per sequence");
        qe_destroy(hq);
    }
    {   // [gptoss-rel3-pentrim] qe_trim takes the trimmed tokens out of the penalty counts (review rel2 L4). The fake
        // model's reply rows are all different tokens, so a presence penalty bites only on a token counted wrongly:
        // after a trim the continuation must be the untrimmed reply again (a stale count would pick the second peak).
        setenv("GPTOSS_MAX_CTX", "4096", 1);
        setenv("GPTOSS_SLOTS", "2", 1);
        void* ht = qe_create("/x", "0", 4096, "");
        gptoss::Handle& T = *(gptoss::Handle*)ht;
        auto peak = [](int row) { return (row * 7) % 1000 + 10; };
        std::vector<int> pr(100, 5);
        const int sd[2] = {3, 4};   // seed tokens the fake never emits (counted once each, kept by a trim)
        auto ref_cnt = [&](const gptoss::Slot& s) {   // seed + the reply tokens held (toks after the prompt + next)
            std::unordered_map<int, int> c{{3, 1}, {4, 1}};
            for (size_t q = 100; q < s.toks.size(); q++) c[s.toks[q]]++;
            if (s.next >= 0) c[s.next]++;
            return c;
        };
        qe_set_sampling(ht, 1.0f, 1, 1, 0, 3);
        qe_set_penalty(ht, 5.0f, 0.5f, sd, 2);
        std::vector<int> ra{qe_start3(ht, pr.data(), 100, 1, 0, 0)};
        const int st = qe_last_slot(ht);
        for (int i = 0; i < 19; i++) { qe_spec_step(ht, out); ra.push_back(out[0]); }
        bool plain = true;
        for (size_t i = 0; i < ra.size(); i++) plain = plain && ra[i] == peak(99 + (int)i);
        CHECK(plain && T.s[st].sp.cnt == ref_cnt(T.s[st]), "pentrim: the untrimmed penalised reply (20 distinct first peaks), counts = seed + reply");
        // a stop trim mid-reply: keep the prompt + 5 reply tokens (the 5th pending), then continue
        CHECK(qe_trim(ht, st, 105) == 1, "pentrim: trim 15 tokens back");
        CHECK(T.s[st].sp.cnt == ref_cnt(T.s[st]) && T.s[st].sp.cnt.size() == 7 && !T.s[st].pen_synced,
              "pentrim: counts after the trim = seed + the 5 reply tokens held (%zu ids)", T.s[st].sp.cnt.size());
        std::vector<int> rb(ra.begin(), ra.begin() + 5);
        for (int i = 0; i < 15; i++) { qe_spec_step(ht, out); rb.push_back(out[0]); }
        CHECK(rb == ra, "pentrim: the continuation after the trim = the untrimmed reply (CPU reference)");
        bool row_ok = gptoss::g_impl->pen[st].count(ra[10]) && gptoss::g_impl->pen[st][ra[10]] == 0.5f + 5.f;
        for (const auto& kv : gptoss::g_impl->pen[st])
            row_ok = row_ok && kv.second == (T.s[st].sp.cnt.count(kv.first) ? T.s[st].sp.pen_val(kv.first) : 0.f);
        CHECK(row_ok && T.s[st].sp.cnt == ref_cnt(T.s[st]), "pentrim: the cards' row made again from the rebuilt counts");
        // a trim inside the prompt: only the seed is left
        CHECK(qe_trim(ht, st, 90) == 1 && T.s[st].sp.cnt.size() == 2 && T.s[st].sp.cnt[3] == 1, "pentrim: a trim into the prompt keeps the seed only");
        // tokens generated after a trim into the prompt are reply tokens: a second trim keeps those it holds (review rel3 L1)
        std::vector<int> g3;
        for (int i = 0; i < 5; i++) { qe_spec_step(ht, out); g3.push_back(out[0]); }
        CHECK(qe_trim(ht, st, 93) == 1, "pentrim: second trim");
        {
            std::unordered_map<int, int> c3{{3, 1}, {4, 1}};
            for (int i = 0; i < 3; i++) c3[g3[i]]++;
            CHECK(T.s[st].sp.cnt == c3, "pentrim: after a trim into the prompt, 5 steps and a trim to 3 of them: seed + those 3");
        }
        // the next turn (the server sends the conversation again with its own seed): counts = its seed + its reply
        std::vector<int> p2(T.s[st].toks);
        p2.push_back(T.s[st].next);
        p2.push_back(6);
        qe_set_sampling(ht, 1.0f, 1, 1, 0, 3);
        const int sd2[1] = {peak((int)p2.size() - 1)};
        qe_set_penalty(ht, 5.0f, 0.f, sd2, 1);
        std::vector<int> rc{qe_start3(ht, p2.data(), (int)p2.size(), 1, 0, 0)};
        for (int i = 0; i < 5; i++) { qe_spec_step(ht, out); rc.push_back(out[0]); }
        bool nt = rc[0] == peak((int)p2.size() - 1) + 1;
        for (size_t i = 1; i < rc.size(); i++) nt = nt && rc[i] == peak((int)p2.size() - 1 + (int)i);
        std::unordered_map<int, int> c2{{sd2[0], 1}};
        for (int t : rc) c2[t]++;
        CHECK(nt && T.s[st].sp.cnt == c2, "pentrim: the next turn's reply and counts (its seed + its reply only)");
        // a trim beyond the ring window drops the sequence and every reply count
        for (int i = 0; i < 140; i++) qe_spec_step(ht, out);
        CHECK(qe_trim(ht, st, (int)p2.size() + 1) == 0 && T.s[st].sp.cnt.size() == 1, "pentrim: a dropped sequence keeps the seed only");
        qe_destroy(ht);
    }
    printf("%d failed\n", fails);
    return fails ? 1 : 0;
}
