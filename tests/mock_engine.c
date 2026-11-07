// CPU mock of the engine C API (server/qe_capi.h, TIER 0, plus TIER 1 unless -DMOCK_TIER0) for the
// server's end-to-end HTTP test (tests/test_server_mock.py). No GPU. Replies are scripted:
//   QE_MOCK_SCRIPT  file, one reply per line as comma-separated token ids (used in turn, cyclic)
//   QE_MOCK_LOG     file: every start appends "P <n> <reused> <ids...>", every set_sampling "S t p k"
// Prefix reuse is modelled like the engine's live sequence: reused = common prefix with the
// previous prompt + its generated tokens (at most n - 1).
//   gcc -O2 -shared -fPIC -o mock.so tests/mock_engine.c [-DMOCK_TIER0] [-DMOCK_GRAMMAR]
// -DMOCK_GRAMMAR adds the grammar tier (TIER 8) the forced close of <think> uses (tests/test_think_budget.py): a
// constrained step emits, per row, the one token its mask allows (else the script's), verifying the caller's drafts;
// after an emitted </think> (248069) the script goes on after its own </think> (the model moves on to its answer).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int max_t, k, reused, nseq, cap;
    int* seq;
    int** replies;
    int* rlen;
    int nrep, next, cur, rpos, limit;
    FILE* log;
#ifdef MOCK_GRAMMAR
    unsigned* mask;
    int io[32];
    int con, ncon, stepc;
#endif
} Mock;

static void push(Mock* m, int t) {
    if (m->nseq == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 4096;
        m->seq = realloc(m->seq, sizeof(int) * m->cap);
    }
    m->seq[m->nseq++] = t;
}

void* qe_create(const char* model_dir, const char* devices, int max_t, const char* split) {
    (void)model_dir, (void)devices, (void)split;
    Mock* m = calloc(1, sizeof(Mock));
    m->max_t = max_t > 0 ? max_t : 262144;
    const char* sp = getenv("QE_MOCK_SCRIPT");
    FILE* f = sp ? fopen(sp, "r") : NULL;
    if (!f) return NULL;
    char* line = NULL;
    size_t sz = 0;
    while (getline(&line, &sz, f) > 0) {
        int n = 0, cap = 64, *r = malloc(sizeof(int) * cap);
        for (char* p = strtok(line, ",\n"); p; p = strtok(NULL, ",\n")) {
            if (n == cap) r = realloc(r, sizeof(int) * (cap *= 2));
            r[n++] = atoi(p);
        }
        if (!n) { free(r); continue; }
        m->replies = realloc(m->replies, sizeof(int*) * (m->nrep + 1));
        m->rlen = realloc(m->rlen, sizeof(int) * (m->nrep + 1));
        m->replies[m->nrep] = r;
        m->rlen[m->nrep++] = n;
    }
    free(line);
    fclose(f);
    const char* lp = getenv("QE_MOCK_LOG");
    m->log = lp ? fopen(lp, "a") : NULL;
    return m->nrep ? m : NULL;
}
void qe_destroy(void* e) { free(e); }
int qe_max_context(void* e) { return ((Mock*)e)->max_t; }
void qe_set_sampling(void* e, float t, float p, int k, float mp, unsigned long long seed) {
    Mock* m = e;
    (void)mp, (void)seed;
    if (m->log) fprintf(m->log, "S %.3f %.3f %d\n", t, p, k), fflush(m->log);
}
int qe_start3(void* e, const int* ids, int n, int k, unsigned long long affinity, int hint) {
    Mock* m = e;
    (void)affinity, (void)hint;
    if (n <= 0 || n > m->max_t) return -1;
    int l = 0;
    while (l < m->nseq && l < n && m->seq[l] == ids[l]) l++;
    m->reused = l < n ? l : n - 1;
    m->nseq = 0;
    for (int i = 0; i < n; i++) push(m, ids[i]);
    if (m->log) {
        fprintf(m->log, "P %d %d", n, m->reused);
        for (int i = 0; i < n; i++) fprintf(m->log, " %d", ids[i]);
        fprintf(m->log, "\n");
        fflush(m->log);
    }
    m->k = k;
    m->limit = 0;
    m->cur = m->next++ % m->nrep;
    m->rpos = 1;
    push(m, m->replies[m->cur][0]);
#ifdef MOCK_GRAMMAR
    m->stepc = 1;
#endif
    return m->replies[m->cur][0];
}
#ifdef MOCK_GRAMMAR
#define MOCK_WORDS ((248320 + 31) / 32)
static int one_bit(const unsigned* row) {  // the only allowed token of a mask row, else -1
    int t = -1;
    for (int w = 0; w < MOCK_WORDS; w++)
        if (row[w]) {
            if (t >= 0 || (row[w] & (row[w] - 1))) return -1;
            t = w * 32 + __builtin_ctz(row[w]);
        }
    return t;
}
static int con_step(Mock* m, int* out) {
    const int* r = m->replies[m->cur];
    int kv = m->io[16] < 0 ? 0 : m->io[16] > 7 ? 7 : m->io[16];
    if (m->k <= 0) kv = 0;
    if (m->limit > 0 && kv + 1 > m->limit) kv = m->limit - 1;
    int c = 0;
    for (int row = 0; row <= kv; row++) {
        int t = one_bit(m->mask + (size_t)row * MOCK_WORDS);
        if (t < 0) t = m->rpos < m->rlen[m->cur] ? r[m->rpos++] : r[m->rlen[m->cur] - 1];
        out[c++] = t;
        push(m, t);
        if (t == 248069)  // forced </think>: the scripted reply continues after its own
            for (int i = m->rpos; i < m->rlen[m->cur]; i++)
                if (r[i] == 248069) { m->rpos = i + 1; break; }
        if (row < kv && t != m->io[17 + row]) break;  // a draft the row did not confirm
    }
    m->io[0] = 0;
    m->ncon++;
    m->stepc = c;
    if (m->limit > 0) m->limit -= c;
    return c;
}
#endif
int qe_spec_step(void* e, int* out) {
    Mock* m = e;
    const char* step_ms = getenv("QE_MOCK_STEP_MS");  // (a reply that takes time, for tests of what waits behind it)
    if (step_ms) usleep(atoi(step_ms) * 1000);
#ifdef MOCK_GRAMMAR
    if (m->con) return con_step(m, out);
#endif
    const int* r = m->replies[m->cur];
    int want = m->k > 0 ? 3 : 1;  // multi-token steps like accepted MTP drafts
    if (m->limit > 0 && want > m->limit) want = m->limit;
    int c = 0;
    for (; c < want; c++) {
        int t = m->rpos < m->rlen[m->cur] ? r[m->rpos++] : r[m->rlen[m->cur] - 1];
        out[c] = t;
        push(m, t);
    }
    if (m->limit > 0) m->limit -= c;
    return c;
}
int qe_last_reused(void* e) { return ((Mock*)e)->reused; }
int qe_last_slot(void* e) { (void)e; return 0; }
void qe_release(void* e, int slot) {
    (void)slot;
#ifdef MOCK_GRAMMAR
    Mock* m = e;
    m->con = 0;
    if (m->log && m->ncon) fprintf(m->log, "C %d\n", m->ncon), fflush(m->log);
    m->ncon = 0;
#else
    (void)e;
#endif
}

#ifndef MOCK_TIER0
int qe_vocab(void* e) { (void)e; return 248320; }
int qe_num_slots(void* e) { (void)e; return 1; }
long long qe_pool_tokens(void* e) { const char* v = getenv("QE_MOCK_POOL_NOW"); return v ? atoll(v) : ((Mock*)e)->max_t; }  // (QE_MOCK_POOL_NOW: the pool has shrunk to this)
int qe_kv_in_ram(void* e) { (void)e; const char* v = getenv("QE_MOCK_KV_RAM"); return v && *v == '1'; }
int qe_can_admit(void* e, int n) { return n <= ((Mock*)e)->max_t && !getenv("QE_MOCK_NO_ADMIT"); }  // (QE_MOCK_NO_ADMIT: a pool shrunk under VRAM pressure)
void qe_set_limit(void* e, int slot, int tokens) { (void)slot; ((Mock*)e)->limit = tokens; }
double qe_warmup(void* e, int modes) { (void)e, (void)modes; return 0.0; }
void qe_reset(void* e) { ((Mock*)e)->nseq = 0; }
#endif

#ifdef MOCK_GRAMMAR
unsigned* qe_mask_buffer(void* e, int slot) {
    Mock* m = e;
    (void)slot;
    if (!m->mask) m->mask = calloc((size_t)8 * MOCK_WORDS, 4);
    return m->mask;
}
int* qe_spec_io(void* e, int slot) { (void)slot; return ((Mock*)e)->io; }
void qe_constrain(void* e, int slot, int on) {
    Mock* m = e;
    (void)slot;
    m->con = on != 0;
    if (on) m->io[0] = m->io[16] = 0;
    if (m->log) fprintf(m->log, "K %d\n", on), fflush(m->log);
}
int qe_constrained(void* e, int slot) { (void)slot; return ((Mock*)e)->con; }
// logprobs: raw logits of a row, fixed pattern around the token the row emitted (tests/test_server_mock.py check_logprobs)
void qe_logits_capture(void* e, int slot, int on) {
    Mock* m = e;
    (void)slot;
    if (m->log) fprintf(m->log, "L %d\n", on), fflush(m->log);
}
int qe_logits_read(void* e, int slot, int row, float* out) {
    Mock* m = e;
    (void)slot;
    if (row < 0 || row >= m->stepc) return 0;
    const int t = m->seq[m->nseq - m->stepc + row];
    for (int i = 0; i < 248320; i++) out[i] = -50.f;
    for (int i = 0; i < 30; i++) out[(t + i) % 248320] = 3.f - 0.5f * i;
    const char* lp = getenv("QE_MOCK_LP");  // nan: one NaN logit; allnan: no finite logit; norow: the read fails
    if (lp && !strcmp(lp, "norow")) return 0;
    if (lp && !strcmp(lp, "nan")) out[(t + 40) % 248320] = 0.f / 0.f;
    if (lp && !strcmp(lp, "allnan")) for (int i = 0; i < 248320; i++) out[i] = 0.f / 0.f;
    return 248320;
}
#endif
