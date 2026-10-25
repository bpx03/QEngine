// CPU mock of the engine C API (server/qe_capi.h, TIER 0, plus TIER 1 unless -DMOCK_TIER0) for the
// server's end-to-end HTTP test (tests/test_server_mock.py). No GPU. Replies are scripted:
//   QE_MOCK_SCRIPT  file, one reply per line as comma-separated token ids (used in turn, cyclic)
//   QE_MOCK_LOG     file: every start appends "P <n> <reused> <ids...>", every set_sampling "S t p k"
// Prefix reuse is modelled like the engine's live sequence: reused = common prefix with the
// previous prompt + its generated tokens (at most n - 1).
//   gcc -O2 -shared -fPIC -o mock.so tests/mock_engine.c [-DMOCK_TIER0]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int max_t, k, reused, nseq, cap;
    int* seq;
    int** replies;
    int* rlen;
    int nrep, next, cur, rpos, limit;
    FILE* log;
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
    return m->replies[m->cur][0];
}
int qe_spec_step(void* e, int* out) {
    Mock* m = e;
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
void qe_release(void* e, int slot) { (void)e, (void)slot; }

#ifndef MOCK_TIER0
int qe_vocab(void* e) { (void)e; return 248320; }
int qe_num_slots(void* e) { (void)e; return 1; }
long long qe_pool_tokens(void* e) { return ((Mock*)e)->max_t; }
int qe_can_admit(void* e, int n) { return n <= ((Mock*)e)->max_t; }
void qe_set_limit(void* e, int slot, int tokens) { (void)slot; ((Mock*)e)->limit = tokens; }
double qe_warmup(void* e, int modes) { (void)e, (void)modes; return 0.0; }
void qe_reset(void* e) { ((Mock*)e)->nseq = 0; }
#endif
