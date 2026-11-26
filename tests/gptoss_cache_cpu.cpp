// [gptoss-rearm] review2 L5: CPU test of the gpt-oss host store's idle-archive logic (src/gptoss/capi_cache.cpp) with a
// stub Engine (no GPU): the fit counts a partial copy once (H1), the per-slot back-off (H1 / review2 L1), the size estimate
// with the rings (review2 L2), the partial kept below the common prefix (M2), the fit on unshared bytes (M1), the
// orphaned partial dropped (review2 L4), mem_guard drops the partial before stored entries (review2 M1),
// GPTOSS_ARCHIVE_SYNC_ROWS on a current partial (review2 L3), no stored entry dropped for an idle archive. Built and run by tests/test_gptoss_cache_cpu.py.
#include "../src/gptoss/capi_srv.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>

using namespace gptoss;

static const size_t RB = 256 << 10;   // stub row bytes: 256 KiB (64 MiB per call = 256 rows, the minimum)
static int g_saves = 0, g_last_src = -1;
static Plan g_plan;

namespace gptoss {
void Engine::reset() {}
void Engine::truncate(int) {}
void Engine::copy_rows(int, int, int) {}
void Engine::ring_save(std::vector<unsigned char>& o) const { o.assign(64 << 10, 1); }
void Engine::ring_load(const std::vector<unsigned char>&, int) {}
size_t Engine::row_bytes() const { return RB; }
void Engine::rows_save(int src, int n, std::vector<unsigned char>& out) const {
    g_saves++;
    g_last_src = src;
    out.assign((size_t)n * RB, 7);   // (touched: MemAvailable goes down as with the real copy)
}
void Engine::rows_load(const std::vector<unsigned char>&, int, int) {}
bool Engine::broken() const { return false; }
const Plan& Engine::plan() const { return g_plan; }
void srv_bind(Handle&, int) {}
void srv_fault(Handle&, int, const char*) {}
}  // namespace gptoss

extern "C" {
int qe_archive_more(void* p, long long budget);
void qe_mem_guard(void* p);
}

static int fails = 0;
static void check(const char* name, bool ok, const std::string& info = "") {
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, info.empty() ? "" : ": ", info.c_str());
    if (!ok) fails++;
}
static double avail_gib() {
    std::ifstream f("/proc/meminfo");
    std::string k, u;
    size_t v = 0;
    while (f >> k >> v >> u)
        if (k == "MemAvailable:") return v / 1048576.0;
    return 0;
}
static void set_headroom(double gib) {   // MemAvailable - QE_KV_ARCHIVE_MARGIN_GB = gib now
    setenv("QE_KV_ARCHIVE_MARGIN_GB", std::to_string(avail_gib() - gib).c_str(), 1);
}
alignas(Engine) static unsigned char eng_mem[sizeof(Engine)];

static void fresh(Handle& h, int slots) {
    h.~Handle();
    new (&h) Handle();
    h.e = reinterpret_cast<Engine*>(eng_mem);
    h.s.resize(slots);
}
static void conv(Handle& h, int j, int n, int first = 1) {
    Slot& s = h.s[j];
    s.toks.clear();
    for (int i = 0; i < n; i++) s.toks.push_back(first + i);
    s.cap = n + 64;
    s.base = j * 100000;
    s.archived = false;
}
static int run_all(Handle& h, long long budget, int max_calls = 100) {
    int r = 0;
    for (int i = 0; i < max_calls && (r = qe_archive_more(&h, budget)) == 0; i++) {}
    return r;
}

int main() {
    g_plan.ngpu = 1;
    setenv("GPTOSS_ARCHIVE_SYNC_ROWS", "400", 1);   // (read once, by the first srv_archive_cheap: T8)
    unsetenv("QE_KV_ARCHIVE_GB");
    const long long B = 64LL << 20;   // 256 rows per call
    Handle h;

    // T1 H1: 2048 rows (512 MiB) with 768 MiB of headroom: a whole first archive fits (before: needed 2 S)
    fresh(h, 1);
    conv(h, 0, 2048);
    set_headroom(0.75);
    int r = run_all(h, B);
    check("H1: archive of S with 1.5 S headroom is stored", r == 1 && h.store.size() == 1 && h.s[0].archived,
          "r " + std::to_string(r) + " store " + std::to_string(h.store.size()));

    // T2 review2 L2 + back-off: cap = rows + 32 KiB (below rows + ring 64 KiB): with the ring estimate the first try is
    // refused before any copy; then held back (no copy, no fit) on later calls
    fresh(h, 1);
    conv(h, 0, 1024);
    set_headroom(4.0);
    h.ring_est = 64 << 10;   // (an earlier archive saw the ring size)
    setenv("QE_KV_ARCHIVE_GB", std::to_string((1024.0 * RB + (32 << 10)) / 1073741824.0).c_str(), 1);
    g_saves = 0;
    r = qe_archive_more(&h, B);
    check("L2: the estimate with the rings refuses before copying", r == 2 && g_saves == 0, "r " + std::to_string(r) +
          " saves " + std::to_string(g_saves));
    // without the ring estimate (first archive of the process): copied, refused at the end, then held back
    fresh(h, 1);
    conv(h, 0, 1024);
    g_saves = 0;
    r = run_all(h, B);
    const int saves1 = g_saves;
    check("H1 back-off: a copy refused at the end returns 2", r == 2 && saves1 > 0 && h.store.empty(),
          "r " + std::to_string(r) + " saves " + std::to_string(saves1));
    for (int i = 0; i < 20; i++) r = qe_archive_more(&h, B);
    check("H1 back-off: later calls copy nothing (2)", r == 2 && g_saves == saves1, "saves " + std::to_string(g_saves));
    conv(h, 0, 1025);   // the conversation changed: tried again (the fit refuses now: ring_est known)
    r = qe_archive_more(&h, B);
    check("back-off lifted when the slot's rows change", r == 2 && h.s[0].arch_fail_n == 1025);
    unsetenv("QE_KV_ARCHIVE_GB");

    // T3 review2 L1: two refused slots do not alternate copies
    fresh(h, 2);
    conv(h, 0, 512, 1);
    conv(h, 1, 512, 5000);
    setenv("QE_KV_ARCHIVE_GB", std::to_string((512.0 * RB + (32 << 10)) / 1073741824.0).c_str(), 1);
    g_saves = 0;
    for (int i = 0; i < 30; i++) qe_archive_more(&h, B);
    const int s2 = g_saves;
    for (int i = 0; i < 30; i++) qe_archive_more(&h, B);
    check("L1: per-slot back-off, two slots: no copies after both were refused", g_saves == s2 && s2 > 0 &&
          h.s[0].arch_fail_t >= 0 && h.s[1].arch_fail_t >= 0, "saves " + std::to_string(s2) + " -> " + std::to_string(g_saves));
    unsetenv("QE_KV_ARCHIVE_GB");

    // T4 M2: a next turn during a partial copy keeps the rows below the common prefix
    fresh(h, 1);
    conv(h, 0, 2048);
    set_headroom(4.0);
    qe_archive_more(&h, B);   // rows [0, 256)
    conv(h, 0, 2056);         // the same prefix, 8 more rows
    qe_archive_more(&h, B);
    check("M2: the copy goes on at row 256 (not 0)", g_last_src == h.s[0].base + 256, "src " + std::to_string(g_last_src));
    r = run_all(h, B);
    check("M2: the archive completes", r == 1 && h.store.size() == 1 && h.store[0].toks.size() == 2056);

    // T5 M1: archiving the next turn of a stored conversation drops no other entry (fit on the unshared rows)
    {
        fresh(h, 1);
        conv(h, 0, 1600);
        set_headroom(8.0);
        run_all(h, B);   // entry A (1,600 rows), the slot shares it (prov)
        Handle tmp;
        tmp.e = h.e;
        tmp.s.resize(1);
        conv(tmp, 0, 1200, 90000);
        run_all(tmp, B);   // another conversation's entry, moved into h's store
        tmp.store[0].use = 0;   // (the oldest: LRU drops it first)
        h.store.push_back(tmp.store[0]);
        h.store_bytes += tmp.store_bytes;
        conv(h, 0, 2048);   // A's next turn: 448 new rows
        h.s[0].prov_n = 1600;
        setenv("QE_KV_ARCHIVE_GB", std::to_string((h.store_bytes + 800.0 * RB) / 1073741824.0).c_str(), 1);
        r = run_all(h, B);
        bool other = false;
        for (const Entry& e : h.store) other |= e.toks.size() == 1200;
        check("M1: the next turn is stored, the other conversation stays", r == 1 && other && h.store.size() == 2,
              "store " + std::to_string(h.store.size()));
        unsetenv("QE_KV_ARCHIVE_GB");
    }

    // T6 review2 L4: a partial whose slot lost its rows is given back
    fresh(h, 1);
    conv(h, 0, 2048);
    set_headroom(4.0);
    qe_archive_more(&h, B);
    h.s[0].toks.clear();
    h.s[0].cap = 0;
    qe_archive_more(&h, B);
    check("L4: orphaned partial dropped", h.arch_slot == -1 && h.arch.rows.empty());

    // T7 review2 M1: below the margin, mem_guard drops the partial and keeps the stored entry
    fresh(h, 2);
    conv(h, 0, 256, 1);
    set_headroom(4.0);
    run_all(h, B);   // entry of slot 0
    conv(h, 1, 2048, 7000);
    qe_archive_more(&h, B);   // partial of slot 1
    const bool had = h.arch_slot == 1 && h.store.size() == 1;
    set_headroom(-0.5);   // MemAvailable now 0.5 GiB below the margin
    qe_mem_guard(&h);
    check("review2 M1: mem_guard drops the partial first, the stored entry stays", had && h.arch_slot == -1 && h.store.size() == 1,
          "store " + std::to_string(h.store.size()) + " arch_slot " + std::to_string(h.arch_slot));

    // T8 review2 L3: GPTOSS_ARCHIVE_SYNC_ROWS=400 holds for a current partial too
    fresh(h, 1);
    conv(h, 0, 2048);
    set_headroom(4.0);
    qe_archive_more(&h, B);   // 256 rows copied, 1,792 left
    const bool before = srv_archive_cheap(h, 0);
    for (int i = 0; i < 6; i++) qe_archive_more(&h, B);   // 1,792 copied, 256 left
    const bool after = h.arch_slot == 0 && srv_archive_cheap(h, 0);
    check("review2 L3: a current partial with 1,792 rows left is not a bounded copy; with 256 left it is", !before && after);

    // T9 (review2 follow-up, H1 repro h1e): the idle background archive drops no stored entry for itself
    fresh(h, 2);
    conv(h, 0, 512, 1);
    set_headroom(4.0);
    run_all(h, B);   // entry of slot 0 (128 MiB)
    conv(h, 1, 2048, 7000);   // 512 MiB idle conversation, 0.3 GiB of headroom
    set_headroom(0.3);
    g_saves = 0;
    r = run_all(h, B);
    check("idle archive without room: refused (2), the stored entry stays", r == 2 && h.store.size() == 1 && g_saves == 0,
          "r " + std::to_string(r) + " store " + std::to_string(h.store.size()));

    // T10 [gptoss-archprot]: a start's eviction archive never drops the entry the start restores
    for (int protect = 0; protect < 2; protect++) {
        fresh(h, 2);
        conv(h, 0, 1024, 1);   // A: 256 MiB, archived, then its slot evicted (only the store holds it)
        set_headroom(4.0);
        run_all(h, B);
        const std::vector<int> a = h.s[0].toks;
        h.s[0].toks.clear();
        h.s[0].cap = 0;
        conv(h, 1, 2048, 7000);   // B: 512 MiB idle in VRAM, not archived
        setenv("QE_KV_ARCHIVE_GB", std::to_string((2100.0 * RB) / 1073741824.0).c_str(), 1);
        std::vector<int> a2 = a;   // A's next turn arrives
        for (int i = 0; i < 20; i++) a2.push_back(50000 + i);
        if (protect) srv_protect_begin(h, a2.data(), (int)a2.size(), -1);
        r = srv_archive_slot(h, 1);   // the start evicts B (a synchronous archive)
        if (protect) srv_protect_end(h);
        bool has_a = false;
        for (const Entry& e : h.store) has_a |= e.toks == a;
        if (protect)
            check("archprot: B's eviction archive does not drop A (B not stored)", r == -1 && has_a, "r " + std::to_string(r));
        else
            check("archprot: without the protection the same archive drops A (the case it fixes)", r == 1 && !has_a,
                  "r " + std::to_string(r));
        unsetenv("QE_KV_ARCHIVE_GB");
    }

    // T11 (archprot review H1): a refused victim archive drops nothing (A protected, C another conversation)
    {
        fresh(h, 3);
        set_headroom(4.0);
        conv(h, 2, 256, 30000);   // C: 64 MiB, archived first (the oldest)
        run_all(h, B);
        conv(h, 0, 1024, 1);      // A: 256 MiB, archived
        run_all(h, B);
        const std::vector<int> a = h.s[0].toks, c = h.s[2].toks;
        h.s[0].toks.clear();
        h.s[0].cap = 0;
        h.s[2].toks.clear();
        h.s[2].cap = 0;
        conv(h, 1, 2048, 7000);   // B: 512 MiB idle in VRAM
        setenv("QE_KV_ARCHIVE_GB", std::to_string((h.store_bytes + 300.0 * (1 << 20)) / 1073741824.0).c_str(), 1);
        std::vector<int> a2 = a;
        for (int i = 0; i < 20; i++) a2.push_back(50000 + i);
        srv_protect_begin(h, a2.data(), (int)a2.size(), -1);
        r = srv_archive_slot(h, 1);
        srv_protect_end(h);
        bool has_a = false, has_c = false;
        for (const Entry& e : h.store) {
            has_a |= e.toks == a;
            has_c |= e.toks == c;
        }
        check("T11: refused victim keeps C", r == -1 && has_a && has_c, "r " + std::to_string(r) + " has_a " +
              std::to_string(has_a) + " has_c " + std::to_string(has_c) + " store " + std::to_string(h.store.size()));
        unsetenv("QE_KV_ARCHIVE_GB");
    }

    // T12 (review L1 / L2): a branch at an older archived turn E while the slot holds the longer turn: E is the store
    // fallback (the slot is the best source), and the slot's own archive keeps E instead of superseding it
    {
        fresh(h, 1);
        set_headroom(4.0);
        conv(h, 0, 512, 1);
        run_all(h, B);   // E = turn 1
        const std::vector<int> e1 = h.s[0].toks;
        conv(h, 0, 1024, 1);   // turn 2 extends it (same first 512 tokens)
        h.s[0].prov_n = 512;
        std::vector<int> br = e1;   // the branch: turn 1 + another follow-up
        for (int i = 0; i < 20; i++) br.push_back(60000 + i);
        srv_protect_begin(h, br.data(), (int)br.size(), -1);   // (a start on another slot)
        const bool on = h.protect_on && h.protect_toks == e1;
        r = srv_archive_slot(h, 0);   // the start's own-slot archive (turn 2)
        srv_protect_end(h);
        bool has_e = false, has_t2 = false;
        for (const Entry& e : h.store) {
            has_e |= e.toks == e1;
            has_t2 |= e.toks.size() == 1024;
        }
        check("T12: store fallback protected, turn 1 kept next to turn 2", on && r == 1 && has_e && has_t2,
              "on " + std::to_string(on) + " r " + std::to_string(r) + " store " + std::to_string(h.store.size()));
        // without a start in progress, turn 2 supersedes turn 1 as before
        conv(h, 0, 1536, 1);
        h.s[0].prov_n = 0;
        r = run_all(h, B);
        int n_t = 0;
        for (const Entry& e : h.store) n_t += !e.hint;
        check("T12: an idle archive still supersedes the older turns", r == 1 && n_t == 1, "entries " + std::to_string(n_t));
    }

    // T13 (review L3): a system-part snapshot (hint entry) as the protected source
    {
        fresh(h, 2);
        set_headroom(4.0);
        Entry hint;
        hint.hint = true;
        for (int i = 0; i < 1024; i++) hint.toks.push_back(1 + i);
        hint.ring.assign(64 << 10, 1);
        hint.ring_hi = 1024;
        hint.rows.push_back(RowPart{0, 1024, std::make_shared<std::vector<unsigned char>>(1024 * RB, 7)});
        hint.use = 1;
        h.store.push_back(hint);
        h.store_bytes = 1024 * RB + (64 << 10) + 4 * 1024;
        conv(h, 1, 2048, 7000);   // B: the victim
        setenv("QE_KV_ARCHIVE_GB", std::to_string((2100.0 * RB) / 1073741824.0).c_str(), 1);
        std::vector<int> q = hint.toks;
        for (int i = 0; i < 30; i++) q.push_back(70000 + i);
        srv_protect_begin(h, q.data(), (int)q.size(), -1);
        const bool on = h.protect_on && h.protect_hint;
        r = srv_archive_slot(h, 1);
        srv_protect_end(h);
        check("T13: a protected system-part snapshot is not dropped for a victim", on && r == -1 && h.store.size() == 1 &&
              h.store[0].hint, "on " + std::to_string(on) + " r " + std::to_string(r));
        unsetenv("QE_KV_ARCHIVE_GB");
    }

    // T14 (archprot_review2 M1, the reviewer's repro): the next turn of idle slot 0's conversation X (stored as X2 by the
    // idle archive) starts on slot 0 itself: X2 is not protected, so another idle conversation Y can still be stored
    for (int own = 0; own >= -1; own--) {
        fresh(h, 2);
        set_headroom(4.0);
        conv(h, 0, 1024, 1);   // X on slot 0, archived (X2), the slot keeps its rows
        run_all(h, B);
        conv(h, 1, 1024, 9000);   // Y idle on slot 1, not archived
        std::vector<int> x = h.s[0].toks;
        for (int i = 0; i < 20; i++) x.push_back(80000 + i);
        setenv("QE_KV_ARCHIVE_GB", std::to_string((h.store_bytes + 100.0 * (1 << 20)) / 1073741824.0).c_str(), 1);
        srv_protect_begin(h, x.data(), (int)x.size(), own);
        const bool on = h.protect_on;
        r = srv_archive_slot(h, 1);
        srv_protect_end(h);
        bool has_y = false;
        for (const Entry& e : h.store) has_y |= e.toks.size() == 1024 && e.toks[0] == 9000;
        if (own == 0)
            check("T14: start on the source slot itself: no fallback, Y stored", !on && r == 1 && has_y,
                  "protect_on " + std::to_string(on) + " r " + std::to_string(r) + " has_y " + std::to_string(has_y));
        else
            check("T14: start on another slot (slot 0 evictable): fallback X2 protected, Y refused", on && r == -1 && !has_y,
                  "protect_on " + std::to_string(on) + " r " + std::to_string(r));
        unsetenv("QE_KV_ARCHIVE_GB");
    }

    printf("%d failed\n", fails);
    return fails ? 1 : 0;
}
