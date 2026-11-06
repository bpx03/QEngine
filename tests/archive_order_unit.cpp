// 27B archive eviction order (src/archive_order.h) on a model of Engine::ram_make_room: no GPU.
#include "archive_order.h"
#include <cstdio>
#include <cstdlib>
#include <list>
#include <string>
#include <vector>

#define EXPECT(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)
static int fails = 0;
static const double kStale = 1800, kNow = 100000;

struct A {
    std::string name;
    int prio = 1;
    double last_s = 0;
    unsigned long long used = 0;
    size_t total = 1;
    bool ram = true, pinned = false;
    int hold = 0;
};
static qe_arch::Rank rank_of(const A& a) {
    return {qe_arch::active_main(a.prio, a.last_s, kNow, kStale), a.last_s, a.used};
}

struct Model {  // Engine's loop: fits first, then one victim at a time
    std::list<A> l;
    size_t used() const {
        size_t u = 0;
        for (auto& a : l) if (a.ram && !a.pinned) u += a.total;
        return u;
    }
    std::vector<std::string> gone;
    bool write(const A& w, size_t limit, bool optional = false) {
        const qe_arch::Rank me = rank_of(w);
        const qe_arch::Rank* older = optional ? &me : nullptr;
        if (used() - qe_arch::can_free(l, rank_of, older) + w.total > limit) return false;
        while (used() + w.total > limit) {
            auto v = qe_arch::victim(l, rank_of);
            if (v == l.end()) return false;
            gone.push_back(v->name);
            l.erase(v);
        }
        l.push_back(w);
        return true;
    }
};
static A conv(const char* n, double age_s, int prio = 1, size_t total = 1) {
    static unsigned long long tick = 0;
    A a;
    a.name = n, a.prio = prio, a.last_s = kNow - age_s, a.used = ++tick, a.total = total;
    return a;
}

int main() {
    using qe_arch::active_main;
    // kStale: a test value of FLASH_MAIN_STALE_S, not its default; only the tracker's main
    EXPECT(active_main(2, kNow - 1799, kNow, kStale));
    EXPECT(!active_main(2, kNow - 1800, kNow, kStale));
    EXPECT(!active_main(1, kNow - 1, kNow, kStale));
    EXPECT(!active_main(0, kNow - 1, kNow, kStale));
    {   // LRU over conversations; a finished one (prio 0) has no class of its own; the active main last
        Model m;
        m.l = {conv("main", 1000, 2), conv("new", 5), conv("old", 900), conv("finished", 100, 0)};
        EXPECT(m.write(conv("w", 0), 3));
        EXPECT(m.gone.size() == 2 && m.gone[0] == "old" && m.gone[1] == "finished");
        EXPECT(m.write(conv("w2", 0), 3));
        EXPECT(m.gone.back() == "new");
        // the active main stays although it is the oldest
        EXPECT(m.write(conv("w3", 0), 3));
        EXPECT(m.gone.back() == "w");
    }
    {   // the old main holds nothing back: plain LRU past the staleness limit, and it does not refuse a write
        Model m;
        m.l = {conv("oldmain", 7200, 2), conv("a", 8000), conv("b", 60)};
        EXPECT(m.write(conv("w", 0), 3));
        EXPECT(m.gone.back() == "a");
        EXPECT(m.write(conv("w2", 0), 3));
        EXPECT(m.gone.back() == "oldmain");
    }
    {   // nothing else left: the active main goes for an ordinary write
        Model m;
        m.l = {conv("main", 10, 2, 2)};
        EXPECT(m.write(conv("w", 0), 2));
        EXPECT(m.gone.size() == 1 && m.gone[0] == "main");
    }
    {   // not evictable: pinned, held (a restore, a paused slot), no memory copy (being loaded for a restore / on disk only)
        Model m;
        A p = conv("pinned", 9000), h = conv("held", 8000), d = conv("loading", 7000);
        p.pinned = true, h.hold = 1, d.ram = false;
        m.l = {p, h, d, conv("x", 5)};
        EXPECT(m.write(conv("w", 0), 2));
        EXPECT(m.gone.size() == 1 && m.gone[0] == "x");
        EXPECT(!m.write(conv("w2", 0, 1, 2), 2));  // the rest cannot go
    }
    {   // a write that cannot fit evicts nothing (not the main either)
        Model m;
        A h = conv("held", 10, 1, 3);
        h.hold = 1;
        m.l = {h, conv("main", 10, 2, 1), conv("x", 10, 1, 1)};
        EXPECT(!m.write(conv("w", 0, 1, 3), 5));
        EXPECT(m.gone.empty() && m.l.size() == 3);
    }
    {   // an optional write: only what goes before the writer; never a newer conversation or, for an ordinary writer, a main
        Model m;
        m.l = {conv("main", 2000 - 1000, 2), conv("older", 500), conv("newer", 5)};
        EXPECT(m.write(conv("w", 100), 3, true));
        EXPECT(m.gone.size() == 1 && m.gone[0] == "older");
        EXPECT(!m.write(conv("w2", 200), 3, true));   // older than all the rest: refused, nothing lost
        EXPECT(m.gone.size() == 1);
        EXPECT(m.write(conv("w3", 1, 2), 3, true));   // an active main takes a non-main, the oldest
        EXPECT(m.gone.back() == "w");
        // a full archive of idle conversations does not churn: the oldest writer stops at once
        Model c;
        c.l = {conv("a", 10), conv("b", 20)};
        EXPECT(!c.write(conv("oldest", 30), 2, true));
        EXPECT(c.gone.empty());
    }
    {   // ties (files found on disk, no last use yet): by tick, oldest modification first
        Model m;
        A a = conv("first", 0), b = conv("second", 0);
        a.last_s = b.last_s = 0;
        m.l = {b, a};
        EXPECT(rank_of(a).used < rank_of(b).used);
        EXPECT(m.write(conv("w", 0), 2));
        EXPECT(m.gone[0] == "first");
    }
    {   // what was just written survives its own make-room and every newer-first rule: repeated writes keep the newest
        Model m;
        for (int i = 0; i < 20; i++) {
            A w = conv("w", 1000 - i);
            w.name += std::to_string(i);
            EXPECT(m.write(w, 3));
            EXPECT(m.l.back().name == w.name);
            EXPECT(m.used() <= 3);
        }
    }
    printf(fails ? "archive_order_unit: %d FAILED\n" : "archive_order_unit: ok\n", fails);
    return fails ? 1 : 0;
}
