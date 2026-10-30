// CPU-only unit test of the graph-instantiate OOM policy (src/flash/graph_oom.h). Built and run by tests/test_env.py.
#include "flash/graph_oom.h"

#include <cstdio>

static int fails = 0;
#define EXPECT(c)                                                     \
    do {                                                              \
        if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

using namespace graph_oom;

int main() {
    // FLASH_TEST_GRAPH_OOM lists
    EXPECT(parse_list(nullptr).empty());
    EXPECT(parse_list("").empty());
    EXPECT((parse_list("3") == std::vector<int>{3}));
    EXPECT((parse_list("3,7, 12") == std::vector<int>{3, 7, 12}));
    EXPECT((parse_list("x,5,,9y,2") == std::vector<int>{5, 9, 2}));  // junk skipped, a number's tail ignored

    // injection: per card call index, optional rank, hard = the retry fails too
    Inject a;
    a.at = {2, 5};
    EXPECT(!a.first(1, 0) && a.first(2, 0) && a.first(5, 1) && !a.first(6, 0));
    EXPECT(!a.retry(2, 0));  // soft: the retry (after dropping the graphs) succeeds
    a.hard = true;
    EXPECT(a.retry(2, 0) && !a.retry(3, 0));
    a.rank = 1;
    EXPECT(!a.first(2, 0) && a.first(2, 1) && !a.retry(2, 0) && a.retry(2, 1));
    Inject none;
    for (int i = 1; i < 100; i++) EXPECT(!none.first(i, 0) && !none.retry(i, 0));

    // cooldown after a failed retry: exactly kCoolSteps missing graphs are not captured, then capture again
    Cool c;
    EXPECT(!c.skip());
    c.trip();
    int skipped = 0;
    while (c.skip()) skipped++;
    EXPECT(skipped == kCoolSteps);
    EXPECT(!c.skip());
    c.trip();
    c.skip();
    c.trip();  // a new failure restarts the count
    skipped = 0;
    while (c.skip()) skipped++;
    EXPECT(skipped == kCoolSteps);

    if (fails) return 1;
    printf("graph_oom_unit: ok\n");
    return 0;
}
