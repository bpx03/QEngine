// CPU-only unit test of src/env.h (empty = unset, unparsable = warning + default). Built and run by
// tests/test_env.py; standalone: g++ -std=c++20 -I src tests/env_unit.cpp -o /tmp/env_unit && /tmp/env_unit
#include "env.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int fails = 0;
#define EXPECT(c)                                                     \
    do {                                                              \
        if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

int main() {
    unsetenv("QE_T");
    EXPECT(env_int("QE_T", 64) == 64);
    EXPECT(!env_set("QE_T") && env_str("QE_T") == nullptr);
    setenv("QE_T", "", 1);  // compose "${QE_T:-}": unset, not 0
    EXPECT(env_int("QE_T", 64) == 64);
    EXPECT(env_double("QE_T", 8.5) == 8.5);
    EXPECT(env_ll("QE_T", 1ll << 40) == (1ll << 40));
    EXPECT(env_str("QE_T") == nullptr && !env_set("QE_T"));
    setenv("QE_T", "  \t", 1);  // whitespace only: unset
    EXPECT(env_int("QE_T", 7) == 7 && env_str("QE_T") == nullptr);
    setenv("QE_T", "0", 1);  // an explicit 0 stays 0 (the off-switches)
    EXPECT(env_int("QE_T", 64) == 0 && env_double("QE_T", 1.3) == 0.0 && env_set("QE_T"));
    setenv("QE_T", " 12 ", 1);
    EXPECT(env_int("QE_T", 64) == 12);
    setenv("QE_T", "-3", 1);
    EXPECT(env_int("QE_T", 64) == -3);
    setenv("QE_T", "1.25", 1);
    EXPECT(std::fabs(env_double("QE_T", 2.0) - 1.25) < 1e-12);
    EXPECT(env_int("QE_T", 64) == 64);  // not an integer: warning + default (atoi gave 1)
    setenv("QE_T", "abc", 1);  // atoi gave 0
    EXPECT(env_int("QE_T", 64) == 64 && env_double("QE_T", 1.3) == 1.3 && env_ll("QE_T", 5) == 5);
    EXPECT(env_set("QE_T") && env_str("QE_T"));  // string switches still see the value
    setenv("QE_T", "12abc", 1);
    EXPECT(env_int("QE_T", 64) == 64);
    setenv("QE_T", "99999999999", 1);  // over int: default; fits long long
    EXPECT(env_int("QE_T", 64) == 64 && env_ll("QE_T", 0) == 99999999999ll);
    setenv("QE_T", "1e3", 1);
    EXPECT(env_double("QE_T", 0) == 1000.0);
    unsetenv("QE_T");
    if (fails) return 1;
    printf("env_unit: ok\n");
    return 0;
}
