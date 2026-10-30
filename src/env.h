#pragma once
// Environment switches (QE_*, FLASH_*). An empty or whitespace-only value means "unset, use the
// default": a Docker Compose pass-through such as `FLASH_BATCH_K: "${FLASH_BATCH_K:-}"` exports an
// empty string, which atoi() used to read as 0 (batched drafts, grouped MoE, big prefill batches
// and the host-RAM margin silently off). A value that does not parse as a number prints a one-line
// warning (once per variable) and also uses the default. No CUDA dependency: tests/env_unit.cpp
// builds this header with a plain C++ compiler.
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

// The value of `name`, or nullptr when it is unset, empty or only whitespace.
inline const char* env_str(const char* name) {
    const char* e = getenv(name);
    if (!e) return nullptr;
    for (const char* p = e; *p; ++p)
        if (!isspace((unsigned char)*p)) return e;
    return nullptr;
}

inline bool env_set(const char* name) { return env_str(name) != nullptr; }

namespace qe_env_detail {
inline void warn(const char* name, const char* val, const char* kind) {
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (!seen.insert(name).second) return;
    fprintf(stderr, "[env] warning: %s=\"%s\" is not %s, using the default\n", name, val, kind);
}
inline bool tail_ok(const char* end) {
    while (*end && isspace((unsigned char)*end)) ++end;
    return *end == 0;
}
}  // namespace qe_env_detail

inline long long env_ll(const char* name, long long def) {
    const char* e = env_str(name);
    if (!e) return def;
    errno = 0;
    char* end = nullptr;
    const long long v = strtoll(e, &end, 10);
    if (end == e || errno == ERANGE || !qe_env_detail::tail_ok(end)) {
        qe_env_detail::warn(name, e, "an integer");
        return def;
    }
    return v;
}

inline int env_int(const char* name, int def) {
    const char* e = env_str(name);
    if (!e) return def;
    const long long v = env_ll(name, (long long)INT_MIN - 1);
    if (v < INT_MIN || v > INT_MAX) {
        if (v != (long long)INT_MIN - 1) qe_env_detail::warn(name, e, "an int");
        return def;
    }
    return (int)v;
}

inline double env_double(const char* name, double def) {
    const char* e = env_str(name);
    if (!e) return def;
    errno = 0;
    char* end = nullptr;
    const double v = strtod(e, &end);
    if (end == e || errno == ERANGE || !qe_env_detail::tail_ok(end)) {
        qe_env_detail::warn(name, e, "a number");
        return def;
    }
    return v;
}
