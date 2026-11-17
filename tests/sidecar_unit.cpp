// CPU-only unit test for the shared sidecar loaders (src/flash/sidecar_load.h): cold_experts.json and
// token_rank.bin rules, acceptance and every rejection path. Built and run by CMake/ctest (tests/sidecar_unit.cpp
// is in the foreach list in CMakeLists.txt; it needs no GPU and no python harness).
#include "flash/sidecar_load.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static int fails = 0;
#define EXPECT(c)                                                                                            \
    do {                                                                                                     \
        if (!(c)) {                                                                                          \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                     \
            fails++;                                                                                         \
        }                                                                                                    \
    } while (0)

namespace fs = std::filesystem;

// the engine's real sidecar contract (flash.h): 48 layers, 512 experts, vocab 248320
static const int LAYERS = 48, NE = 512, VOCAB = 248320;

static void write_file(const fs::path& p, const std::string& data) {
    std::ofstream o(p, std::ios::binary);
    o.write(data.data(), (std::streamsize)data.size());
    o.close();
}

static std::string cold_json(const std::vector<std::vector<std::string>>& layers) {
    std::string js = "{\"layers\": [";
    for (size_t li = 0; li < layers.size(); ++li) {
        js += li ? ", [" : "[";
        for (size_t i = 0; i < layers[li].size(); ++i) {
            if (i) js += ",";
            js += layers[li][i];
        }
        js += "]";
    }
    return js + "]}";
}

static bool cold_throws(const fs::path& p, const char* needle) {
    try {
        sidecar::load_cold_experts(p.string(), LAYERS, NE);
        return false;
    } catch (const std::exception& ex) {
        return std::strstr(ex.what(), needle) != nullptr;
    }
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "flash_sidecar_unit";
    fs::remove_all(dir);
    fs::create_directories(dir);

    // cold_experts.json: missing file is empty (optional sidecar), as before the check
    EXPECT(sidecar::load_cold_experts((dir / "absent.json").string(), LAYERS, NE).empty());

    // valid: 48 layers, whole ids in range, no duplicates per layer
    std::vector<std::vector<std::string>> good(LAYERS);
    for (int li = 0; li < LAYERS; li++) good[li] = {"0", "7", "511"};
    write_file(dir / "cold.json", cold_json(good));
    const auto cold = sidecar::load_cold_experts((dir / "cold.json").string(), LAYERS, NE);
    EXPECT(cold.size() == (size_t)LAYERS && cold[0].size() == 3 && cold[47][2] == 511);

    // rejections: wrong top level, wrong layer count, non-array layer, non-whole id, out-of-range id, duplicate
    write_file(dir / "c1.json", "{\"layers\": 3}");
    EXPECT(cold_throws(dir / "c1.json", "'layers' array"));
    write_file(dir / "c2.json", cold_json(std::vector<std::vector<std::string>>(LAYERS - 1, {"1"})));
    EXPECT(cold_throws(dir / "c2.json", "47 layers, expected 48"));
    std::string nums;
    for (int li = 0; li < LAYERS; li++) nums += (li ? ",1" : "1");
    write_file(dir / "c3.json", "{\"layers\": [" + nums + "]}");
    EXPECT(cold_throws(dir / "c3.json", "is not an array"));
    std::vector<std::vector<std::string>> frac = good;
    frac[3] = {"2.5"};
    write_file(dir / "c4.json", cold_json(frac));
    EXPECT(cold_throws(dir / "c4.json", "not a whole number"));
    std::vector<std::vector<std::string>> big = good;
    big[5] = {"512"};
    write_file(dir / "c5.json", cold_json(big));
    EXPECT(cold_throws(dir / "c5.json", "outside [0, 512)"));
    std::vector<std::vector<std::string>> dup = good;
    dup[6] = {"9", "9"};
    write_file(dir / "c6.json", cold_json(dup));
    EXPECT(cold_throws(dir / "c6.json", "duplicate expert id 9"));
    // strict parse: trailing garbage is refused, not silently accepted
    write_file(dir / "c7.json", cold_json(good) + " oops");
    EXPECT(cold_throws(dir / "c7.json", ""));  // parse_strict must fail (any message)

    // token_rank.bin: missing empty; whole int32 ids in [0, vocab); cap; short read; junk tail
    EXPECT(sidecar::load_token_rank((dir / "absent.bin").string(), 65536, VOCAB).empty());
    std::string ids;
    for (int t : {5, 0, VOCAB - 1, 12345}) {
        const int v = t;
        ids.append((const char*)&v, 4);
    }
    write_file(dir / "rank.bin", ids);
    const auto rank = sidecar::load_token_rank((dir / "rank.bin").string(), 65536, VOCAB);
    EXPECT(rank.size() == 4 && rank[2] == VOCAB - 1);
    EXPECT(sidecar::load_token_rank((dir / "rank.bin").string(), 2, VOCAB).size() == 2);  // cap: reads only 2
    std::string tail = ids;
    tail.push_back(0x41);  // 17 bytes: not whole int32 ids
    write_file(dir / "ragged.bin", tail);
    try {
        sidecar::load_token_rank((dir / "ragged.bin").string(), 65536, VOCAB);
        EXPECT(false);
    } catch (const std::exception& ex) {
        EXPECT(std::strstr(ex.what(), "not whole int32 ids") != nullptr);
    }
    std::string bad = ids;
    const int neg = -1;
    bad.insert(0, (const char*)&neg, 4);
    write_file(dir / "neg.bin", bad);
    try {
        sidecar::load_token_rank((dir / "neg.bin").string(), 65536, VOCAB);
        EXPECT(false);
    } catch (const std::exception& ex) {
        EXPECT(std::strstr(ex.what(), "outside [0, 248320)") != nullptr);
    }
    // 10 bytes: 2 whole ids + 2 stray bytes. On a regular file the read itself cannot come up short
    // (the loader sizes the read to the file), so this case lands in the whole-int32 size check; either
    // rejection message is accepted to pin the rule, not the code path that reports it.
    write_file(dir / "short.bin", ids.substr(0, 10));
    try {
        sidecar::load_token_rank((dir / "short.bin").string(), 65536, VOCAB);
        EXPECT(false);
    } catch (const std::exception& ex) {
        EXPECT(std::strstr(ex.what(), "short read") != nullptr || std::strstr(ex.what(), "not whole int32 ids") != nullptr);
    }

    printf("sidecar_unit: %s (%d failures)\n", fails ? "FAILED" : "ok", fails);
    fs::remove_all(dir);
    return fails ? 1 : 0;
}
