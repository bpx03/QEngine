// CPU-only unit test for the shared sidecar loaders (src/flash/sidecar_load.h): cold_experts.json and
// token_rank.bin rules, acceptance and every rejection path. Built and run by CMake/ctest (tests/sidecar_unit.cpp
// is in the foreach list in CMakeLists.txt; it needs no GPU and no python harness). The acceptance side is pinned
// against the released reader: every token_rank.bin that main's fread(data, 4, cap, f) served (trailing bytes,
// short and empty files included) must still load, and only ids that can hurt the device tables (outside the
// vocabulary) plus the malformed-JSON paths are refused.
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

static std::string rank_blob(const std::vector<int>& ts) {
    std::string s;
    for (int t : ts) s.append((const char*)&t, 4);
    return s;
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
    // size cap: cold_experts.json is read bounded before the parse (kMaxColdJsonBytes = 16 MiB, the
    // manifest cap's size in checkpoint.h); a bigger file is refused with the cap in its message.
    // Without the cap the bytes reach the JSON parser first and the message is the parser's, without
    // this number - so this case fails there.
    {
        const fs::path huge = dir / "huge.json";
        write_file(huge, std::string((size_t)sidecar::kMaxColdJsonBytes + 16, 'x'));
        EXPECT(cold_throws(huge, "16777216"));
        fs::remove(huge);
    }

    // token_rank.bin: missing empty; the list is capped at cap; ids must be whole and in [0, VOCAB).
    EXPECT(sidecar::load_token_rank((dir / "absent.bin").string(), 65536, VOCAB).empty());
    const std::string ids = rank_blob({5, 0, VOCAB - 1, 12345});
    write_file(dir / "rank.bin", ids);
    const auto rank = sidecar::load_token_rank((dir / "rank.bin").string(), 65536, VOCAB);
    EXPECT(rank.size() == 4 && rank[2] == VOCAB - 1);
    EXPECT(sidecar::load_token_rank((dir / "rank.bin").string(), 2, VOCAB).size() == 2);  // cap: reads only 2

    // main's reader read whole int32s with fread and never looked at bytes past the last whole id, so
    // a legitimate token_rank.bin with a trailing tail served fine on every released server. A size
    // check on the whole file (bytes % 4) refuses those models again - an acceptance regression: the
    // cap region is all that gets validated. These cases fail on the pre-fix header.
    for (int extra = 1; extra <= 3; extra++) {
        const fs::path p = dir / ("tail" + std::to_string(extra) + ".bin");
        write_file(p, ids + std::string((size_t)extra, 0x41));
        const auto t = sidecar::load_token_rank(p.string(), 65536, VOCAB);
        EXPECT(t.size() == 4 && t[3] == 12345);
    }
    // junk past the cap too: the released fread stopped at cap ids and ignored the rest, ragged size
    // or not - only the first cap ids load, and the trailing bytes are not an error
    write_file(dir / "overcap.bin", ids + std::string(7, 0x55));
    const auto capped = sidecar::load_token_rank((dir / "overcap.bin").string(), 2, VOCAB);
    EXPECT(capped.size() == 2 && capped[1] == 0);

    // empty file and a sub-element file: fread returned 0 ids on main, and an empty list is what "no
    // sidecar" means (the Worker guards the draft vocab on !draft_ids_.empty()), so both load empty.
    write_file(dir / "empty.bin", "");
    EXPECT(sidecar::load_token_rank((dir / "empty.bin").string(), 65536, VOCAB).empty());
    write_file(dir / "stub.bin", std::string("\x07\x00", 2));
    EXPECT(sidecar::load_token_rank((dir / "stub.bin").string(), 65536, VOCAB).empty());

    // what is genuinely poison: ids outside the vocabulary land as out-of-bounds row gathers on the
    // device. Refused whole, both signs.
    write_file(dir / "neg.bin", rank_blob({-1, 5}));
    try {
        sidecar::load_token_rank((dir / "neg.bin").string(), 65536, VOCAB);
        EXPECT(false);
    } catch (const std::exception& ex) {
        EXPECT(std::strstr(ex.what(), "outside [0, 248320)") != nullptr);
    }
    write_file(dir / "big.bin", rank_blob({5, VOCAB}));
    try {
        sidecar::load_token_rank((dir / "big.bin").string(), 65536, VOCAB);
        EXPECT(false);
    } catch (const std::exception& ex) {
        EXPECT(std::strstr(ex.what(), "outside [0, 248320)") != nullptr);
    }

    printf("sidecar_unit: %s (%d failures)\n", fails ? "FAILED" : "ok", fails);
    fs::remove_all(dir);
    return fails ? 1 : 0;
}
