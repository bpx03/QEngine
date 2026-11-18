// CPU-only unit test for the shared sidecar loaders (src/flash/sidecar_load.h): cold_experts.json and
// token_rank.bin rules, acceptance and every rejection path. Built and run by CMake/ctest (tests/sidecar_unit.cpp
// is in the foreach list in CMakeLists.txt; it needs no GPU and no python harness). The acceptance side is pinned
// against the released reader: every token_rank.bin that main's fread(data, 4, cap, f) served (trailing bytes,
// short and empty files included) must still load, and only ids that can hurt the device tables (outside the
// vocabulary) plus the malformed-JSON paths are refused. Streams the size probe cannot size - pipes,
// FIFOs - are exercised through a real FIFO with a forked writer.
#include "flash/sidecar_load.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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

// A real FIFO stands in for a rank file piped in at load: fork a writer that emits payload and
// closes, read the far end with load_token_rank. Returns whether the load threw; *out carries the
// ids on success, *err the message on a throw (or a setup failure).
static bool fifo_rank(const fs::path& dir, const char* name, const std::string& payload,
                      std::vector<int>* out, std::string* err) {
    const fs::path fifo = dir / name;
    fs::remove(fifo);
    if (mkfifo(fifo.c_str(), 0600) != 0) {
        *err = "mkfifo failed";
        return true;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        fs::remove(fifo);
        *err = "fork failed";
        return true;
    }
    if (pid == 0) {
        const int fd = ::open(fifo.c_str(), O_WRONLY);  // waits for the reader's fopen
        if (fd >= 0) {
            size_t off = 0;
            while (off < payload.size()) {
                const ssize_t w = ::write(fd, payload.data() + off, payload.size() - off);
                if (w <= 0) break;
                off += (size_t)w;
            }
            ::close(fd);  // end of stream
        }
        _exit(0);
    }
    bool threw = false;
    try {
        *out = sidecar::load_token_rank(fifo.string(), 65536, VOCAB);
    } catch (const std::exception& ex) {
        threw = true;
        *err = ex.what();
    }
    int st = 0;
    waitpid(pid, &st, 0);
    fs::remove(fifo);
    return threw;
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
    // rejection clauses that had no case yet: a JSON root that is not an object at all, and an object
    // with no 'layers' key. Relaxing the root/'layers' clause leaves the parser's own "json: missing key
    // layers" text where the sidecar rule used to report, so the needle below stops matching and these
    // cases fail.
    write_file(dir / "c8.json", "[1,2,3]");
    EXPECT(cold_throws(dir / "c8.json", "'layers' array"));
    write_file(dir / "c9.json", "42");
    EXPECT(cold_throws(dir / "c9.json", "'layers' array"));
    write_file(dir / "c10.json", "{}");
    EXPECT(cold_throws(dir / "c10.json", "'layers' array"));
    // negative expert id: the range clause must catch x.num < 0, because (int)x.num == -1 otherwise
    // indexes seen[-1], out of bounds for the per-layer duplicate guard (the OOB usually passes
    // silently, so the refusal itself is what this case pins)
    std::vector<std::vector<std::string>> negl = good;
    negl[7] = {"-1"};
    write_file(dir / "c11.json", cold_json(negl));
    EXPECT(cold_throws(dir / "c11.json", "outside [0, 512)"));
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
    // short read: on a regular file sized from ftell the read cannot come up short (only a truncate
    // under the size probe would), so the clause is tested at the seam it guards - the shared id
    // reader fed a stream that ends one id early. Dropping the short-read throw lets a vector with an
    // uninitialized tail through and this case fails.
    if (FILE* tf = tmpfile()) {
        const int two[2] = {7, 9};
        fwrite(two, 4, 2, tf);
        rewind(tf);
        std::vector<int> want(3);
        bool refused = false;
        try {
            sidecar::detail::read_ids(tf, want, "short-stream");  // read_ids closes tf
        } catch (const std::exception& ex) {
            refused = std::strstr(ex.what(), "short read (2 of 3 ids)") != nullptr;
        }
        EXPECT(refused);
    } else EXPECT(false);

    // non-seekable rank files (no pipe|fifo case existed before these two). The size probe is
    // fseek(END)+ftell, and both fail on a stream: the old sizing turned ftell's -1 into 0 ids and
    // load_token_rank returned the empty vocabulary SILENTLY - indistinguishable from a missing
    // sidecar - where the released fread(data, 4, cap, f) read the ids off the very same stream.
    // A piped rank file must load its ids; only evidence that the stream itself broke is refused.
    {
        std::vector<int> got;
        std::string err;
        EXPECT(!fifo_rank(dir, "rank.fifo", rank_blob({42, 9001}), &got, &err));  // no throw...
        EXPECT(got.size() == 2 && got[0] == 42 && got[1] == 9001);                // ...and both ids, in order
    }
    // premature close: the stream dies mid-id. A pipe has no file size to call a short read
    // against, so the bytes are the evidence - a partial last id is data the sender died carrying
    // and gets an explicit error, never the silent empty vocabulary the failed probe used to
    // return (nor a quietly short list). A clean end, empty included, still just loads what came.
    {
        std::vector<int> got;
        std::string err;
        EXPECT(fifo_rank(dir, "cut.fifo", rank_blob({7}) + std::string("ab"), &got, &err));
        EXPECT(err.find("cut.fifo") != std::string::npos && err.find("mid-id") != std::string::npos);
    }

    printf("sidecar_unit: %s (%d failures)\n", fails ? "FAILED" : "ok", fails);
    fs::remove_all(dir);
    return fails ? 1 : 0;
}
