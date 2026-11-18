// Shared sidecar loaders: cold_experts.json and token_rank.bin.
// These files feed the isc[e * NGRP + g] tables (Worker::set_cold and load_deferred_groups) and the
// lm_head row gather of the balanced draft head (set_draft_vocab) - ids past the end there are silent
// device-side out-of-bounds writes/reads, so every id is checked whole and in range here, at the one
// parse point both entry points and the unit test share. Rules:
//   cold_experts.json - read only up to kMaxColdJsonBytes (the cap runs while reading, before the
//     parse, so a junk file of any size gets a clean refusal instead of being slurped whole), then a
//     strict JSON object {"layers": [...]} with exactly `layers` arrays of whole expert ids in
//     [0, experts), no duplicates within a layer;
//   token_rank.bin    - the released reader's read, validated: at most `cap` whole little-endian
//     int32 ids (the draft head consumes only the first `cap` rows) and every id read must lie in
//     [0, vocab). Bytes past the last whole id inside the cap are trailing padding the old
//     fread(data, 4, cap, f) never looked at, and a short or empty file meant "fewer rows / draft
//     vocab off", exactly like a missing one: all of those load, as on every server released before
//     this check existed. Only ids outside the vocabulary - what used to reach the tables as silent
//     device-side OOB - and a read that comes up short under a concurrent truncate are refused.
//     A stream ftell cannot size - a pipe, a FIFO - gets the same capped read without a size, to
//     the end of the stream: a failed size probe must never stand in for "no sidecar" there, while
//     a read error or a close mid-id is refused (a stream has no trailing padding to excuse a half
//     id, and a clean end, an empty one included, is just the list the stream carried).
// A bad sidecar throws std::runtime_error with the path and the rule; the C entry point catches it at
// its boundary and refuses the load. A missing file is not an error (both sidecars are optional).
#pragma once
#include "../json.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace sidecar {

// Upper bound for cold_experts.json itself: the real file is 48 layers x at most 512 expert ids, a
// little over 100 KB. 16 MiB is the same generous cap checkpoint.h puts on the manifest file
// (kMaxManifestBytes), and it is applied while reading, before the JSON parse ever runs.
inline constexpr unsigned long long kMaxColdJsonBytes = 16ull << 20;

namespace detail {
// fread exactly ids.size() int32s into ids and close f. On a regular file sized from ftell the read
// cannot come up short; it fires on a file truncated under the size probe, where an uninitialized
// tail of ids would otherwise reach the device tables. Streams the probe cannot size never get
// here - they are read blind to their end by load_token_rank.
inline void read_ids(FILE* f, std::vector<int>& ids, const std::string& path) {
    const size_t got = fread(ids.data(), 4, ids.size(), f);
    fclose(f);
    if (got != ids.size())
        throw std::runtime_error(path + ": token_rank.bin: short read (" + std::to_string(got) + " of " +
                                 std::to_string(ids.size()) + " ids)");
}
}  // namespace detail

inline std::vector<int> load_token_rank(const std::string& path, int cap, int vocab) {
    std::vector<int> ids;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return ids;
    const int seeked = fseek(f, 0, SEEK_END);
    const long long bytes = seeked == 0 ? ftell(f) : -1;
    if (bytes < 0) {
        // the size probe cannot see this stream - not a regular file: a pipe, a FIFO. A failed
        // probe is not "empty": sizing the list from it would hand back the empty vocabulary for
        // every piped rank file, silently, where the released fread(data, 4, cap, f) read the ids
        // off the same stream without ever seeking. Run that capped read blind, to the end of the
        // stream, and do not call a broken read success: ferror is an I/O error, and a last id cut
        // in half is a premature close - both refused. A clean end is the list the stream carried,
        // an empty one included, exactly like an empty regular file.
        rewind(f);  // no-op where seeking failed (the stream is at its start); clears the seek error
        // byte count, not element count: an element-sized fread consumes a partial last id and
        // hides exactly the evidence this stream needs, where the regular-file branch can afford
        // to ignore trailing padding because the file size said what was meant to arrive.
        ids.resize(cap);
        const size_t got = fread(ids.data(), 1, (size_t)cap * 4, f);
        const int err = ferror(f);
        fclose(f);
        if (err)
            throw std::runtime_error(path + ": token_rank.bin: read error on a stream ftell cannot size");
        if (got % 4 != 0)
            throw std::runtime_error(path + ": token_rank.bin: stream ended mid-id (" + std::to_string(got / 4) +
                                     " whole ids, then a partial one) - premature close");
        ids.resize(got / 4);
    } else {
        fseek(f, 0, SEEK_SET);
        // whole ids inside the cap region only - what the released reader read, and nothing more: bytes
        // past the last whole id (or past cap * 4) were never part of the list, as on main
        const long long whole = bytes / 4;
        ids.resize(std::min<long long>(whole, cap));
        detail::read_ids(f, ids, path);
    }
    for (int t : ids)
        if (t < 0 || t >= vocab)
            throw std::runtime_error(path + ": token_rank.bin: token id " + std::to_string(t) + " outside [0, " +
                                     std::to_string(vocab) + ")");
    return ids;
}

inline std::vector<std::vector<int>> load_cold_experts(const std::string& path, int layers, int experts) {
    std::vector<std::vector<int>> cold;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return cold;
    std::string js;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        js.append(buf, n);
        if (js.size() > kMaxColdJsonBytes) {
            fclose(f);
            throw std::runtime_error(path + ": cold_experts.json: over the " + std::to_string(kMaxColdJsonBytes) +
                                     " byte cap (not a cold-expert list)");
        }
    }
    fclose(f);
    const Json j = Json::parse_strict(js);
    if (j.type != Json::Obj || !j.has("layers") || j["layers"].type != Json::Arr)
        throw std::runtime_error(path + ": cold_experts.json: expected an object with a 'layers' array");
    const std::vector<Json>& ls = j["layers"].arr;
    if ((int)ls.size() != layers)
        throw std::runtime_error(path + ": cold_experts.json: " + std::to_string(ls.size()) + " layers, expected " +
                                 std::to_string(layers));
    for (size_t li = 0; li < ls.size(); li++) {
        if (ls[li].type != Json::Arr)
            throw std::runtime_error(path + ": cold_experts.json: layer " + std::to_string(li) + " is not an array");
        std::vector<int> v;
        std::vector<char> seen(experts, 0);
        for (const Json& x : ls[li].arr) {
            if (x.type != Json::Num || !std::isfinite(x.num) || x.num != std::floor(x.num))
                throw std::runtime_error(path + ": cold_experts.json: layer " + std::to_string(li) +
                                         ": expert id is not a whole number");
            if (x.num < 0 || x.num >= experts)
                throw std::runtime_error(path + ": cold_experts.json: layer " + std::to_string(li) + ": expert id " +
                                         std::to_string(x.num) + " outside [0, " + std::to_string(experts) + ")");
            const int e = (int)x.num;
            if (seen[e])
                throw std::runtime_error(path + ": cold_experts.json: layer " + std::to_string(li) +
                                         ": duplicate expert id " + std::to_string(e));
            seen[e] = 1;
            v.push_back(e);
        }
        cold.push_back(std::move(v));
    }
    return cold;
}

}  // namespace sidecar
