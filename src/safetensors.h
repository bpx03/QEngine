#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "json.h"

struct TensorInfo {
    std::string dtype;  // "BF16", "I32", "I64", ...
    std::vector<long long> shape;
    const char* data = nullptr;
    size_t nbytes = 0;
    size_t elem_size() const { return dtype == "BF16" ? 2 : dtype == "I64" ? 8 : 4; }
};

// what a page-out did: bytes that left the page cache, and whether every file could be measured (cachestat, or
// mincore where the kernel allows it); else `dropped` covers only the measured files and `requested` all of them
struct PageOut {
    size_t dropped = 0, requested = 0;
    bool measured = true;
};
// Cached of /proc/meminfo in bytes (the whole host's page cache; -1: unreadable): the estimate when !measured
long long meminfo_cached();
// drops the page cache of one read-only shared file mapping (path, its mapping p, n bytes): MADV_PAGEOUT, then always
// MADV_DONTNEED (MADV_PAGEOUT silently skips files the process neither owns nor may write: docker --user, models of
// another user) and POSIX_FADV_DONTNEED (also the cached pages never faulted into this mapping); adds to *out
void drop_file_cache(const std::string& path, void* p, size_t n, PageOut* out);

// Guarded open for checkpoint files: lstat must show a regular file — a symlink is
// rejected outright when nofollow (the sandbox case), a FIFO or directory before any open() — then
// O_RDONLY|O_CLOEXEC (|O_NOFOLLOW), then fstat re-checks the opened file is still regular. A blocking
// special file can never turn into a stuck read, and the sandbox never follows a symlink. Throws
// runtime_error with "missing"/"symbolic link"/"not a regular file"/"cannot open" and returns the fd.
int open_regular_file(const std::string& path, bool nofollow);

// Test seam: counts every safetensors header parsed through any entry point. A checkpoint
// must parse each shard header exactly once — the unit asserts the increment across Checkpoint::open
// equals the number of safetensors files it opens. Previously it was two parses per file: a strict
// validate pass over the bytes, then SafeTensors::open REOPENED the path and reparsed it permissively
// (double work plus a symlink/swap window between the two opens).
inline std::size_t& safetensors_header_parses() { static std::size_t n = 0; return n; }

// mmap()s one or more .safetensors files and indexes tensors by name.
class SafeTensors {
public:
    // only != nullptr: register just these names (an index that maps a name to this file); shadow: this file's tensors
    // replace earlier ones of the same name on purpose (tuned weights), no duplicate check
    Json open(const std::string& path, const std::set<std::string>* only = nullptr, bool shadow = false);
    // Single-open checkpoint facade (used only by flash/checkpoint.h): guarded open (open_regular_file:
    // S_ISREG + O_NOFOLLOW when nofollow + fstat), mmap, and the header is parsed STRICTLY from the
    // mapped bytes exactly once; the payload is validated against those same bytes and the tensor table
    // is indexed from THAT parsed object, which is returned for manifest cross-validation. No reopen,
    // no second parse, no TOCTOU window between validate and open. max_size (0: unbounded) rejects an
    // oversized file before mapping. Legacy open() keeps its historical behaviour for other subsystems.
    Json open_strict_validated(const std::string& path, bool nofollow, uint64_t max_size = 0);
    ~SafeTensors();
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorInfo& get(const std::string& name) const;
    // a tensor that lives elsewhere (another container mapped by the caller, which keeps the memory alive);
    // replaces an entry of the same name
    void put(const std::string& name, const TensorInfo& t) { tensors_[name] = t; }
    // the file and offset behind a mapped address (empty path: not in any mapping)
    std::string file_of(const void* p, size_t* off) const;
    // drops the page cache of every mapped file for which keep(base, size) is false (see drop_file_cache); the
    // mappings stay valid: a later read faults the page in again from the file
    PageOut page_out(const std::function<bool(const void* base, size_t size)>& keep);
    // drops the file pages of every tensor whose name starts with `prefix` (whole pages inside the tensors: a page shared
    // with a tensor outside the prefix stays): MADV_DONTNEED on the mapping, POSIX_FADV_DONTNEED on the file. The mapping
    // stays valid (a later read faults the page in from the file). Returns the bytes asked for.
    size_t drop_prefix(const std::string& prefix, const std::vector<std::string>& keep = {}) const;  // keep: name parts never dropped
    size_t mapped_bytes() const;

private:
    int map_of(const void* p) const;
    // index one parsed (and already payload-validated) header into the tensor table
    void index(const Json& j, const char* data, size_t dsize, const std::string& path,
               const std::set<std::string>* only, bool shadow);
    std::map<std::string, TensorInfo> tensors_;
    std::vector<std::pair<void*, size_t>> maps_;
    std::vector<std::string> paths_;
};

// Load-time page cache: the model files would otherwise fill the page cache next to the memory the engine pins during
// the load (a 110 GB cgroup: 28M pages scanned in direct reclaim; with swap the reclaim may page out the server). A
// group of tensors (one layer: a name prefix) is dropped once every loader that reads it is done with it: each loader
// declares its groups (expect, then ready); nothing is dropped before all `loaders` are ready, so a group read by
// several cards (column slices share pages) is read from the file once.
class LoadDrop {
public:
    // keep: tensors whose name contains one of these stay cached (Flash-Next: the n-gram table, "ple_embedding", sits under
    // layer 1's prefix and is read from the mapping while serving in disk mode)
    void begin(int loaders, bool on, std::vector<std::string> keep = {});
    bool on() const { return on_; }
    void expect(const SafeTensors* st, const std::string& prefix);
    void ready();
    void done(const SafeTensors* st, const std::string& prefix);
    size_t dropped() const;  // bytes asked to drop so far

private:
    void sweep(std::vector<std::pair<const SafeTensors*, std::string>>* go);  // (mu_ held) complete groups -> go
    void drop(const std::vector<std::pair<const SafeTensors*, std::string>>& go);  // (mu_ not held)
    mutable std::mutex mu_;
    bool on_ = false;
    int loaders_ = 0, ready_ = 0;
    std::vector<std::string> keep_;
    std::map<std::pair<const SafeTensors*, std::string>, std::pair<int, int>> g_;  // expected, done
    size_t bytes_ = 0;
};
