#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
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

// mmap()s one or more .safetensors files and indexes tensors by name.
class SafeTensors {
public:
    void open(const std::string& path);
    ~SafeTensors();
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorInfo& get(const std::string& name) const;
    // the file and offset behind a mapped address (empty path: not in any mapping)
    std::string file_of(const void* p, size_t* off) const;
    // drops the page cache of every mapped file for which keep(base, size) is false (see drop_file_cache); the
    // mappings stay valid: a later read faults the page in again from the file
    PageOut page_out(const std::function<bool(const void* base, size_t size)>& keep);

private:
    std::map<std::string, TensorInfo> tensors_;
    std::vector<std::pair<void*, size_t>> maps_;
    std::vector<std::string> paths_;
};
