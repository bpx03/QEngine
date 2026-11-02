#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

void SafeTensors::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;
    void* base = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) throw std::runtime_error("mmap failed " + path);
    maps_.push_back({base, size});
    paths_.push_back(path);

    uint64_t hlen;
    memcpy(&hlen, base, 8);
    std::string header((const char*)base + 8, hlen);
    Json j = Json::parse(header);
    const char* data = (const char*)base + 8 + hlen;
    for (auto& [name, v] : j.obj) {
        if (name == "__metadata__") continue;
        TensorInfo t;
        t.dtype = v["dtype"].str;
        for (auto& d : v["shape"].arr) t.shape.push_back(d.i());
        size_t b = v["data_offsets"][0].i(), e = v["data_offsets"][1].i();
        t.data = data + b;
        t.nbytes = e - b;
        // a name in two files is allowed only with identical content (MTP: mtp.safetensors and mtp-int4 both carry the
        // unquantised tensors); a differing duplicate (e.g. a stale expert link next to a re-converted layer file)
        // would otherwise override silently, depending on the directory order
        auto it = tensors_.find(name);
        if (it != tensors_.end() && (it->second.dtype != t.dtype || it->second.shape != t.shape ||
                                     it->second.nbytes != t.nbytes || memcmp(it->second.data, t.data, t.nbytes) != 0)) {
            size_t off = 0;
            throw std::runtime_error("tensor " + name + " in both " + file_of(it->second.data, &off) + " and " + path +
                                     " with different content");
        }
        tensors_[name] = t;
    }
}

std::string SafeTensors::file_of(const void* p, size_t* off) const {
    for (size_t i = 0; i < maps_.size(); i++) {
        const char* b = (const char*)maps_[i].first;
        if ((const char*)p >= b && (const char*)p < b + maps_[i].second) {
            *off = (const char*)p - b;
            return paths_[i];
        }
    }
    return std::string();
}

#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif
#ifndef __NR_cachestat
#define __NR_cachestat 451  // (x86_64, aarch64 and most others since Linux 6.5)
#endif
// the bytes of a file in the page cache: cachestat (Linux 6.5, needs only the fd); else mincore on its mapping, which
// the kernel answers truthfully only for a file the process owns or may write (else every page reads resident);
// -1: unknown
static long long cached_bytes(int fd, void* base, size_t n) {
    struct { uint64_t off, len; } range = {0, 0};  // (len 0: to the end of the file)
    struct { uint64_t nr_cache, nr_dirty, nr_writeback, nr_evicted, nr_recently_evicted; } cs = {};
    const long pg = sysconf(_SC_PAGESIZE);
    if (fd >= 0 && syscall(__NR_cachestat, fd, &range, &cs, 0) == 0) return (long long)(cs.nr_cache * (uint64_t)pg);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0 || (st.st_uid != geteuid() && faccessat(fd, "", W_OK, AT_EMPTY_PATH) != 0)) return -1;
    std::vector<unsigned char> v;
    unsigned long long res = 0;
    for (size_t off = 0; off < n; off += (size_t)1 << 30) {  // 1 GiB pieces: a small vector
        const size_t len = std::min(n - off, (size_t)1 << 30);
        v.resize((len + pg - 1) / pg);
        if (mincore((char*)base + off, len, v.data()) != 0) return -1;
        for (unsigned char c : v) res += (c & 1) ? (unsigned long long)pg : 0;
    }
    return (long long)std::min<unsigned long long>(res, n);
}

long long meminfo_cached() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long long kb = -1;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "Cached:", 7)) { kb = atoll(line + 7); break; }
    fclose(f);
    return kb < 0 ? -1 : kb * 1024;
}

void drop_file_cache(const std::string& path, void* p, size_t n, PageOut* out) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    const long long before = cached_bytes(fd, p, n);
    madvise(p, n, MADV_PAGEOUT);   // reclaims at once where allowed (unsupported before Linux 5.4: EINVAL, harmless)
    madvise(p, n, MADV_DONTNEED);  // unmaps the rest from this process (clean read-only shared mapping: no data lost)
    if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // drops the cached pages no process maps
    const long long after = cached_bytes(fd, p, n);
    if (fd >= 0) ::close(fd);
    out->requested += n;
    if (before < 0 || after < 0) out->measured = false;
    else out->dropped += before > after ? (size_t)(before - after) : 0;
}

PageOut SafeTensors::page_out(const std::function<bool(const void* base, size_t size)>& keep) {
    PageOut o;
    for (size_t i = 0; i < maps_.size(); i++)
        if (!keep(maps_[i].first, maps_[i].second)) drop_file_cache(paths_[i], maps_[i].first, maps_[i].second, &o);
    return o;
}

SafeTensors::~SafeTensors() {
    for (auto& [p, n] : maps_) munmap(p, n);
}

const TensorInfo& SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}
