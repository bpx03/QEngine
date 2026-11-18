#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

int open_regular_file(const std::string& path, bool nofollow) {
    struct stat st;
    if (lstat(path.c_str(), &st) != 0)
        throw std::runtime_error(path + ": missing file (" + std::strerror(errno) + ")");
    if (nofollow && S_ISLNK(st.st_mode))
        throw std::runtime_error(path + ": symbolic link (the sandbox never follows symlinks)");
    if (!S_ISLNK(st.st_mode) && !S_ISREG(st.st_mode))
        throw std::runtime_error(path + ": not a regular file");  // FIFO, dir, socket, device: refuse before opening
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (nofollow ? O_NOFOLLOW : 0));
    if (fd < 0) throw std::runtime_error(path + ": cannot open (" + std::strerror(errno) + ")");
    if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) return fd;
    ::close(fd);
    throw std::runtime_error(path + ": not a regular file");  // swapped to a special file between lstat and open
}

Json SafeTensors::open(const std::string& path, const std::set<std::string>* only, bool shadow) {
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

    uint64_t hlen = 0;
    if (size >= 8) memcpy(&hlen, base, 8);
    // bounds (a truncated file, or not a safetensors file at all, e.g. a PyTorch .pt zip): a clear error, no wild read
    if (size < 10 || hlen > size - 8 || ((const char*)base)[8] != '{')
        throw std::runtime_error("not a safetensors file (bad header): " + path);
    std::string header((const char*)base + 8, hlen);
    Json j = Json::parse(header);
    ++safetensors_header_parses();
    index(j, (const char*)base + 8 + hlen, size - 8 - hlen, path, only, shadow);
    return j;
}

static size_t st_width_of(const std::string& d) {  // payload width of a supported safetensors dtype
    if (d == "U8" || d == "I8" || d == "BOOL" || d == "F8_E4M3" || d == "F8_E5M2") return 1;
    if (d == "BF16" || d == "F16" || d == "I16" || d == "U16") return 2;
    if (d == "F32" || d == "I32" || d == "U32") return 4;
    if (d == "F64" || d == "I64" || d == "U64") return 8;
    return 0;
}

Json SafeTensors::open_strict_validated(const std::string& path, bool nofollow, uint64_t max_size) {
    const int fd = open_regular_file(path, nofollow);
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) { ::close(fd); throw std::runtime_error(path + ": cannot stat file"); }
    if (max_size && (uint64_t)st.st_size > max_size) {
        ::close(fd);
        throw std::runtime_error(path + ": shard size " + std::to_string((long long)st.st_size) +
                                 " bytes exceeds the manifest limit " + std::to_string(max_size));
    }
    const size_t size = (size_t)st.st_size;
    void* base = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) throw std::runtime_error("mmap failed " + path);
    try {
        uint64_t hlen = 0;
        if (size >= 8) memcpy(&hlen, base, 8);
        if (size < 10 || hlen > size - 8 || hlen > (64ull << 20) || ((const char*)base)[8] != '{')
            throw std::runtime_error("not a safetensors file (bad header): " + path);
        const std::string header((const char*)base + 8, hlen);
        Json j = Json::parse_strict(header);  // the one and only parse of these header bytes
        ++safetensors_header_parses();
        if (j.type != Json::Obj) throw std::runtime_error(path + ": header is not a JSON object");
        // structural validation against the mapped bytes: supported dtype, bounded shape, and
        // contiguous, non-overlapping, exactly-covering payload (the previous file-validation checks)
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (const auto& e : j.obj) {
            if (e.first == "__metadata__") continue;
            const auto w = st_width_of(e.second.has("dtype") ? e.second["dtype"].str : "");
            bool shape_ok = w && e.second.has("shape") && e.second["shape"].type == Json::Arr &&
                            e.second.has("data_offsets") && e.second["data_offsets"].type == Json::Arr;
            uint64_t n = 1;
            if (shape_ok)
                for (const auto& d : e.second["shape"].arr) {
                    if (d.type != Json::Num || d.num < 0 || d.num > 9007199254740991.0 || (double)d.i() != d.num ||
                        n > size / (uint64_t)std::max<long long>(d.i(), 1) + 1)
                        { shape_ok = false; break; }
                    n *= (uint64_t)d.i();
                }
            const auto& off = shape_ok ? e.second["data_offsets"] : Json{};
            const bool offs_ok = shape_ok && off.arr.size() == 2 && off.arr[0].type == Json::Num && off.arr[1].type == Json::Num &&
                                 off.arr[0].num >= 0 && off.arr[1].num >= 0 && (double)off.arr[0].i() == off.arr[0].num &&
                                 (double)off.arr[1].i() == off.arr[1].num &&
                                 (uint64_t)off.arr[1].i() <= size - 8 - hlen && off.arr[1].i() >= off.arr[0].i();
            if (!offs_ok || (uint64_t)(off.arr[1].i() - off.arr[0].i()) != n * w)
                throw std::runtime_error(path + ": " + e.first + ": malformed dtype, shape or data_offsets");
            ranges.emplace_back((uint64_t)off.arr[0].i(), (uint64_t)off.arr[1].i());
        }
        std::sort(ranges.begin(), ranges.end());
        uint64_t next = 0;
        for (const auto& r : ranges) {
            if (r.first != next) throw std::runtime_error(path + ": data_offsets overlap or leave a gap");
            next = r.second;
        }
        if (next != size - 8 - hlen) throw std::runtime_error(path + ": payload bytes differ from the tensor extents");
        const char* data = (const char*)base + 8 + hlen;
        maps_.push_back({base, size});
        paths_.push_back(path);
        index(j, data, size - 8 - hlen, path, nullptr, false);  // the tensors come from the object validated above
        return j;
    } catch (...) { munmap(base, size); throw; }
}

void SafeTensors::index(const Json& j, const char* data, size_t dsize, const std::string& path,
                        const std::set<std::string>* only, bool shadow) {
    for (auto& [name, v] : j.obj) {
        if (name == "__metadata__") continue;
        if (only && !only->count(name)) continue;
        TensorInfo t;
        t.dtype = v["dtype"].str;
        for (auto& d : v["shape"].arr) t.shape.push_back(d.i());
        size_t b = v["data_offsets"][0].i(), e = v["data_offsets"][1].i();
        if (b > e || e > dsize) throw std::runtime_error("tensor " + name + ": data offsets outside the file " + path);
        t.data = data + b;
        t.nbytes = e - b;
        // a name in two files is allowed only with identical content (MTP: mtp.safetensors and mtp-int4 both carry the
        // unquantised tensors); a differing duplicate (e.g. a stale expert link next to a re-converted layer file)
        // would otherwise override silently, depending on the directory order
        auto it = tensors_.find(name);
        if (!shadow && it != tensors_.end() && (it->second.dtype != t.dtype || it->second.shape != t.shape ||
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

int SafeTensors::map_of(const void* p) const {
    for (size_t i = 0; i < maps_.size(); i++) {
        const char* b = (const char*)maps_[i].first;
        if ((const char*)p >= b && (const char*)p < b + maps_[i].second) return (int)i;
    }
    return -1;
}

size_t SafeTensors::mapped_bytes() const {
    size_t n = 0;
    for (auto& m : maps_) n += m.second;
    return n;
}

size_t SafeTensors::drop_prefix(const std::string& prefix, const std::vector<std::string>& keep) const {
    const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    std::vector<std::vector<std::pair<uintptr_t, uintptr_t>>> rs(maps_.size());  // per mapping: page ranges
    for (auto it = tensors_.lower_bound(prefix); it != tensors_.end() && it->first.compare(0, prefix.size(), prefix) == 0; ++it) {
        const int m = map_of(it->second.data);
        if (m < 0 || !it->second.nbytes) continue;
        bool kept = false;
        for (auto& k : keep) kept |= it->first.find(k) != std::string::npos;
        if (kept) continue;
        const uintptr_t a = ((uintptr_t)it->second.data + pg - 1) / pg * pg, e = ((uintptr_t)it->second.data + it->second.nbytes) / pg * pg;
        if (e > a) rs[m].push_back({a, e});
    }
    size_t asked = 0;
    for (size_t m = 0; m < maps_.size(); m++) {
        if (rs[m].empty()) continue;
        auto& v = rs[m];
        std::sort(v.begin(), v.end());
        std::vector<std::pair<uintptr_t, uintptr_t>> mg;  // merged (a layer's tensors are mostly adjacent)
        for (auto& r : v)
            if (!mg.empty() && r.first <= mg.back().second) mg.back().second = std::max(mg.back().second, r.second);
            else mg.push_back(r);
        const int fd = ::open(paths_[m].c_str(), O_RDONLY);
        const uintptr_t base = (uintptr_t)maps_[m].first;
        for (auto& [a, e] : mg) {
            madvise((void*)a, e - a, MADV_DONTNEED);  // (clean read-only shared mapping: no data lost)
            if (fd >= 0) posix_fadvise(fd, (off_t)(a - base), (off_t)(e - a), POSIX_FADV_DONTNEED);
            asked += e - a;
        }
        if (fd >= 0) ::close(fd);
    }
    return asked;
}

void LoadDrop::begin(int loaders, bool on, std::vector<std::string> keep) {
    std::lock_guard<std::mutex> lk(mu_);
    on_ = on;
    keep_ = std::move(keep);
    loaders_ = loaders;
    ready_ = 0;
    g_.clear();
    bytes_ = 0;
}

void LoadDrop::expect(const SafeTensors* st, const std::string& prefix) {
    std::lock_guard<std::mutex> lk(mu_);
    if (on_) g_[{st, prefix}].first++;
}

void LoadDrop::ready() {
    std::vector<std::pair<const SafeTensors*, std::string>> go;
    {
        std::lock_guard<std::mutex> lk(mu_);
        ready_++;
        sweep(&go);
    }
    drop(go);
}

void LoadDrop::done(const SafeTensors* st, const std::string& prefix) {
    std::vector<std::pair<const SafeTensors*, std::string>> go;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = g_.find({st, prefix});
        if (it == g_.end()) return;
        it->second.second++;
        sweep(&go);
    }
    drop(go);
}

void LoadDrop::sweep(std::vector<std::pair<const SafeTensors*, std::string>>* go) {
    if (!on_ || ready_ < loaders_) return;
    for (auto it = g_.begin(); it != g_.end();)
        if (it->second.second >= it->second.first) {
            go->push_back(it->first);
            it = g_.erase(it);
        } else {
            ++it;
        }
}

void LoadDrop::drop(const std::vector<std::pair<const SafeTensors*, std::string>>& go) {
    // (outside the lock: the other loaders' done() calls do not wait for the madvise / fadvise work)
    if (go.empty()) return;
    std::vector<std::string> keep;
    {
        std::lock_guard<std::mutex> lk(mu_);
        keep = keep_;
    }
    size_t n = 0;
    for (auto& [st, prefix] : go) n += st->drop_prefix(prefix, keep);
    std::lock_guard<std::mutex> lk(mu_);
    bytes_ += n;
}

size_t LoadDrop::dropped() const {
    std::lock_guard<std::mutex> lk(mu_);
    return bytes_;
}

SafeTensors::~SafeTensors() {
    for (auto& [p, n] : maps_) munmap(p, n);
}

const TensorInfo& SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}
