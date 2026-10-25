#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

SafeTensors::~SafeTensors() {
    for (auto& [p, n] : maps_) munmap(p, n);
}

const TensorInfo& SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}
