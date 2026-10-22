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

SafeTensors::~SafeTensors() {
    for (auto& [p, n] : maps_) munmap(p, n);
}

const TensorInfo& SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}
