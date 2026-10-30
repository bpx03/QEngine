#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

size_t TensorInfo::elem_size() const {
    static const std::map<std::string, size_t> sizes = {
        {"F64", 8}, {"I64", 8}, {"U64", 8}, {"F32", 4}, {"I32", 4}, {"U32", 4},
        {"BF16", 2}, {"F16", 2}, {"I16", 2}, {"U16", 2},
        {"I8", 1}, {"U8", 1}, {"BOOL", 1}, {"F8_E4M3", 1}, {"F8_E5M2", 1}};
    auto it = sizes.find(dtype);
    if (it == sizes.end()) throw std::runtime_error("unknown safetensors dtype " + dtype);
    return it->second;
}

std::vector<std::string> safetensors_files(const std::string& dir) {
    struct stat st;
    if (stat((dir + "/model.safetensors").c_str(), &st) == 0) return {dir + "/model.safetensors"};
    std::ifstream f(dir + "/model.safetensors.index.json");
    if (!f) throw std::runtime_error("no model.safetensors or model.safetensors.index.json in " + dir);
    std::stringstream ss;
    ss << f.rdbuf();
    std::set<std::string> names;
    const Json index = Json::parse(ss.str());  // named: a range-for does not keep a temporary's member alive
    for (auto& [tensor, file] : index["weight_map"].obj)
        if (file.str != "model_mtp.safetensors") names.insert(file.str);
    std::vector<std::string> out;
    for (auto& n : names) out.push_back(dir + "/" + n);
    return out;
}

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
