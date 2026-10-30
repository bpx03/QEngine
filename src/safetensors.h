#pragma once
#include <map>
#include <string>
#include <vector>

#include "json.h"

struct TensorInfo {
    std::string dtype;  // "BF16", "I32", "I64", ...
    std::vector<long long> shape;
    const char* data = nullptr;
    size_t nbytes = 0;
    size_t elem_size() const;
};

// Weight files of a checkpoint directory: model.safetensors if present, else the shards named in
// model.safetensors.index.json (sorted, without model_mtp.safetensors, which is opened separately).
std::vector<std::string> safetensors_files(const std::string& dir);

// mmap()s one or more .safetensors files and indexes tensors by name.
class SafeTensors {
public:
    void open(const std::string& path);
    ~SafeTensors();
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorInfo& get(const std::string& name) const;
    // the file and offset behind a mapped address (empty path: not in any mapping)
    std::string file_of(const void* p, size_t* off) const;

private:
    std::map<std::string, TensorInfo> tensors_;
    std::vector<std::pair<void*, size_t>> maps_;
    std::vector<std::string> paths_;
};
