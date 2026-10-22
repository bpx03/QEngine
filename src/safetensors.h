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
    size_t elem_size() const { return dtype == "BF16" ? 2 : dtype == "I64" ? 8 : 4; }
};

// mmap()s one or more .safetensors files and indexes tensors by name.
class SafeTensors {
public:
    void open(const std::string& path);
    ~SafeTensors();
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorInfo& get(const std::string& name) const;

private:
    std::map<std::string, TensorInfo> tensors_;
    std::vector<std::pair<void*, size_t>> maps_;
};
