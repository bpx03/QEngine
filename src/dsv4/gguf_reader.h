#pragma once
// R16 [dsv4-gguf-iq]: read-only GGUF reader (format v2 / v3) for the DSV4 GGUF expert mode. Header, key / values,
// tensor infos, data offsets with the file's alignment; files mapped PROT_READ (never written). A directory holds ONE
// split set (same name stem, consistent split.count / split.no, no tensor name twice); missing split files are allowed
// (Maya-S keeps its Engram shards elsewhere), a caller checks the tensors it needs. Routed experts:
// blk.<l>.ffn_{gate,up,down}_exps.weight, ggml dims [K, N, 384] (row-major [384][N][K] in blocks of 256).
// Hardened against corrupt files (review M1-M3): alignment validated, overflow-safe bounds, string length caps.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dsv4 {

struct GgTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> ne;        // ggml order: ne[0] = innermost (row length)
    uint64_t offset = 0;             // from the file's data start
    uint64_t nbytes = 0;             // 0 when the type's block size is unknown here (sized = false)
    bool sized = false;
    const uint8_t* data = nullptr;   // mapped (nullptr when not sized)
    int file = -1;
};

struct GgFile {
    std::string path;
    int fd = -1;
    const uint8_t* base = nullptr;
    size_t size = 0;
    uint32_t version = 0;
    uint64_t alignment = 32, data_off = 0;
    int split_count = 1, split_no = 0;
    std::map<std::string, std::string> kv;   // scalar values as text, arrays as "[n x type]"
    struct Arr { uint32_t et = 0; uint64_t n = 0; const uint8_t* data = nullptr; };   // [dsv4-rel-gguf]: scalar arrays
    std::map<std::string, Arr> arr;          // numeric arrays in place (mapped; et = gguf value type)
};

class GgufModel {
public:
    GgufModel() = default;
    GgufModel(const GgufModel&) = delete;
    GgufModel& operator=(const GgufModel&) = delete;
    ~GgufModel();
    // map one file (checks the header and that every sized tensor's bytes lie inside the file: a partial download
    // fails); a tensor name already present (another file) fails
    bool add_file(const std::string& path, std::string* err);
    // the *.gguf files of a directory, name order: one split set (see above); false on the first bad file
    bool open_dir(const std::string& dir, std::string* err);
    const GgTensor* find(const std::string& name) const;
    // [dsv4-rel-gguf]: a key of the first file (split.no 0) as text / a numeric array as int64 or double; false if absent
    bool kv(const std::string& key, std::string* v) const;
    bool arr_i64(const std::string& key, std::vector<int64_t>* v) const;
    bool arr_f64(const std::string& key, std::vector<double>* v) const;
    const std::vector<GgTensor>& tensors() const { return t_; }
    const std::vector<GgFile>& files() const { return f_; }
    // routed expert matrix m (0 gate = w1, 1 up = w3, 2 down = w2) of layer l: the tensor, or nullptr
    const GgTensor* expert_tensor(int l, int m) const;
    // the raw GGUF bytes of expert e's matrix (one [N][K] slice) and its byte count; nullptr when t is not a sized
    // 3-d tensor or e is out of range
    const uint8_t* expert_slice(const GgTensor& t, int e, size_t* nbytes) const;

private:
    std::vector<GgFile> f_;
    std::vector<GgTensor> t_;
    std::map<std::string, size_t> idx_;
};

}  // namespace dsv4
