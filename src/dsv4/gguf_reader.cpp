// R16 [dsv4-gguf-iq]: GGUF reader (gguf_reader.h). Own implementation of the documented GGUF container format.
#include "gguf_reader.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <set>

#include "gguf_iq.h"

namespace dsv4 {

namespace {
constexpr uint64_t MAX_KEY = 65536, MAX_NAME = 4096, MAX_STR = 1u << 26;   // review M3: length caps
struct Cur {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if ((size_t)(end - p) < sizeof(T)) { ok = false; return v; }
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    // a string of at most cap bytes; keep = false: skipped without a copy
    std::string str(uint64_t cap, bool keep = true) {
        const uint64_t n = get<uint64_t>();
        if (!ok || n > cap || (uint64_t)(end - p) < n) { ok = false; return {}; }
        std::string s = keep ? std::string((const char*)p, n) : std::string();
        p += n;
        return s;
    }
};
// gguf value types: 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 string 9 array 10 u64 11 i64 12 f64
size_t scalar_size(uint32_t t) {
    static const size_t s[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    return t < 13 ? s[t] : 0;
}
// value as text; *u = the value when it is an unsigned / signed integer >= 0 (else untouched), *isint set
std::string read_value(Cur& c, uint32_t t, uint64_t* u, bool* isint) {
    char b[64];
    *isint = false;
    auto ui = [&](uint64_t v) { *u = v; *isint = true; snprintf(b, sizeof b, "%llu", (unsigned long long)v); return std::string(b); };
    auto si = [&](int64_t v) { if (v >= 0) { *u = (uint64_t)v; *isint = true; } snprintf(b, sizeof b, "%lld", (long long)v); return std::string(b); };
    switch (t) {
        case 0: return ui(c.get<uint8_t>());
        case 1: return si(c.get<int8_t>());
        case 2: return ui(c.get<uint16_t>());
        case 3: return si(c.get<int16_t>());
        case 4: return ui(c.get<uint32_t>());
        case 5: return si(c.get<int32_t>());
        case 6: snprintf(b, sizeof b, "%g", c.get<float>()); return b;
        case 7: return c.get<uint8_t>() ? "true" : "false";
        case 8: return c.str(MAX_STR);
        case 10: return ui(c.get<uint64_t>());
        case 11: return si(c.get<int64_t>());
        case 12: snprintf(b, sizeof b, "%g", c.get<double>()); return b;
        case 9: {
            const uint32_t et = c.get<uint32_t>();
            const uint64_t n = c.get<uint64_t>();
            if (!c.ok) return {};
            if (et == 8) {
                if (n > (uint64_t)(c.end - c.p) / 8) { c.ok = false; return {}; }   // each string >= 8 B
                for (uint64_t i = 0; i < n && c.ok; i++) c.str(MAX_STR, false);
            } else {
                const size_t es = scalar_size(et);   // nested arrays (et 9): not written by GGUF writers
                if (!es || n > (uint64_t)(c.end - c.p) / es) { c.ok = false; return {}; }
                c.p += es * n;
            }
            snprintf(b, sizeof b, "[%llu x t%u]", (unsigned long long)n, et);
            return b;
        }
    }
    c.ok = false;
    return {};
}
// "<stem>-NNNNN-of-NNNNN.gguf" -> stem; any other name -> the name without ".gguf"
std::string stem_of(const std::string& n) {
    std::string s = n.substr(0, n.size() - 5);
    const size_t p = s.rfind("-of-");
    if (p != std::string::npos && p >= 6 && s[p - 6] == '-') return s.substr(0, p - 6);
    return s;
}
}  // namespace

GgufModel::~GgufModel() {
    for (auto& f : f_) {
        if (f.base) munmap((void*)f.base, f.size);
        if (f.fd >= 0) close(f.fd);
    }
}

bool GgufModel::add_file(const std::string& path, std::string* err) {
    GgFile F;
    F.path = path;
    auto fail = [&](const std::string& m) {
        if (err) *err = path + ": " + m;
        if (F.base) munmap((void*)F.base, F.size);
        if (F.fd >= 0) close(F.fd);
        return false;
    };
    F.fd = open(path.c_str(), O_RDONLY);
    if (F.fd < 0) return fail("open");
    struct stat sb;
    if (fstat(F.fd, &sb) != 0) return fail("stat");
    F.size = (size_t)sb.st_size;
    if (F.size < 24) return fail("too small for a GGUF header");
    void* m = mmap(nullptr, F.size, PROT_READ, MAP_SHARED, F.fd, 0);
    if (m == MAP_FAILED) return fail("mmap");
    F.base = (const uint8_t*)m;
    madvise(m, F.size, MADV_RANDOM);   // the header pass touches only the start; expert reads are explicit
    posix_fadvise(F.fd, 0, 0, POSIX_FADV_RANDOM);   // review L3: no readahead past the slices the load reads
    Cur c{F.base, F.base + F.size};
    const uint32_t magic = c.get<uint32_t>();
    F.version = c.get<uint32_t>();
    if (magic != 0x46554747u) return fail("not a GGUF file");
    if (F.version < 2 || F.version > 3) return fail("GGUF version " + std::to_string(F.version));
    const uint64_t nt = c.get<uint64_t>(), nkv = c.get<uint64_t>();
    // every kv / tensor info takes >= 8 + 4 bytes: counts beyond the file are corrupt (no huge reserve)
    if (nt > F.size / 12 || nkv > F.size / 12) return fail("implausible tensor / kv count");
    bool have_align = false;
    for (uint64_t i = 0; i < nkv && c.ok; i++) {
        const std::string k = c.str(MAX_KEY);
        const uint32_t t = c.get<uint32_t>();
        uint64_t u = 0;
        bool isint = false;
        const uint8_t* vp = c.p;
        const std::string v = read_value(c, t, &u, &isint);
        if (!c.ok) break;
        if (t == 9) {   // [dsv4-rel-gguf]: numeric arrays kept in place (element type, count, data)
            uint32_t et;
            uint64_t n;
            memcpy(&et, vp, 4);
            memcpy(&n, vp + 4, 8);
            if (et != 8 && et != 9) F.arr[k] = {et, n, vp + 12};
        }
        if (k == "general.alignment") {   // review M1: u32, power of two, 1 .. 1 MiB
            if (t != 4 || u == 0 || (u & (u - 1)) || u > (1u << 20)) return fail("bad general.alignment " + v);
            F.alignment = u;
            have_align = true;
        } else if (k == "split.count" && isint && u >= 1 && u <= 4096) F.split_count = (int)u;
        else if (k == "split.no" && isint && u < 4096) F.split_no = (int)u;
        F.kv[k] = v;
    }
    (void)have_align;
    std::vector<GgTensor> ts;
    for (uint64_t i = 0; i < nt && c.ok; i++) {
        GgTensor T;
        T.name = c.str(MAX_NAME);
        const uint32_t nd = c.get<uint32_t>();
        if (nd == 0 || nd > 4) c.ok = false;
        for (uint32_t d = 0; d < nd && c.ok; d++) T.ne.push_back(c.get<uint64_t>());
        T.type = c.get<uint32_t>();
        T.offset = c.get<uint64_t>();
        T.file = (int)f_.size();
        ts.push_back(std::move(T));
    }
    if (!c.ok) return fail("truncated or corrupt header");
    const uint64_t hdr = (uint64_t)(c.p - F.base);
    F.data_off = (hdr + F.alignment - 1) / F.alignment * F.alignment;   // hdr < size <= 2^63, alignment <= 2^20
    if (F.data_off > F.size) return fail("data section starts past the file end");
    const uint64_t room = F.size - F.data_off;
    for (auto& T : ts) {
        if (T.offset % F.alignment) return fail("tensor " + T.name + ": offset not aligned");
        int bw, bb;
        gg_type_block(T.type, bw, bb);
        uint64_t n = 1;
        for (auto x : T.ne)   // review M2: overflow-safe element count
            if (x == 0 || __builtin_mul_overflow(n, x, &n)) return fail("tensor " + T.name + ": bad dims");
        if (!bw) continue;   // review L1: type unknown here: kept unsized, fails only if requested
        if (T.ne[0] % bw) return fail("tensor " + T.name + ": row not whole blocks");
        uint64_t nbytes;
        if (__builtin_mul_overflow(n / bw, (uint64_t)bb, &nbytes)) return fail("tensor " + T.name + ": size overflow");
        if (T.offset > room || nbytes > room - T.offset) return fail("tensor " + T.name + " ends past the file end (partial file?)");
        T.nbytes = nbytes;
        T.sized = true;
        T.data = F.base + F.data_off + T.offset;
    }
    for (auto& T : ts)   // review M4: one tensor name once over the model's files
        if (idx_.count(T.name)) return fail("tensor " + T.name + " is also in " + f_[t_[idx_[T.name]].file].path);
    std::set<std::string> here;
    for (auto& T : ts)
        if (!here.insert(T.name).second) return fail("tensor " + T.name + " twice in the file");
    f_.push_back(std::move(F));
    for (auto& T : ts) {
        idx_[T.name] = t_.size();
        t_.push_back(std::move(T));
    }
    return true;
}

bool GgufModel::open_dir(const std::string& dir, std::string* err) {
    DIR* d = opendir(dir.c_str());
    if (!d) { if (err) *err = dir + ": opendir"; return false; }
    std::vector<std::string> names;
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0) names.push_back(n);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    if (names.empty()) { if (err) *err = dir + ": no .gguf files"; return false; }
    // review M4: one split set only (another quant of the same model in the same directory would mix silently)
    const std::string stem = stem_of(names[0]);
    for (auto& n : names)
        if (stem_of(n) != stem) {
            if (err) *err = dir + ": more than one GGUF model (" + names[0] + ", " + n + "): keep one split set per directory";
            return false;
        }
    std::set<int> nos;
    for (auto& n : names) {
        if (!add_file(dir + "/" + n, err)) return false;
        const GgFile& F = f_.back();
        if (F.split_count != f_[0].split_count || F.split_no >= F.split_count || !nos.insert(F.split_no).second) {
            if (err) *err = F.path + ": split.no " + std::to_string(F.split_no) + " / split.count " + std::to_string(F.split_count) +
                            " does not fit the set of " + f_[0].path;
            return false;
        }
    }
    return true;
}

const GgTensor* GgufModel::find(const std::string& name) const {
    auto it = idx_.find(name);
    return it == idx_.end() ? nullptr : &t_[it->second];
}

const GgTensor* GgufModel::expert_tensor(int l, int m) const {
    static const char* nm[3] = {"gate", "up", "down"};
    if (m < 0 || m > 2) return nullptr;
    return find("blk." + std::to_string(l) + ".ffn_" + nm[m] + "_exps.weight");
}

const uint8_t* GgufModel::expert_slice(const GgTensor& t, int e, size_t* nbytes) const {
    if (!t.sized || t.ne.size() < 3 || e < 0 || (uint64_t)e >= t.ne[2] || t.nbytes % t.ne[2]) return nullptr;   // review L2
    const size_t per = t.nbytes / t.ne[2];
    if (nbytes) *nbytes = per;
    return t.data + (size_t)e * per;
}

static const GgFile* first_file(const std::vector<GgFile>& f) {
    for (const auto& x : f)
        if (x.split_no == 0) return &x;
    return f.empty() ? nullptr : &f[0];
}
bool GgufModel::kv(const std::string& key, std::string* v) const {
    const GgFile* f = first_file(f_);
    if (!f) return false;
    auto it = f->kv.find(key);
    if (it == f->kv.end()) return false;
    *v = it->second;
    return true;
}
template <class T>
static bool arr_get(const std::vector<GgFile>& fs, const std::string& key, std::vector<T>* v) {
    const GgFile* f = first_file(fs);
    if (!f) return false;
    auto it = f->arr.find(key);
    if (it == f->arr.end()) return false;
    const GgFile::Arr& a = it->second;
    v->resize(a.n);
    for (uint64_t i = 0; i < a.n; i++) {
        const uint8_t* p = a.data + i * scalar_size(a.et);
        T x = 0;
        switch (a.et) {
            case 0: x = (T)*(const uint8_t*)p; break;
            case 1: x = (T)*(const int8_t*)p; break;
            case 2: { uint16_t y; memcpy(&y, p, 2); x = (T)y; } break;
            case 3: { int16_t y; memcpy(&y, p, 2); x = (T)y; } break;
            case 4: { uint32_t y; memcpy(&y, p, 4); x = (T)y; } break;
            case 5: { int32_t y; memcpy(&y, p, 4); x = (T)y; } break;
            case 6: { float y; memcpy(&y, p, 4); x = (T)y; } break;
            case 7: x = (T)*(const uint8_t*)p; break;
            case 10: { uint64_t y; memcpy(&y, p, 8); x = (T)y; } break;
            case 11: { int64_t y; memcpy(&y, p, 8); x = (T)y; } break;
            case 12: { double y; memcpy(&y, p, 8); x = (T)y; } break;
            default: return false;
        }
        (*v)[i] = x;
    }
    return true;
}
bool GgufModel::arr_i64(const std::string& key, std::vector<int64_t>* v) const { return arr_get(f_, key, v); }
bool GgufModel::arr_f64(const std::string& key, std::vector<double>* v) const { return arr_get(f_, key, v); }

}  // namespace dsv4
