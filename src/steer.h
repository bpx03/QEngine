#pragma once
// Activation-steering vectors (QE_STEER / QE_STEER_DIR / QE_STEER_DEFAULT), shared by both engines (27B: src/capi.cpp,
// Flash-Next: src/flash/flash_steer.h): a llama.cpp control-vector GGUF (general.architecture = controlvector,
// tensors direction.<L> F32 [H], any contiguous layers) plus a JSON sidecar (mode project | add, scale, layer_range,
// per_layer_strength: the defaults of a request; "modes".<other mode>: that mode's defaults; hidden_size,
// direction / effects.intended documented). A request picks the mode and the layers (steer_resolve). Host-only, no
// CUDA; the format is the public README's "Activation steering" section (spec: steering-vector-format v1). Every
// check failure throws std::runtime_error with the reason; the caller refuses to start.
//
// Where the vector acts (each engine): at the output of layer L (llama.cpp's "l_out" / "l_last": the residual after
// the layer's last update, before the next norm): add h += alpha v, project h -= alpha (h . v) v (Flash-Next: on each
// of its 4 hyper-connection streams).
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <algorithm>
#include <dirent.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"

namespace qsteer {

// One vector file: unit-or-not directions for the file's layers flo..fhi. The sidecar's mode, layer_range, scale and
// per_layer_strength are only the defaults of a request; a request may pick either mode and any layer range inside
// flo..fhi (steer_resolve). Each mode has its own defaults: the top-level sidecar fields are those of the default
// mode, "modes": {"<other mode>": {layer_range, scale, per_layer_strength}} those of the other one (absent: the
// file's layers, scale 1, strength 1).
struct SteerModeDef {
    int lo = 0, hi = -1;  // default layer range of the mode
    float scale = 1.f;    // folded into w
};
struct SteerSpec {
    std::string file, meta, name, hint, description, intended;
    int mode = 0;            // default mode: 0 add, 1 project
    int flo = 0, fhi = -1;   // layers present in the file
    SteerModeDef md[2];      // per-mode defaults
    int lo = 0, hi = -1;     // the default mode's layer range (= md[mode])
    float scale = 1.f;       // the default mode's scale (= md[mode])
    // [2 modes][layers][hidden], strength and the mode's sidecar scale folded in (add: alpha v; project: sqrt(alpha)
    // unit v); zero outside flo..fhi
    std::vector<float> w;
    std::string log_line() const {
        char b[1024];
        snprintf(b, sizeof b, "steering: %s, mode=%s, layers %d..%d (%d steered), scale %g, name %s; file layers %d..%d, "
                 "other mode %s layers %d..%d", file.c_str(), mode ? "project" : "add", lo, hi, hi - lo + 1, scale, name.c_str(),
                 flo, fhi, mode ? "add" : "project", md[1 - mode].lo, md[1 - mode].hi);
        return b;
    }
};

// a request's (mode, layers) for vector v: mode -1 = the vector's default mode, lo / hi -1 = the mode's default
// range. Returns "" (mode / lo / hi resolved in place) or the reason it is refused.
inline std::string steer_resolve(const SteerSpec& v, int& mode, int& lo, int& hi) {
    if (mode < -1 || mode > 1) return "steering mode must be add or project";
    if (mode < 0) mode = v.mode;
    if ((lo < 0) != (hi < 0)) return "steering layers need both ends [lo, hi]";
    if (lo < 0) { lo = v.md[mode].lo; hi = v.md[mode].hi; }
    if (lo > hi || lo < v.flo || hi > v.fhi)
        return "steering layers [" + std::to_string(lo) + ", " + std::to_string(hi) + "] not within the layers of vector '" + v.name +
               "' (" + std::to_string(v.flo) + ".." + std::to_string(v.fhi) + ")";
    return "";
}

namespace steer_detail {

inline std::string read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) throw std::runtime_error("steering: cannot open " + p);
    std::string s;
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

// SHA-256 (FIPS 180-4), for vector_file_sha256
inline std::string sha256(const std::string& m) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::string s = m;
    const uint64_t bits = (uint64_t)m.size() * 8;
    s += (char)0x80;
    while (s.size() % 64 != 56) s += (char)0;
    for (int i = 7; i >= 0; i--) s += (char)((bits >> (i * 8)) & 0xff);
    auto rr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t o = 0; o < s.size(); o += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)(uint8_t)s[o + 4 * i] << 24 | (uint32_t)(uint8_t)s[o + 4 * i + 1] << 16 |
                   (uint32_t)(uint8_t)s[o + 4 * i + 2] << 8 | (uint32_t)(uint8_t)s[o + 4 * i + 3];
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rr(w[i - 15], 7) ^ rr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rr(w[i - 2], 17) ^ rr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t t1 = hh + (rr(e, 6) ^ rr(e, 11) ^ rr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rr(a, 2) ^ rr(a, 13) ^ rr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    char out[65];
    for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", h[i]);
    return out;
}

// minimal GGUF (v3) reader: string KVs, integer KVs, tensor infos and data
struct Gguf {
    std::map<std::string, std::string> str;
    std::map<std::string, long long> num;
    struct T { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
    std::vector<T> tensors;
    size_t data0 = 0;
    std::string bytes;
};
inline Gguf parse_gguf(const std::string& path) {
    Gguf g;
    g.bytes = read_file(path);
    const std::string& b = g.bytes;
    size_t p = 0;
    auto need = [&](size_t n) {
        if (p + n > b.size()) throw std::runtime_error("steering: " + path + " is truncated (not a GGUF file?)");
    };
    auto u32 = [&]() { need(4); uint32_t v; memcpy(&v, b.data() + p, 4); p += 4; return v; };
    auto u64 = [&]() { need(8); uint64_t v; memcpy(&v, b.data() + p, 8); p += 8; return v; };
    auto gstr = [&]() {
        const uint64_t n = u64();
        need(n);
        std::string s = b.substr(p, n);
        p += n;
        return s;
    };
    need(4);
    if (b.compare(0, 4, "GGUF") != 0) throw std::runtime_error("steering: " + path + " is not a GGUF file (bad magic)");
    p = 4;
    const uint32_t ver = u32();
    if (ver != 3) throw std::runtime_error("steering: " + path + ": GGUF version " + std::to_string(ver) + ", v3 required");
    const uint64_t nt = u64(), nkv = u64();
    static const int tsz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    std::function<void(uint32_t, const std::string&)> value = [&](uint32_t t, const std::string& key) {
        if (t == 8) { std::string s = gstr(); if (!key.empty()) g.str[key] = s; return; }
        if (t == 9) {
            const uint32_t et = u32();
            const uint64_t n = u64();
            for (uint64_t i = 0; i < n; i++) value(et, "");
            return;
        }
        if (t > 12) throw std::runtime_error("steering: " + path + ": bad GGUF value type " + std::to_string(t));
        need(tsz[t]);
        long long v = 0;
        switch (t) {
            case 0: v = (uint8_t)b[p]; break;
            case 1: v = (int8_t)b[p]; break;
            case 2: { uint16_t x; memcpy(&x, b.data() + p, 2); v = x; break; }
            case 3: { int16_t x; memcpy(&x, b.data() + p, 2); v = x; break; }
            case 4: { uint32_t x; memcpy(&x, b.data() + p, 4); v = x; break; }
            case 5: { int32_t x; memcpy(&x, b.data() + p, 4); v = x; break; }
            case 7: v = (uint8_t)b[p]; break;
            case 10: { uint64_t x; memcpy(&x, b.data() + p, 8); v = (long long)x; break; }
            case 11: { int64_t x; memcpy(&x, b.data() + p, 8); v = x; break; }
            default: break;  // floats: not used
        }
        p += tsz[t];
        if (!key.empty() && t != 6 && t != 12) g.num[key] = v;
    };
    for (uint64_t i = 0; i < nkv; i++) {
        const std::string k = gstr();
        value(u32(), k);
    }
    for (uint64_t i = 0; i < nt; i++) {
        Gguf::T t;
        t.name = gstr();
        const uint32_t nd = u32();
        if (nd > 4) throw std::runtime_error("steering: " + path + ": tensor " + t.name + " has " + std::to_string(nd) + " dims");
        for (uint32_t d = 0; d < nd; d++) t.dims.push_back(u64());
        t.type = u32();
        t.off = u64();
        g.tensors.push_back(t);
    }
    const long long al = g.num.count("general.alignment") ? g.num["general.alignment"] : 32;
    g.data0 = (p + al - 1) / al * al;
    return g;
}

inline bool placeholder(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && isspace((unsigned char)s[i])) i++;
    return i == s.size() || s[i] == '<';
}
inline const Json* get(const Json& j, const std::string& k) { return j.type == Json::Obj && j.has(k) ? &j[k] : nullptr; }

}  // namespace steer_detail

// spec: "<file.gguf>[:<scale>]" (QE_STEER; the scale overrides the sidecar's, as llama.cpp --control-vector-scaled);
// meta: sidecar path (QE_STEER_META) or empty (<stem>.json, else <file>.json); model_dir: the checkpoint (its
// config.json and directory name are checked against the sidecar)
inline SteerSpec steer_load(const std::string& spec_in, const std::string& meta_in, int hidden, int layers,
                            const std::string& model_dir) {
    using namespace steer_detail;
    SteerSpec r;
    std::string spec = spec_in;
    for (char sep : {',', ';'})
        if (spec.find(sep) != std::string::npos) {
            const size_t k = spec.find(sep);
            throw std::runtime_error("steering: QE_STEER takes one vector file and names a second one: '" + spec.substr(0, k) +
                                     "' and '" + spec.substr(k + 1) + "' (several vectors: QE_STEER_DIR; one is active per request)");
        }
    float scale_override = NAN;
    {
        const size_t c = spec.rfind(':');
        if (c != std::string::npos && c + 1 < spec.size()) {
            char* end = nullptr;
            const float v = strtof(spec.c_str() + c + 1, &end);
            if (end && *end == 0) { scale_override = v; spec = spec.substr(0, c); }
        }
    }
    r.file = spec;
    Gguf g = parse_gguf(spec);
    // architecture
    if (!g.str.count("general.architecture") || g.str["general.architecture"] != "controlvector")
        throw std::runtime_error("steering: " + spec + ": general.architecture is '" +
                                 (g.str.count("general.architecture") ? g.str["general.architecture"] : std::string("(missing)")) +
                                 "', 'controlvector' required");
    if (g.str.count("controlvector.model_hint")) r.hint = g.str["controlvector.model_hint"];
    // sidecar
    std::string meta = meta_in;
    if (meta.empty()) {
        std::string stem = spec;
        if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".gguf") stem = stem.substr(0, stem.size() - 5);
        FILE* f = fopen((stem + ".json").c_str(), "rb");
        if (f) { fclose(f); meta = stem + ".json"; }
        else meta = spec + ".json";
    }
    r.meta = meta;
    Json j;
    {
        std::string js;
        try { js = read_file(meta); } catch (...) {
            throw std::runtime_error("steering: no sidecar for " + spec + " (looked for " + meta +
                                     "; set QE_STEER_META): its direction and effects must be documented");
        }
        try { j = Json::parse(js); } catch (std::exception& e) {
            throw std::runtime_error("steering: sidecar " + meta + " is not valid JSON (" + e.what() + ")");
        }
    }
    if (j.type != Json::Obj) throw std::runtime_error("steering: sidecar " + meta + " is not a JSON object");
    // required documentation
    {
        const Json* d = get(j, "direction");
        if (!d || d->type != Json::Str || placeholder(d->str))
            throw std::runtime_error("steering: sidecar " + meta + " lacks 'direction' (how the vector was measured): refused");
        const Json* ef = get(j, "effects");
        const Json* in = ef ? get(*ef, "intended") : nullptr;
        if (!in || in->type != Json::Str || placeholder(in->str))
            throw std::runtime_error("steering: sidecar " + meta + " lacks 'effects.intended' (the documented behavior change): refused");
    }
    if (const Json* n = get(j, "name"); n && n->type == Json::Str && !placeholder(n->str)) r.name = n->str;
    if (r.name.empty()) {  // the file stem
        std::string b = spec.substr(spec.find_last_of('/') + 1);
        if (b.size() > 5 && b.substr(b.size() - 5) == ".gguf") b = b.substr(0, b.size() - 5);
        r.name = b;
    }
    if (const Json* n = get(j, "description"); n && n->type == Json::Str && !placeholder(n->str)) r.description = n->str;
    r.intended = j["effects"]["intended"].str;
    // mode / dir
    {
        const Json* m = get(j, "mode");
        if (!m || m->type != Json::Str || (m->str != "project" && m->str != "add"))
            throw std::runtime_error("steering: sidecar " + meta + ": 'mode' must be \"project\" or \"add\"");
        r.mode = m->str == "project" ? 1 : 0;
        const Json* d = get(j, "dir");
        if (d && (d->type != Json::Str || d->str != "per-layer"))
            throw std::runtime_error("steering: sidecar " + meta + ": dir '" + (d->type == Json::Str ? d->str : std::string("?")) +
                                     "' is not supported (v1: only per-layer directions; --cvec-dir shared is refused)");
    }
    // checkpoint identity: hidden size, layer count, model name
    auto mismatch = [&](const std::string& what) {
        throw std::runtime_error("steering: " + spec + " was measured on another checkpoint (" + what +
                                 "); a vector is valid only for the checkpoint it was measured from: re-measure it");
    };
    for (const Json* hs : {get(j, "hidden_size"), get(j, "compatibility") ? get(j["compatibility"], "hidden_size") : nullptr})
        if (hs && hs->type == Json::Num && hs->i() != 0 && hs->i() != hidden)
            mismatch("hidden_size " + std::to_string(hs->i()) + ", this checkpoint " + std::to_string(hidden));
    if (!model_dir.empty()) {
        try {
            Json cfg = Json::parse(read_file(model_dir + "/config.json"));
            const Json& tc = cfg.has("text_config") ? cfg["text_config"] : cfg;
            if (tc.has("hidden_size") && tc["hidden_size"].i() != hidden)
                throw std::runtime_error("steering: engine hidden size " + std::to_string(hidden) + " != config.json " +
                                         std::to_string(tc["hidden_size"].i()));
        } catch (std::runtime_error& e) {
            if (std::string(e.what()).rfind("steering:", 0) == 0) throw;
        }
        const Json* cp = get(j, "compatibility");
        const Json* cm = cp ? get(*cp, "model") : nullptr;
        static const bool chk = !getenv("QE_STEER_MODEL_CHECK") || atoi(getenv("QE_STEER_MODEL_CHECK"));
        if (chk && cm && cm->type == Json::Str && !placeholder(cm->str)) {
            std::string dn = model_dir;
            while (dn.size() > 1 && dn.back() == '/') dn.pop_back();
            dn = dn.substr(dn.find_last_of('/') + 1);
            if (dn != cm->str && dn.find(cm->str) == std::string::npos && cm->str.find(dn) == std::string::npos)
                mismatch("compatibility.model '" + cm->str + "', this checkpoint '" + dn + "'; QE_STEER_MODEL_CHECK=0 skips the name check");
        }
    }
    // tensors
    std::map<int, const Gguf::T*> dirs;
    for (auto& t : g.tensors) {
        if (t.name.rfind("direction.", 0) != 0)
            throw std::runtime_error("steering: " + spec + ": unexpected tensor '" + t.name + "' (only direction.<L>)");
        const std::string ls = t.name.substr(10);
        char* end = nullptr;
        const long L = strtol(ls.c_str(), &end, 10);
        if (ls.empty() || *end || L < 0) throw std::runtime_error("steering: " + spec + ": bad tensor name '" + t.name + "'");
        if (dirs.count((int)L))
            throw std::runtime_error("steering: " + spec + ": two directions for layer " + std::to_string(L) +
                                     " (one direction per layer)");
        if (t.type != 0) throw std::runtime_error("steering: " + spec + ": " + t.name + " is not F32");
        if (t.dims.size() != 1) throw std::runtime_error("steering: " + spec + ": " + t.name + " is not 1-D [H]");
        if ((long long)t.dims[0] != hidden) mismatch(t.name + " has " + std::to_string(t.dims[0]) + " values, hidden size " + std::to_string(hidden));
        if (g.data0 + t.off + (size_t)hidden * 4 > g.bytes.size()) throw std::runtime_error("steering: " + spec + ": " + t.name + " data truncated");
        dirs[(int)L] = &t;
    }
    if (dirs.empty()) throw std::runtime_error("steering: " + spec + ": no direction tensors");
    // the file's layers: contiguous, within the checkpoint, layer_count = their number
    r.flo = dirs.begin()->first;
    r.fhi = dirs.rbegin()->first;
    if (r.fhi >= layers) mismatch("direction." + std::to_string(r.fhi) + ", this checkpoint has layers 0.." + std::to_string(layers - 1));
    for (int L = r.flo; L <= r.fhi; L++)
        if (!dirs.count(L)) throw std::runtime_error("steering: " + spec + ": layers not contiguous (direction." + std::to_string(L) +
                                                     " missing in " + std::to_string(r.flo) + ".." + std::to_string(r.fhi) + ")");
    if (!g.num.count("controlvector.layer_count")) throw std::runtime_error("steering: " + spec + ": controlvector.layer_count missing");
    if (g.num["controlvector.layer_count"] != r.fhi - r.flo + 1)
        throw std::runtime_error("steering: " + spec + ": controlvector.layer_count " + std::to_string(g.num["controlvector.layer_count"]) +
                                 " != the " + std::to_string(r.fhi - r.flo + 1) + " directions of the file");
    // integrity
    if (const Json* sh = get(j, "vector_file_sha256"); sh && sh->type == Json::Str && !placeholder(sh->str)) {
        const std::string got = sha256(g.bytes);
        if (got != sh->str) throw std::runtime_error("steering: " + spec + ": sha256 " + got + " != sidecar vector_file_sha256 " + sh->str);
    }
    // per-mode defaults: the top-level fields for the default mode, "modes".<other> for the other one
    const Json* modes = get(j, "modes");
    if (modes && modes->type != Json::Obj) throw std::runtime_error("steering: sidecar " + meta + ": 'modes' must be an object {\"add\"|\"project\": {...}}");
    if (modes)
        for (auto& [k, v] : modes->obj) {
            if (k != "add" && k != "project") throw std::runtime_error("steering: sidecar " + meta + ": modes key '" + k + "' is not add or project");
            if ((k == "project") == (r.mode == 1))
                throw std::runtime_error("steering: sidecar " + meta + ": modes." + k + " repeats the default mode (its defaults are the top-level fields)");
            if (v.type != Json::Obj) throw std::runtime_error("steering: sidecar " + meta + ": modes." + k + " must be an object");
        }
    std::vector<float> pls[2] = {std::vector<float>(layers, 1.f), std::vector<float>(layers, 1.f)};
    for (int m = 0; m < 2; m++) {
        const std::string mn = m ? "project" : "add";
        const Json* o = m == r.mode ? &j : (modes ? get(*modes, mn) : nullptr);
        const std::string where = m == r.mode ? std::string("sidecar ") + meta : "sidecar " + meta + " modes." + mn;
        SteerModeDef& d = r.md[m];
        d.lo = r.flo;
        d.hi = r.fhi;
        if (const Json* lr = o ? get(*o, "layer_range") : nullptr) {
            if (lr->type != Json::Arr || lr->arr.size() != 2) throw std::runtime_error("steering: " + where + ": layer_range must be [lo, hi]");
            d.lo = (int)(*lr)[0].i();
            d.hi = (int)(*lr)[1].i();
            if (d.lo < 0 || d.hi < d.lo)
                throw std::runtime_error("steering: " + where + ": layer_range [" + std::to_string(d.lo) + ", " + std::to_string(d.hi) + "] is empty");
            if (d.hi >= layers) mismatch("layer_range ends at " + std::to_string(d.hi) + ", this checkpoint has layers 0.." + std::to_string(layers - 1));
            if (d.lo < r.flo || d.hi > r.fhi)
                throw std::runtime_error("steering: " + where + ": layer_range [" + std::to_string(d.lo) + ", " + std::to_string(d.hi) +
                                         "] lies outside the file's layers " + std::to_string(r.flo) + ".." + std::to_string(r.fhi));
        }
        if (const Json* sc = o ? get(*o, "scale") : nullptr) {
            if (sc->type != Json::Num) throw std::runtime_error("steering: " + where + ": scale must be a number");
            d.scale = (float)sc->f();
        }
        if (m == r.mode && !std::isnan(scale_override)) d.scale = scale_override;  // QE_STEER file:scale: the default mode's
        if (const Json* p = o ? get(*o, "per_layer_strength") : nullptr) {
            if (p->type != Json::Obj) throw std::runtime_error("steering: " + where + ": per_layer_strength must be an object {\"L\": alpha}");
            for (auto& [k, v] : p->obj) {
                char* end = nullptr;
                const long L = strtol(k.c_str(), &end, 10);
                if (k.empty() || *end || L < r.flo || L > r.fhi || v.type != Json::Num)
                    throw std::runtime_error("steering: " + where + ": per_layer_strength key '" + k + "' outside the file's layers or not a number");
                pls[m][L] = (float)v.f();
            }
        }
    }
    r.lo = r.md[r.mode].lo;
    r.hi = r.md[r.mode].hi;
    r.scale = r.md[r.mode].scale;
    // fold: per mode, every layer of the file
    r.w.assign((size_t)2 * layers * hidden, 0.f);
    for (int L = r.flo; L <= r.fhi; L++) {
        const float* src = reinterpret_cast<const float*>(g.bytes.data() + g.data0 + dirs[L]->off);
        std::vector<float> v(src, src + hidden);
        double n2 = 0;
        for (float x : v) {
            if (!std::isfinite(x)) throw std::runtime_error("steering: " + spec + ": direction." + std::to_string(L) + " has a non-finite value");
            n2 += (double)x * x;
        }
        for (int m = 0; m < 2; m++) {
            const float a = r.md[m].scale * pls[m][L];
            if (!std::isfinite(a)) throw std::runtime_error("steering: non-finite strength at layer " + std::to_string(L));
            float* dst = r.w.data() + ((size_t)m * layers + L) * hidden;
            if (m == 1) {
                if (n2 < 1e-12) {  // a zero direction: refused only where project is the default mode
                    if (r.mode == 1) throw std::runtime_error("steering: " + spec + ": direction." + std::to_string(L) + " is zero (project needs a direction)");
                    continue;
                }
                if (a < 0) throw std::runtime_error("steering: project mode needs strength >= 0 (layer " + std::to_string(L) + ": " + std::to_string(a) + ")");
                const float sc = sqrtf(a) / (float)sqrt(n2);
                for (int d = 0; d < hidden; d++) dst[d] = v[d] * sc;
            } else {
                for (int d = 0; d < hidden; d++) dst[d] = v[d] * a;
            }
        }
    }
    return r;
}

// the server's vector set: QE_STEER (one file, sidecar QE_STEER_META) and / or every *.gguf of QE_STEER_DIR (sidecar
// next to it); names unique. *def: QE_STEER_DEFAULT by name, else the QE_STEER vector when it is the only one, else -1.
inline std::vector<SteerSpec> steer_load_env(int hidden, int layers, const std::string& model_dir, int* def) {
    std::vector<SteerSpec> v;
    *def = -1;
    const char* one = getenv("QE_STEER");
    const char* dir = getenv("QE_STEER_DIR");
    if (one && *one) v.push_back(steer_load(one, getenv("QE_STEER_META") ? getenv("QE_STEER_META") : "", hidden, layers, model_dir));
    if (dir && *dir) {
        std::vector<std::string> files;
        if (DIR* d = opendir(dir)) {
            for (dirent* de; (de = readdir(d));) {
                const std::string n = de->d_name;
                if (n.size() > 5 && n.substr(n.size() - 5) == ".gguf") files.push_back(std::string(dir) + "/" + n);
            }
            closedir(d);
        } else {
            throw std::runtime_error(std::string("steering: cannot open QE_STEER_DIR ") + dir);
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) throw std::runtime_error(std::string("steering: no *.gguf in QE_STEER_DIR ") + dir);
        for (auto& f : files) v.push_back(steer_load(f, "", hidden, layers, model_dir));
    }
    for (size_t a = 0; a < v.size(); a++)
        for (size_t b = 0; b < a; b++)
            if (v[a].name == v[b].name)
                throw std::runtime_error("steering: two vectors named '" + v[a].name + "': " + v[b].file + " and " + v[a].file);
    if (const char* dn = getenv("QE_STEER_DEFAULT"); dn && *dn) {
        for (size_t a = 0; a < v.size(); a++)
            if (v[a].name == dn) *def = (int)a;
        if (*def < 0 && strcmp(dn, "off") != 0) {
            std::string names;
            for (auto& x : v) names += (names.empty() ? "" : ", ") + x.name;
            throw std::runtime_error(std::string("steering: QE_STEER_DEFAULT '") + dn + "' is not a loaded vector (" + names + ")");
        }
    } else if (one && *one && v.size() == 1) {
        *def = 0;
    }
    return v;
}

inline std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o + "\"";
}
// {"default": name or null, "vectors": [{name, mode, layers, scale (the defaults), file_layers (a request may pick any
// range inside), modes: {add|project: {layers, scale}} (the defaults per mode), description, intended, file}]}
inline std::string steer_info_json(const std::vector<SteerSpec>& v, int def) {
    std::string o = "{\"default\": " + (def >= 0 ? json_str(v[def].name) : std::string("null")) + ", \"vectors\": [";
    for (size_t i = 0; i < v.size(); i++) {
        const SteerSpec& x = v[i];
        char b[512];
        snprintf(b, sizeof b,
                 ", \"mode\": \"%s\", \"layers\": [%d, %d], \"scale\": %g, \"file_layers\": [%d, %d], \"modes\": {\"add\": "
                 "{\"layers\": [%d, %d], \"scale\": %g}, \"project\": {\"layers\": [%d, %d], \"scale\": %g}}",
                 x.mode ? "project" : "add", x.lo, x.hi, x.scale, x.flo, x.fhi, x.md[0].lo, x.md[0].hi, x.md[0].scale, x.md[1].lo,
                 x.md[1].hi, x.md[1].scale);
        o += (i ? ", " : "") + std::string("{\"name\": ") + json_str(x.name) + b + ", \"description\": " + json_str(x.description) +
             ", \"intended\": " + json_str(x.intended) + ", \"file\": " + json_str(x.file) + "}";
    }
    return o + "]}";
}

}  // namespace qsteer
