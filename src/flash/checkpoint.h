#pragma once
#include "../safetensors.h"
#include "../env.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace flash {
enum class WeightFormat { Int4, Int8 };
namespace checkpoint_detail {
// Manifest schema stamp. Pre-v2 manifests (single body string, absolute external n-gram path, pinned
// upstream source) are NOT backward compatible: both readers reject them with "schema updated, reconvert".
inline const std::string kManifestSchema = "qfn-int8-v2";
// Upper bound for one body shard file (20 GiB). tools/quant_int8.py mirrors this as MAX_SHARD_BYTES and
// writes at most this per shard; the manifest may record a smaller limit, never a larger one.
inline constexpr uint64_t kMaxShardBytes = 20ull << 30;
// mtp.fc_embedding maps [token_embedding ; hidden] to hidden: the loader accepts the logical shapes
// [hidden, hidden] (inputs fed separately) and [hidden, 2*hidden] (fused). tools/quant_int8.py mirrors
// this as MTP_FC_EMBEDDING_K_MULTIPLIERS; converter-side validation must stay aligned with this list.
inline constexpr std::array<int, 2> kMtpFcEmbeddingKMultipliers{1, 2};
inline void fail(const std::string& context, const std::string& key, const std::string& expected,
                 const std::string& actual) {
    throw std::runtime_error(context + ": " + key + ": expected " + expected + ", actual " + actual);
}
inline std::string resolve_path(const std::string& directory, const std::string& path) {
    if (path.empty()) throw std::runtime_error(directory + ": empty checkpoint path");
    return (std::filesystem::path(path).is_absolute() ? std::filesystem::path(path)
            : std::filesystem::path(directory) / path).lexically_normal().string();
}
// Path sandbox: every manifest file field is relative to the checkpoint directory. Absolute paths,
// drive/UNC roots, embedded NULs and any '..' component are rejected before the value is joined; the
// lexically normalised result must additionally stay inside the checkpoint directory.
inline std::string sandbox_file(const std::string& directory, const std::string& value,
                                const std::string& context) {
    if (value.empty()) fail(context, "path", "relative file name", "empty");
    if (value.find('\0') != std::string::npos) fail(context, "path", "no NUL bytes", "embedded NUL");
    const std::filesystem::path p(value);
    // POSIX std::filesystem sees "C:/x" as a relative path; reject Windows drive roots explicitly
    if (p.is_absolute() || !p.root_name().empty() ||
        (value.size() >= 2 && ((value[0]>='A'&&value[0]<='Z')||(value[0]>='a'&&value[0]<='z')) && value[1]==':'))
        fail(context, "path", "relative path (sandboxed to the checkpoint directory)", value);
    for (const auto& part : p) if (part == "..") fail(context, "path", "no parent-directory components", value);
    const auto joined = (std::filesystem::path(directory) / p).lexically_normal();
    const auto rel = joined.lexically_relative(std::filesystem::path(directory).lexically_normal());
    if (rel.empty() || rel.is_absolute() || *rel.begin() == "..")
        fail(context, "path", "inside the checkpoint directory", value);
    return joined.string();
}
inline bool is_file(const std::string& path) { return std::filesystem::is_regular_file(path); }
// Sandbox symlink gate: is_regular_file/status/ifstream all FOLLOW symlinks, so the lexical sandbox
// above would be escapable if a manifest-named file were replaced by a symlink. Every sandbox-resolved
// manifest path — the manifest itself, the body shards, and the recorded MTP/config/self-contained
// n-gram names — must pass this check before its first open. Scope: operator-supplied absolute inputs
// (FLASH_NGRAM_PATH, the explicit n-gram/MTP override arguments) keep their documented absolute
// allowance and are deliberately NOT symlink-gated; only paths resolved through the sandbox are.
// A missing file passes here unchanged so the established "missing" diagnostics stay as they were.
inline void require_no_symlink(const std::string& path, const std::string& context) {
    std::error_code ec;
    const auto type = std::filesystem::symlink_status(path, ec).type();
    if (!ec && type == std::filesystem::file_type::symlink)
        fail(context, "path", "regular file (the sandbox never follows symlinks)", "symbolic link: " + path);
}
inline std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(path + ": cannot open file");
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
// stat the file first: it must exist, be a regular file (S_ISREG) and fit the bound before any read.
inline std::string read_bounded_file(const std::string& path, uint64_t max_bytes, const std::string& context) {
    std::error_code ec;
    const auto st = std::filesystem::status(path, ec);
    if (ec || st.type() == std::filesystem::file_type::not_found)
        fail(context, "file", "existing regular file", "missing");
    if (st.type() != std::filesystem::file_type::regular)
        fail(context, "file", "regular file", "not a regular file");
    const uint64_t bytes=std::filesystem::file_size(path,ec);
    if(ec || bytes>max_bytes)
        fail(context, "file size", "at most " + std::to_string(max_bytes) + " bytes", std::to_string(bytes));
    return read_file(path);
}
inline Json parse_json(const std::string& bytes, const std::string& context) {
    try { return Json::parse_strict(bytes); }
    catch(const std::runtime_error& ex) { throw std::runtime_error(context+": "+ex.what()); }
}
inline const Json& field(const Json& j, const std::string& key, const std::string& context) {
    if (j.type != Json::Obj || !j.has(key)) fail(context, key, "present", "missing");
    return j[key];
}
inline std::string text(const Json& j, const std::string& key, const std::string& context) {
    const auto& v = field(j, key, context);
    if (v.type != Json::Str || v.str.empty()) fail(context, key, "nonempty string", "invalid type/value");
    return v.str;
}
inline long long integer(const Json& j, const std::string& key, const std::string& context) {
    const auto& v = field(j, key, context);
    if (v.type != Json::Num || !std::isfinite(v.num) || v.num < 0 || v.num > 9007199254740991.0 || (double)v.i() != v.num)
        fail(context, key, "nonnegative integer", "invalid type/value");
    return v.i();
}
inline std::string shape_text(const std::vector<long long>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? "," : "") + std::to_string(shape[i]);
    return s + "]";
}
inline size_t elements(const std::vector<long long>& shape, const std::string& context) {
    size_t n = 1;
    for (auto d : shape) {
        if (d < 0 || (d && n > std::numeric_limits<size_t>::max() / (size_t)d))
            fail(context, "shape", "nonnegative bounded extent", shape_text(shape));
        n *= (size_t)d;
    }
    return n;
}
inline std::vector<long long> shape_array(const Json& j, const std::string& context) {
    if (j.type != Json::Arr) fail(context, "shape", "array", "other JSON type");
    std::vector<long long> shape;
    for (const auto& d : j.arr) {
        if (d.type != Json::Num || !std::isfinite(d.num) || d.num < 0 || d.num > 9007199254740991.0 || (double)d.i() != d.num)
            fail(context, "shape", "nonnegative integer dimensions", "invalid dimension");
        shape.push_back(d.i());
    }
    return shape;
}
// Validate structure before the legacy mmap reader accesses any header or payload byte. The safetensors
// header is parsed with the strict reader as well: a duplicate key could otherwise hide an extra tensor.
inline Json validate_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error(path + ": cannot open safetensors file");
    const auto end = in.tellg();
    if (end < 8) fail(path, "file size", "at least 8 bytes", std::to_string((long long)end));
    const uint64_t size = (uint64_t)end;
    in.seekg(0);
    unsigned char prefix[8];
    in.read((char*)prefix, 8);
    uint64_t header = 0;
    for (int i = 0; i < 8; ++i) header |= (uint64_t)prefix[i] << (8 * i);
    if (header == 0 || header > size - 8 || header > (64ull << 20))
        fail(path, "header length", "1..min(file size-8,64MiB)", std::to_string(header));
    std::string bytes((size_t)header, '\0');
    in.read(bytes.data(), (std::streamsize)header);
    if (!in) throw std::runtime_error(path + ": truncated safetensors header");
    const Json j = parse_json(bytes,path);
    if (j.type != Json::Obj) fail(path, "header", "JSON object", "other JSON type");
    std::vector<std::pair<uint64_t,uint64_t>> ranges;
    for (const auto& entry : j.obj) {
        if (entry.first == "__metadata__") continue;
        const auto context = path + ": " + entry.first;
        const auto dtype = text(entry.second, "dtype", context);
        size_t width = 0;
        if (dtype == "U8" || dtype == "I8" || dtype == "BOOL" || dtype == "F8_E4M3" || dtype == "F8_E5M2") width = 1;
        else if (dtype == "BF16" || dtype == "F16" || dtype == "I16" || dtype == "U16") width = 2;
        else if (dtype == "F32" || dtype == "I32" || dtype == "U32") width = 4;
        else if (dtype == "F64" || dtype == "I64" || dtype == "U64") width = 8;
        else fail(context, "dtype", "supported safetensors dtype", dtype);
        const auto shape = shape_array(field(entry.second, "shape", context), context);
        const auto offsets = shape_array(field(entry.second, "data_offsets", context), context);
        const auto n = elements(shape, context);
        if (offsets.size() != 2 || offsets[1] < offsets[0] || (uint64_t)offsets[1] > size - 8 - header
            || n > std::numeric_limits<size_t>::max() / width || (uint64_t)(offsets[1] - offsets[0]) != n * width)
            fail(context, "data_offsets", "in-file extent matching dtype and shape", shape_text(offsets));
        ranges.emplace_back((uint64_t)offsets[0], (uint64_t)offsets[1]);
    }
    std::sort(ranges.begin(), ranges.end());
    uint64_t next = 0;
    for (auto r : ranges) {
        if (r.first != next) fail(path, "data_offsets", "contiguous non-overlapping payload", std::to_string(r.first));
        next = r.second;
    }
    if (next != size - 8 - header) fail(path, "payload bytes", std::to_string(next), std::to_string(size - 8 - header));
    return j;
}
inline void open_file(SafeTensors& st, const std::string& path) { validate_file(path); st.open(path); }
inline void validate_recipe(const Json& recipe, const std::string& context) {
    const auto format = text(recipe, "format", context);
    if (format != "symmetric INT8") fail(context, "recipe.format", "symmetric INT8", format);
    for (const auto& p : {std::pair<const char*,long long>{"group_size",128}, {"zero_point",0}}) {
        const auto actual = integer(recipe, p.first, context);
        if (actual != p.second) fail(context, std::string("recipe.") + p.first, std::to_string(p.second), std::to_string(actual));
    }
    const auto dtype = text(recipe, "scale_dtype", context);
    if (dtype != "BF16") fail(context, "recipe.scale_dtype", "BF16", dtype);
}
inline void validate_manifest_top(const Json& j, const std::string& context) {
    // Clean break over the pre-sharding layout: an unrecognised (or missing) schema stamp must never be
    // half-read; the only correct action is re-converting with the current tools/quant_int8.py.
    if (!j.has("schema") || j["schema"].type != Json::Str || j["schema"].str != kManifestSchema)
        fail(context, "schema", kManifestSchema, "missing or stale: schema updated, reconvert with the current tools/quant_int8.py");
    if (integer(j, "format_version", context) != 1) fail(context, "format_version", "1", "other version");
    const auto family = text(j, "family", context);
    if (family != "qwen4_exp") fail(context, "family", "qwen4_exp", family);
    const auto& outputs = field(j, "outputs", context);
    if (outputs.type != Json::Obj) fail(context, "outputs", "object", "other JSON type");
    if (outputs["body"].type != Json::Arr || outputs["body"].arr.empty())
        fail(context, "outputs.body", "nonempty array of shard files", "invalid type");
    text(outputs, "mtp", context); text(outputs, "config", context);
}
inline void validate_int8_tensor(const SafeTensors& st, const std::string& base,
                                 const std::vector<long long>& logical_shape, const std::string& context = "checkpoint") {
    if ((logical_shape.size() != 2 && logical_shape.size() != 3) || logical_shape.back() <= 0 || logical_shape.back() % 128)
        fail(context, base, "rank 2/3 logical shape with positive K divisible by 128", shape_text(logical_shape));
    for (auto d : logical_shape) if (d <= 0) fail(context, base, "positive dimensions", shape_text(logical_shape));
    for (int kind = 0; kind < 3; ++kind) {
        const auto key = base + (kind == 0 ? ".weight_packed" : kind == 1 ? ".weight_scale" : ".weight_shape");
        if (!st.has(key)) fail(context, key, "present", "missing");
        const auto& t = st.get(key);
        auto shape = logical_shape;
        if (kind == 2) shape = {(long long)logical_shape.size()};
        else shape.back() /= kind == 0 ? 4 : 128;
        const std::string dtype = kind == 0 ? "I32" : kind == 1 ? "BF16" : "I64";
        const size_t width = kind == 0 ? 4 : kind == 1 ? 2 : 8;
        const auto count=elements(shape,context);
        if (count>std::numeric_limits<size_t>::max()/width || t.dtype != dtype || t.shape != shape || t.nbytes != count * width)
            fail(context, key, dtype + " " + shape_text(shape) + " with exact byte extent",
                 t.dtype + " " + shape_text(t.shape) + " bytes=" + std::to_string(t.nbytes));
        if (kind == 2) for (size_t i = 0; i < logical_shape.size(); ++i) {
            int64_t d; memcpy(&d, t.data + i * 8, 8);
            if (d != logical_shape[i]) fail(context, key, shape_text(logical_shape), "logical dimension " + std::to_string(d));
        }
    }
    if (st.has(base + ".weight") || st.has(base)) fail(context, base, "packed INT8 only", "mixed BF16 and packed selected matrix");
}
struct Converted { std::string source_key, base, file; std::vector<long long> shape; };
inline Converted parse_converted(const Json& j, const std::string& context) {
    Converted c{text(j,"source_key",context),text(j,"base",context),text(j,"file",context),
                shape_array(field(j,"shape",context),context)};
    if((c.shape.size()!=2 && c.shape.size()!=3) || c.shape.back()<=0 || c.shape.back()%128)
        fail(context,c.base,"rank 2/3, K divisible by 128",shape_text(c.shape));
    for(auto d:c.shape) if(d<=0) fail(context,c.base,"positive dimensions",shape_text(c.shape));
    std::string base = c.source_key;
    if (base.size() >= 7 && base.compare(base.size()-7,7,".weight") == 0) base.resize(base.size()-7);
    if (base != c.base) fail(context,"base",base,c.base);
    if (text(j,"dtype",context) != "BF16") fail(context,"dtype","BF16","other dtype");
    const auto& o = field(j,"outputs",context);
    for (const auto& p : {std::pair<const char*,const char*>{"packed",".weight_packed"}, {"scale",".weight_scale"}, {"shape",".weight_shape"}})
        if (text(o,p.first,context) != c.base+p.second) fail(context,p.first,c.base+p.second,text(o,p.first,context));
    return c;
}
// Identity of the 128-shard n-gram table. The manifest stores NO path: a self-contained checkpoint keeps
// the table as a sandboxed outputs.ngram next to the weights; an external table is supplied at open time
// (override argument or FLASH_NGRAM_PATH). sha256 is recorded by the converter and re-verified by
// tools/quant_int8.py --check; the runtime keeps the identity-field checks only (hashing 50+ GB at load
// time is not worth the startup cost).
struct NgramExternal { std::string path; long long size_bytes=0, rows_per_shard=0; int width_bytes=0, shards=0; };
inline NgramExternal parse_ngram_identity(const Json& j, const std::string& context) {
    const auto width=integer(j,"width_bytes",context), shards=integer(j,"shards",context);
    if(width!=162 || shards!=128) fail(context,"ngram","width 162, 128 shards","invalid metadata");
    NgramExternal n{{},integer(j,"size_bytes",context),integer(j,"rows_per_shard",context),162,128};
    if (n.size_bytes <= 0 || n.rows_per_shard <= 0 || (uint64_t)n.rows_per_shard>std::numeric_limits<size_t>::max()/162)
        fail(context,"ngram","positive bounded size/rows","invalid metadata");
    const auto& sha = text(j, "sha256", context);
    if (sha.size() != 64 || !std::all_of(sha.begin(), sha.end(), [](char c) { return (c>='0'&&c<='9')||(c>='a'&&c<='f'); }))
        fail(context, "ngram.sha256", "64 lowercase hex digits", "invalid digest");
    if (j.has("path")) fail(context, "ngram", "no path in the manifest (identity only)", "path present");
    return n;
}

struct ResolvedOutputs { std::vector<std::string> body; std::string mtp, ngram; };
inline ResolvedOutputs resolve_outputs(const Json& manifest, const std::string& directory,
                                       bool want_mtp, const std::string& mtp_override,
                                       const std::string& ngram_override = {}) {
    validate_manifest_top(manifest,directory);
    const auto& outputs=manifest["outputs"];
    ResolvedOutputs result;
    std::set<std::string> seen;
    for (const auto& item : outputs["body"].arr) {
        if (item.type != Json::Str) fail(directory,"outputs.body","array of file names","non-string entry");
        const auto shard=sandbox_file(directory,item.str,directory+": outputs.body entry");
        if (!seen.insert(shard).second) fail(directory,"outputs.body","distinct shard files","duplicate "+shard);
        result.body.push_back(shard);
    }
    const auto mtp_name=sandbox_file(directory,text(outputs,"mtp",directory),directory+": outputs.mtp");
    if (seen.count(mtp_name)) fail(directory,"outputs","body/MTP files distinct",mtp_name);
    if (outputs.has("ngram")) {
        result.ngram = sandbox_file(directory,text(outputs,"ngram",directory),directory+": outputs.ngram");
        require_no_symlink(result.ngram, directory+": outputs.ngram");  // sandboxed name: never followed
    }
    else if (!ngram_override.empty())
        result.ngram = resolve_path(directory,ngram_override);  // operator-supplied: absolute is allowed here
    else if (const char* env = env_str("FLASH_NGRAM_PATH"); env && *env)
        result.ngram = env;
    else fail(directory,"outputs.ngram","self-contained table or FLASH_NGRAM_PATH / explicit override","external table not configured");
    result.mtp = want_mtp ? resolve_path(directory,mtp_override.empty()?mtp_name:mtp_override) : std::string{};
    if (want_mtp && mtp_override.empty()) require_no_symlink(result.mtp, directory+": outputs.mtp");  // recorded name only; overrides unchanged
    if (want_mtp && !mtp_override.empty() && std::find(result.body.begin(),result.body.end(),result.mtp) != result.body.end())
        fail(directory,"outputs","separate body/MTP paths",result.mtp);
    if(!is_file(result.ngram)) throw std::runtime_error(result.ngram+": missing INT8 ngram dependency");
    return result;
}

struct NgramSpan { const char* data; size_t bytes, offset; std::string file; };
inline NgramSpan ngram_span(const SafeTensors& tensors, bool int8) {
    std::array<const TensorInfo*,128> shards{};
    const std::string pre="model.language_model.layers.1.ple.ple_embedding.";
    for(int i=0;i<128;++i) shards[i]=&tensors.get(pre+(int8?"ngram_q8.shard_":"ngram_embedding.shard_")
        +std::to_string(i)+(int8?"":".weight"));
    std::sort(shards.begin(),shards.end(),[](const auto* a,const auto* b){return (uintptr_t)a->data<(uintptr_t)b->data;});
    NgramSpan span{shards[0]->data,0,0,{}};
    span.file=tensors.file_of(span.data,&span.offset);
    if(span.file.empty()) throw std::runtime_error("ngram: shard has no file owner");
    for(const auto* t:shards) {
        size_t offset=0;
        const auto file=tensors.file_of(t->data,&offset);
        if(file!=span.file || offset!=span.offset+span.bytes || (uintptr_t)t->data!=(uintptr_t)span.data+span.bytes)
            throw std::runtime_error("ngram: one-span pin requires contiguous shards in one owned file");
        span.bytes+=t->nbytes;
    }
    return span;
}
inline void validate_ngram_tensors(const SafeTensors& st, const NgramExternal& n, const std::string& context) {
    for (int i = 0; i < 128; ++i) {
        const auto key = "model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_" + std::to_string(i);
        if (!st.has(key)) fail(context,key,"present","missing");
        const auto& t = st.get(key);
        if (t.dtype != "U8" || t.shape != std::vector<long long>{n.rows_per_shard,162} || t.nbytes != (size_t)n.rows_per_shard*162)
            fail(context,key,"U8 ["+std::to_string(n.rows_per_shard)+",162] with exact bytes",t.dtype+" "+shape_text(t.shape));
    }
}
struct InventorySpec {
    int layers, experts, hidden, intermediate, q_heads, q_head_dim, kv_heads, lnk_heads, lnk_dim, lnv_heads, lnv_dim, shared_inter, hc, ple_layer;
};
inline void validate_inventory(const std::vector<Converted>& converted, const InventorySpec& s, const std::string& context) {
    std::map<std::string,std::vector<long long>> actual;
    for (const auto& c : converted) if (!actual.emplace(c.base,c.shape).second) fail(context,c.base,"one converted record","duplicate");
    auto take = [&](const std::string& base, std::vector<long long> shape) {
        auto it = actual.find(base);
        if (it == actual.end()) fail(context,base,"converted matrix "+shape_text(shape),"missing");
        if (it->second != shape) fail(context,base,shape_text(shape),shape_text(it->second));
        actual.erase(it);
    };
    auto layer = [&](const std::string& pre, bool full) {
        if (full) {
            take(pre+"self_attn.q_proj",{2LL*s.q_heads*s.q_head_dim,s.hidden});
            take(pre+"self_attn.k_proj",{(long long)s.kv_heads*s.q_head_dim,s.hidden});
            take(pre+"self_attn.v_proj",{(long long)s.kv_heads*s.q_head_dim,s.hidden});
            take(pre+"self_attn.o_proj",{s.hidden,(long long)s.q_heads*s.q_head_dim});
        } else {
            take(pre+"linear_attn.in_proj_qkv",{2LL*s.lnk_heads*s.lnk_dim+(long long)s.lnv_heads*s.lnv_dim,s.hidden});
            take(pre+"linear_attn.in_proj_z",{(long long)s.lnv_heads*s.lnv_dim,s.hidden});
            take(pre+"linear_attn.out_proj",{s.hidden,(long long)s.lnv_heads*s.lnv_dim});
        }
        for (const char* name : {"gate_proj","up_proj"}) take(pre+"mlp.shared_expert."+name,{s.shared_inter,s.hidden});
        take(pre+"mlp.shared_expert.down_proj",{s.hidden,s.shared_inter});
        take(pre+"mlp.experts.gate_up_proj",{s.experts,2LL*s.intermediate,s.hidden});
        take(pre+"mlp.experts.down_proj",{s.experts,s.hidden,s.intermediate});
    };
    for (int i=0;i<s.layers;++i) {
        const auto pre="model.language_model.layers."+std::to_string(i)+".";
        const bool full=actual.count(pre+"self_attn.q_proj")!=0;
        layer(pre,full);
        if (i==s.ple_layer) {
            take(pre+"ple.key_proj",{(long long)s.hc*s.hidden,s.hidden});
            take(pre+"ple.value_proj",{s.hidden,s.hidden});
        }
    }
    layer("mtp.layers.0.",true);
    { // shared rule with tools/quant_int8.py: see kMtpFcEmbeddingKMultipliers above
        auto it = actual.find("mtp.fc_embedding");
        if (it == actual.end()) fail(context,"mtp.fc_embedding","converted matrix","missing");
        bool ok = false;
        for (const int m : kMtpFcEmbeddingKMultipliers)
            ok = ok || it->second == std::vector<long long>{s.hidden, (long long)m * s.hidden};
        if (!ok) fail(context,"mtp.fc_embedding","[hidden,hidden] or [hidden,2*hidden]",shape_text(it->second));
        actual.erase(it);
    }
    take("mtp.fc_hidden",{s.hidden,s.hidden});
    if (!actual.empty()) fail(context,actual.begin()->first,"no unexpected converted matrices","extra record");
}
} // namespace checkpoint_detail

struct Checkpoint {
    SafeTensors body, mtp, ngram;
    WeightFormat format=WeightFormat::Int4;
    bool has_manifest=false;
    std::string model_dir, ngram_path;
    std::vector<checkpoint_detail::Converted> converted;
    Checkpoint() = default;
    Checkpoint(const Checkpoint&)=delete; Checkpoint& operator=(const Checkpoint&)=delete;
    Checkpoint(Checkpoint&&)=delete; Checkpoint& operator=(Checkpoint&&)=delete;
    const SafeTensors& ngram_tensors() const { return has_manifest ? ngram : body; }
    void validate_inventory(const checkpoint_detail::InventorySpec& spec) const {
        if (has_manifest) checkpoint_detail::validate_inventory(converted,spec,model_dir+"/qfn-int8-manifest.json");
    }
    void open(const std::string& directory, bool want_mtp, const std::string& mtp_override,
              const std::string& ngram_override = "") {
        using namespace checkpoint_detail;
        model_dir=directory;
        const auto path=resolve_path(directory,"qfn-int8-manifest.json");
        if(std::filesystem::exists(path) && !is_file(path))
            fail(path,"manifest","regular file","non-file path");
        if (!is_file(path)) {
            // the format comes from the tensor headers below, never from config.json: an AutoRound INT4
            // directory legitimately carries quantization_config groups with 8-bit entries (its lm_head /
            // KV quantizers), and reading the format there would refuse a working INT4 checkpoint
            std::vector<std::string> files;
            for (const auto& e : std::filesystem::directory_iterator(directory)) {
                const auto name=e.path().filename().string();
                // real INT4 directories contain a stray ngram_int8.safetensors DIRECTORY: only regular files qualify
                if (e.is_regular_file() && e.path().extension()==".safetensors" && name.rfind("mtp",0)!=0 && name!="model_mtp.safetensors") files.push_back(e.path().string());
            }
            std::sort(files.begin(),files.end());
            if (files.empty()) throw std::runtime_error(directory+": no body safetensors files");
            for (const auto& f:files) {
                const auto j=validate_file(f);
                for(const auto& t:j.obj) if(t.first.size()>14 && t.first.compare(t.first.size()-14,14,".weight_packed")==0) {
                    const auto base=t.first.substr(0,t.first.size()-14);
                    // INT8 detection: the .weight_shape companion, not a scale-width heuristic (which
                    // also matches INT4 group-256 checkpoints)
                    if(j.has(base+".weight_scale") && j.has(base+".weight_shape"))
                        throw std::runtime_error(f+": packed INT8 checkpoint requires qfn-int8-manifest.json");
                }
                body.open(f);
            }
            if(want_mtp) {
                if(!mtp_override.empty()) open_file(mtp,resolve_path(directory,mtp_override));
                else {
                    bool found=false;
                    for(const auto* name:{"mtp.safetensors","mtp-int4.safetensors"}) {
                        const auto f=resolve_path(directory,name);
                        if(is_file(f)){open_file(mtp,f);found=true;}
                    }
                    if(!found) throw std::runtime_error(directory+": requested MTP checkpoint missing");
                }
            }
            return;
        }
        require_no_symlink(path,path);  // the manifest itself is a sandboxed name: never a symlink
        has_manifest=true; format=WeightFormat::Int8;
        const auto j=parse_json(read_file(path),path);
        validate_manifest_top(j,path); validate_recipe(field(j,"recipe",path),path);
        const auto& records=field(field(j,"tensors",path),"converted",path);
        if(records.type!=Json::Arr) fail(path,"tensors.converted","array","other JSON type");
        for(const auto& r:records.arr) converted.push_back(parse_converted(r,path));
        const size_t mtp_count=(size_t)std::count_if(converted.begin(),converted.end(),[](const auto& c){return c.source_key.rfind("mtp.",0)==0;});
        if(converted.size()-mtp_count!=398 || mtp_count!=11)
            fail(path,"converted inventory","398 body + 11 MTP",std::to_string(converted.size()-mtp_count)+" body + "+std::to_string(mtp_count)+" MTP");
        const auto resolved=resolve_outputs(j,directory,want_mtp,mtp_override,ngram_override);
        const std::string mtp_record_name=text(j["outputs"],"mtp",path);
        // Source honesty: repo/revision are recorded only when the converter was told what was ACTUALLY
        // converted; an unlisted or absent source is never a rejection, the index digest is the integrity record.
        {
            const auto& source=field(j,"source",path);
            if (source.has("repo") != source.has("revision"))
                fail(path,"source","repo and revision recorded together","half-recorded source identity");
            if (source.has("repo")) { text(source,"repo",path); text(source,"revision",path); }
            const auto kind=text(source,"index_sha256_kind",path);
            if (kind != "index_json" && kind != "weight_files") fail(path,"source.index_sha256_kind","index_json or weight_files",kind);
            const auto digest=text(source,"index_sha256",path);
            if (digest.size()!=64 || !std::all_of(digest.begin(),digest.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))
                fail(path,"source.index_sha256","64 lowercase hex digits","invalid digest");
        }
        const auto limit=integer(j,"shard_limit_bytes",path);
        if (limit<=0 || (uint64_t)limit>kMaxShardBytes) fail(path,"shard_limit_bytes","1..20 GiB",std::to_string(limit));
        std::set<std::string> shard_names;
        for (const auto& item : j["outputs"]["body"].arr) shard_names.insert(item.str);
        // map every manifest-listed tensor to its shard; a tensor recorded twice fails at insert
        std::map<std::string,std::string> tensor_shard;
        const auto& preserved=field(field(j,"tensors",path),"preserved",path);
        const auto& preserved_body=field(preserved,"body",path);
        if(preserved_body.type!=Json::Obj) fail(path,"tensors.preserved.body","object of tensor -> shard","invalid type");
        for(const auto& entry:preserved_body.obj) {
            if(entry.second.type!=Json::Str) fail(path,entry.first,"shard file name","invalid type");
            if(!shard_names.count(entry.second.str)) fail(path,entry.first,"a body shard of this checkpoint",entry.second.str);
            if(!tensor_shard.emplace(entry.first,entry.second.str).second) fail(path,entry.first,"one shard","duplicate mapping");
        }
        const auto& preserved_mtp=field(preserved,"mtp",path);
        if(preserved_mtp.type!=Json::Arr) fail(path,"tensors.preserved.mtp","array","invalid type");
        for(const auto& name:preserved_mtp.arr) {
            if(name.type!=Json::Str) fail(path,"tensors.preserved.mtp","tensor names","invalid entry");
            if(!tensor_shard.emplace(name.str,mtp_record_name).second) fail(path,name.str,"one shard","duplicate mapping");
        }
        for(const auto& c:converted) {
            const bool is_mtp=c.source_key.rfind("mtp.",0)==0;
            if(!is_mtp && !shard_names.count(c.file)) fail(path,c.base+".file","a body shard of this checkpoint",c.file);
            if(is_mtp && c.file!=mtp_record_name) fail(path,c.base+".file",mtp_record_name,c.file);
            for(const auto* suffix:{".weight_packed",".weight_scale",".weight_shape"})
                if(!tensor_shard.emplace(c.base+suffix,c.file).second) fail(path,c.base,"unique selected output","duplicate");
        }
        // enforce the size cap, then match every shard header against the manifest mapping exactly
        std::vector<Json> shard_headers;
        for(const auto& shard:resolved.body) {
            require_no_symlink(shard,shard);  // sandboxed manifest name: reject a substituted symlink
            if(std::filesystem::file_size(shard)>(uintmax_t)limit)
                fail(shard,"shard size","at most "+std::to_string(limit)+" bytes",std::to_string(std::filesystem::file_size(shard)));
            shard_headers.push_back(validate_file(shard));
        }
        std::map<std::string,std::set<std::string>> expected_per_shard;
        for(const auto& [tensor,shard_file]:tensor_shard) expected_per_shard[shard_file].insert(tensor);
        for(size_t index=0;index<resolved.body.size();++index) {
            const auto name=j["outputs"]["body"][index].str;
            auto expected=expected_per_shard[name];
            if(expected.empty()) fail(path,name,"at least one tensor","empty shard");
            for(const auto& t:shard_headers[index].obj) if(t.first!="__metadata__") {
                if(!expected.erase(t.first)) fail(path,t.first,"manifest-listed tensor of "+name,"unlisted or misplaced output");
            }
            if(!expected.empty()) fail(path,*expected.begin(),"output tensor of "+name,"missing header");
        }
        if(want_mtp) {
            auto expected=expected_per_shard[mtp_record_name];
            const auto header=validate_file(resolved.mtp);
            for(const auto& t:header.obj) if(t.first!="__metadata__") {
                if(!expected.erase(t.first)) fail(path,t.first,"manifest-listed MTP tensor","unlisted or misplaced output");
            }
            if(!expected.empty()) fail(path,*expected.begin(),"MTP output tensor","missing header");
            mtp.open(resolved.mtp);
        }
        const auto config_path=resolve_path(directory,sandbox_file(directory,text(j["outputs"],"config",path),path+": outputs.config"));
        require_no_symlink(config_path,config_path);  // sandboxed manifest name
        const auto config=parse_json(read_bounded_file(config_path,4u<<20,config_path),config_path);
        const auto& text_config=config.has("text_config")?config["text_config"]:config;
        { // required keys: never skip a missing one; the error lists everything that is absent
            std::string missing;
            for(const auto* key:{"hidden_size","num_hidden_layers","num_attention_heads","head_dim","num_key_value_heads",
                    "linear_num_key_heads","linear_key_head_dim","linear_num_value_heads","linear_value_head_dim",
                    "num_experts","moe_intermediate_size","hc_count"})
                if(!text_config.has(key)) missing += (missing.empty()?"":", ")+std::string(key);
            if(!missing.empty()) fail(path,"config","all required text_config keys","missing: "+missing);
        }
        for(const auto& expected: {std::pair<const char*,long long>{"hidden_size",2560},{"num_hidden_layers",48},
                {"num_attention_heads",24},{"head_dim",256},{"num_key_value_heads",2},{"linear_num_key_heads",16},
                {"linear_key_head_dim",128},{"linear_num_value_heads",48},{"linear_value_head_dim",128},
                {"num_experts",512},{"moe_intermediate_size",640},{"hc_count",4}}) {
            if(integer(text_config,expected.first,path)!=expected.second)
                fail(path,std::string("config.")+expected.first,std::to_string(expected.second),
                     std::to_string(integer(text_config,expected.first,path)));
        }
        for(const auto& shard:resolved.body) body.open(shard);
        {
            const auto& types=field(text_config,"layer_types",path);
            if(types.type!=Json::Arr || types.arr.size()!=48) fail(path,"config.layer_types","48 entries","invalid count/type");
            for(int i=0;i<48;++i) {
                const auto key="model.language_model.layers."+std::to_string(i)+".self_attn.q_proj.weight_packed";
                const auto expected=body.has(key)?"full_attention":"linear_attention";
                if(types[i].type!=Json::Str || types[i].str!=expected)
                    fail(path,"config.layer_types["+std::to_string(i)+"]",expected,types[i].str);
            }
        }
        // Architecture is fixed; fixtures exercise pure helpers, never a runtime bypass.
        validate_inventory({48,512,2560,640,24,256,2,16,128,48,128,640,4,1});
        for(const auto& c:converted) {
            const bool is_mtp=c.source_key.rfind("mtp.",0)==0;
            if(!is_mtp || want_mtp) validate_int8_tensor(is_mtp?mtp:body,c.base,c.shape,c.file);
        }
        const char* mode=env_str("FLASH_NGRAM");
        if(mode && std::string(mode)=="bf16") throw std::runtime_error(path+": FLASH_NGRAM=bf16 conflicts with the INT8 ngram table");
        const auto& identity=field(j,"ngram",path);
        const auto n=parse_ngram_identity(identity,path);
        if(n.rows_per_shard!=2500012) fail(path,"ngram.rows_per_shard","2500012",std::to_string(n.rows_per_shard));
        const auto& external_keys=field(identity,"keys",path);
        if(external_keys.type!=Json::Arr || external_keys.arr.size()!=128)
            fail(path,"ngram.keys","128 shard names","invalid inventory");
        for(int i=0;i<128;++i) {
            const auto expected="model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_"+std::to_string(i);
            if(external_keys[i].type!=Json::Str || external_keys[i].str!=expected)
                fail(path,"ngram.keys["+std::to_string(i)+"]",expected,external_keys[i].str);
        }
        if(identity.has("equivalence_verified") && identity["equivalence_verified"].type==Json::Bool && identity["equivalence_verified"].b)
            fail(path,"ngram.equivalence_verified","false","true");
        ngram_path=resolved.ngram;
        if(std::filesystem::file_size(ngram_path)!=(uintmax_t)n.size_bytes)
            fail(ngram_path,"size_bytes",std::to_string(n.size_bytes),std::to_string(std::filesystem::file_size(ngram_path)));
        const auto ngram_header=validate_file(ngram_path);
        if(ngram_header.obj.size()-(ngram_header.has("__metadata__")?1:0)!=128)
            fail(ngram_path,"tensor inventory","exactly 128 shards","unexpected shard count");
        ngram.open(ngram_path); validate_ngram_tensors(ngram,n,ngram_path);
        const std::string pre="model.language_model.layers.1.ple.ple_embedding.";
        long long total=0;
        if(!body.has(pre+"layer_multipliers")) fail(path,pre+"layer_multipliers","I64 [3]","missing");
        const auto& multipliers=body.get(pre+"layer_multipliers");
        if(multipliers.dtype!="I64" || multipliers.shape!=std::vector<long long>{3} || multipliers.nbytes!=24)
            fail(path,pre+"layer_multipliers","I64 [3]",multipliers.dtype+" "+shape_text(multipliers.shape));
        for(const auto* name:{"ngram_heads_vocab_sizes","ngram_heads_offsets"}) {
            if(!body.has(pre+name)) fail(path,pre+name,"I64 [16]","missing");
            const auto& t=body.get(pre+name);
            if(t.dtype!="I64" || t.shape!=std::vector<long long>{16} || t.nbytes!=128)
                fail(path,pre+name,"I64 [16]",t.dtype+" "+shape_text(t.shape));
        }
        for(int i=0;i<16;++i) {
            int64_t count,offset;
            memcpy(&count,body.get(pre+"ngram_heads_vocab_sizes").data+i*8,8);
            memcpy(&offset,body.get(pre+"ngram_heads_offsets").data+i*8,8);
            if(count<=0 || offset!=total || count>128LL*n.rows_per_shard-total)
                fail(path,"PLE offsets/vocab sizes","contiguous bounded row ranges","invalid metadata at head "+std::to_string(i));
            total+=count;
        }
        if((total+127)/128!=n.rows_per_shard)
            fail(path,"PLE padded row total",std::to_string(128LL*n.rows_per_shard),std::to_string((total+127)/128*128));
        fprintf(stderr,"[flash] validated manifest: 398 body + 11 MTP native INT8 matrices in %zu shards; MTP %s; %s ngram %lld rows plus %lld padding rows, %lld bytes\n",
            resolved.body.size(),want_mtp?"headers validated":"not requested",
            j["outputs"].has("ngram")?"self-contained":"external",total,128LL*n.rows_per_shard-total,n.size_bytes);
    }
};
} // namespace flash
