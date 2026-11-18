#pragma once
#include "../safetensors.h"
#include "../env.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>
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
// Fixed architecture inventory, shared with tools/quant_int8.py (the converter mirrors these names;
// tests/test_quant_int8.py parses this header to prove the mirrors cannot drift):
inline constexpr int kBodyMatrices = 398;               // converted body matrices of a full QFN checkpoint
inline constexpr int kMtpMatrices = 11;                 // converted MTP matrices
inline constexpr int kLayerCount = 48;                  // language-model layers
inline constexpr long long kNgramRowsPerShard = 2500012; // pinned QFN n-gram rows per 128-shard table
// Upper bound for the manifest file itself: a 409-record v2 manifest is ~1 MB, 16 MiB is generous.
inline constexpr uint64_t kMaxManifestBytes = 16ull << 20;
// Engine-side shapes of preserved tensors read in fixed-size regions by Worker::load
// (flash.h mirrors these constants; the converter preserves the upstream tensors untouched):
inline constexpr int kHcLowRank = 320;              // hyper-connection low-rank width (LOWRANK)
inline constexpr int kIndexerHeads = 4;             // indexer key heads; the projection adds one query head (IDX_NH)
inline constexpr int kIndexerHeadDim = 128;         // indexer head dimension (IDX_HD)
inline constexpr int kConvKernel = 4;               // short-convolution width (linear_attn and PLE conv1d)
inline constexpr long long kVocabRows = 248320;     // shared embedding and lm_head rows (VOCAB)
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
// Sandbox symlink gate: EVERY component of a sandbox-resolved manifest path below
// the checkpoint root must be a non-symlink. Checking only the final component left names like
// "sub/shard.safetensors" escapable through an intermediate directory replaced by a symlink to an
// outside directory. Missing components pass through so the established "missing" diagnostics stay;
// the checkpoint root itself is operator-chosen and deliberately not gated. Scope: operator-supplied
// absolute inputs (FLASH_NGRAM_PATH, the explicit n-gram/MTP override arguments) keep their documented
// absolute allowance and are NOT symlink-gated; only sandbox-joined names are.
inline void require_no_symlink(const std::string& root, const std::string& path, const std::string& context) {
    std::error_code ec;
    const auto rel = std::filesystem::path(path).lexically_relative(std::filesystem::path(root));
    if (rel.empty() || rel.is_absolute()) {  // not under the root: keep the final-component check alone
        const auto type = std::filesystem::symlink_status(path, ec).type();
        if (!ec && type == std::filesystem::file_type::symlink)
            fail(context, "path", "regular file (the sandbox never follows symlinks)", "symbolic link: " + path);
        return;
    }
    auto prefix = std::filesystem::path(root);
    for (const auto& part : rel) {
        prefix /= part;
        const auto status = std::filesystem::symlink_status(prefix, ec);
        if (ec) return;  // missing or unreadable component: the later guarded open reports it
        if (status.type() == std::filesystem::file_type::symlink)
            fail(context, "path", "no symlink component (the sandbox never follows symlinks)",
                 "symbolic link component: " + prefix.string());
    }
}
inline std::string read_bounded_file(const std::string& path, uint64_t max_bytes, const std::string& context) {
    // Same guarded open as the safetensors files: the type/regular check happens before a single
    // byte is read, so a FIFO or directory at a manifest-recorded name fails cleanly instead of blocking.
    const int fd = open_regular_file(path, true);
    struct stat st;
    const int rc = fstat(fd, &st);
    if (rc != 0 || st.st_size < 0 || (uint64_t)st.st_size > max_bytes) {
        const std::string actual = rc ? "unreadable" : std::to_string((long long)st.st_size);
        ::close(fd);
        fail(context, "file size", "at most " + std::to_string(max_bytes) + " bytes", actual);
    }
    std::string bytes((size_t)st.st_size, '\0');
    size_t off = 0;
    while (off < bytes.size()) {
        const ssize_t n = ::pread(fd, bytes.data() + off, bytes.size() - off, (off_t)off);
        if (n <= 0) { ::close(fd); throw std::runtime_error(path + ": cannot read file"); }
        off += (size_t)n;
    }
    ::close(fd);
    return bytes;
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
// Last stored dimension of a .weight_shape companion. The header JSON only records the offsets: the
// companion is a 1-D I64 vector whose element count comes from (end-start)/8, so the last element —
// the stored logical K — is read without trusting the header's shape list.
inline long long stored_shape_last_dim(const std::string& path, const Json& entry) {
    if (text(entry,"dtype",path) != "I64") fail(path,".weight_shape","I64 logical shape",text(entry,"dtype",path));
    const auto& offsets = field(entry,"data_offsets",path);
    if (offsets.type != Json::Arr || offsets.arr.size() != 2 ||
        offsets.arr[0].type != Json::Num || offsets.arr[1].type != Json::Num ||
        offsets.arr[1].i() <= offsets.arr[0].i() || (offsets.arr[1].i()-offsets.arr[0].i()) % 8 != 0)
        fail(path,".weight_shape","I64 data offsets with a whole number of int64s","malformed companion");
    const long long count = (offsets.arr[1].i()-offsets.arr[0].i())/8;
    // Guarded open (never a blocking open on a special file); the scan route is not sandbox-gated,
    // so symlinks keep their legacy allowance but the regular-file check still applies via fstat.
    const int fd = open_regular_file(path,false);
    uint64_t header = 0;
    if (::pread(fd,(char*)&header,8,0)!=8) { ::close(fd); fail(path,".weight_shape","readable stored shape","file shorter than 8 bytes"); }
    const off_t off = (off_t)(8 + header + (uint64_t)offsets.arr[0].i() + (uint64_t)(count-1)*8);
    int64_t last = 0;
    const ssize_t got = ::pread(fd,(char*)&last,8,off);
    ::close(fd);
    if (got != 8) fail(path,".weight_shape","readable stored shape","truncated payload");
    return last;
}
// Manifest-backed safetensors files reach the loader through SafeTensors::open_strict_validated — one guarded
// open per file, strict header parse from the mapped bytes exactly once, payload validated against
// those same bytes and the tensor table indexed from THAT parsed object (returned for manifest
// cross-checks). The former validate-then-reopen pair (ifstream strict parse + permissive
// SafeTensors::open on the path again) was double work and a swap/symlink window between the opens.
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
    // N-gram precedence: a recorded self-contained outputs.ngram ALWAYS wins; a
    // supplied override/env is then reported as IGNORED by Checkpoint::open, never followed. External
    // checkpoints (no recorded name) resolve the argument and FLASH_NGRAM_PATH identically: relative
    // values resolve against the model directory, absolute paths are allowed.
    if (outputs.has("ngram")) {
        result.ngram = sandbox_file(directory,text(outputs,"ngram",directory),directory+": outputs.ngram");
        require_no_symlink(directory,result.ngram,directory+": outputs.ngram");  // sandboxed name: never followed
    }
    else if (!ngram_override.empty())
        result.ngram = resolve_path(directory,ngram_override);  // operator-supplied: absolute or checkpoint-relative
    else if (const char* env = env_str("FLASH_NGRAM_PATH"); env && *env)
        result.ngram = resolve_path(directory,env);             // same resolution as the argument
    else fail(directory,"outputs.ngram","self-contained table or FLASH_NGRAM_PATH / explicit override","external table not configured");
    // sandbox_file already joined the recorded MTP name with the checkpoint directory: wrapping the
    // result in resolve_path a second time doubled a RELATIVE checkpoint dir
    // (models/int8/models/int8/mtp.safetensors). Only operator-supplied overrides join again.
    result.mtp = want_mtp ? (mtp_override.empty() ? mtp_name : resolve_path(directory,mtp_override)) : std::string{};
    if (want_mtp && mtp_override.empty()) require_no_symlink(directory,result.mtp,directory+": outputs.mtp");  // recorded name only; overrides unchanged
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
// Runtime consumers read FIXED-SIZE regions straight out of the mapping for every preserved
// tensor below (Worker::load, the load_w16/load_w8/bf_dev call sites) — dtype,
// shape and exact byte extent are therefore pinned at open: a mistyped or truncated tensor must fail
// there, not overread at load. Names the text engine never reads (the preserved visual tower, extra
// layer norms) and the mixers' absent block_inject_weight stay optional.
inline void require_preserved(const SafeTensors& st, const std::string& name,
                              const std::vector<long long>& shape, const std::string& context) {
    if (!st.has(name)) fail(context, name, "engine-required BF16 " + shape_text(shape), "missing");
    const auto& t = st.get(name);
    const size_t bytes = elements(shape, context) * 2;
    if (t.dtype != "BF16" || t.shape != shape || t.nbytes != bytes)
        fail(context, name, "BF16 " + shape_text(shape) + " with exact byte extent",
             t.dtype + " " + shape_text(t.shape) + " bytes=" + std::to_string(t.nbytes));
}
inline void validate_preserved(const SafeTensors& body, const SafeTensors* mtp, const InventorySpec& s,
                               const std::string& context) {
    const long long rw = (long long)s.hc * s.hidden;  // RW = HC*H in the engine
    auto mixer = [&](const SafeTensors& st, const std::string& pre, bool inject) {
        require_preserved(st, pre+"hc_norm.weight", {rw}, context);
        require_preserved(st, pre+"input_mix_weight_down.weight", {kHcLowRank, rw}, context);
        require_preserved(st, pre+"input_mix_weight_up.weight", {rw, kHcLowRank}, context);
        if (inject) require_preserved(st, pre+"block_inject_weight.weight", {s.hc, rw}, context);
    };
    auto layer = [&](const SafeTensors& st, const std::string& pre, bool full) {
        mixer(st, pre+"attn_hyper_connection.", true);
        mixer(st, pre+"mlp_hyper_connection.", true);
        require_preserved(st, pre+"mlp.gate.weight", {s.experts, s.hidden}, context);
        require_preserved(st, pre+"mlp.shared_expert_gate.weight", {1, s.hidden}, context);
        if (full) {
            require_preserved(st, pre+"self_attn.q_norm.weight", {s.q_head_dim}, context);
            require_preserved(st, pre+"self_attn.k_norm.weight", {s.q_head_dim}, context);
            require_preserved(st, pre+"self_attn.indexer.index_qk_proj.weight",
                              {(long long)(kIndexerHeads+1)*kIndexerHeadDim, s.hidden}, context);
            require_preserved(st, pre+"self_attn.indexer.q_layernorm.weight", {kIndexerHeadDim}, context);
            require_preserved(st, pre+"self_attn.indexer.k_layernorm.weight", {kIndexerHeadDim}, context);
        } else {
            require_preserved(st, pre+"linear_attn.A_log", {s.lnv_heads}, context);
            require_preserved(st, pre+"linear_attn.dt_bias", {s.lnv_heads}, context);
            require_preserved(st, pre+"linear_attn.conv1d.weight", {rw, 1, kConvKernel}, context);
            require_preserved(st, pre+"linear_attn.norm.weight", {s.lnv_dim}, context);
            require_preserved(st, pre+"linear_attn.in_proj_a.weight", {s.lnv_heads, s.hidden}, context);
            require_preserved(st, pre+"linear_attn.in_proj_b.weight", {s.lnv_heads, s.hidden}, context);
        }
    };
    require_preserved(body, "model.language_model.embed_tokens.weight", {kVocabRows, s.hidden}, context);
    require_preserved(body, "lm_head.weight", {kVocabRows, s.hidden}, context);
    mixer(body, "model.language_model.hyper_connection_mixer.", false);  // head mixer: no inject rows
    for (int i=0;i<s.layers;++i) {
        const auto pre="model.language_model.layers."+std::to_string(i)+".";
        layer(body, pre, body.has(pre+"self_attn.q_proj.weight_packed"));  // same family rule as layer_types
        if (i==s.ple_layer) {
            require_preserved(body, pre+"ple.norm_key.weight", {rw}, context);
            require_preserved(body, pre+"ple.norm_query.weight", {rw}, context);
            require_preserved(body, pre+"ple.norm_conv.weight", {rw}, context);
            require_preserved(body, pre+"ple.conv1d.weight", {rw, 1, kConvKernel}, context);
        }
    }
    if (!mtp) return;  // want_mtp: the opened MTP block is read the same way
    mixer(*mtp, "mtp.hyper_connection_mixer.", false);
    layer(*mtp, "mtp.layers.0.", true);  // the MTP block is a full-attention layer
    require_preserved(*mtp, "mtp.pre_fc_norm_embedding.weight", {s.hidden}, context);
    require_preserved(*mtp, "mtp.pre_fc_norm_hidden.weight", {rw}, context);
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
                // (that skip is part of the accepted format layer); every regular *.safetensors the released loader
                // would open stays in the scan, model_mtp.safetensors included - an INT8 directory without manifest
                // is refused by the format check below, exactly one file earlier
                if (e.path().extension()==".safetensors" && e.is_symlink() && !std::filesystem::exists(e.path()))
                    // as on main: the released loader tried to open every *.safetensors name in the directory, so a
                    // dangling link fails the start there; silently skipping it is a manifest-route behavior only
                    throw std::runtime_error(e.path().string()+": cannot open (symbolic link with a missing target)");
                if (e.is_regular_file() && e.path().extension()==".safetensors" && name.rfind("mtp",0)!=0) files.push_back(e.path().string());
            }
            std::sort(files.begin(),files.end());
            if (files.empty()) throw std::runtime_error(directory+": no body safetensors files");
            for (const auto& f:files) {
                // Keep the legacy loader for manifest-less INT4. Detection reuses its parsed header
                // without imposing manifest-only contiguity, dtype or header-cap checks.
                const auto j=body.open(f);  // legacy header and payload semantics, including alignment gaps
                for(const auto& t:j.obj) if(t.first.size()>14 && t.first.compare(t.first.size()-14,14,".weight_packed")==0) {
                    const auto base=t.first.substr(0,t.first.size()-14);
                    // The .weight_shape companion alone does NOT decide the format: compressed-tensors
                    // INT4 stores one too. The stored logical K against the packed width does: INT8 packs
                    // 4 codes per I32 (K == width*4 -> needs the manifest), INT4 packs 8 nibbles per I32
                    // (K == width*8 -> normal INT4 no-manifest route). Anything else is a broken file.
                    if(j.has(base+".weight_scale") && j.has(base+".weight_shape")) {
                        // The discriminator only makes sense for a rank>=1 I32-packed matrix with a
                        // rank>=1 .weight_shape companion. A scalar packed entry (shape []) used to reach
                        // arr.back() on the empty shape vector (undefined behaviour); the guard rejects
                        // the degenerate trio cleanly before any file byte or dimension is inspected.
                        const auto& packed=j[t.first];
                        const auto& stored=j[base+".weight_shape"];
                        const long long packed_rank=(long long)packed["shape"].arr.size();
                        const long long stored_rank=(long long)stored["shape"].arr.size();
                        if(packed["dtype"].str!="I32" || packed_rank<1 || stored_rank<1)
                            fail(f,base,"rank>=1 I32 .weight_packed with a rank>=1 .weight_shape companion",
                                 packed["dtype"].str+" packed rank "+std::to_string(packed_rank)+
                                 ", companion rank "+std::to_string(stored_rank));
                        const long long k = stored_shape_last_dim(f,j[base+".weight_shape"]);
                        const long long width = j[t.first]["shape"].arr.back().i();
                        if (width*4 == k)
                            throw std::runtime_error(f+": packed INT8 checkpoint requires qfn-int8-manifest.json");
                        if (width*8 != k)
                            fail(f,base+".weight_shape","packed width x4 (INT8) or x8 (INT4) matching stored K "+std::to_string(k),
                                 std::to_string(width)+"-wide packing");
                    }
                }
            }
            if(want_mtp) {
                if(!mtp_override.empty()) mtp.open(resolve_path(directory,mtp_override));
                else {
                    // as on main: the released loader opened BOTH MTP names and relied on the identical-duplicate
                    // rule to cross-check them; taking either file alone is a manifest-route allowance, not this route
                    for(const auto* name:{"mtp.safetensors","mtp-int4.safetensors"}) {
                        const auto f=resolve_path(directory,name);
                        if(!is_file(f))
                            throw std::runtime_error(f+": the manifest-less route opens both MTP files, as the released loader did");
                        mtp.open(f);
                    }
                }
            }
            return;
        }
        require_no_symlink(directory,path,path);  // the manifest itself is a sandboxed name: never a symlink
        has_manifest=true; format=WeightFormat::Int8;
        // An mtp_override must be a BYTE-COMPATIBLE replacement of the recorded MTP inventory:
        // the header-vs-manifest matching below rejects a replacement with missing, extra or altered
        // tensors, so an override can relocate the file but never change what it contains.
        const auto j=parse_json(read_bounded_file(path,kMaxManifestBytes,path),path);
        validate_manifest_top(j,path); validate_recipe(field(j,"recipe",path),path);
        const auto& records=field(field(j,"tensors",path),"converted",path);
        if(records.type!=Json::Arr) fail(path,"tensors.converted","array","other JSON type");
        for(const auto& r:records.arr) converted.push_back(parse_converted(r,path));
        const size_t mtp_count=(size_t)std::count_if(converted.begin(),converted.end(),[](const auto& c){return c.source_key.rfind("mtp.",0)==0;});
        if(converted.size()-mtp_count!=(size_t)kBodyMatrices || mtp_count!=(size_t)kMtpMatrices)
            fail(path,"converted inventory",std::to_string(kBodyMatrices)+" body + "+std::to_string(kMtpMatrices)+" MTP",
                 std::to_string(converted.size()-mtp_count)+" body + "+std::to_string(mtp_count)+" MTP");
        const auto resolved=resolve_outputs(j,directory,want_mtp,mtp_override,ngram_override);
        const std::string mtp_record_name=text(j["outputs"],"mtp",path);
        // Source honesty: repo/revision are recorded only when the converter was told what was ACTUALLY
        // converted; an unlisted or absent source is never a rejection, the index digest is the traceability record.
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
            require_no_symlink(directory,shard,shard);  // sandboxed manifest name: reject a substituted symlink
            // Single guarded open per shard — the cap is checked against the fstat'ed size before
            // mapping, the header is parsed strictly once from the mapped bytes and the tensors below
            // are indexed from that same parse (no reopen, no second parse, no window in between)
            shard_headers.push_back(body.open_strict_validated(shard,true,(uint64_t)limit));
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
            const auto header=mtp.open_strict_validated(resolved.mtp, mtp_override.empty());  // recorded name: O_NOFOLLOW
            for(const auto& t:header.obj) if(t.first!="__metadata__") {
                if(!expected.erase(t.first)) fail(path,t.first,"manifest-listed MTP tensor","unlisted or misplaced output");
            }
            if(!expected.empty()) fail(path,*expected.begin(),"MTP output tensor","missing header");
        }
        // recorded manifest paths are already sandbox-joined: resolving the joined value against the
        // directory again doubled a relative checkpoint dir (models/int8/models/int8/config.json)
        const auto config_path=sandbox_file(directory,text(j["outputs"],"config",path),path+": outputs.config");
        require_no_symlink(directory,config_path,config_path);  // sandboxed manifest name
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
        for(const auto& expected: {std::pair<const char*,long long>{"hidden_size",2560},{"num_hidden_layers",kLayerCount},
                {"num_attention_heads",24},{"head_dim",256},{"num_key_value_heads",2},{"linear_num_key_heads",16},
                {"linear_key_head_dim",128},{"linear_num_value_heads",48},{"linear_value_head_dim",128},
                {"num_experts",512},{"moe_intermediate_size",640},{"hc_count",4}}) {
            if(integer(text_config,expected.first,path)!=expected.second)
                fail(path,std::string("config.")+expected.first,std::to_string(expected.second),
                     std::to_string(integer(text_config,expected.first,path)));
        }
        {   // body shards are already opened (single guarded open above); layer_types cross-checks the mapping
            const auto& types=field(text_config,"layer_types",path);
            if(types.type!=Json::Arr || types.arr.size()!=(size_t)kLayerCount)
                fail(path,"config.layer_types",std::to_string(kLayerCount)+" entries","invalid count/type");
            for(int i=0;i<kLayerCount;++i) {
                const auto key="model.language_model.layers."+std::to_string(i)+".self_attn.q_proj.weight_packed";
                const auto expected=body.has(key)?"full_attention":"linear_attention";
                if(types[i].type!=Json::Str || types[i].str!=expected)
                    fail(path,"config.layer_types["+std::to_string(i)+"]",expected,types[i].str);
            }
        }
        // Architecture is fixed; fixtures exercise pure helpers, never a runtime bypass.
        const InventorySpec spec{kLayerCount,512,2560,640,24,256,2,16,128,48,128,640,4,1};
        validate_inventory(spec);
        validate_preserved(body, want_mtp ? &mtp : nullptr, spec, path);
        for(const auto& c:converted) {
            const bool is_mtp=c.source_key.rfind("mtp.",0)==0;
            if(!is_mtp || want_mtp) validate_int8_tensor(is_mtp?mtp:body,c.base,c.shape,c.file);
        }
        const char* mode=env_str("FLASH_NGRAM");
        if(mode && std::string(mode)=="bf16") throw std::runtime_error(path+": FLASH_NGRAM=bf16 conflicts with the INT8 ngram table");
        const auto& identity=field(j,"ngram",path);
        const auto n=parse_ngram_identity(identity,path);
        if(n.rows_per_shard!=kNgramRowsPerShard) fail(path,"ngram.rows_per_shard",std::to_string(kNgramRowsPerShard),std::to_string(n.rows_per_shard));
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
        const auto ngram_header=ngram.open_strict_validated(ngram_path, j["outputs"].has("ngram"));  // recorded table: O_NOFOLLOW
        if(ngram_header.obj.size()-(ngram_header.has("__metadata__")?1:0)!=128)
            fail(ngram_path,"tensor inventory","exactly 128 shards","unexpected shard count");
        validate_ngram_tensors(ngram,n,ngram_path);
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
        // UNIFIED RULE reporting: a self-contained checkpoint IGNORES any override/env — say so on
        // one line each instead of leaving the operator guessing; the summary states the ACTUAL source.
        if (j["outputs"].has("ngram")) {
            if (!ngram_override.empty())
                fprintf(stderr,"[flash] WARNING: ngram override %s IGNORED — checkpoint is self-contained (outputs.ngram)\n",
                    ngram_override.c_str());
            if (const char* env = env_str("FLASH_NGRAM_PATH"); env && *env)
                fprintf(stderr,"[flash] WARNING: ngram override %s IGNORED — checkpoint is self-contained (outputs.ngram)\n",env);
        }
        const std::string ngram_src = j["outputs"].has("ngram") ? std::string("self-contained ngram")
                                                                : "external ngram (override): "+resolved.ngram;
        fprintf(stderr,"[flash] validated manifest: %d body + %d MTP native INT8 matrices in %zu shards; MTP %s; %s — %lld rows plus %lld padding rows, %lld bytes\n",
            kBodyMatrices,kMtpMatrices,resolved.body.size(),want_mtp?"headers validated":"not requested",
            ngram_src.c_str(),total,128LL*n.rows_per_shard-total,n.size_bytes);
    }
};
} // namespace flash
