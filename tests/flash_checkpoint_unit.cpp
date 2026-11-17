#include "flash/checkpoint.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <sys/stat.h>  // mkfifo fixture
#include <stdexcept>
#include <string>
#include <vector>

namespace fs=std::filesystem;
using namespace flash;
using namespace flash::checkpoint_detail;
static int fails=0;
#define require(c,why) do { if(!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,(why)); ++fails; } } while(0)
template<class F> static void rejects(F&& run) {
    bool rejected=false;
    try { run(); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"malformed checkpoint accepted");
}
struct Temp {
    fs::path path=fs::temp_directory_path()/("flash-checkpoint-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temp() { fs::create_directories(path); }
    ~Temp() { std::error_code ignored; fs::remove_all(path,ignored); }
};
static void matrix_file(const fs::path& path,const std::string& packed_dtype="I32",int scale_columns=1,int packed_bytes=2048,long long logical_k=128,bool mixed=false) {
    // 16x128 logical matrix; packed bytes include the full file extent, even in invalid logical fixtures.
    const int scale_bytes=16*scale_columns*2, shape_begin=packed_bytes+scale_bytes;
    std::string header="{\"a.weight_packed\":{\"dtype\":\""+packed_dtype+"\",\"shape\":[16,32],\"data_offsets\":[0,"+std::to_string(packed_bytes)+"]},"
        "\"a.weight_scale\":{\"dtype\":\"BF16\",\"shape\":[16,"+std::to_string(scale_columns)+"],\"data_offsets\":["+std::to_string(packed_bytes)+","+std::to_string(shape_begin)+"]},"
        "\"a.weight_shape\":{\"dtype\":\"I64\",\"shape\":[2],\"data_offsets\":["+std::to_string(shape_begin)+","+std::to_string(shape_begin+16)+"]}}";
    if(mixed) {
        header.pop_back();
        header+=",\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[16,128],\"data_offsets\":["
            +std::to_string(shape_begin+16)+","+std::to_string(shape_begin+16+4096)+"]}}";
    }
    while(header.size()%8) header+=' ';
    std::ofstream out(path,std::ios::binary); const uint64_t length=header.size();
    out.write(reinterpret_cast<const char*>(&length),8); out.write(header.data(),header.size());
    std::vector<char> payload(shape_begin,0); out.write(payload.data(),payload.size());
    const int64_t shape[2]={16,logical_k}; out.write(reinterpret_cast<const char*>(shape),16);
    if(mixed) { const std::array<char,4096> bf16{}; out.write(bf16.data(),bf16.size()); }
}
// compressed-tensors-style INT4 companion files: I32-packed width = K/8 (8 nibbles per word), a BF16
// scale and an I64 .weight_shape storing the logical shape. stored_k lets fixtures write a stored
// logical K that contradicts the packing (the ambiguous case must fail clearly).
static void int4_shaped_file(const fs::path& path,long long logical_k=128,long long stored_k=0) {
    if(!stored_k) stored_k=logical_k;
    const long long width=logical_k/8;
    const int packed_bytes=(int)(16*width*4), scale_bytes=32, shape_begin=packed_bytes+scale_bytes;
    std::string header="{\"a.weight_packed\":{\"dtype\":\"I32\",\"shape\":[16,"+std::to_string(width)+"],\"data_offsets\":[0,"+std::to_string(packed_bytes)+"]},"
        "\"a.weight_scale\":{\"dtype\":\"BF16\",\"shape\":[16,1],\"data_offsets\":["+std::to_string(packed_bytes)+","+std::to_string(shape_begin)+"]},"
        "\"a.weight_shape\":{\"dtype\":\"I64\",\"shape\":[2],\"data_offsets\":["+std::to_string(shape_begin)+","+std::to_string(shape_begin+16)+"]}}";
    while(header.size()%8) header+=' ';
    std::ofstream out(path,std::ios::binary); const uint64_t length=header.size();
    out.write(reinterpret_cast<const char*>(&length),8); out.write(header.data(),header.size());
    std::vector<char> payload(shape_begin,0); out.write(payload.data(),payload.size());
    const int64_t shape[2]={16,stored_k}; out.write(reinterpret_cast<const char*>(shape),16);
}
static Converted converted(const std::string& base,const std::vector<long long>& shape,bool mtp=false) {
    Converted c; c.base=base; c.source_key=base+(base.find(".experts.")==std::string::npos?".weight":"");
    c.file=mtp?"model_mtp.safetensors":"model.safetensors"; c.shape=shape; return c;
}
static std::vector<Converted> inventory() {
    std::vector<Converted> records;
    auto layer=[&](const std::string& prefix,bool full,bool mtp) {
        auto add=[&](const std::string& suffix,std::vector<long long> shape) { records.push_back(converted(prefix+suffix,shape,mtp)); };
        if(full) { add("self_attn.q_proj",{12288,2560}); add("self_attn.k_proj",{512,2560}); add("self_attn.v_proj",{512,2560}); add("self_attn.o_proj",{2560,6144}); }
        else { add("linear_attn.in_proj_qkv",{10240,2560}); add("linear_attn.in_proj_z",{6144,2560}); add("linear_attn.out_proj",{2560,6144}); }
        add("mlp.shared_expert.gate_proj",{640,2560}); add("mlp.shared_expert.up_proj",{640,2560}); add("mlp.shared_expert.down_proj",{2560,640});
        add("mlp.experts.gate_up_proj",{512,1280,2560}); add("mlp.experts.down_proj",{512,2560,640});
    };
    for(int i=0;i<48;++i) layer("model.language_model.layers."+std::to_string(i)+".",i%4==3,false);
    records.push_back(converted("model.language_model.layers.1.ple.key_proj",{10240,2560}));
    records.push_back(converted("model.language_model.layers.1.ple.value_proj",{2560,2560}));
    layer("mtp.layers.0.",true,true);
    records.push_back(converted("mtp.fc_embedding",{2560,2560},true)); records.push_back(converted("mtp.fc_hidden",{2560,2560},true));
    return records;
}
static void ngram_file(const fs::path& path,bool bad_dtype=false,bool missing=false,int first=0,int shards=128,long long rows=1) {
    std::string header="{";
    const int count=missing?shards-1:shards;
    for(int i=0;i<count;++i) {
        if(i) header+=",";
        header+="\"model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_"+std::to_string(first+i)+"\":{\"dtype\":\""
            +(bad_dtype && i==64?"BF16":"U8")+"\",\"shape\":["+std::to_string(rows)+",162],\"data_offsets\":["
            +std::to_string((long long)i*162*rows)+","+std::to_string((long long)(i+1)*162*rows)+"]}";
    }
    header+="}"; while(header.size()%8) header+=' ';
    std::ofstream out(path,std::ios::binary); const uint64_t size=header.size();
    out.write(reinterpret_cast<const char*>(&size),8); out.write(header.data(),header.size());
    if(rows==1) { std::vector<char> bytes(size_t(count)*162,0); out.write(bytes.data(),bytes.size()); }
    else { // sparse: only the declared last byte is written; holes cost nothing
        out.seekp(std::streamoff(8+size+(long long)count*162*rows-1)); out.put(0);
    }
}

// ---------------------------------------------------------------- full checkpoint fixture
// A complete v2 checkpoint at the fixed 48-layer architecture: sharded body (sparse files), MTP file,
// n-gram table (self-contained or external), config.json and manifest. Mutations are applied at build
// time so every tamper test gets an independent, otherwise-valid checkpoint.
struct MatrixDef { std::string base; std::vector<long long> shape; bool mtp; };
struct BuildOptions {
    std::string schema = kManifestSchema; // empty: omit the schema field entirely (legacy manifest)
    bool external_ngram = false;
    bool source_identity = false;
    bool legacy_body_string = false;   // outputs.body as a plain string, not a shard list
    bool body_absolute = false;        // first body entry recorded as an absolute path
    bool dotdot_shard = false;         // first body entry recorded with a '..' component
    bool ngram_absolute = false;       // self-contained outputs.ngram recorded as an absolute path
    bool drop_shard = false;           // shard file deleted after the build
    long long ngram_size_override = 0; // manifest size_bytes disagrees with the file
    bool config_dir = false;           // config.json is a directory
    std::string external_table;        // absolute override path passed to Checkpoint::open in external mode
    size_t config_pad = 0;             // pad config.json past the 4 MiB read bound
    std::string missing_config_key;    // drop one required text_config key
    long long fc_k = 2560;             // mtp.fc_embedding K (5120 = fused form, 7680 = invalid)
    bool misplaced_record = false;     // one converted record names the wrong shard
    bool extra_header_tensor = false;  // a shard header carries an unlisted tensor
    bool duplicate_record = false;     // one converted record is recorded twice
    bool bad_ngram_key = false;        // one ngram key renamed
    bool ngram_no_sha = false;         // ngram identity without sha256
    long long ngram_rows = 2500012;    // table rows (also the manifest rows_per_shard)
    bool drop_ngram_file = false;      // ngram table file deleted after the build
    bool hc_shrink = false;            // layer 0 attn hc_norm.weight shrunk to BF16 [1] (engine overread fixture)
    bool optional_shrink = false;      // a preserved name outside the engine table shrunk (must still open)
};
static std::vector<MatrixDef> fixture_matrices(long long fc_k) {
    std::vector<MatrixDef> out;
    auto add=[&](const std::string& base,std::vector<long long> shape,bool mtp){ out.push_back({base,shape,mtp}); };
    auto layer=[&](const std::string& pre,bool full,bool mtp){
        if(full){ add(pre+"self_attn.q_proj",{12288,2560},mtp); add(pre+"self_attn.k_proj",{512,2560},mtp);
                  add(pre+"self_attn.v_proj",{512,2560},mtp); add(pre+"self_attn.o_proj",{2560,6144},mtp); }
        else{ add(pre+"linear_attn.in_proj_qkv",{10240,2560},mtp); add(pre+"linear_attn.in_proj_z",{6144,2560},mtp);
              add(pre+"linear_attn.out_proj",{2560,6144},mtp); }
        add(pre+"mlp.shared_expert.gate_proj",{640,2560},mtp); add(pre+"mlp.shared_expert.up_proj",{640,2560},mtp);
        add(pre+"mlp.shared_expert.down_proj",{2560,640},mtp);
        add(pre+"mlp.experts.gate_up_proj",{512,1280,2560},mtp); add(pre+"mlp.experts.down_proj",{512,2560,640},mtp);
    };
    for(int i=0;i<48;++i) {
        const auto pre="model.language_model.layers."+std::to_string(i)+".";
        layer(pre,i%4==3,false);
        if(i==1){ add(pre+"ple.key_proj",{10240,2560},false); add(pre+"ple.value_proj",{2560,2560},false); }
    }
    layer("mtp.layers.0.",true,true);
    add("mtp.fc_embedding",{2560,fc_k},true); add("mtp.fc_hidden",{2560,2560},true);
    return out;
}
static std::string json_str(const std::string& s){ return "\""+s+"\""; }
static std::string shape_json(const std::vector<long long>& s) {
    std::string r="[";
    for(size_t i=0;i<s.size();++i) r+=(i?",":"")+std::to_string(s[i]);
    return r+"]";
}
static uint64_t prod_bytes(const std::vector<long long>& s){ uint64_t n=1; for(auto d:s) n*=uint64_t(d); return n; }
// header-first safetensors file; payload writes land inside the declared holes, everything else stays sparse
static void write_sparse(const fs::path& path,const std::string& header,
                         const std::vector<std::pair<uint64_t,std::string>>& payload,uint64_t total) {
    std::string h=header; while(h.size()%8) h+=' ';
    std::ofstream out(path,std::ios::binary);
    const uint64_t length=h.size();
    out.write(reinterpret_cast<const char*>(&length),8);
    out.write(h.data(),h.size());
    for(const auto& w:payload) { out.seekp(std::streamoff(8+length+w.first)); out.write(w.second.data(),w.second.size()); }
    if(total) { out.seekp(std::streamoff(8+length+total-1)); out.put(0); }
}
static std::string i64_payload(const std::vector<int64_t>& v) {
    return std::string(reinterpret_cast<const char*>(v.data()),v.size()*8);
}
struct TensorEntry { std::string name,dtype; std::vector<long long> shape; std::string payload; bool no_map=false; };
struct CheckpointFixture {
    fs::path dir;
    std::vector<std::string> shards;
    std::string external_table;  // absolute path, external mode only
};
static CheckpointFixture build_checkpoint(const fs::path& dir,const BuildOptions& o) {
    fs::create_directories(dir);
    CheckpointFixture fixture{dir,{},{}};
    const auto matrices=fixture_matrices(o.fc_k);
    // preserved tensors: a couple of small BF16 tensors plus the PLE n-gram metadata (n-gram-adjacent:
    // these must ride in the final shard, like the converter does)
    const std::string pre="model.language_model.layers.1.ple.ple_embedding.";
    const long long rows=o.ngram_rows>0?o.ngram_rows:2500012;
    std::vector<int64_t> vocab(16,20000096),offsets(16,0);
    for(int i=1;i<16;++i) offsets[i]=offsets[i-1]+vocab[i-1];
    std::vector<TensorEntry> mtp_tensors;
    auto push_matrix=[&](const MatrixDef& m,std::vector<TensorEntry>& list){
        const auto& s=m.shape;
        auto packed=s; packed.back()/=4; auto scale=s; scale.back()/=128;
        std::vector<long long> shape_rank{(long long)s.size()};
        list.push_back({m.base+".weight_packed","I32",packed,{}});
        list.push_back({m.base+".weight_scale","BF16",scale,{}});
        std::vector<int64_t> dims(s.begin(),s.end());
        list.push_back({m.base+".weight_shape","I64",shape_rank,i64_payload(dims)});
    };
    // engine-required preserved tensors (mirror of flash/checkpoint.h validate_preserved): hyper-
    // connection mixers, routers, indexer/norms per layer family, PLE norms and the shared head
    auto hc_preserved=[&](const std::string& pre,std::vector<TensorEntry>& list,bool inject,bool shrink){
        list.push_back({pre+"hc_norm.weight","BF16",shrink?std::vector<long long>{1}:std::vector<long long>{10240},{}});
        list.push_back({pre+"input_mix_weight_down.weight","BF16",{320,10240},{}});
        list.push_back({pre+"input_mix_weight_up.weight","BF16",{10240,320},{}});
        if(inject) list.push_back({pre+"block_inject_weight.weight","BF16",{4,10240},{}});
    };
    auto layer_preserved=[&](const std::string& pre,std::vector<TensorEntry>& list,bool full,bool shrink){
        hc_preserved(pre+"attn_hyper_connection.",list,true,shrink);
        hc_preserved(pre+"mlp_hyper_connection.",list,true,false);
        list.push_back({pre+"mlp.gate.weight","BF16",{512,2560},{}});
        list.push_back({pre+"mlp.shared_expert_gate.weight","BF16",{1,2560},{}});
        if(full){
            list.push_back({pre+"self_attn.q_norm.weight","BF16",{256},{}});
            list.push_back({pre+"self_attn.k_norm.weight","BF16",{256},{}});
            list.push_back({pre+"self_attn.indexer.index_qk_proj.weight","BF16",{640,2560},{}});
            list.push_back({pre+"self_attn.indexer.q_layernorm.weight","BF16",{128},{}});
            list.push_back({pre+"self_attn.indexer.k_layernorm.weight","BF16",{128},{}});
        } else {
            list.push_back({pre+"linear_attn.A_log","BF16",{48},{}});
            list.push_back({pre+"linear_attn.dt_bias","BF16",{48},{}});
            list.push_back({pre+"linear_attn.conv1d.weight","BF16",{10240,1,4},{}});
            list.push_back({pre+"linear_attn.norm.weight","BF16",{128},{}});
            list.push_back({pre+"linear_attn.in_proj_a.weight","BF16",{48,2560},{}});
            list.push_back({pre+"linear_attn.in_proj_b.weight","BF16",{48,2560},{}});
        }
    };
    // group body tensors per layer in order; embed tokens and PLE metadata form the trailing groups
    // (n-gram-adjacent tensors ride the final shard, like the converter)
    std::vector<std::vector<TensorEntry>> groups;
    groups.reserve(50);
    for(int layer=0;layer<48;++layer) {
        groups.emplace_back();
        for(const auto& m:matrices) {
            if(m.mtp) continue;
            const auto pos=m.base.find("layers.");
            if(pos==std::string::npos||std::stoi(m.base.substr(pos+7))!=layer) continue;
            push_matrix(m,groups.back());
        }
        layer_preserved("model.language_model.layers."+std::to_string(layer)+".",groups.back(),
                        layer%4==3,o.hc_shrink&&layer==0);
        if(layer==1) {
            const std::string pl="model.language_model.layers.1.ple.";
            groups.back().push_back({pl+"norm_key.weight","BF16",{10240},{}});
            groups.back().push_back({pl+"norm_query.weight","BF16",{10240},{}});
            groups.back().push_back({pl+"norm_conv.weight","BF16",{10240},{}});
            groups.back().push_back({pl+"conv1d.weight","BF16",{10240,1,4},{}});
        }
    }
    groups.emplace_back(std::vector<TensorEntry>{
        {"model.language_model.embed_tokens.weight","BF16",{248320,2560},{}},
        {"lm_head.weight","BF16",{248320,2560},{}}});
    groups.emplace_back(std::vector<TensorEntry>{
        {"model.language_model.hyper_connection_mixer.hc_norm.weight","BF16",{10240},{}},
        {"model.language_model.hyper_connection_mixer.input_mix_weight_down.weight","BF16",{320,10240},{}},
        {"model.language_model.hyper_connection_mixer.input_mix_weight_up.weight","BF16",{10240,320},{}}});
    groups.emplace_back(std::vector<TensorEntry>{
        {pre+"layer_multipliers","I64",{3},i64_payload({0,0,0})},
        {pre+"ngram_heads_vocab_sizes","I64",{16},i64_payload(vocab)},
        {pre+"ngram_heads_offsets","I64",{16},i64_payload(offsets)}});
    for(const auto& m:matrices) if(m.mtp) push_matrix(m,mtp_tensors);
    layer_preserved("mtp.layers.0.",mtp_tensors,true,false);
    mtp_tensors.push_back({"mtp.hyper_connection_mixer.hc_norm.weight","BF16",{10240},{}});
    mtp_tensors.push_back({"mtp.hyper_connection_mixer.input_mix_weight_down.weight","BF16",{320,10240},{}});
    mtp_tensors.push_back({"mtp.hyper_connection_mixer.input_mix_weight_up.weight","BF16",{10240,320},{}});
    mtp_tensors.push_back({"mtp.pre_fc_norm_embedding.weight","BF16",{2560},{}});
    mtp_tensors.push_back({"mtp.pre_fc_norm_hidden.weight","BF16",{10240},{}});
    mtp_tensors.push_back({"mtp.layers.0.input_layernorm.weight","BF16",
                           o.optional_shrink?std::vector<long long>{1}:std::vector<long long>{2560},{}});
    // pack groups into shards
    const uint64_t limit=(20ull<<30)-(1<<20);
    std::vector<std::vector<TensorEntry>> shards;
    shards.emplace_back();
    uint64_t bytes=0;
    for(auto& g:groups) {
        uint64_t gbytes=0;
        for(const auto& t:g) {
            if(t.dtype=="I32") gbytes+=prod_bytes(t.shape)*4;
            else if(t.dtype=="BF16") gbytes+=prod_bytes(t.shape)*2;
            else gbytes+=prod_bytes(t.shape)*8;
        }
        if(bytes && bytes+gbytes>limit) { shards.emplace_back(); bytes=0; }
        for(auto& t:g) shards.back().push_back(std::move(t));
        bytes+=gbytes;
    }
    if(o.extra_header_tensor) shards.front().push_back({"fixture.extra.weight","BF16",{1,1},{},true});
    const size_t count=shards.size();
    for(size_t i=0;i<count;++i) fixture.shards.push_back("model-"+[&]{char b[32];snprintf(b,32,"%05zu",i+1);return std::string(b);}()
                                                          +"-of-"+[&]{char b[32];snprintf(b,32,"%05zu",count);return std::string(b);}()+".safetensors");
    // write shard files + mapping
    std::map<std::string,std::string> tensor_shard;
    for(size_t i=0;i<count;++i) {
        auto& entries=shards[i];
        std::sort(entries.begin(),entries.end(),[](const auto&a,const auto&b){return a.name<b.name;});
        std::string header="{"; uint64_t off=0;
        std::vector<std::pair<uint64_t,std::string>> payload;
        for(const auto& t:entries) {
            uint64_t nbytes=0;
            if(t.dtype=="I32") nbytes=prod_bytes(t.shape)*4;
            else if(t.dtype=="BF16") nbytes=prod_bytes(t.shape)*2;
            else nbytes=prod_bytes(t.shape)*8;
            if(header.size()>1) header+=",";
            header+=json_str(t.name)+":{\"dtype\":\""+t.dtype+"\",\"shape\":"+shape_json(t.shape)
                +",\"data_offsets\":["+std::to_string(off)+","+std::to_string(off+nbytes)+"]}";
            if(!t.payload.empty()) payload.push_back({off,t.payload});
            off+=nbytes;
            if(!t.no_map) tensor_shard[t.name]=fixture.shards[i];
        }
        header+="}";
        write_sparse(dir/fixture.shards[i],header,payload,off);
    }
    { // MTP file
        auto& entries=mtp_tensors;
        std::sort(entries.begin(),entries.end(),[](const auto&a,const auto&b){return a.name<b.name;});
        std::string header="{"; uint64_t off=0;
        std::vector<std::pair<uint64_t,std::string>> payload;
        for(const auto& t:entries) {
            uint64_t nbytes=0;
            if(t.dtype=="I32") nbytes=prod_bytes(t.shape)*4;
            else if(t.dtype=="BF16") nbytes=prod_bytes(t.shape)*2;
            else nbytes=prod_bytes(t.shape)*8;
            if(header.size()>1) header+=",";
            header+=json_str(t.name)+":{\"dtype\":\""+t.dtype+"\",\"shape\":"+shape_json(t.shape)
                +",\"data_offsets\":["+std::to_string(off)+","+std::to_string(off+nbytes)+"]}";
            if(!t.payload.empty()) payload.push_back({off,t.payload});
            off+=nbytes;
            tensor_shard[t.name]="model_mtp.safetensors";
        }
        header+="}";
        write_sparse(dir/"model_mtp.safetensors",header,payload,off);
    }
    { // n-gram table: 128 U8 [rows,162] shards, contiguous, sparse payload
        const auto table=o.external_ngram ? dir/"external-ngram.safetensors" : dir/"ngram_int8.safetensors";
        ngram_file(table,false,false,0,128,rows);
        if(o.external_ngram) fixture.external_table=table.string();
    }
    { // config.json
        std::string text="{";
        auto key=[&](const char* name,const char* value){
            if(o.missing_config_key==name) return;
            if(text.size()>1) text+=",";
            text+=std::string("\"")+name+"\":"+value;
        };
        key("model_type","\"qwen4_exp\""); key("hidden_size","2560"); key("num_hidden_layers","48");
        key("num_attention_heads","24"); key("head_dim","256"); key("num_key_value_heads","2");
        key("linear_num_key_heads","16"); key("linear_key_head_dim","128");
        key("linear_num_value_heads","48"); key("linear_value_head_dim","128");
        key("num_experts","512"); key("moe_intermediate_size","640"); key("hc_count","4");
        std::string types="[";
        for(int i=0;i<48;++i) types+=std::string(i?",":"")+"\""+std::string(i%4==3?"full_attention":"linear_attention")+"\"";
        types+="]";
        text+=",\"layer_types\":"+types+"}";
        std::string config="{\"model_type\":\"qwen4_exp\",\"text_config\":"+text+"}";
        if(o.config_dir) { fs::create_directories(dir/"config.json"); }
        else { std::ofstream out(dir/"config.json"); out<<config; if(o.config_pad) out<<std::string(o.config_pad,' '); }
    }
    { // manifest v2
        std::string recs;
        for(const auto& m:matrices) {
            if(!recs.empty()) recs+=",";
            const std::string source_key=m.base.find(".experts.")==std::string::npos?m.base+".weight":m.base;
            const std::string shard=m.mtp?"model_mtp.safetensors":tensor_shard[m.base+".weight_packed"];
            std::string misplaced=o.misplaced_record && m.base=="model.language_model.layers.0.linear_attn.in_proj_qkv"
                && count>1 ? fixture.shards[1] : shard;
            recs+="{\"source_key\":"+json_str(source_key)+",\"base\":"+json_str(m.base)
                +",\"file\":"+json_str(misplaced)+",\"shape\":"+shape_json(m.shape)+",\"dtype\":\"BF16\""
                +",\"outputs\":{\"packed\":"+json_str(m.base+".weight_packed")
                +",\"scale\":"+json_str(m.base+".weight_scale")
                +",\"shape\":"+json_str(m.base+".weight_shape")+"}}";
        }
        if(o.duplicate_record) {
            const auto& m=matrices.front();
            const std::string source_key=m.base+".weight";
            recs+=",{\"source_key\":"+json_str(source_key)+",\"base\":"+json_str(m.base)
                +",\"file\":"+json_str(tensor_shard[m.base+".weight_packed"])+",\"shape\":"+shape_json(m.shape)+",\"dtype\":\"BF16\""
                +",\"outputs\":{\"packed\":"+json_str(m.base+".weight_packed")
                +",\"scale\":"+json_str(m.base+".weight_scale")
                +",\"shape\":"+json_str(m.base+".weight_shape")+"}}";
        }
        std::string preserved_body,preserved_mtp;
        for(const auto& [name,shard]:tensor_shard) {
            bool is_padded=true;
            for(const auto& m:matrices) {
                const std::string packed=m.base+".weight_packed";
                if(name==packed||name==m.base+".weight_scale"||name==m.base+".weight_shape") { is_padded=false; break; }
            }
            if(!is_padded) continue;
            if(shard=="model_mtp.safetensors") preserved_mtp+=(preserved_mtp.empty()?"":",")+json_str(name);
            else preserved_body+=(preserved_body.empty()?"":",")+json_str(name)+":"+json_str(shard);
        }
        std::string keys;
        for(int i=0;i<128;++i)
            keys+=(i?",":"")+json_str(i==5&&o.bad_ngram_key?"bad.key":
                "model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_"+std::to_string(i));
        const auto table=o.external_ngram ? dir/"external-ngram.safetensors" : dir/"ngram_int8.safetensors";
        const long long real_size=(long long)fs::file_size(table);
        std::string body_json;
        if(o.legacy_body_string) body_json=json_str("model.safetensors");
        else {
            body_json="[";
            for(size_t i=0;i<count;++i) {
                body_json+=(i?",":"")+json_str(o.body_absolute&&i==0 ? (dir/"evil.safetensors").string()
                                 : o.dotdot_shard&&i==0 ? "../evil.safetensors" : fixture.shards[i]);
            }
            body_json+="]";
        }
        std::string manifest="{";
        if(!o.schema.empty()) manifest+="\"schema\":"+json_str(o.schema)+",";
        manifest+="\"format_version\":1,\"family\":\"qwen4_exp\","
            "\"source\":{\"index_sha256_kind\":\"index_json\",\"index_sha256\":\""+std::string(64,'0')+"\""
            +(o.source_identity?",\"repo\":\"acme/QFN-fork\",\"revision\":\"0123456789abcdef\"":"")+"},"
            "\"recipe\":{\"format\":\"symmetric INT8\",\"group_size\":128,\"rounding\":\"nearest-even\","
            "\"scale_dtype\":\"BF16\",\"zero_point\":0,\"search\":0,\"algorithm\":\"RTN\"},"
            "\"outputs\":{\"body\":"+body_json+",\"mtp\":\"model_mtp.safetensors\",\"config\":\"config.json\","
            "\"manifest\":\"qfn-int8-manifest.json\""
            +(o.external_ngram?"" : ",\"ngram\":"+json_str(o.ngram_absolute ? (dir/"ngram_int8.safetensors").string()
                                                                            : "ngram_int8.safetensors"))+"},"
            "\"shard_limit_bytes\":"+std::to_string(20ll<<30)+","
            "\"tensors\":{\"converted\":["+recs+"],\"preserved\":{\"body\":{"+preserved_body+"},\"mtp\":["+preserved_mtp+
            "]},\"excluded\":[]},"
            "\"ngram\":{\"size_bytes\":"+std::to_string(o.ngram_size_override?o.ngram_size_override:real_size)
            +",\"rows_per_shard\":"+std::to_string(rows)+",\"width_bytes\":162,\"shards\":128,"
            "\"layout\":\"U8 [rows,162]: 160 signed INT8 values then one FP16 scale\",\"keys\":["+keys+"]"
            +(o.ngram_no_sha?"" : ",\"sha256\":\""+std::string(64,'a')+"\"")+",\"equivalence_verified\":false},"
            "\"runtime_supported\":true,\"runtime_support_reason\":\"fixture\"}";
        std::ofstream out(dir/"qfn-int8-manifest.json"); out<<manifest;
    }
    if(o.drop_shard) fs::remove(dir/fixture.shards[0]);
    if(o.drop_ngram_file) fs::remove(o.external_ngram ? dir/"external-ngram.safetensors" : dir/"ngram_int8.safetensors");
    return fixture;
}
static void open_checkpoint(const fs::path& dir,const BuildOptions& o) {
    Checkpoint ck;
    if(o.external_ngram) ck.open(dir.string(),true,"",o.external_table);
    else ck.open(dir.string(),true,"");
    require(ck.format==WeightFormat::Int8 && ck.has_manifest,"format/manifest flags wrong");
    require(ck.converted.size()==409,"converted inventory changed");
}

static void alignment_gap_test() {
    Temp temp;
        { // Legacy INT4 alignment padding must not enter the manifest-only strict route.
            const auto d=temp.path/"int4-gap"; fs::create_directories(d);
            const auto p=d/"model.safetensors";
            std::string h=R"({"m.weight_packed":{"dtype":"I32","shape":[1,1],"data_offsets":[0,4]},"m.weight_scale":{"dtype":"F16","shape":[1],"data_offsets":[8,10]}})";
            while(h.size()%8) h+=' ';
            const std::array<char,10> bytes{{1,2,3,4,0,0,0,0,5,6}};
            { std::ofstream out(p,std::ios::binary); const uint64_t n=h.size();
              out.write(reinterpret_cast<const char*>(&n),8); out.write(h.data(),h.size()); out.write(bytes.data(),bytes.size()); }
            auto check=[&](const SafeTensors& st) {
                require(st.get("m.weight_packed").nbytes==4 && std::memcmp(st.get("m.weight_packed").data,bytes.data(),4)==0,"gapped INT4 packed bytes changed");
                require(st.get("m.weight_scale").nbytes==2 && std::memcmp(st.get("m.weight_scale").data,bytes.data()+8,2)==0,"gapped INT4 scale bytes changed");
            };
            { SafeTensors legacy; legacy.open(p.string()); check(legacy); }
            rejects([&]{ SafeTensors strict; strict.open_strict_validated(p.string(),false); });
            bool opened=false;
            try { Checkpoint ck; ck.open(d.string(),false,""); check(ck.body); opened=!ck.has_manifest && ck.format==WeightFormat::Int4; }
            catch(const std::runtime_error& e) { fprintf(stderr,"alignment-gap INT4: %s\n",e.what()); }
            require(opened,"manifest-less INT4 alignment gap rejected by runtime loader");
        }

}

int main(int argc, char** argv) {
    if(argc==2 && std::string(argv[1])=="--alignment-gap-only") {
        try { alignment_gap_test(); }
        catch(const std::exception& e) { fprintf(stderr,"alignment-gap: %s\n",e.what()); return 1; }
        return fails ? 1 : 0;
    }
    try {
        Temp temp;
        for(const char* invalid : {"", "{", "[", "{\"x\":", "{\"x\":1,}", "[1,]", "{\"x\":1,\"x\":2}", "01", "-", "1.", "1e",
                                   "NaN", "true trailing", "\"unterminated", "\"bad\\q\"", "\"\\uZZZZ\"", "\"a\tb\"", "+1", ".5"})
            rejects([&] { Json::parse_strict(invalid); });
        const auto valid_json=Json::parse_strict(" {\"array\":[true,false,null,-1.25e2],\"s\":\"a\\\\b\\n\"} ");
        require(valid_json["array"][3].f()==-125.0 && valid_json["s"].str=="a\\b\n","strict valid JSON changed");
        require(Json::parse_strict("0").i()==0 && Json::parse_strict("-0.5e1").f()==-5.0,"strict number grammar changed");
        // the lenient reader keeps its historical behaviour: duplicate keys resolve to the first entry,
        // trailing data is ignored, strtod numbers are not bounded — strict is a separate entry point
        const Json lenient=Json::parse("{\"a\":1,\"a\":2} trailing");
        require(lenient["a"].i()==1,"lenient parse changed");
        const Json recipe=Json::parse(R"({"format":"symmetric INT8","group_size":128,"scale_dtype":"BF16","zero_point":0})");
        validate_recipe(recipe,"fixture");
        for(const char* bad : {R"({"format":"symmetric INT4","group_size":128,"scale_dtype":"BF16","zero_point":0})",R"({"format":"symmetric INT8","group_size":64,"scale_dtype":"BF16","zero_point":0})",R"({"format":"symmetric INT8","group_size":128,"scale_dtype":"F16","zero_point":0})",R"({"format":"symmetric INT8","group_size":128,"scale_dtype":"BF16","zero_point":128})"})
            rejects([&] { validate_recipe(Json::parse(bad),"fixture"); });
        Json manifest=Json::parse(R"({"schema":"qfn-int8-v2","format_version":1,"family":"qwen4_exp","outputs":{"body":["model-00001-of-00001.safetensors"],"mtp":"model_mtp.safetensors","config":"config.json"},"runtime_supported":true,"equivalence_verified":false})");
        manifest.obj["recipe"]=recipe;
        validate_manifest_top(manifest,"historical metadata");
        { auto legacy=manifest; legacy.obj.erase("schema"); rejects([&]{ validate_manifest_top(legacy,"legacy"); }); }
        { auto other=manifest; other.obj["schema"].str="qfn-int8-v3"; rejects([&]{ validate_manifest_top(other,"other schema"); }); }
        { Json body_str=manifest; body_str.obj["outputs"].obj["body"]=Json::parse("\"model.safetensors\"");
          rejects([&]{ validate_manifest_top(body_str,"legacy body string"); }); }
        { Json no_mtp=manifest; no_mtp.obj["outputs"].obj.erase("mtp"); rejects([&]{ validate_manifest_top(no_mtp,"missing mtp"); }); }
        const std::string root=temp.path.string();
        require(resolve_path(root,"model.safetensors")== (temp.path/"model.safetensors").string(),"body relative path");
        const auto absolute=(temp.path/"external-table.safetensors").string();
        { std::ofstream dependency(absolute); dependency<<"fixture"; }
        require(resolve_path(root,absolute)==absolute,"operator override path relocated");
        require(sandbox_file(root,"model.safetensors","fixture")==(temp.path/"model.safetensors").string(),"sandbox relative path");
        require(sandbox_file(root,"sub/table.safetensors","fixture")==(temp.path/"sub/table.safetensors").string(),"sandbox nested relative path");
        rejects([&]{ sandbox_file(root,absolute,"fixture"); });
        rejects([&]{ sandbox_file(root,"../outside.safetensors","fixture"); });
        rejects([&]{ sandbox_file(root,"sub/../outside.safetensors","fixture"); });
        rejects([&]{ sandbox_file(root,"C:/evil.safetensors","fixture"); });
        rejects([&]{ sandbox_file(root,"","fixture"); });
        { // resolve_outputs over the new outputs shape
            Json resolve_manifest=manifest;
            const auto set_str=[](Json& j,const std::string& v){ Json s; s.type=Json::Str; s.str=v; return s; };
            resolve_manifest.obj["outputs"].obj["ngram"]=set_str(resolve_manifest.obj["outputs"],"ngram_int8.safetensors");
            (void)set_str;
            { std::ofstream table(temp.path/"ngram_int8.safetensors"); table<<"x"; }
            const auto selected=resolve_outputs(resolve_manifest,root,true,"");
            require(selected.body.size()==1 && selected.body[0]==(temp.path/"model-00001-of-00001.safetensors").string(),"body shard not sandbox-joined");
            require(selected.mtp==(temp.path/"model_mtp.safetensors").string(),"MTP output changed");
            require(selected.ngram==(temp.path/"ngram_int8.safetensors").string(),"self-contained ngram path changed");
            // UNIFIED rule: a recorded self-contained table wins; override/env are
            // reported as IGNORED by Checkpoint::open and never reach the resolver result.
            require(resolve_outputs(resolve_manifest,root,true,"",absolute).ngram==(temp.path/"ngram_int8.safetensors").string(),
                    "operator ngram override must not beat a recorded self-contained table");
            setenv("FLASH_NGRAM_PATH",absolute.c_str(),1);
            require(resolve_outputs(resolve_manifest,root,true,"").ngram==(temp.path/"ngram_int8.safetensors").string(),
                    "FLASH_NGRAM_PATH must not beat a recorded self-contained table");
            unsetenv("FLASH_NGRAM_PATH");
            require(resolve_outputs(resolve_manifest,root,false,"").mtp.empty(),"unrequested MTP opened");
            const auto override_path=(temp.path/"explicit-mtp.safetensors").string();
            require(resolve_outputs(resolve_manifest,root,true,override_path).mtp==override_path,"explicit MTP override ignored");
            { Json dup=resolve_manifest; dup.obj["outputs"].obj["body"]=Json::parse("[\"model_mtp.safetensors\"]");
              rejects([&]{ resolve_outputs(dup,root,true,""); }); }
            { Json abs_ngram=resolve_manifest; Json s; s.type=Json::Str; s.str=absolute;
              abs_ngram.obj["outputs"].obj["ngram"]=s;
              rejects([&]{ resolve_outputs(abs_ngram,root,true,""); }); }
            { Json dot=resolve_manifest; dot.obj["outputs"].obj["body"]=Json::parse("[\"../evil.safetensors\"]");
              rejects([&]{ resolve_outputs(dot,root,true,""); }); }
            { Json missing_table=resolve_manifest; missing_table.obj["outputs"].obj.erase("ngram");
              unsetenv("FLASH_NGRAM_PATH");
              rejects([&]{ resolve_outputs(missing_table,root,true,""); });  // external mode without any table source
              setenv("FLASH_NGRAM_PATH",absolute.c_str(),1);
              require(resolve_outputs(missing_table,root,true,"").ngram==absolute,"FLASH_NGRAM_PATH external table ignored");
              require(resolve_outputs(missing_table,root,true,"",absolute).ngram==absolute,"ngram override ignored");
              rejects([&]{ resolve_outputs(missing_table,root,true,"","/nonexistent/table.safetensors"); });  // override must exist too
              unsetenv("FLASH_NGRAM_PATH"); }
        }
        for(int test=0;test<6;++test) {
            const auto path=temp.path/(std::to_string(test)+".safetensors");
            matrix_file(path,test==1?"BF16":"I32",test==2?2:1,test==3?2044:2048,test==4?256:128,test==5);
            SafeTensors tensors; tensors.open(path.string());
            if(test==0) validate_int8_tensor(tensors,"a",{16,128},path.string());
            else rejects([&] { validate_int8_tensor(tensors,"a",{16,128},path.string()); });
        }
        for(int test=0;test<3;++test) {
            const auto path=temp.path/("ngram-"+std::to_string(test)+".safetensors");
            ngram_file(path,test==1,test==2);
            SafeTensors tensors; tensors.open(path.string());
            const NgramExternal table{path.string(),static_cast<long long>(fs::file_size(path)),1,162,128};
            if(test==0) {
                validate_ngram_tensors(tensors,table,path.string());
                const auto span=ngram_span(tensors,true);
                require(span.bytes==128*162 && span.file==path.string(),"ngram pin span escaped owning file");
            }
            else rejects([&] { validate_ngram_tensors(tensors,table,path.string()); });
        }
        {
            const auto a=temp.path/"ngram-a.safetensors",b=temp.path/"ngram-b.safetensors";
            ngram_file(a,false,false,0,64); ngram_file(b,false,false,64,64);
            SafeTensors split; split.open(a.string()); split.open(b.string());
            rejects([&] { ngram_span(split,true); });
        }
        rejects([&] { SafeTensors missing; missing.open((temp.path/"missing.safetensors").string()); });
        {
            const auto short_file=temp.path/"short.safetensors";
            { std::ofstream out(short_file,std::ios::binary); out<<"tiny"; }
            auto strict_open=[&](const std::string& p){ SafeTensors st; st.open_strict_validated(p,false); };
            rejects([&] { strict_open(short_file.string()); });
            const auto long_header=temp.path/"long-header.safetensors";
            { std::ofstream out(long_header,std::ios::binary); uint64_t length=4096; out.write(reinterpret_cast<const char*>(&length),8); out<<"{}"; }
            rejects([&] { strict_open(long_header.string()); });
            const auto truncated=temp.path/"truncated.safetensors";
            matrix_file(truncated); fs::resize_file(truncated,fs::file_size(truncated)-1);
            rejects([&] { strict_open(truncated.string()); });
            const auto dup_key=temp.path/"dup-key.safetensors";
            { // strict header parsing: a duplicate key cannot smuggle a second tensor past the inventory
              std::string header="{\"a.weight_packed\":{\"dtype\":\"I32\",\"shape\":[16,32],\"data_offsets\":[0,2048]},"
                  "\"a.weight_packed\":{\"dtype\":\"I32\",\"shape\":[16,32],\"data_offsets\":[0,2048]}}";
              while(header.size()%8) header+=' ';
              std::ofstream out(dup_key,std::ios::binary); const uint64_t length=header.size();
              out.write(reinterpret_cast<const char*>(&length),8); out.write(header.data(),header.size());
              std::vector<char> payload(2048,0); out.write(payload.data(),payload.size());
            }
            rejects([&] { strict_open(dup_key.string()); });
        }
        auto records=inventory();
        const InventorySpec spec{48,512,2560,640,24,256,2,16,128,48,128,640,4,1};
        validate_inventory(records,spec,"full inventory");
        auto bad=records; bad.erase(bad.begin()); rejects([&] { validate_inventory(bad,spec,"missing selected projection"); });
        bad=records; bad.push_back(records.front()); rejects([&] { validate_inventory(bad,spec,"duplicate matrix"); });
        bad=records; bad.back().shape={2560,7680}; rejects([&] { validate_inventory(bad,spec,"unsupported MTP FC"); });
        { // shared kMtpFcEmbeddingKMultipliers rule: the fused [hidden,2*hidden] form is valid
          auto fused=records;
          for(auto& r:fused) if(r.base=="mtp.fc_embedding") r.shape={2560,5120};
          validate_inventory(fused,spec,"fused MTP FC embedding");
        }
        bad=records; bad.push_back(converted("model.language_model.layers.0.self_attn.q_proj",{12288,2560}));
        rejects([&] { validate_inventory(bad,spec,"mixed mixer families"); });

        { // no-manifest directory: INT4-style safetensors (packed+scale, no .weight_shape) opens on the
          // legacy route; the stray ngram_int8.safetensors DIRECTORY seen in real INT4 checkpoints must
          // not break the scan; a packed INT8 body without a manifest is refused
            const auto int4=[&](const fs::path& p){
                std::string header="{\"m.weight_packed\":{\"dtype\":\"I32\",\"shape\":[16,16],\"data_offsets\":[0,1024]},"
                    "\"m.weight_scale\":{\"dtype\":\"F16\",\"shape\":[16,8],\"data_offsets\":[1024,1280]}}";
                while(header.size()%8) header+=' ';
                std::ofstream out(p,std::ios::binary); const uint64_t len=header.size();
                out.write(reinterpret_cast<const char*>(&len),8); out.write(header.data(),header.size());
                std::vector<char> payload(1280,0); out.write(payload.data(),payload.size());
            };
            const auto legacy=temp.path/"int4-dir"; fs::create_directories(legacy);
            int4(legacy/"model-00001-of-00001.safetensors");
            fs::create_directories(legacy/"ngram_int8.safetensors");
            { Checkpoint ck; ck.open(legacy.string(),false,"");
              require(!ck.has_manifest && ck.format==WeightFormat::Int4,"INT4 directory route changed"); }
            { // manifest existence is probed with symlink_status, so a DANGLING symlink named
              // qfn-int8-manifest.json is not "no manifest": the packed INT8 directory (the bug: it fell
              // through to the scan route and died on the width discriminator "requires
              // qfn-int8-manifest.json") and the otherwise-valid INT4 directory (it scanned as if the
              // reserved name were absent) are both rejected by the symlink gate instead, and only a
              // true absence keeps the scan route.
                const auto dangling_manifest=[&](const fs::path& d,bool packed8){
                    fs::create_directories(d);
                    if (packed8) matrix_file(d/"model.safetensors"); else int4(d/"model-00001-of-00001.safetensors");
                    fs::create_symlink("no-such-manifest.json",d/"qfn-int8-manifest.json");
                    bool via_symlink_gate=false;
                    try { Checkpoint ck; ck.open(d.string(),false,""); }
                    catch(const std::runtime_error& e){ const std::string m=e.what();
                        via_symlink_gate = m.find("symbolic link")!=std::string::npos &&
                                           m.find("requires qfn-int8-manifest")==std::string::npos; }
                    require(via_symlink_gate,(d.string()+": a dangling qfn-int8-manifest.json link did not reach the symlink gate").c_str());
                };
                dangling_manifest(temp.path/"manifest-link-int8",true);   // the width-error case of the bug
                dangling_manifest(temp.path/"manifest-link-int4",false);  // the silently-scanning INT4 case
                const auto absent=temp.path/"manifest-absent"; fs::create_directories(absent);
                int4(absent/"model-00001-of-00001.safetensors");
                { Checkpoint ck; ck.open(absent.string(),false,"");  // a true absence still scans, as today
                  require(!ck.has_manifest && ck.format==WeightFormat::Int4,"a directory without a manifest stopped scanning"); }
            }
            { // the manifest-less route keeps main's behavior for the MTP pair and for dangling links:
              // BOTH names (mtp.safetensors, mtp-int4.safetensors) are opened and cross-checked, a missing one
              // fails the start, a dangling *.safetensors link is an open failure and not a skip - except the
              // mtp-prefixed names, which main filters before it ever opens a name - and an explicit MTP
              // override replaces the pair. This is the layer the FLASH_MTP_K=0 fix lives on: qe_create asks
              // for the MTP block unconditionally (kmax only switches the draft off), pinned here as want_mtp=true.
                // mtp_file takes a payload fill byte: mtp.safetensors, mtp-int4.safetensors and the explicit
                // override no longer share 32 zero bytes, so has("blk.weight") is not the only observable -
                // payload_is tells WHICH file's bytes the loader actually mapped. The pair must stay
                // byte-identical for the duplicate rule; the conflict case pins that a differing byte in
                // one of the pair rejects the start, and that with the conflict in place the override opens
                // its own bytes.
                const auto mtp_file=[&](const fs::path& p,unsigned char fill){
                    std::string header="{\"blk.weight\":{\"dtype\":\"F16\",\"shape\":[4,4],\"data_offsets\":[0,32]}}";
                    while(header.size()%8) header+=' ';
                    std::ofstream out(p,std::ios::binary); const uint64_t len=header.size();
                    out.write(reinterpret_cast<const char*>(&len),8); out.write(header.data(),header.size());
                    std::vector<char> payload(32,(char)fill); out.write(payload.data(),payload.size());
                };
                const auto payload_is=[&](const Checkpoint& ck,unsigned char fill){
                    const auto& t=ck.mtp.get("blk.weight");
                    char expected[32]; memset(expected,(char)fill,sizeof expected);
                    return t.nbytes==32 && memcmp(t.data,expected,sizeof expected)==0;
                };
                const auto mroot=temp.path/"mtp-route"; fs::create_directories(mroot);
                int4(mroot/"model-00001-of-00001.safetensors");
                mtp_file(mroot/"mtp.safetensors",0xA1); mtp_file(mroot/"mtp-int4.safetensors",0xA1);
                { Checkpoint ck; ck.open(mroot.string(),true,"");
                  require(ck.mtp.has("blk.weight"),"manifest-less MTP pair did not open");
                  require(payload_is(ck,0xA1),"the opened MTP pair is not the fixture's own bytes"); }
                { Checkpoint ck; ck.open(mroot.string(),false,"");   // want_mtp=false opens no MTP at all
                  require(!ck.mtp.has("blk.weight"),"MTP opened without want_mtp"); }
                fs::remove(mroot/"mtp-int4.safetensors");
                rejects([&]{ Checkpoint ck; ck.open(mroot.string(),true,""); });  // main needs both names
                mtp_file(mroot/"mtp-int4.safetensors",0xA1);  // identical pair restored: the route opens again
                { Checkpoint ck; ck.open(mroot.string(),true,"");
                  require(ck.mtp.has("blk.weight"),"restored MTP pair did not open");
                  require(payload_is(ck,0xA1),"restored pair opened different bytes"); }
                mtp_file(mroot/"mtp-int4.safetensors",0xB2);  // one of the pair carries different content
                rejects([&]{ Checkpoint ck; ck.open(mroot.string(),true,""); });  // the cross-check is content, not names
                mtp_file(temp.path/"explicit-mtp2.safetensors",0xC3);
                { Checkpoint ck; ck.open(mroot.string(),true,(temp.path/"explicit-mtp2.safetensors").string());
                  require(ck.mtp.has("blk.weight"),"MTP override did not open");
                  // with the pair conflict in place, only the override's own bytes can satisfy this:
                  // neither the pair's 0xA1 nor the conflicting 0xB2 may appear
                  require(payload_is(ck,0xC3),"the MTP override opened somebody else's bytes"); }
                const auto dangle=temp.path/"dangling-dir"; fs::create_directories(dangle);
                int4(dangle/"model-00001-of-00001.safetensors");
                fs::create_symlink("no-such-target.safetensors",dangle/"link-00002-of-00009.safetensors");
                rejects([&]{ Checkpoint ck; ck.open(dangle.string(),false,""); });  // main fails on the dangling link
                const auto dangle_mtp=temp.path/"dangling-mtp-dir"; fs::create_directories(dangle_mtp);
                int4(dangle_mtp/"model-00001-of-00001.safetensors");
                fs::create_symlink("no-such-mtp.safetensors",dangle_mtp/"mtp-int4.safetensors");
                { Checkpoint ck; ck.open(dangle_mtp.string(),false,"");  // main filtered mtp names before opening:
                  require(!ck.mtp.has("blk.weight"),"a dangling mtp link broke a start that did not want MTP"); }
            }
            const auto legacy8=temp.path/"int8-nomanifest"; fs::create_directories(legacy8);
            matrix_file(legacy8/"model.safetensors");  // packed + scale + weight_shape without a manifest
            rejects([&]{ Checkpoint ck; ck.open(legacy8.string(),false,""); });  // width*4 == stored K: INT8
            { // Compressed-tensors INT4 stores .weight_shape too — width*8 == stored K must open as INT4
            const auto legacy4=temp.path/"int4-shaped"; fs::create_directories(legacy4);
            int4_shaped_file(legacy4/"model-00001-of-00001.safetensors");
            { Checkpoint ck; ck.open(legacy4.string(),false,"");
              require(!ck.has_manifest && ck.format==WeightFormat::Int4,"INT4 with .weight_shape companion rejected"); }
            const auto odd=temp.path/"int4-odd-shape"; fs::create_directories(odd);
            int4_shaped_file(odd/"model-00001-of-00001.safetensors",128,96);  // width 16: 64 != 96 and 128 != 96
            rejects([&]{ Checkpoint ck; ck.open(odd.string(),false,""); });  // neither x4 nor x8: clear error
            { // A scalar I32 .weight_packed (shape []) with companions used to reach arr.back() on
              // the empty shape vector (crash); it must now be refused cleanly before any dimension read
            const auto scalar8=temp.path/"int8-scalar-packed"; fs::create_directories(scalar8);
            std::string scalar_header="{\"a.weight_packed\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]},"
                "\"a.weight_scale\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[4,6]},"
                "\"a.weight_shape\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[6,14]}}";
            while(scalar_header.size()%8) scalar_header+=' ';
            { std::ofstream out(scalar8/"model.safetensors",std::ios::binary); const uint64_t len=scalar_header.size();
              out.write(reinterpret_cast<const char*>(&len),8); out.write(scalar_header.data(),scalar_header.size());
              std::vector<char> payload(6,0); out.write(payload.data(),payload.size());
              const int64_t stored=128; out.write(reinterpret_cast<const char*>(&stored),8); }
            rejects([&]{ Checkpoint ck; ck.open(scalar8.string(),false,""); });
            }
            }
        }

        { // ---- Checkpoint::open fixtures (complete v2 checkpoints; sparse files) ----
        unsetenv("FLASH_NGRAM"); unsetenv("FLASH_NGRAM_PATH");
        const auto base=temp.path/"happy";
        build_checkpoint(base,BuildOptions{});
        open_checkpoint(base,BuildOptions{});
        { // explicit re-open of the untouched fixture (second reader must stay deterministic)
            Checkpoint ck; ck.open(base.string(),true,"");
            require(ck.ngram.has("model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_127"),"ngram table not opened");
            require(ck.body.has("model.language_model.layers.47.mlp.experts.gate_up_proj.weight_packed"),"body shard tensor missing");
            require(ck.mtp.has("mtp.fc_embedding.weight_packed"),"MTP tensor missing");
        }
        { // One guarded open parses each safetensors header exactly once — the previous loader
          // strict-validated the bytes, then SafeTensors::open REOPENED the path and reparsed it
          // permissively: double work plus a swap/symlink window between the two opens
            const auto d=temp.path/"single-parse"; const auto fix=build_checkpoint(d,BuildOptions{});
            const auto before=safetensors_header_parses();
            Checkpoint ck; ck.open(d.string(),true,"");
            require(safetensors_header_parses()-before==fix.shards.size()+2,
                    "checkpoint open did not parse each safetensors header exactly once");
        }
        { // An engine-required preserved tensor with a bad shape is refused at open — before the fix
          // this opened and Worker::load memcpy'd its fixed RW-sized region from a 2-byte tensor
            auto o=BuildOptions{}; o.hc_shrink=true; const auto d=temp.path/"hc-shrunk";
            build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); });
            auto q=BuildOptions{}; q.optional_shrink=true; const auto e=temp.path/"opt-shrunk";
            build_checkpoint(e,q); open_checkpoint(e,q);  // names outside the engine table stay free
        }
        { auto o=BuildOptions{}; o.source_identity=true; const auto d=temp.path/"source-id";
          build_checkpoint(d,o); open_checkpoint(d,o); }
        { auto o=BuildOptions{}; o.legacy_body_string=true; const auto d=temp.path/"legacy";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.schema=""; const auto d=temp.path/"no-schema";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.body_absolute=true; const auto d=temp.path/"absolute-body";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.dotdot_shard=true; const auto d=temp.path/"dotdot";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.ngram_absolute=true; const auto d=temp.path/"absolute-ngram";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.drop_shard=true; const auto d=temp.path/"drop-shard";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.ngram_size_override=100; const auto d=temp.path/"wrong-size";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.config_dir=true; const auto d=temp.path/"config-dir";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.config_pad=(4u<<20)+1; const auto d=temp.path/"config-big";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.missing_config_key="hc_count"; const auto d=temp.path/"config-key";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.fc_k=5120; const auto d=temp.path/"fc-fused";
          build_checkpoint(d,o); open_checkpoint(d,o); }  // [hidden,2*hidden] accepted by the shared rule
        { auto o=BuildOptions{}; o.fc_k=7680; const auto d=temp.path/"fc-invalid";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.misplaced_record=true; const auto d=temp.path/"misplaced";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.extra_header_tensor=true; const auto d=temp.path/"extra-tensor";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.duplicate_record=true; const auto d=temp.path/"dup-record";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.bad_ngram_key=true; const auto d=temp.path/"bad-key";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.ngram_no_sha=true; const auto d=temp.path/"no-sha";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.ngram_rows=2500011; const auto d=temp.path/"wrong-rows";
          build_checkpoint(d,o); rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); }); }
        { auto o=BuildOptions{}; o.external_ngram=true; const auto d=temp.path/"external";
          const auto ext=build_checkpoint(d,o);
          rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); });  // no path in the manifest, none configured
          setenv("FLASH_NGRAM_PATH",ext.external_table.c_str(),1);
          { Checkpoint ck; ck.open(d.string(),true,"");
            require(ck.ngram_path==ext.external_table,"external table path wrong"); }
          unsetenv("FLASH_NGRAM_PATH");
          Checkpoint ck; ck.open(d.string(),true,"",ext.external_table);  // explicit override wins
          require(ck.ngram_path==ext.external_table,"ngram override ignored");
          std::string manifest_text; { std::ifstream in(d/"qfn-int8-manifest.json"); manifest_text.assign(std::istreambuf_iterator<char>(in),{}); }
          require(manifest_text.find("ngram_int8.safetensors")==std::string::npos,
                  "external manifest recorded an n-gram path");
        }
        { // A self-contained checkpoint ignores override/env; the recorded path wins
            setenv("FLASH_NGRAM_PATH",(temp.path/"decoy-table.safetensors").string().c_str(),1);
            Checkpoint ck; ck.open(base.string(),true,"");
            unsetenv("FLASH_NGRAM_PATH");
            require(ck.ngram_path==(base/"ngram_int8.safetensors").string(),"a configured override must not beat outputs.ngram");
        }
        { // External mode resolves FLASH_NGRAM_PATH relative to the model directory, like the argument
            const auto d=temp.path/"external-rel"; auto o=BuildOptions{}; o.external_ngram=true; const auto ext=build_checkpoint(d,o);
            setenv("FLASH_NGRAM_PATH","external-ngram.safetensors",1);
            Checkpoint ck; ck.open(d.string(),true,"");
            unsetenv("FLASH_NGRAM_PATH");
            require(ck.ngram_path==ext.external_table,"relative FLASH_NGRAM_PATH must resolve against the model directory");
        }
        { // A relative checkpoint directory: every manifest-recorded path joins with the directory
          // exactly once (the config and the recorded MTP path used to be re-resolved against it again,
          // doubling the directory and rejecting every relative-path open)
            const auto save=fs::current_path();
            fs::current_path(temp.path);
            build_checkpoint("rel-ckpt",BuildOptions{});
            bool opened_mtp=false, opened_plain=false;
            try { Checkpoint ck; ck.open("rel-ckpt",true,""); opened_mtp=ck.has_manifest; }
            catch(const std::exception& e) { fprintf(stderr,"rel-ckpt want_mtp: %s\n",e.what()); }
            try { Checkpoint ck; ck.open("rel-ckpt",false,""); opened_plain=ck.has_manifest; }
            catch(const std::exception& e) { fprintf(stderr,"rel-ckpt no mtp: %s\n",e.what()); }
            fs::current_path(save);
            require(opened_mtp,"relative checkpoint dir with want_mtp did not open");
            require(opened_plain,"relative checkpoint dir without want_mtp did not open");
        }
        { // manifest present but not a regular file
            const auto d=temp.path/"manifest-dir"; fs::create_directories(d/"qfn-int8-manifest.json");
            rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); });
        }
        { // sandbox symlink gate: a manifest-named regular file replaced by a symlink is rejected,
          // never followed (otherwise the lexical sandbox would be escapable with one ln -s)
            const auto d=temp.path/"symlink-shard"; const auto fixture=build_checkpoint(d,BuildOptions{});
            fs::rename(d/fixture.shards[0],d/"moved-shard.safetensors");
            fs::create_symlink("moved-shard.safetensors",d/fixture.shards[0]);
            rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); });
        }
        { const auto d=temp.path/"symlink-ngram"; build_checkpoint(d,BuildOptions{});
          fs::rename(d/"ngram_int8.safetensors",d/"moved-table.safetensors");
          fs::create_symlink("moved-table.safetensors",d/"ngram_int8.safetensors");
          rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); });
        }
        { const auto d=temp.path/"symlink-manifest"; build_checkpoint(d,BuildOptions{});
          fs::rename(d/"qfn-int8-manifest.json",d/"moved-manifest.json");
          fs::create_symlink("moved-manifest.json",d/"qfn-int8-manifest.json");
          rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); });
        }
        { const auto d=temp.path/"symlink-mtp"; build_checkpoint(d,BuildOptions{});
          fs::rename(d/"model_mtp.safetensors",d/"moved-mtp.safetensors");
          fs::create_symlink("moved-mtp.safetensors",d/"model_mtp.safetensors");
          rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); });
        }
        { // Every checkpoint file is opened through the guarded open (lstat S_ISREG, O_NOFOLLOW,
          // fstat) — a substituted directory or FIFO at a manifest-recorded name is a clean rejection;
          // the recorded MTP path used to go straight to an ifstream, which blocked forever on a FIFO.
            const auto md=temp.path/"mtp-dir"; build_checkpoint(md,BuildOptions{});
            fs::remove(md/"model_mtp.safetensors"); fs::create_directories(md/"model_mtp.safetensors");
            rejects([&]{ Checkpoint ck; ck.open(md.string(),true,""); });
            const auto sd=temp.path/"shard-dir"; const auto sfix=build_checkpoint(sd,BuildOptions{});
            fs::remove(sd/sfix.shards[0]); fs::create_directories(sd/sfix.shards[0]);
            rejects([&]{ Checkpoint ck; ck.open(sd.string(),true,""); });
            const auto cd=temp.path/"config-dir2"; const auto cfix=build_checkpoint(cd,BuildOptions{});
            fs::remove(cd/"config.json"); fs::create_directories(cd/"config.json");
            rejects([&]{ Checkpoint ck; ck.open(cd.string(),false,""); });
            (void)cfix;
            const auto ff=temp.path/"mtp-fifo"; build_checkpoint(ff,BuildOptions{});
            fs::remove(ff/"model_mtp.safetensors");
            require(::mkfifo((ff/"model_mtp.safetensors").string().c_str(),0644)==0,"mkfifo fixture failed");
            rejects([&]{ Checkpoint ck; ck.open(ff.string(),true,""); });  // must reject, never block
        }
        { // An intermediate directory component replaced by a symlink to an OUTSIDE directory must
          // not smuggle the shard through (the gate used to inspect only the final component); a real
          // subdirectory with the same manifest rewrite keeps working
            auto reroute=[&](const fs::path& dir,const std::string& shard){  // record "sub/<shard>" everywhere
                std::ifstream in(dir/"qfn-int8-manifest.json");
                const std::string m((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
                const std::string needle="\""+shard+"\"", repl="\"sub/"+shard+"\"";
                std::string out;
                for(size_t p=0;;){ const auto q=m.find(needle,p);
                    if(q==std::string::npos){ out+=m.substr(p); break; }
                    out+=m.substr(p,q-p)+repl; p=q+needle.size(); }
                std::ofstream o(dir/"qfn-int8-manifest.json"); o<<out;
            };
            const auto d=temp.path/"symlink-sub"; const auto fixture=build_checkpoint(d,BuildOptions{});
            const auto outside=temp.path/"outside-sub"; fs::create_directories(outside);
            fs::rename(d/fixture.shards[0],outside/fixture.shards[0]);   // regular file outside the checkpoint dir
            fs::create_symlink(outside.string(),d/"sub");
            reroute(d,fixture.shards[0]);
            rejects([&]{ Checkpoint ck; ck.open(d.string(),true,""); });
            const auto ok=temp.path/"real-sub"; const auto okfix=build_checkpoint(ok,BuildOptions{});
            fs::create_directories(ok/"sub");
            fs::rename(ok/okfix.shards[0],ok/"sub"/okfix.shards[0]);
            reroute(ok,okfix.shards[0]);
            Checkpoint ck; ck.open(ok.string(),true,""); require(ck.has_manifest,"legit subdirectory shard rejected");
        }
        { setenv("FLASH_NGRAM","bf16",1);
          rejects([&]{ Checkpoint ck; ck.open(base.string(),true,""); });
          unsetenv("FLASH_NGRAM"); }
        }
        if(fails) return 1; printf("flash_checkpoint_unit: ok\n"); return 0;
    } catch(const std::exception& e) { fprintf(stderr,"flash_checkpoint_unit: %s\n",e.what()); return 1; }
}
