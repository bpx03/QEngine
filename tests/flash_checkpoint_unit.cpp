#include "flash/checkpoint.h"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs=std::filesystem;
using namespace flash;
using namespace flash::checkpoint_detail;
static void require(bool condition,const char* why) { if(!condition) throw std::runtime_error(why); }
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
static void ngram_file(const fs::path& path,bool bad_dtype=false,bool missing=false,int first=0,int shards=128) {
    std::string header="{";
    const int count=missing?shards-1:shards;
    for(int i=0;i<count;++i) {
        if(i) header+=",";
        header+="\"model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_"+std::to_string(first+i)+"\":{\"dtype\":\""
            +(bad_dtype && i==64?"BF16":"U8")+"\",\"shape\":[1,162],\"data_offsets\":["
            +std::to_string(i*162)+","+std::to_string((i+1)*162)+"]}";
    }
    header+="}"; while(header.size()%8) header+=' ';
    std::ofstream out(path,std::ios::binary); const uint64_t size=header.size();
    out.write(reinterpret_cast<const char*>(&size),8); out.write(header.data(),header.size());
    std::vector<char> bytes(size_t(count)*162,0); out.write(bytes.data(),bytes.size());
}
int main() {
    try {
        Temp temp;
        for(const char* invalid : {"", "{", "[", "{\"x\":", "{\"x\":1,}", "[1,]", "{\"x\":1,\"x\":2}", "01", "-", "1.", "1e", "NaN", "true trailing", "\"unterminated", "\"bad\\q\"", "\"\\uZZZZ\"" })
            rejects([&] { Json::parse(invalid); });
        const auto valid_json=Json::parse(" {\"array\":[true,false,null,-1.25e2],\"s\":\"a\\\\b\\n\"} ");
        require(valid_json["array"][3].f()==-125.0 && valid_json["s"].str=="a\\b\n","valid JSON changed");
        const Json recipe=Json::parse(R"({"format":"symmetric INT8","group_size":128,"scale_dtype":"BF16","zero_point":0})");
        validate_recipe(recipe,"fixture");
        for(const char* bad : {R"({"format":"symmetric INT4","group_size":128,"scale_dtype":"BF16","zero_point":0})",R"({"format":"symmetric INT8","group_size":64,"scale_dtype":"BF16","zero_point":0})",R"({"format":"symmetric INT8","group_size":128,"scale_dtype":"F16","zero_point":0})",R"({"format":"symmetric INT8","group_size":128,"scale_dtype":"BF16","zero_point":128})"})
            rejects([&] { validate_recipe(Json::parse(bad),"fixture"); });
        Json manifest=Json::parse(R"({"format_version":1,"family":"qwen4_exp","outputs":{"body":"model.safetensors","mtp":"model_mtp.safetensors"},"runtime_supported":false,"equivalence_verified":false})");
        manifest.obj["recipe"]=recipe;
        validate_manifest_top(manifest,"historical metadata");
        const std::string root=temp.path.string();
        require(resolve_path(root,"model.safetensors")== (temp.path/"model.safetensors").string(),"body relative path");
        const auto absolute=(temp.path/"external-table.safetensors").string();
        require(resolve_path(root,absolute)==absolute,"absolute external dependency relocated");
        require(resolve_path(root,"table.safetensors")== (temp.path/"table.safetensors").string(),"relative external dependency");
        { std::ofstream dependency(absolute); dependency<<"fixture"; }
        manifest.obj["ngram_external"]=Json::parse("{\"path\":\""+absolute+"\",\"size_bytes\":7,\"rows_per_shard\":1,\"width_bytes\":162,\"shards\":128}");
        const auto selected=resolve_outputs(manifest,root,true,"");
        require(selected.body==(temp.path/"model.safetensors").string() && selected.mtp==(temp.path/"model_mtp.safetensors").string(),"body/MTP outputs mixed");
        require(selected.ngram==absolute,"manifest external path changed");
        require(resolve_outputs(manifest,root,false,"").mtp.empty(),"unrequested MTP opened");
        const auto override_path=(temp.path/"explicit-mtp.safetensors").string();
        require(resolve_outputs(manifest,root,true,override_path).mtp==override_path,"explicit MTP override ignored");
        Json mixed=manifest; mixed.obj["outputs"].obj["mtp"]=mixed["outputs"]["body"];
        rejects([&] { resolve_outputs(mixed,root,true,""); });
        fs::remove(absolute);
        rejects([&] { resolve_outputs(manifest,root,true,""); });
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
            rejects([&] { validate_file(short_file.string()); });
            const auto long_header=temp.path/"long-header.safetensors";
            { std::ofstream out(long_header,std::ios::binary); uint64_t length=4096; out.write(reinterpret_cast<const char*>(&length),8); out<<"{}"; }
            rejects([&] { validate_file(long_header.string()); });
            const auto truncated=temp.path/"truncated.safetensors";
            matrix_file(truncated); fs::resize_file(truncated,fs::file_size(truncated)-1);
            rejects([&] { validate_file(truncated.string()); });
        }
        auto records=inventory();
        const InventorySpec spec{48,512,2560,640,24,256,2,16,128,48,128,640,4,1};
        validate_inventory(records,spec,"full inventory");
        auto bad=records; bad.erase(bad.begin()); rejects([&] { validate_inventory(bad,spec,"missing selected projection"); });
        bad=records; bad.push_back(records.front()); rejects([&] { validate_inventory(bad,spec,"duplicate matrix"); });
        bad=records; bad.back().shape={2560,5120}; rejects([&] { validate_inventory(bad,spec,"unsupported MTP FC"); });
        bad=records; bad.push_back(converted("model.language_model.layers.0.self_attn.q_proj",{12288,2560}));
        rejects([&] { validate_inventory(bad,spec,"mixed mixer families"); });
        std::cout<<"flash_checkpoint_unit: PASS\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"flash_checkpoint_unit: "<<e.what()<<'\n'; return 1; }
}
