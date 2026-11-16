#include "flash/flash_kernels.cuh"
#include "flash/flash_pfq.cuh"
#include "flash/checkpoint.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

using namespace flash;
static int fails = 0;
#define require(ok, why) \
    do { if (!(ok)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (why)); throw std::runtime_error(why); } } while (0)
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { CUDA_CHECK(cudaFree(p)); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Pinned {
    uint8_t* p = nullptr;
    explicit Pinned(size_t n) { CUDA_CHECK(cudaHostAlloc((void**)&p, n, cudaHostAllocPortable)); }
    ~Pinned() { if (p) CUDA_CHECK(cudaFreeHost(p)); }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
};
static uint16_t bits(bf16 x) { uint16_t v; std::memcpy(&v, &x, 2); return v; }
static void compare(const std::vector<float>& actual, const std::vector<float>& expected, double tolerance) {
    double error = 0, norm = 0;
    require(actual.size() == expected.size(), "output extent");
    if (actual.size() != expected.size()) return;
    bool nonfinite = false;
    for (size_t i = 0; i < actual.size(); ++i) {
        nonfinite = nonfinite || !std::isfinite(actual[i]);
        const double d = actual[i] - expected[i]; error += d*d; norm += double(expected[i])*expected[i];
    }
    require(!nonfinite, "nonfinite output");
    const double relative = std::sqrt(error / std::max(norm, 1e-30));
    if (relative > tolerance) { std::fprintf(stderr,"FAIL normalized RMS %.9g exceeds %.9g\n",relative,tolerance); ++fails; }
}
// Compare through lane/register coordinates, independent of repack's row-major traversal.
static void check_packed_mapping(const uint8_t* packed, const bf16* packed_s, int n, int k,
                                 const std::vector<uint8_t>& source, const std::vector<bf16>& scales) {
    for (int tile=0;tile<n/16;++tile) for (int chunk=0;chunk<k/32;++chunk) for (int lane=0;lane<32;++lane)
        for (int step=0;step<2;++step) for (int half=0;half<2;++half) for (int high=0;high<2;++high) for (int odd=0;odd<2;++odd) {
            const int row=tile*16+lane/4+high*8, col=chunk*32+step*16+half*8+(lane%4)*2+odd;
            const size_t p=(((size_t(tile)*(k/32)+chunk)*32+lane)*16)+step*8+(half*2+high)*2+odd;
            require(packed[p]==uint8_t(source[size_t(row)*k+col]^0x80),"INT8 code bits changed");
        }
    for(int r=0;r<n;++r) for(int g=0;g<k/128;++g)
        require(bits(packed_s[(size_t(r/16)*(k/128)+g)*16+r%16])==bits(scales[size_t(r)*(k/128)+g]),"BF16 scale bits changed");
}
static void packing_case(int n, int k, std::vector<uint8_t> source={},std::vector<bf16> scales={}) {
    if(source.empty()) {
        source.resize(size_t(n)*k);
        for (int r=0;r<n;++r) for(int c=0;c<k;++c) {
            int q=((r*31+c*17)%255)-127;
            if ((r+c)%13==0) q=0;
            if ((r+c)%29==0) q=127;
            if ((r+c)%31==0) q=-127;
            if ((r==0 && c<128) || r==n-1) q=0;
            source[size_t(r)*k+c]=uint8_t(q+128);
        }
    }
    if(scales.empty()) {
        scales.resize(size_t(n)*(k/128));
        for(size_t i=0;i<scales.size();++i) scales[i]=__float2bfloat16((i%7==0)?1.f:float(i%11+1)/128.f);
        for(int group=0;group<k/128;++group) scales[size_t(n-1)*(k/128)+group]=__float2bfloat16(1.f);
    }
    require(source.size()==size_t(n)*k && scales.size()==size_t(n)*(k/128),"packing input extent");
    Device<uint8_t> src(source.size()), dst(source.size()); Device<bf16> ss(scales.size()), ds(scales.size());
    CUDA_CHECK(cudaMemcpy(src.p,source.data(),source.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ss.p,scales.data(),scales.size()*2,cudaMemcpyHostToDevice));
    repack_w8(reinterpret_cast<int32_t*>(src.p),ss.p,n,k,reinterpret_cast<uint4*>(dst.p),ds.p,0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> packed(source.size()); std::vector<bf16> packed_s(scales.size());
    CUDA_CHECK(cudaMemcpy(packed.data(),dst.p,packed.size(),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(packed_s.data(),ds.p,packed_s.size()*2,cudaMemcpyDeviceToHost));
    check_packed_mapping(packed.data(),packed_s.data(),n,k,source,scales);
    W8 w{reinterpret_cast<uint4*>(dst.p),ds.p,n,k};
    for(int m : {1,2,3,8,9,16,17,32,33,64,65}) {
        const int ldx=k+16, ldy=n+16;
        std::vector<bf16> x(size_t(m)*ldx);
        for(int t=0;t<m;++t) for(int c=0;c<k;++c) x[size_t(t)*ldx+c]=__float2bfloat16(float((t*7+c*3)%31-15)/32.f);
        Device<bf16> dx(x.size()); Device<float> dy(size_t(m)*ldy);
        CUDA_CHECK(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
        for (int off=0;off<m;off+=kMaxBatchRows)
            gemv_w8(w,dx.p+size_t(off)*ldx,ldx,dy.p+size_t(off)*ldy,ldy,std::min(m-off,kMaxBatchRows),0);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> y(size_t(m)*ldy), actual, expected;
        CUDA_CHECK(cudaMemcpy(y.data(),dy.p,y.size()*4,cudaMemcpyDeviceToHost));
        for(int t=0;t<m;++t) for(int r=0;r<n;++r) {
            double v=0;
            for(int c=0;c<k;++c) v+=float(int(source[size_t(r)*k+c])-128)*__bfloat162float(scales[size_t(r)*(k/128)+c/128])*__bfloat162float(x[size_t(t)*ldx+c]);
            actual.push_back(y[size_t(t)*ldy+r]); expected.push_back(float(v));
        }
        compare(actual,expected,1e-4);
        // Production R8 quantizer launches whole warps; small tiles above still exercise lossless repack/GEMV.
        if (k % 512 != 0) continue;
        Device<int8_t> xq(size_t(m)*k); Device<float2> xs(size_t(m)*(k/128));
        k_x_to_r8<<<m,k/16>>>(dx.p,ldx,k,xq.p,xs.p);
        CUDA_CHECK(cudaFuncSetAttribute(k_rows_q,cudaFuncAttributeMaxDynamicSharedMemorySize,RQ_SMEM));
        k_rows_q<<<dim3((m+RQ_TB-1)/RQ_TB,(n+RQ_NB-1)/RQ_NB),256,RQ_SMEM>>>(
            w.q,w.s,k,n,xq.p,xs.p,dy.p,ldy,m);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(y.data(),dy.p,y.size()*4,cudaMemcpyDeviceToHost));
        actual.clear(); expected.clear();
        for(int t=0;t<m;++t) for(int r=0;r<n;++r) {
            double v=0;
            for(int group=0;group<k/128;++group) {
                float mx=0;
                for(int c=group*128;c<(group+1)*128;++c)
                    mx=std::max(mx,std::fabs(__bfloat162float(x[size_t(t)*ldx+c])));
                float scale=mx>0?mx/127.f:1.f;
                uint32_t raw; std::memcpy(&raw,&scale,4); raw&=~3u; std::memcpy(&scale,&raw,4);
                int dot=0;
                for(int c=group*128;c<(group+1)*128;++c) {
                    const int q=int(std::nearbyint(__bfloat162float(x[size_t(t)*ldx+c])/scale));
                    dot+=(int(source[size_t(r)*k+c])-128)*q;
                }
                v+=dot*double(scale)*__bfloat162float(scales[size_t(r)*(k/128)+group]);
            }
            actual.push_back(y[size_t(t)*ldy+r]); expected.push_back(float(v));
        }
        compare(actual,expected,1e-4);
    }
}
static int host_xh_pos(int column) {
    const int chunk=column/64, c=column%64;
    return chunk*64+(c&1)+((c/8)%2)*2+((c/16)%2)*4+(c/32)*8+((c/2)%4)*16;
}
static float xh_round(float value) {
    return __half2float(__float2half_rn(std::clamp(value,-2047.f,2047.f)/16.f))*16.f;
}
template<int Bits> struct GroupFixture {
    std::vector<int> gate, down;
    std::vector<char> packed;
    static float scale_at(int row,int group) {  // distinct per (row, group), exact in both bf16 and fp16 scales
        return (Bits==8?1.f/8192.f:1.f/512.f)*(1.f+((row*5+group*3)%4)/16.f);
    }
    GroupFixture() : gate(size_t(2*GRP)*H),down(size_t(H)*GRP),packed(ExpertLayout<Bits>::bytes) {
        const int bound=Bits==8?127:7;
        for(size_t i=0;i<gate.size();++i) gate[i]=int((i*17+i/H*11)%(2*bound+1))-bound;
        for(size_t i=0;i<down.size();++i) down[i]=int((i*13+i/GRP*3)%(2*bound+1))-bound;
        for(int c=0;c<H;++c) { gate[c]=bound; gate[size_t(8)*H+c]=bound; }
        for(int c=0;c<GRP;++c) down[c]=bound;
        pack(gate,2*GRP,H,0,ExpertLayout<Bits>::gs);
        pack(down,H,GRP,ExpertLayout<Bits>::dq,ExpertLayout<Bits>::ds);
    }
    void pack(const std::vector<int>& values,int n,int k,size_t qo,size_t so) {
        std::vector<uint8_t> source(size_t(n)*k*Bits/8,0);
        if constexpr(Bits==8) {
            for(size_t i=0;i<values.size();++i) source[i]=uint8_t(values[i]+128);
        } else {
            for(size_t i=0;i<values.size();++i) source[i/2]|=uint8_t(values[i]+8)<<((i%2)*4);
        }
        std::vector<bf16> scales(size_t(n)*(k/128));
        for (int row=0;row<n;++row) for (int group=0;group<k/128;++group) scales[size_t(row)*(k/128)+group]=__float2bfloat16(scale_at(row,group));
        Device<uint8_t> src(source.size()),q(source.size()); Device<bf16> ss(scales.size()),s(scales.size());
        CUDA_CHECK(cudaMemcpy(src.p,source.data(),source.size(),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ss.p,scales.data(),scales.size()*2,cudaMemcpyHostToDevice));
        if constexpr(Bits==8) repack_w8(reinterpret_cast<int32_t*>(src.p),ss.p,n,k,reinterpret_cast<uint4*>(q.p),s.p,0);
        else repack_w4(reinterpret_cast<int32_t*>(src.p),ss.p,n,k,reinterpret_cast<uint4*>(q.p),s.p,0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(packed.data()+qo,q.p,source.size(),cudaMemcpyDeviceToHost));
        if constexpr(Bits==8) CUDA_CHECK(cudaMemcpy(packed.data()+so,s.p,scales.size()*2,cudaMemcpyDeviceToHost));
        else {
            std::vector<__half> hs(scales.size());  // scales live in the slot layout repack_scales_kernel writes
            for (int row=0;row<n;++row) for (int group=0;group<k/128;++group)
                hs[size_t((row/16)*(k/128)+group)*16+row%16]=__float2half_rn(scale_at(row,group));
            std::memcpy(packed.data()+so,hs.data(),hs.size()*2);
        }
    }
};
template<int Bits,int NT> static void decode_launch(const MoeJob* jobs,const int* count,const float* weights,
                                                    const __half* x,__half* act,float* y,int m) {
    k_moe_gu<Bits,NT><<<dim3(16,2),128>>>(jobs,count,weights,x,act,m);
    k_moe_dn<Bits,NT><<<H/16,256>>>(jobs,count,act,y,m);
}
template<int Bits> static void decode_rows(const MoeJob* jobs,const int* count,const float* weights,
                                           const __half* x,__half* act,float* y,int m) {
    if(m<=8) decode_launch<Bits,1>(jobs,count,weights,x,act,y,m);
    else if(m<=16) decode_launch<Bits,2>(jobs,count,weights,x,act,y,m);
    else if(m<=24) decode_launch<Bits,3>(jobs,count,weights,x,act,y,m);
    else decode_launch<Bits,4>(jobs,count,weights,x,act,y,m);
}
template<int Bits> static void expert_case() {
    GroupFixture<Bits> group;
    const size_t bytes=2*group.packed.size();
    Device<char> resident(bytes),cache(bytes),stage(bytes);
    char *host=nullptr,*mapped=nullptr;
    CUDA_CHECK(cudaHostAlloc(&host,bytes,cudaHostAllocMapped));
    std::memcpy(host,group.packed.data(),group.packed.size());
    std::memcpy(host+group.packed.size(),group.packed.data(),group.packed.size());
    CUDA_CHECK(cudaHostGetDevicePointer(&mapped,host,0));
    CUDA_CHECK(cudaMemcpy(resident.p,host,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache.p,host,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(stage.p,cache.p,bytes,cudaMemcpyDeviceToDevice));
    Device<MoeJob> jobs(2); Device<int> count(1); const int nj=2;
    CUDA_CHECK(cudaMemcpy(count.p,&nj,4,cudaMemcpyHostToDevice));
    Device<__half> x(size_t(MAXM)*H),act(size_t(2)*MAXM*GRP); Device<float> y(size_t(MAXM)*H),jw(2*MAXM);
    Device<int> jrow(2*(MAXM+1)); Device<float> grouped_weights(2*MAXM);
    Device<int> map((NE+1)*NGRP),selection(MAXM*TOPK);
    Device<float> route_weight(MAXM*TOPK),shared_weight(MAXM);
    Device<uintptr_t> cold_ptr(2); Device<unsigned char> touch(2); Device<unsigned> cold_count(1);
    std::vector<int> host_map((NE+1)*NGRP,-1);
    host_map[0]=0; host_map[NE*NGRP]=1;
    CUDA_CHECK(cudaMemcpy(map.p,host_map.data(),host_map.size()*4,cudaMemcpyHostToDevice));
    std::vector<bf16> router(size_t(528)*H),router_input(size_t(MAXM)*H);
    std::vector<float> actual_logits(528,0);
    for(int e=0;e<=NE;++e) {
        const float logit=e==NE?-127.f/128.f:(e==0?0.f:(e<TOPK?-127.f/32.f:-127.f));
        router[size_t(e)*H]=__float2bfloat16(logit);
        const float value=__bfloat162float(router[size_t(e)*H]);
        const float scale=__bfloat162float(__float2bfloat16(std::fabs(value)/127.f));
        actual_logits[e]=scale>0?float(std::clamp(int(std::nearbyint(value/scale)),-127,127))*scale:0.f;
    }
    for(int t=0;t<MAXM;++t) router_input[size_t(t)*H]=__float2bfloat16(1.f);
    double normalization=0;
    for(int e=0;e<NE;++e) normalization+=std::exp(double(actual_logits[e])-actual_logits[0]);
    const float routed=1.f/float(normalization),shared=1.f/(1.f+std::exp(-actual_logits[NE]));
    Device<bf16> rw(router.size()),rx(router_input.size()),rs(size_t(528)*(H/128));
    Device<uint4> rq(size_t(528)*H/16),r16(size_t(528)*H/8);
    Device<float> logits(size_t(MAXM)*528),fused_part(size_t(RF_SPLIT)*RF_TILES*32*4);
    Device<unsigned> route_counter(1);
    CUDA_CHECK(cudaMemcpy(rw.p,router.data(),router.size()*2,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rx.p,router_input.data(),router_input.size()*2,cudaMemcpyHostToDevice));
    quantize_w8(rw.p,528,H,rq.p,rs.p,0);
    repack_w16(rw.p,528,H,r16.p,0);
    std::vector<float> all_logits(size_t(MAXM)*528);
    for(int t=0;t<MAXM;++t) std::copy(actual_logits.begin(),actual_logits.end(),all_logits.begin()+size_t(t)*528);
    CUDA_CHECK(cudaMemcpy(logits.p,all_logits.data(),all_logits.size()*4,cudaMemcpyHostToDevice));
    int one_warp=0; CUDA_CHECK(cudaMemcpyFromSymbol(&one_warp,g_moe_dn_1w,sizeof(one_warp)));
    std::vector<float> alone;
    for(int case_id : {1,2,3,8,9,16,17,32,-1}) {
        const bool stress=case_id<0;
        const int m=std::abs(case_id);
        std::vector<__half> input(size_t(m)*H); std::vector<float> weights(2*MAXM,0);
        for(int j=0;j<2;++j) for(int t=0;t<m;++t) weights[j*MAXM+t]=j?shared:routed;
        for(int t=0;t<m;++t) for(int c=0;c<H;++c)
            input[size_t(t)*H+host_xh_pos(c)]=__float2half_rn(stress?64.f:float((t*7+c*3)%31-15)/512.f);
        CUDA_CHECK(cudaMemcpy(x.p,input.data(),input.size()*2,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(jw.p,weights.data(),weights.size()*4,cudaMemcpyHostToDevice));
        std::vector<int> selected(size_t(m)*TOPK);
        std::vector<float> selected_weights(size_t(m)*TOPK,0),shared_weights(m);
        for(int t=0;t<m;++t) {
            for(int k=0;k<TOPK;++k) selected[t*TOPK+k]=k;
            selected_weights[t*TOPK]=weights[t]; shared_weights[t]=weights[MAXM+t];
        }
        CUDA_CHECK(cudaMemcpy(selection.p,selected.data(),selected.size()*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(route_weight.p,selected_weights.data(),selected_weights.size()*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(shared_weight.p,shared_weights.data(),shared_weights.size()*4,cudaMemcpyHostToDevice));
        std::vector<int> rows(2*(MAXM+1),0);
        std::vector<float> compact_weights(2*MAXM,0);
        for(int j=0;j<2;++j) {
            rows[j*(MAXM+1)]=m;
            for(int slot=0;slot<m;++slot) {
                const int token=m-1-slot;
                rows[j*(MAXM+1)+1+slot]=token;
                compact_weights[j*MAXM+slot]=weights[j*MAXM+token];
            }
        }
        CUDA_CHECK(cudaMemcpy(jrow.p,rows.data(),rows.size()*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(grouped_weights.p,compact_weights.data(),compact_weights.size()*4,cudaMemcpyHostToDevice));
        std::vector<float> expected(size_t(m)*H,0),expected_act(size_t(2)*m*GRP);
        for(int j=0;j<2;++j) for(int t=0;t<m;++t) for(int c=0;c<GRP;++c) {
            const int gate_row=(c/8)*16+c%8;
            double g=0,u=0;
            for(int k=0;k<H;++k) {
                const float v=__half2float(input[size_t(t)*H+host_xh_pos(k)])*16.f;
                g+=group.gate[size_t(gate_row)*H+k]*double(group.scale_at(gate_row,k/128))*v;
                u+=group.gate[size_t(gate_row+8)*H+k]*double(group.scale_at(gate_row+8,k/128))*v;
            }
            expected_act[(size_t(j)*m+t)*GRP+c]=xh_round(float(g/(1+std::exp(-g))*u*weights[j*MAXM+t]));
        }
        for(int t=0;t<m;++t) for(int r=0;r<H;++r) {
            double v=0;
            for(int j=0;j<2;++j) for(int c=0;c<GRP;++c)
                v+=group.down[size_t(r)*GRP+c]*double(group.scale_at(r,0))*expected_act[(size_t(j)*m+t)*GRP+c];
            expected[size_t(t)*H+r]=float(v);
        }
        std::vector<float> first;
        for(const char* placement : {resident.p,mapped,cache.p,stage.p}) {
            const bool cold=placement==mapped;
            const uintptr_t pointers[2]={reinterpret_cast<uintptr_t>(placement)|1u,
                reinterpret_cast<uintptr_t>(placement+ExpertLayout<Bits>::bytes)|1u};
            CUDA_CHECK(cudaMemcpy(cold_ptr.p,pointers,sizeof(pointers),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(touch.p,0,2)); CUDA_CHECK(cudaMemset(cold_count.p,0,4));
            k_jobs<Bits><<<1,512>>>(m,map.p,placement,ColdTab(cold_ptr.p,touch.p),cold?0:2,
                selection.p,route_weight.p,shared_weight.p,jobs.p,jw.p,count.p,cold_count.p,nullptr);
            decode_rows<Bits>(jobs.p,count.p,jw.p,x.p,act.p,y.p,m);
            CUDA_CHECK(cudaDeviceSynchronize());
            unsigned observed_cold=0;
            CUDA_CHECK(cudaMemcpy(&observed_cold,cold_count.p,4,cudaMemcpyDeviceToHost));
            require(observed_cold==(cold?2u:0u),"cold-group accounting mismatch");
            std::vector<float> actual(size_t(m)*H);
            CUDA_CHECK(cudaMemcpy(actual.data(),y.p,actual.size()*4,cudaMemcpyDeviceToHost));
            compare(actual,expected,1e-2);
            if(first.empty()) first=actual;
            else require(std::memcmp(first.data(),actual.data(),actual.size()*4)==0,"placement changes expert arithmetic");
            for(int gy : {1,2}) {
                k_moe_gu_g<Bits><<<dim3(16,gy),128>>>(jobs.p,count.p,jrow.p,grouped_weights.p,x.p,act.p);
                k_moe_dn_g<Bits><<<H/16,256>>>(jobs.p,count.p,jrow.p,act.p,y.p,m);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(actual.data(),y.p,actual.size()*4,cudaMemcpyDeviceToHost));
                compare(actual,expected,1e-2);
            }
            for(int route : {2,4,5,16}) {
                if(route==5 && m>8) continue;
                if(route==2) {
                    k_route2<Bits><<<1,512>>>(logits.p,m,map.p,placement,jobs.p,jw.p,count.p,1,
                        nullptr,ColdTab(cold_ptr.p,touch.p),cold?0:2,cold_count.p);
                } else if(route==4 || route==16) {
                    if(route==4) gemv_w8(W8{rq.p,rs.p,528,H},rx.p,H,logits.p,528,m,0);
                    else gemv_w16(W16{r16.p,528,H},rx.p,H,logits.p,528,m,0);
                    k_topk_rows<<<m,512>>>(logits.p,1,selection.p,route_weight.p,shared_weight.p,nullptr);
                    k_jobs<Bits><<<1,512>>>(m,map.p,placement,ColdTab(cold_ptr.p,touch.p),cold?0:2,
                        selection.p,route_weight.p,shared_weight.p,jobs.p,jw.p,count.p,cold_count.p,nullptr);
                } else {
                    CUDA_CHECK(cudaMemset(route_counter.p,0,4));
                    k_route_f<Bits><<<dim3((RF_TILES+3)/4,RF_SPLIT),512>>>(rq.p,rs.p,rx.p,m,fused_part.p,
                        route_counter.p,logits.p,selection.p,route_weight.p,shared_weight.p,nullptr,map.p,placement,
                        ColdTab(cold_ptr.p,touch.p),cold?0:2,jobs.p,jw.p,count.p,cold_count.p);
                }
                decode_rows<Bits>(jobs.p,count.p,jw.p,x.p,act.p,y.p,m);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(actual.data(),y.p,actual.size()*4,cudaMemcpyDeviceToHost));
                compare(actual,expected,1e-2);
            }
        }
        if(one_warp && !stress) {
            if(alone.empty()) alone.assign(first.begin(),first.begin()+H);
            else require(std::memcmp(alone.data(),first.data(),H*4)==0,"batch differs from alone with ordered down reduction");
        }
    }
    CUDA_CHECK(cudaFreeHost(host));
}
static std::vector<float> quantized_values(const std::vector<float>& values,int width) {
    std::vector<float> result(values.size());
    for(size_t off=0;off<values.size();off+=width) {
        float mx=0;
        for(int i=0;i<width;++i) mx=std::max(mx,std::fabs(values[off+i]/16.f));
        float scale=mx>0?mx/127.f:1.f;
        uint32_t raw; std::memcpy(&raw,&scale,4); raw&=~3u; std::memcpy(&scale,&raw,4);
        for(int i=0;i<width;++i) result[off+i]=float(std::nearbyint((values[off+i]/16.f)/scale))*scale*16.f;
    }
    return result;
}
template<int Bits,int G,bool BuildJobs=false> static void prefill_case() {
    GroupFixture<Bits> group;
    Device<char> device(group.packed.size()*(BuildJobs?2:1));
    CUDA_CHECK(cudaMemcpy(device.p,group.packed.data(),group.packed.size(),cudaMemcpyHostToDevice));
    if constexpr(BuildJobs)
        CUDA_CHECK(cudaMemcpy(device.p+group.packed.size(),device.p,group.packed.size(),cudaMemcpyDeviceToDevice));
    const MoeJob w{reinterpret_cast<const uint4*>(device.p),device.p+ExpertLayout<Bits>::gs,
                  reinterpret_cast<const uint4*>(device.p+ExpertLayout<Bits>::dq),device.p+ExpertLayout<Bits>::ds};
    for(int m : {1,2,3,8,9,16,17,32,33,64,65}) {
        if constexpr(BuildJobs) { if(m!=3 && m!=65) continue; }
        std::vector<__half> input(size_t(m)*H);
        std::vector<float> logical(size_t(m)*H),weights(m);
        std::vector<int> tokens(m);
        for(int t=0;t<m;++t) {
            tokens[t]=(t*3)%m; weights[t]=0.5f+float(t%3)/16.f;
            for(int c=0;c<H;++c) {
                const __half v=__float2half_rn(float((t*11+c*7)%31-15)/512.f);
                input[size_t(t)*H+host_xh_pos(c)]=v; logical[size_t(t)*H+c]=__half2float(v)*16.f;
            }
        }
        const auto qinput=quantized_values(logical,G);
        std::vector<PfJob> descriptors;
        for(int off=0;off<m;off+=PF_SHJ) for(int g=0;g<2;++g)
            descriptors.push_back(PfJob{w,off,std::min(PF_SHJ,m-off),int(descriptors.size())*PF_SHJ,g==0?2:0});
        const int nj=int(descriptors.size()),nact=nj*PF_SHJ;
        Device<PfJob> jobs(nj); Device<int> count(1),tl(size_t(m)*(TOPK+1)); Device<float> wl(size_t(m)*(TOPK+1)),output(size_t(m)*H);
        Device<__half> x(input.size()),act(size_t(nact)*GRP),saved(size_t(nact)*GRP),slots(size_t(m)*H);
        Device<int8_t> xq(size_t(m)*XQ_LD); Device<float2> xqs(size_t(m)*XQS_LD);
        Device<unsigned long long> fixed(size_t(m)*H);
        CUDA_CHECK(cudaMemcpy(jobs.p,descriptors.data(),nj*sizeof(PfJob),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(count.p,&nj,4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tl.p,tokens.data(),m*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(wl.p,weights.data(),m*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(x.p,input.data(),input.size()*2,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(act.p,0,size_t(nact)*GRP*2));
        std::vector<float> dn_pf_baseline[2]; std::vector<__half> dn_q_baseline[2];  // per-variant byte baselines across the four placements (BuildJobs)
        auto run_jobs=[&](const std::vector<PfJob>& list) {  // every kernel over the job list of the current placement
            k_xh_to_q<G><<<m,H/16>>>(x.p,xq.p,xqs.p);
            k_moe_gu_q<Bits,G,1><<<2*nj,256>>>(jobs.p,count.p,tl.p,wl.p,xq.p,xqs.p,act.p);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(saved.p,act.p,size_t(nact)*GRP*2,cudaMemcpyDeviceToDevice));
            std::vector<float> expected_act(size_t(m)*GRP);
            for(int t=0;t<m;++t) for(int c=0;c<GRP;++c) {
                const int r=(c/8)*16+c%8; double gate=0,up=0;
                for(int k=0;k<H;++k) {
                    const float value=qinput[size_t(tokens[t])*H+k];
                    gate+=group.gate[size_t(r)*H+k]*double(group.scale_at(r,k/128))*value;
                    up+=group.gate[size_t(r+8)*H+k]*double(group.scale_at(r+8,k/128))*value;
                }
                expected_act[size_t(t)*GRP+c]=xh_round(float(gate/(1+std::exp(-gate))*up*weights[t]));
            }
            // gu_q writes act[aoff + p] from the list position off + p (input row tl[off + p], weight wl[off + p]):
            // check the rows the ng > 0 jobs wrote (their ng == 0 pair duplicates them, rows past n stay memset);
            // with BuildJobs the list is the device router's (aoffs compact the groups per expert, not per 64-row job)
            std::vector<__half> saved_host(size_t(nact)*GRP);
            CUDA_CHECK(cudaMemcpy(saved_host.data(),saved.p,saved_host.size()*2,cudaMemcpyDeviceToHost));
            std::vector<float> act_actual,act_expected;
            for (const PfJob& pj : list) if (pj.ng > 0)
                for (int p = 0; p < pj.n; ++p) for (int c = 0; c < GRP; ++c) {
                    act_actual.push_back(__half2float(saved_host[size_t(pj.aoff+p)*GRP+host_xh_pos(c)])*16.f);
                    act_expected.push_back(expected_act[size_t(pj.off+p)*GRP+c]);
                }
            compare(act_actual,act_expected,1e-3);
            const auto qact=quantized_values(expected_act,GRP);
            std::vector<float> slot_expected(size_t(m)*H),float_expected(size_t(m)*H,0);
            for(int t=0;t<m;++t) for(int r=0;r<H;++r) {
                double qv=0,v=0;
                for(int c=0;c<GRP;++c) {
                    const double weight=group.down[size_t(r)*GRP+c]*double(group.scale_at(r,0))*2;
                    qv+=weight*qact[size_t(t)*GRP+c]; v+=weight*expected_act[size_t(t)*GRP+c];
                }
                slot_expected[size_t(t)*H+r]=__half2float(__float2half_rn(float(qv)/16.f))*16.f;
                float_expected[size_t(tokens[t])*H+r]+=float(v);
            }
            for(bool deterministic : {false,true}) {
                CUDA_CHECK(cudaMemset(output.p,0,size_t(m)*H*4));
                CUDA_CHECK(cudaMemset(fixed.p,0,size_t(m)*H*8));
                k_moe_dn_pf<Bits,2><<<dim3(H/16/8,nj),128>>>(jobs.p,count.p,tl.p,saved.p,output.p,deterministic?fixed.p:nullptr);
                if(deterministic) k_fx_to_f<<<256,256>>>(fixed.p,output.p,size_t(m)*H);
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<float> actual(size_t(m)*H);
                CUDA_CHECK(cudaMemcpy(actual.data(),output.p,actual.size()*4,cudaMemcpyDeviceToHost));
                compare(actual,float_expected,1e-2);
                if constexpr(BuildJobs) {  // same variant, same bytes at every placement (the RMS compare above alone would pass on per-placement drift)
                    std::vector<float>& ref = dn_pf_baseline[deterministic?1:0];
                    if (ref.empty()) ref = actual;
                    else require(ref.size() == actual.size() && std::memcmp(ref.data(), actual.data(), actual.size()*4) == 0,
                        deterministic ? "deterministic dn_pf outputs differ across placements"
                                      : "non-deterministic dn_pf outputs differ across placements");
                }
            }
            k_act_to_q<<<(nact*8+255)/256,256>>>(act.p,nact);
            const int split=m/2; SlotBuf slot{slots.p,split,slots.p+size_t(split)*H};
            for(int tile : {1,2}) {
                CUDA_CHECK(cudaMemset(slots.p,0,size_t(m)*H*2));  // dn_q only rewrites the rows its jobs cover: every tile starts from zeroed slots
                if(tile==1) k_moe_dn_q<Bits,1><<<dim3(H/16/4,nj),128>>>(jobs.p,count.p,act.p,slot);
                else k_moe_dn_q<Bits,2><<<dim3(H/16/8,nj),128>>>(jobs.p,count.p,act.p,slot);
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<__half> halves(size_t(m)*H); std::vector<float> actual(halves.size());
                CUDA_CHECK(cudaMemcpy(halves.data(),slots.p,halves.size()*2,cudaMemcpyDeviceToHost));
                for(size_t i=0;i<actual.size();++i) actual[i]=__half2float(halves[i])*16.f;
                compare(actual,slot_expected,1e-2);
                if (m == 65) {
                    const int bounds[3] = {0, 64, m};
                    Device<int> cuts(3);
                    Device<__half> window(size_t(66) * H);
                    CUDA_CHECK(cudaMemcpy(cuts.p, bounds, sizeof(bounds), cudaMemcpyHostToDevice));
                    const __half sentinel = __float2half(123.f);
                    std::vector<__half> guarded(size_t(66) * H, sentinel);
                    for (int w = 0; w < 2; ++w) {
                        CUDA_CHECK(cudaMemcpy(window.p, guarded.data(), guarded.size() * 2, cudaMemcpyHostToDevice));
                        SlotBuf narrow{window.p + H, 64, nullptr};
                        narrow.cut = cuts.p;
                        narrow.p = w;
                        if (tile == 1) k_moe_dn_q<Bits,1><<<dim3(H/16/4,nj),128>>>(jobs.p,count.p,act.p,narrow);
                        else k_moe_dn_q<Bits,2><<<dim3(H/16/8,nj),128>>>(jobs.p,count.p,act.p,narrow);
                        CUDA_CHECK(cudaDeviceSynchronize());
                        std::vector<__half> got(guarded.size());
                        CUDA_CHECK(cudaMemcpy(got.data(), window.p, got.size() * 2, cudaMemcpyDeviceToHost));
                        require(std::memcmp(got.data(), guarded.data(), H * 2) == 0 &&
                                std::memcmp(got.data() + size_t(65) * H, guarded.data(), H * 2) == 0,
                                "quantized prefill window overwrote its canaries");
                        const size_t count = size_t(bounds[w + 1] - bounds[w]) * H;
                        require(std::memcmp(got.data() + H, halves.data() + size_t(bounds[w]) * H, count * 2) == 0,
                                "quantized prefill window differs from unsliced slot rows");
                    }
                }
                if constexpr(BuildJobs) {  // rows are slot-ordered, the slot order is the router's atomic fill order: compare by token
                    std::vector<__half> ordered(halves.size());
                    for(int s=0;s<m;++s) std::copy_n(halves.data()+size_t(s)*H,H,ordered.data()+size_t(tokens[s])*H);
                    std::vector<__half>& ref = dn_q_baseline[tile-1];
                    if (ref.empty()) ref = std::move(ordered);
                    else require(ref.size() == ordered.size() && std::memcmp(ref.data(), ordered.data(), ordered.size()*2) == 0,
                        tile==1 ? "dn_q tile 1 slot rows differ across placements" : "dn_q tile 2 slot rows differ across placements");
                }
            }
        };
        if constexpr(BuildJobs) {
            Device<int> map((NE+1)*NGRP),selected(size_t(m)*TOPK);
            Device<float> route_weights(size_t(m)*TOPK),shared_weights(m);
            std::vector<int> mapping((NE+1)*NGRP,-1),ids(size_t(m)*TOPK);
            std::vector<float> rw(size_t(m)*TOPK,0);
            mapping[0]=0; mapping[1]=1;
            for(int t=0;t<m;++t) {
                for(int k=0;k<TOPK;++k) ids[t*TOPK+k]=k;
                rw[t*TOPK]=weights[t];
            }
            CUDA_CHECK(cudaMemcpy(map.p,mapping.data(),mapping.size()*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(selected.p,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(route_weights.p,rw.data(),rw.size()*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(shared_weights.p,0,m*4));
            // grp_ptr must resolve the pair (mapping 0,1 are expert 0's groups) at every placement: resident (ndev
            // covers both, no table read), mapped host (entries tagged |1), device cache and staging ring (untagged
            // entries, ndev=0: the stand-ins for the slots the expert cache lends; the cold table itself is the real one)
            const size_t bytes=group.packed.size();
            Device<char> cache(2*bytes),stage(2*bytes);
            char* host=nullptr; char* mapped=nullptr;
            CUDA_CHECK(cudaHostAlloc(&host,2*bytes,cudaHostAllocMapped));
            std::memcpy(host,group.packed.data(),bytes); std::memcpy(host+bytes,group.packed.data(),bytes);
            CUDA_CHECK(cudaHostGetDevicePointer(&mapped,host,0));
            CUDA_CHECK(cudaMemcpy(cache.p,host,2*bytes,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(stage.p,host,2*bytes,cudaMemcpyHostToDevice));
            Device<uintptr_t> cold_ptr(2); Device<unsigned char> touch(2);
            for(int place=0;place<4;++place) {
                const bool table=place>0;  // the resident placement never reaches the table (li < ndev)
                const uintptr_t base=place==0?reinterpret_cast<uintptr_t>(device.p)
                    :place==1?reinterpret_cast<uintptr_t>(mapped)
                    :place==2?reinterpret_cast<uintptr_t>(cache.p):reinterpret_cast<uintptr_t>(stage.p);
                const uintptr_t tag=place==1?uintptr_t(1):uintptr_t(0);  // host-mapped entries carry |1, grp_ptr strips it
                const uintptr_t cold[2]={base|tag,(base+bytes)|tag};
                CUDA_CHECK(cudaMemcpy(cold_ptr.p,cold,sizeof(cold),cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(touch.p,0,2)); CUDA_CHECK(cudaMemset(count.p,0,4));
                CUDA_CHECK(cudaMemset(act.p,0,size_t(nact)*GRP*2));
                k_jobs_pf<Bits><<<1,512>>>(m,map.p,device.p,table?ColdTab(cold_ptr.p,touch.p):ColdTab(),table?0:2,
                    selected.p,route_weights.p,shared_weights.p,jobs.p,tl.p,wl.p,count.p,nullptr);
                CUDA_CHECK(cudaDeviceSynchronize());
                int njobs=0; CUDA_CHECK(cudaMemcpy(&njobs,count.p,4,cudaMemcpyDeviceToHost));
                require(njobs==2*((m+PF_SHJ-1)/PF_SHJ),"prefill router emitted the wrong job count");
                std::vector<PfJob> emitted(njobs);
                CUDA_CHECK(cudaMemcpy(emitted.data(),jobs.p,njobs*sizeof(PfJob),cudaMemcpyDeviceToHost));
                for(int j=0;j<njobs;++j) {  // expert 0's chunks x local groups: group k is the placement's copy li=k
                    const int q=j/2,k=j%2; const MoeJob& gj=emitted[j].w;
                    require(reinterpret_cast<uintptr_t>(gj.gq)==base+uintptr_t(k)*bytes,"job gate_up address");
                    require(reinterpret_cast<uintptr_t>(gj.gs)==base+uintptr_t(k)*bytes+ExpertLayout<Bits>::gs,"job gate_up scale address");
                    require(reinterpret_cast<uintptr_t>(gj.dq)==base+uintptr_t(k)*bytes+ExpertLayout<Bits>::dq,"job down address");
                    require(reinterpret_cast<uintptr_t>(gj.ds)==base+uintptr_t(k)*bytes+ExpertLayout<Bits>::ds,"job down scale address");
                    require(emitted[j].off==q*PF_SHJ && emitted[j].n==std::min(PF_SHJ,m-q*PF_SHJ),"prefill job row extent");
                    require(emitted[j].aoff==k*m+q*PF_SHJ && emitted[j].ng==(k==0?2:0),"prefill job activation layout");
                }
                unsigned char touched[2]={0,0};
                CUDA_CHECK(cudaMemcpy(touched,touch.p,2,cudaMemcpyDeviceToHost));
                require(table?(touched[0]==1&&touched[1]==1):(touched[0]==0&&touched[1]==0),
                    "cold-group table touch does not match the placement");  // every read through the table touches
                CUDA_CHECK(cudaMemcpy(tokens.data(),tl.p,m*4,cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(weights.data(),wl.p,m*4,cudaMemcpyDeviceToHost));
                std::vector<int> occurrences(m,0);
                for(int i=0;i<m;++i) {
                    require(tokens[i]>=0 && tokens[i]<m,"prefill router emitted out-of-range token");
                    ++occurrences[tokens[i]];
                    require(weights[i]==0.5f+float(tokens[i]%3)/16.f,"prefill route weight belongs to another token");
                }
                for(int times:occurrences) require(times==1,"prefill router lost/duplicated token");
                run_jobs(emitted);
            }
            CUDA_CHECK(cudaFreeHost(host));
        } else run_jobs(descriptors);
    }
}
static uint8_t group_code(int group, int plane, int row, int col) {
    const unsigned key = (unsigned)(group*7919 + plane*104729 + row*37 + col*101);
    switch (key % 23) {
        case 0: return 0;
        case 1: return 128;
        case 2: return 255;
        case 3: return 1;
        case 4: return 254;
        default: return (uint8_t)(key*29u + row*7u + col*13u);
    }
}
static bf16 group_scale(int group, int plane, int row, int part) {
    return __float2bfloat16(0.5f + float((group*13 + plane*29 + row*3 + part*7) % 97) / 64.f);
}
static void make_group_record(int group, char* record) {
    auto* gate = (uint8_t*)record;
    auto* gs = (bf16*)(record + ExpertLayout<8>::gs);
    auto* down = (uint8_t*)(record + ExpertLayout<8>::dq);
    auto* ds = (bf16*)(record + ExpertLayout<8>::ds);
    for (int half=0;half<2;++half) for (int i=0;i<GRP;++i) {
        const int row=(i/8)*16+half*8+i%8;
        for (int c=0;c<H;++c) gate[(size_t)row*H+c]=group_code(group,half,i,c);
        for (int part=0;part<H/128;++part) gs[(size_t)row*(H/128)+part]=group_scale(group,half,i,part);
    }
    for (int row=0;row<H;++row) {
        for (int c=0;c<GRP;++c) down[(size_t)row*GRP+c]=group_code(group,2,row,c);
        ds[row]=group_scale(group,2,row,0);
    }
}
static void cold_group_copy_case() {
    constexpr int BATCH=4, GROUPS=2*BATCH+1;
    constexpr size_t GB=ExpertLayout<8>::bytes;
    Pinned staging(2*BATCH*GB), output(GROUPS*GB);
    Device<uint8_t> staged_device(2*BATCH*GB), scratch(BATCH*GB);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaEvent_t events[2];
    for (auto& event : events) CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
    bool recorded[2]={false,false};
    for (int first=0,batch=0;first<GROUPS;first+=BATCH,++batch) {
        const int slot=batch&1, count=std::min(BATCH,GROUPS-first);
        if (recorded[slot]) CUDA_CHECK(cudaEventSynchronize(events[slot]));
        char* hb=(char*)staging.p+(size_t)slot*BATCH*GB;
        char* db=(char*)staged_device.p+(size_t)slot*BATCH*GB;
        for (int b=0;b<count;++b) make_group_record(first+b,hb+(size_t)b*GB);
        CUDA_CHECK(cudaMemcpyAsync(db,hb,(size_t)count*GB,cudaMemcpyHostToDevice,stream));
        for (int b=0;b<count;++b) {
            char* src=db+(size_t)b*GB;
            char* dst=(char*)scratch.p+(size_t)b*GB;
            repack_w8((const int32_t*)src,(const bf16*)(src+ExpertLayout<8>::gs),2*GRP,H,
                      (uint4*)dst,(bf16*)(dst+ExpertLayout<8>::gs),stream);
            repack_w8((const int32_t*)(src+ExpertLayout<8>::dq),(const bf16*)(src+ExpertLayout<8>::ds),H,GRP,
                      (uint4*)(dst+ExpertLayout<8>::dq),(bf16*)(dst+ExpertLayout<8>::ds),stream);
            CUDA_CHECK(cudaMemcpyAsync(output.p+(size_t)(first+b)*GB,dst,GB,cudaMemcpyDeviceToHost,stream));
        }
        CUDA_CHECK(cudaEventRecord(events[slot],stream));
        recorded[slot]=true;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (auto& event : events) CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaStreamDestroy(stream));
    bool saw_zero=false,saw_center=false,saw_max=false,saw_one=false,saw_near_max=false;
    for (int group=0;group<GROUPS;++group) {
        std::vector<uint8_t> gate((size_t)2*GRP*H), down((size_t)H*GRP);
        std::vector<bf16> gs((size_t)2*GRP*(H/128)), ds(H);
        for (int half=0;half<2;++half) for (int i=0;i<GRP;++i) {
            const int row=(i/8)*16+half*8+i%8;
            for (int c=0;c<H;++c) {
                const uint8_t code=group_code(group,half,i,c);
                gate[(size_t)row*H+c]=code;
                saw_zero|=code==0; saw_center|=code==128; saw_max|=code==255; saw_one|=code==1; saw_near_max|=code==254;
            }
            for (int part=0;part<H/128;++part) gs[(size_t)row*(H/128)+part]=group_scale(group,half,i,part);
        }
        for (int row=0;row<H;++row) {
            for (int c=0;c<GRP;++c) down[(size_t)row*GRP+c]=group_code(group,2,row,c);
            ds[row]=group_scale(group,2,row,0);
        }
        const char* got=(const char*)output.p+(size_t)group*GB;
        check_packed_mapping((const uint8_t*)got,(const bf16*)(got+ExpertLayout<8>::gs),2*GRP,H,gate,gs);
        check_packed_mapping((const uint8_t*)(got+ExpertLayout<8>::dq),(const bf16*)(got+ExpertLayout<8>::ds),H,GRP,down,ds);
    }
    require(saw_zero&&saw_center&&saw_max&&saw_one&&saw_near_max,"cold group codes cover zero and byte extremes");
}
static void real_samples(const std::string& model) {
    SafeTensors tensors;
    // v2 body is a shard list (model-00001-of-000NN.safetensors); the legacy single model.safetensors
    // also works. MTP files are never body. SafeTensors indexes tensors across every opened shard.
    const auto path=model;
    std::vector<std::string> files;
    for (const auto& e : std::filesystem::directory_iterator(model)) {
        const auto name=e.path().filename().string();
        if (e.is_regular_file() && e.path().extension()==".safetensors" && name.rfind("model_mtp",0)!=0)
            files.push_back(e.path().string());
    }
    std::sort(files.begin(),files.end());
    if (files.empty()) throw std::runtime_error(model+": no body safetensors files");
    for (const auto& f : files) { checkpoint_detail::validate_file(f); tensors.open(f); }
    const std::string prefix="model.language_model.layers.0.mlp.experts.";
    const auto& gq=tensors.get(prefix+"gate_up_proj.weight_packed");
    const auto& gs=tensors.get(prefix+"gate_up_proj.weight_scale");
    const auto& dq=tensors.get(prefix+"down_proj.weight_packed");
    const auto& ds=tensors.get(prefix+"down_proj.weight_scale");
    checkpoint_detail::validate_int8_tensor(tensors,prefix+"gate_up_proj",{512,1280,2560},path);
    checkpoint_detail::validate_int8_tensor(tensors,prefix+"down_proj",{512,2560,640},path);
    for(int expert : {0,511}) for(int group : {0,4}) {
        std::vector<uint8_t> gate_codes(16*H),down_codes(16*GRP);
        std::vector<bf16> gate_scales(16*(H/128)),down_scales(16);
        for(int row=0;row<16;++row) {
            const int source_row=group*GRP+(group==4?120:0)+(row%8)+(row/8)*INTER;
            const size_t grow=size_t(expert)*1280+source_row;
            std::memcpy(gate_codes.data()+size_t(row)*H,gq.data+grow*H,H);
            std::memcpy(gate_scales.data()+size_t(row)*(H/128),gs.data+grow*(H/128)*2,(H/128)*2);
            const size_t drow=size_t(expert)*H+(row==15?H-1:row);
            std::memcpy(down_codes.data()+size_t(row)*GRP,dq.data+drow*INTER+group*GRP,GRP);
            std::memcpy(down_scales.data()+row,ds.data+(drow*NGRP+group)*2,2);
        }
        packing_case(16,H,std::move(gate_codes),std::move(gate_scales));
        packing_case(16,GRP,std::move(down_codes),std::move(down_scales));
    }
}
static void prefill_borrow_tails_case() {  // Worker::prefill_borrow_tails with a base of 2560 rows (host only, no CUDA)
    struct Row { int n; WeightFormat format; bool staged, lent, want; };
    static const Row rows[]={
        // at or below the base buffer: INT4 never takes the cache tails; INT8 only while it stages cold groups AND the
        // ring extension still lives in cache slots (then the staging needs the whole ring back)
        {512,WeightFormat::Int4,false,false,false},{512,WeightFormat::Int4,false,true,false},{512,WeightFormat::Int4,true,false,false},{512,WeightFormat::Int4,true,true,false},
        {512,WeightFormat::Int8,false,false,false},{512,WeightFormat::Int8,false,true,false},{512,WeightFormat::Int8,true,false,false},{512,WeightFormat::Int8,true,true,true},
        {2047,WeightFormat::Int4,false,false,false},{2047,WeightFormat::Int4,false,true,false},{2047,WeightFormat::Int4,true,false,false},{2047,WeightFormat::Int4,true,true,false},
        {2047,WeightFormat::Int8,false,false,false},{2047,WeightFormat::Int8,false,true,false},{2047,WeightFormat::Int8,true,false,false},{2047,WeightFormat::Int8,true,true,true},
        {2048,WeightFormat::Int4,false,false,false},{2048,WeightFormat::Int4,false,true,false},{2048,WeightFormat::Int4,true,false,false},{2048,WeightFormat::Int4,true,true,false},
        {2048,WeightFormat::Int8,false,false,false},{2048,WeightFormat::Int8,false,true,false},{2048,WeightFormat::Int8,true,false,false},{2048,WeightFormat::Int8,true,true,true},
        {2560,WeightFormat::Int4,false,false,false},{2560,WeightFormat::Int4,false,true,false},{2560,WeightFormat::Int4,true,false,false},{2560,WeightFormat::Int4,true,true,false},
        {2560,WeightFormat::Int8,false,false,false},{2560,WeightFormat::Int8,false,true,false},{2560,WeightFormat::Int8,true,false,false},{2560,WeightFormat::Int8,true,true,true},
        // past the base buffer the batch takes the tails whatever the format or the staging
        {2561,WeightFormat::Int4,false,false,true},{2561,WeightFormat::Int4,false,true,true},{2561,WeightFormat::Int4,true,false,true},{2561,WeightFormat::Int4,true,true,true},
        {2561,WeightFormat::Int8,false,false,true},{2561,WeightFormat::Int8,false,true,true},{2561,WeightFormat::Int8,true,false,true},{2561,WeightFormat::Int8,true,true,true},
    };
    for(const Row& r : rows)
        require(Worker::prefill_borrow_tails(r.format,r.n,2560,r.staged,r.lent)==r.want,"prefill borrow-tails rule");
}
int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        gemv_workspace_init();
        prefill_borrow_tails_case();
        packing_case(16,128); packing_case(32,256); packing_case(48,640); packing_case(32,2560);
        cold_group_copy_case();
        for(int one_warp : {0,1}) {
            CUDA_CHECK(cudaMemcpyToSymbol(g_moe_dn_1w,&one_warp,sizeof(one_warp)));
            expert_case<4>(); expert_case<8>();
        }
        prefill_case<4,64>(); prefill_case<8,64>(); prefill_case<8,128>();
        prefill_case<4,64,true>(); prefill_case<8,64,true>();
        if(const char* model=std::getenv("INT8_MODEL")) real_samples(model);
        if (fails) return 1;
        std::printf("flash_int8_unit: ok\n");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"flash_int8_unit: %s\n",e.what()); return 1; }
}
