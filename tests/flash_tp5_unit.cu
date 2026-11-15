// Real CUDA ranks: host LL collectives and staged QSA exchanges, independent CPU oracles.
#include "flash/flash_kernels.cuh"
#include "kernels/ops.cuh"
#include "flash/pp_plan.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <exception>
#include <vector>

using namespace flash;
#define require(ok, why) do { if (!(ok)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,why); throw std::runtime_error(why); } } while (0)

// Same Device/require/normalized RMS pattern as flash_int8_unit; guards surround every allocation.
template<class T> struct Device {
    T* p = nullptr;
    T* allocation = nullptr;
    size_t n;
    int device;
    explicit Device(size_t count) : n(count) {
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaMalloc(&allocation,(n+64)*sizeof(T)));
        CUDA_CHECK(cudaMemset(allocation,0xa5,(n+64)*sizeof(T)));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        p=allocation+32;
    }
    ~Device() { if(allocation) { cudaSetDevice(device); cudaFree(allocation); } }
    void check() const {
        CUDA_CHECK(cudaSetDevice(device));
        unsigned char a[32*sizeof(T)], b[32*sizeof(T)];
        CUDA_CHECK(cudaMemcpy(a,allocation,sizeof(a),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b,p+n,sizeof(b),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<sizeof(a);++i) require(a[i]==0xa5 && b[i]==0xa5,"device buffer canary overwritten");
    }
    // Captured graphs use nonblocking streams: pageable H2D copies/default-stream fills must finish first.
    void zero() { CUDA_CHECK(cudaMemset(p,0,n*sizeof(T))); CUDA_CHECK(cudaStreamSynchronize(nullptr)); }
    void put(const std::vector<T>& v) {
        require(v.size()==n,"input extent");
        CUDA_CHECK(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    std::vector<T> get() const { std::vector<T> v(n); CUDA_CHECK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
};
struct Pinned {
    unsigned char *allocation=nullptr,*p=nullptr;
    size_t n;
    explicit Pinned(size_t count):n(count) {
        CUDA_CHECK(cudaHostAlloc(&allocation,n+256,cudaHostAllocPortable|cudaHostAllocMapped));
        std::memset(allocation,0xa5,n+256); p=allocation+128;
    }
    ~Pinned() { if(allocation) cudaFreeHost(allocation); }
    template<class T> T* mapped() const { void* d=nullptr; CUDA_CHECK(cudaHostGetDevicePointer(&d,allocation,0)); return reinterpret_cast<T*>(static_cast<unsigned char*>(d)+128); }
    void check() const { for(int i=0;i<128;++i) require(allocation[i]==0xa5 && p[n+i]==0xa5,"pinned staging canary overwritten"); }
    Pinned(const Pinned&)=delete;
    Pinned& operator=(const Pinned&)=delete;
};
struct Graph {
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t exec=nullptr;
    int device;
    Graph() { CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); }
    ~Graph() { cudaSetDevice(device); if(exec) cudaGraphExecDestroy(exec); if(graph) cudaGraphDestroy(graph); if(stream) cudaStreamDestroy(stream); }
    template<class F> void capture(F launch) {
        CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); launch(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    void run() { CUDA_CHECK(cudaSetDevice(device)); CUDA_CHECK(cudaGraphLaunch(exec,stream)); }
    void sync() { CUDA_CHECK(cudaSetDevice(device)); CUDA_CHECK(cudaStreamSynchronize(stream)); }
};
static void compare(const std::vector<float>& actual,const std::vector<float>& expected,double tolerance) {
    require(actual.size()==expected.size(),"output extent");
    double error=0,norm=0;
    for(size_t i=0;i<actual.size();++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]),"nonfinite output/reference");
        const double d=actual[i]-expected[i]; error+=d*d; norm+=double(expected[i])*expected[i];
    }
    const double relative=std::sqrt(error/std::max(norm,1e-30));
    if(relative>tolerance) std::fprintf(stderr,"normalized RMS %.9g exceeds %.9g\n",relative,tolerance);
    require(relative<=tolerance,"CPU oracle mismatch");
}
static float half_round(float v) { return __half2float(__float2half_rn(v)); }
static void plan_checks() {
    flash_pp::PickIn pick{};
    pick.n=5;
    require(flash_pp::pick_par(pick).par==flash_pp::Par::PP,"automatic five-card policy must remain PP");
    flash_pp::Par explicit_mode;
    require(flash_pp::parse_par("tp",&explicit_mode) && explicit_mode==flash_pp::Par::TP,
            "explicit TP must not silently become PP");
    require(flash_pp::comm_wanted(5,explicit_mode,false,false),"explicit TP5 must allocate rank communication");
    for(int n=1;n<=5;++n) {
        const auto plans=make_plan(n,std::vector<double>(n,1.0));
        std::vector<int> qs(NQ),vs(LNV),voc(VOCAB),groups((NE+1)*NGRP);
        for(int r=0;r<n;++r) {
            const Plan& p=plans[r];
            require(p.q0==q_begin(r,n) && p.q1==q_end(r,n),"Q plan/helpers disagree");
            require(p.v0==v_begin(r,n) && p.v1==v_end(r,n),"V plan/helpers disagree");
            require(p.k0==k_begin(r,n) && p.k1==k_end(r,n),"K plan/helpers disagree");
            require(q_count(r,n)<=q_capacity(n),"head exceeds padded capacity");
            if(n<5) require(q_count(r,n)==NQ/n,"TP1-4 Q split changed");
            for(int h=p.q0;h<p.q1;++h) { ++qs[h]; require(q_owner(h,n)==r,"Q owner inverse"); }
            require(p.voc0>=0 && p.voc1<=VOCAB && p.voc0%128==0 && p.voc1%128==0,"unaligned/out-of-range vocab partition");
            for(int word=p.voc0;word<p.voc1;++word) ++voc[word];
            for(int h=p.v0;h<p.v1;++h) { ++vs[h]; require(h/3>=p.k0 && h/3<p.k1,"value maps outside local keys"); }
            for(size_t g=0;g<groups.size();++g) if(p.gcard[g]==r) ++groups[g];
            if(n==5) require(plan_groups(p)==513,"TP5 must assign 513 expert groups/rank");
        }
        for(int count:qs) require(count==1,"Q split gap/overlap");
        for(int count:vs) require(count==1,"V split gap/overlap");
        for(int count:groups) require(count==1,"expert split gap/overlap");
        for(int count:voc) require(count==1,"vocab split gap/overlap");
    }
    for(int n:{0,6}) { bool rejected=false; try { make_plan(n,std::vector<double>(n,1)); } catch(const std::runtime_error&) { rejected=true; } require(rejected,"invalid TP count accepted"); }
    unsigned used=0;
    for(int r=0;r<5;++r) for(int o=0;o<5;++o) {
        const unsigned bit=comm_pair_bit(r,o);
        require(bit<32,"pair bit index exceeds unsigned"); require(!(used&(1u<<bit)),"pair index collision"); used|=1u<<bit;
        if(r<4 && o<4) require(bit==unsigned(4*r+o),"legacy TP4 pair index changed");
    }
    require(used==((1u<<25)-1),"pair mask must cover 25 distinct bits");
    const int boundaries[]={4,9,14,19,23};
    const int owners[]={1,2,3,4,4};
    for(int i=0;i<5;++i) require(q_owner(boundaries[i],5)==owners[i],"TP5 boundary head ownership");
}
struct CollectiveRank {
    Device<float> input,residual,gather,ss;
    Device<unsigned> seq,done,go,bar,epoch,ready;
    CommCounters ctr{};
    Graph graph;
    CollectiveRank(int width,int ranks):input(width),residual(width),gather(size_t(ranks)*width),ss(kCommBlocks*kMaxBatchRows),
        seq(1),done(2),go(1),bar(2),epoch(1),ready(4096) {
        seq.zero(); done.zero(); go.zero(); bar.zero(); epoch.zero(); ready.zero(); ss.zero();
        ctr.seq=seq.p; ctr.done=done.p; ctr.go=go.p; ctr.bar=bar.p; ctr.ss=ss.p; ctr.epoch=epoch.p; ctr.ready=ready.p;
    }
    void check() { input.check(); residual.check(); gather.check(); ss.check(); seq.check(); done.check(); go.check(); bar.check(); epoch.check(); ready.check(); }
};
static void collective_checks(int n) {
    constexpr int width=256;
    Comm c=comm_create(n,width,width); // No peer aliases: every rank uses its own CUDA device and buffers.
    std::vector<std::unique_ptr<CollectiveRank>> ranks;
    for(int r=0;r<n;++r) { CUDA_CHECK(cudaSetDevice(r)); ranks.emplace_back(new CollectiveRank(width,n)); }
    for(int r=0;r<n;++r) {
        CUDA_CHECK(cudaSetDevice(r)); auto& x=*ranks[r];
        comm_bind_ll_scrub(x.ctr,c,r);
        x.graph.capture([&](cudaStream_t st) {
            comm_begin_unit(x.ctr,st);
            CUDA_CHECK(cudaMemsetAsync(x.residual.p,0,width*sizeof(float),st));
            allreduce_add(x.residual.p,x.input.p,1,width,c,r,x.ctr,st,nullptr,0,nullptr,0);
            allgather_small(x.input.p,width,x.gather.p,c,r,x.ctr,st);
        });
    }
    // More than 64 launch units: cross LL tag reuse and both 32-unit host scrub barriers, also in --small.
    for(int unit=0;unit<70;++unit) {
        std::vector<float> expected(width),gather(size_t(n)*width);
        for(int r=0;r<n;++r) {
            CUDA_CHECK(cudaSetDevice(r)); std::vector<float> input(width);
            for(int d=0;d<width;++d) { input[d]=float((r+1)*8+(d%7)+(unit%3))/16.f; expected[d]+=input[d]; gather[size_t(r)*width+d]=input[d]; }
            ranks[r]->input.put(input);
        }
        for(auto& x:ranks) x->graph.run();
        for(auto& x:ranks) x->graph.sync();
        for(int r=0;r<n;++r) {
            CUDA_CHECK(cudaSetDevice(r)); compare(ranks[r]->residual.get(),expected,1e-4); compare(ranks[r]->gather.get(),gather,0);
            if(n==5) require(expected[0]-gather[0]>gather[0],"fifth-rank oracle lacks substantial contribution");
            ranks[r]->check();
        }
    }
    ranks.clear(); CUDA_CHECK(cudaFreeHost(c.ll)); CUDA_CHECK(cudaFreeHost(c.data)); CUDA_CHECK(cudaFreeHost(c.flags));
}
enum class Exchange { Q32,Q16,Local16,Partial32,Partial16 };
struct ExchangeRank {
    Device<float> input,gate,recv;
    Device<bf16> output;
    Device<unsigned> flags,seq,done,scnt;
    Pinned stage,host_flags;
    Graph sender,consumer;
    Xchg send{},read{};
    ExchangeRank(size_t input_count,size_t recv_count,int n,int rows,int heads):
        input(input_count),gate(size_t(rows)*heads*HD),recv(recv_count),output(size_t(rows)*heads*HD),
        flags(n),seq(1),done(1),scnt(n),stage(recv_count*sizeof(float)),host_flags(n*sizeof(unsigned)) {
        gate.zero(); flags.zero(); seq.zero(); done.zero(); scnt.zero();
        std::memset(stage.p,0x5a,stage.n); std::memset(host_flags.p,0,host_flags.n);
    }
    void check() { input.check(); gate.check(); recv.check(); output.check(); flags.check(); seq.check(); done.check(); scnt.check(); stage.check(); host_flags.check(); }
};
static void store_float(std::vector<unsigned char>& bytes,size_t offset,float v) { std::memcpy(bytes.data()+offset,&v,sizeof(v)); }
static void store_half(std::vector<unsigned char>& bytes,size_t offset,float v) { const __half h=__float2half_rn(v); std::memcpy(bytes.data()+offset,&h,sizeof(h)); }
static void exchange_checks(int n,int rows,Exchange mode) {
    const auto ps=make_plan(n,std::vector<double>(n,1.0));
    const bool partial=mode==Exchange::Partial32 || mode==Exchange::Partial16;
    const bool h16=mode!=Exchange::Q32 && mode!=Exchange::Partial32;
    const int cap=q_capacity(n),per=cap*(partial?HD+2:HD);
    const size_t rc=size_t(XSLOT)*n*XROWS*per;
    const size_t input_count=size_t(rows)*(partial?NQ*(HD+2):(NQ+4)*HD);
    std::vector<std::unique_ptr<ExchangeRank>> rs;
    std::vector<std::vector<unsigned char>> expected(n,std::vector<unsigned char>(rc*sizeof(float),0x5a));
    std::vector<std::vector<float>> input(n,std::vector<float>(input_count));
    for(int r=0;r<n;++r) { CUDA_CHECK(cudaSetDevice(r)); rs.emplace_back(new ExchangeRank(input_count,rc,n,rows,q_count(r,n))); }
    for(int r=0;r<n;++r) {
        CUDA_CHECK(cudaSetDevice(r)); auto& x=*rs[r];
        x.send.seq=x.seq.p; x.send.done=x.done.p; x.send.scnt=x.scnt.p;
        for(int peer=0;peer<n;++peer) { x.send.recv[peer]=rs[peer]->stage.mapped<float>(); x.send.flags[peer]=rs[peer]->host_flags.mapped<unsigned>(); }
        x.read=x.send; x.read.recv[r]=x.recv.p; x.read.flags[r]=x.flags.p;
        x.sender.capture([&](cudaStream_t st) {
            const dim3 grid(n-1,3); // Multi-block publication, including blocks with no payload at tiny m.
            switch(mode) {
                case Exchange::Q32: k_qxsend<<<grid,256,0,st>>>(x.input.p,rows,x.send,ps[r]); break;
                case Exchange::Q16: k_qxsend16<<<grid,256,0,st>>>(x.input.p,rows,x.send,ps[r]); break;
                case Exchange::Local16: k_qxlocal16<<<grid,256,0,st>>>(x.input.p,rows,x.send,ps[r]); break;
                case Exchange::Partial32: k_xsend<<<grid,256,0,st>>>(x.input.p,rows,x.send,ps[r]); break;
                case Exchange::Partial16: k_xsend16<<<grid,256,0,st>>>(x.input.p,rows,x.send,ps[r]); break;
            }
        });
        if(partial) x.consumer.capture([&](cudaStream_t st) {
            if(h16) k_qsa_comb16<<<dim3(q_count(r,n),rows),HD,0,st>>>(x.input.p,rows,x.gate.p,q_count(r,n)*HD,x.output.p,q_count(r,n)*HD,x.read,ps[r],0);
            else k_qsa_comb2<<<dim3(q_count(r,n),rows),HD,0,st>>>(x.input.p,rows,x.gate.p,q_count(r,n)*HD,x.output.p,q_count(r,n)*HD,x.read,ps[r],0);
        });
    }
    for(unsigned unit=1;unit<=4;++unit) { // Both parities, then reuse each staged slot with changed data.
        for(int r=0;r<n;++r) {
            CUDA_CHECK(cudaSetDevice(r));
            if(partial) for(int t=0;t<rows;++t) for(int h=0;h<NQ;++h) {
                const size_t off=(size_t(t)*NQ+h)*(HD+2);
                const float L=float(1<<(r%3));
                for(int d=0;d<HD;++d) input[r][off+d]=L*float((r+1)*32+h*2+t+int(unit)+(d%13))/512.f;
                input[r][off+HD]=float(r)/16.f; input[r][off+HD+1]=L;
            } else for(int t=0;t<rows;++t) for(int h=0;h<NQ+4;++h) for(int d=0;d<HD;++d)
                input[r][(size_t(t)*(NQ+4)+h)*HD+d]=float((r+1)*64+h*3+t+int(unit)+(d%17))/32.f;
            rs[r]->input.put(input[r]);
        }
        for(auto& r:rs) r->sender.run();
        for(auto& r:rs) r->sender.sync();
        // CPU expected bytes: each receiver is distinct. Check the entire ring, not just result heads:
        // untouched parity, own-sender slots, unused head capacity, row tails and FP16 slot slack stay sentinel.
        for(int sender=0;sender<n;++sender) for(int receiver=0;receiver<n;++receiver) if(sender!=receiver) {
            const int target=mode==Exchange::Local16?sender:receiver;
            const int owner=mode==Exchange::Local16?receiver:sender;
            const int first=partial?q_begin(receiver,n):q_begin(owner,n);
            const int count=partial?q_count(receiver,n):q_count(owner,n);
            const size_t base=((size_t(unit&(XSLOT-1))*n+owner)*XROWS*per)*sizeof(float);
            for(int t=0;t<rows;++t) for(int hl=0;hl<count;++hl) {
                if(partial) {
                    const size_t src=(size_t(t)*NQ+first+hl)*(HD+2);
                    const int record=h16?XP16:HD+2;
                    const size_t dst=base+(size_t(t)*cap+hl)*record*sizeof(float);
                    for(int d=0;d<HD;++d) {
                        if(h16) store_half(expected[target],dst+d*sizeof(__half),input[sender][src+d]/input[sender][src+HD+1]);
                        else store_float(expected[target],dst+d*sizeof(float),input[sender][src+d]);
                    }
                    const int meta=h16?HD/2:HD;
                    store_float(expected[target],dst+meta*sizeof(float),input[sender][src+HD]);
                    store_float(expected[target],dst+(meta+1)*sizeof(float),input[sender][src+HD+1]);
                } else {
                    const size_t src=(size_t(t)*(NQ+4)+first+hl)*HD;
                    const size_t dst=base+(size_t(t)*cap+hl)*HD*(h16?sizeof(__half):sizeof(float));
                    for(int d=0;d<HD;++d) {
                        if(h16) store_half(expected[target],dst+d*sizeof(__half),input[sender][src+d]/16.f);
                        else store_float(expected[target],dst+d*sizeof(float),input[sender][src+d]);
                    }
                }
            }
        }
        for(int r=0;r<n;++r) {
            CUDA_CHECK(cudaSetDevice(r)); auto& x=*rs[r];
            if (std::memcmp(x.stage.p, expected[r].data(), x.stage.n) != 0)
                for (size_t byte = 0; byte < x.stage.n; ++byte)
                    if (x.stage.p[byte] != expected[r][byte]) {
                        std::fprintf(stderr, "exchange mode=%d ranks=%d unit=%u rank=%d byte=%zu actual=%02x expected=%02x\n",
                                     int(mode), n, unit, r, byte, unsigned(x.stage.p[byte]), unsigned(expected[r][byte]));
                        break;
                    }
            require(std::memcmp(x.stage.p,expected[r].data(),x.stage.n)==0,"exchange payload/padding/ring parity mismatch");
            const auto* flags=reinterpret_cast<const unsigned*>(x.host_flags.p);
            for(int g=0;g<n;++g) require(flags[g]==(g==r?0u:unit),"exchange publish flag or empty-block signalling mismatch");
            CUDA_CHECK(cudaMemcpy(x.recv.p,x.stage.p,x.stage.n,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(x.flags.p,x.host_flags.p,x.host_flags.n,cudaMemcpyHostToDevice));
            if(partial) x.consumer.run();
        }
        if(partial) for(auto& r:rs) r->consumer.sync();
        for(int r=0;r<n;++r) {
            CUDA_CHECK(cudaSetDevice(r)); auto& x=*rs[r];
            if(partial) {
                const auto got=x.output.get(); std::vector<float> actual(got.size()),want(got.size()),without_last(got.size());
                for(int t=0;t<rows;++t) for(int hl=0;hl<q_count(r,n);++hl) for(int d=0;d<HD;++d) {
                    const size_t i=(size_t(t)*q_count(r,n)+hl)*HD+d;
                    const size_t src=(size_t(t)*NQ+q_begin(r,n)+hl)*(HD+2);
                    double a=0,L=0,a4=0,L4=0;
                    for(int g=0;g<n;++g) {
                        const float scale=input[g][src+HD+1];
                        const double w=std::exp(double(input[g][src+HD]-float(n-1)/16.f));
                        const float v=h16 && g!=r?half_round(input[g][src+d]/scale)*scale:input[g][src+d];
                        a+=w*v; L+=w*scale; if(g!=n-1) { a4+=w*v; L4+=w*scale; }
                    }
                    actual[i]=__bfloat162float(got[i]); want[i]=float(a/L*0.5); without_last[i]=float(a4/L4*0.5);
                }
                compare(actual,want,1e-2);
                if(n==5) require(std::fabs(want[0]-without_last[0])>0.005f,"fifth partial not observable in CPU oracle");
                require(x.seq.get()[0]==unit && x.done.get()[0]==0,"combine seq/done mismatch");
            } else CUDA_CHECK(cudaMemcpy(x.seq.p,&unit,sizeof(unit),cudaMemcpyHostToDevice));
            for(unsigned value:x.scnt.get()) require(value==0,"send block counter not reset");
            x.check();
        }
    }
}
// CPU recurrence in transposed production state order [value head][value column][key column].
static std::vector<float> delta_reference(const std::vector<float>& cv,const std::vector<float>& ab,
        const std::vector<float>& A,const std::vector<float>& dt,std::vector<float>& state,
        int rows,int kl,int vl,int k0,int v0) {
    const int ch=2*kl*128+vl*128,ld=dn_ab_ld(vl);
    std::vector<float> out(size_t(rows)*vl*128);
    for(int t=0;t<rows;++t) for(int h=0;h<vl;++h) {
        const int kh=(v0+h)/3-k0; double qq=0,kk=0;
        const size_t row=size_t(t)*ch;
        for(int k=0;k<128;++k) { const double q=cv[row+kh*128+k],v=cv[row+kl*128+kh*128+k]; qq+=q*q; kk+=v*v; }
        const double beta=1/(1+std::exp(-double(ab[size_t(t)*ld+h])));
        const double xa=ab[size_t(t)*ld+vl+h]+dt[h];
        const double decay=std::exp(A[h]*(xa>20?xa:std::log1p(std::exp(xa))));
        for(int d=0;d<128;++d) {
            double mem=0; const size_t base=(size_t(h)*128+d)*128;
            for(int k=0;k<128;++k) { state[base+k]=float(state[base+k]*decay); mem+=state[base+k]*cv[row+kl*128+kh*128+k]/std::sqrt(kk+1e-6); }
            const double delta=(cv[row+2*kl*128+h*128+d]-mem)*beta;
            double value=0;
            for(int k=0;k<128;++k) {
                state[base+k]+=float(cv[row+kl*128+kh*128+k]/std::sqrt(kk+1e-6)*delta);
                value+=state[base+k]*cv[row+kh*128+k]/std::sqrt(qq+1e-6)/std::sqrt(128.0);
            }
            out[(size_t(t)*vl+h)*128+d]=float(value);
        }
    }
    return out;
}
static void deltanet_case(int device,const Plan& p,bool small) {
    CUDA_CHECK(cudaSetDevice(device));
    const int vl=p.v1-p.v0,kl=p.k1-p.k0,ch=2*kl*128+vl*128,ld=dn_ab_ld(vl),rows=small?17:65;
    if(vl==9 || vl==10) require(ld==32,"9/10-head DeltaNet must use stride32");
    const size_t sn=size_t(vl)*128*128;
    std::vector<float> cv(size_t(rows)*ch),ab(size_t(rows)*ld,2048.f),A(vl),dt(vl),initial(sn);
    for(int t=0;t<rows;++t) for(int c=0;c<ch;++c) cv[size_t(t)*ch+c]=float((t*19+c*7+device*3)%101-50)/128.f;
    for(int h=0;h<vl;++h) { A[h]=-0.03f-float(h)/512.f; dt[h]=float(h)/32.f;
        for(int t=0;t<rows;++t) { ab[size_t(t)*ld+h]=float((h+t)%7-3)/8.f; ab[size_t(t)*ld+vl+h]=float((h*3+t)%9-4)/8.f; } }
    for(size_t i=0;i<sn;++i) initial[i]=float(int(i%23)-11)/4096.f;
    // Full DeltaNet front end: padded projection rows, causal conv and parallel 16-row tiles.
    const int projld=ch+vl*128;
    std::vector<float> projected(size_t(rows)*projld,8192.f),history(size_t(3)*ch),final_history(size_t(3)*ch);
    std::vector<bf16> weights(size_t(ch)*4);
    for(int t=0;t<rows;++t) std::copy_n(cv.data()+size_t(t)*ch,ch,projected.data()+size_t(t)*projld);
    for(int c=0;c<ch;++c) {
        float hist[3]={float(c%11-5)/128.f,float(c%7-3)/128.f,float(c%5-2)/128.f};
        for(int j=0;j<3;++j) history[size_t(j)*ch+c]=hist[j];
        for(int j=0;j<4;++j) weights[size_t(c)*4+j]=__float2bfloat16(float(j+1)/8.f);
        for(int t=0;t<rows;++t) {
            const float x=projected[size_t(t)*projld+c];
            float sum=0;
            for(int j=0;j<3;++j) sum+=hist[j]*__bfloat162float(weights[size_t(c)*4+j]);
            sum+=x*__bfloat162float(weights[size_t(c)*4+3]);
            cv[size_t(t)*ch+c]=sum/(1.f+std::exp(-sum));
            hist[0]=hist[1]; hist[1]=hist[2]; hist[2]=x;
        }
        for(int j=0;j<3;++j) final_history[size_t(j)*ch+c]=hist[j];
    }
    Device<float> projection(projected.size()),hist(history.size()),conv_pending(size_t(MAXP)*ch),conv_out(cv.size());
    Device<bf16> conv_weights(weights.size()); Device<int> conv_accepted(1);
    projection.put(projected); hist.put(history); conv_weights.put(weights); conv_pending.zero(); conv_accepted.zero();
    k_dn_conv_d<<<(ch+255)/256,256>>>(projection.p,projld,ch,conv_weights.p,hist.p,conv_pending.p,conv_accepted.p,conv_out.p,rows,true);
    CUDA_CHECK(cudaDeviceSynchronize()); compare(conv_out.get(),cv,1e-4); compare(hist.get(),final_history,1e-4);
    hist.put(history);
    k_dn_conv_par<<<dim3((ch+255)/256,(rows+15)/16),256>>>(projection.p,projld,ch,conv_weights.p,hist.p,conv_pending.p,conv_accepted.p,conv_out.p,rows);
    k_dn_conv_hist<<<(ch+255)/256,256>>>(projection.p,projld,ch,hist.p,conv_pending.p,conv_accepted.p,rows);
    CUDA_CHECK(cudaDeviceSynchronize()); compare(conv_out.get(),cv,1e-4); compare(hist.get(),final_history,1e-4);
    auto reference_state=initial; const auto expected=delta_reference(cv,ab,A,dt,reference_state,rows,kl,vl,p.k0,p.v0);
    Device<float> dc(cv.size()),da(ab.size()),dA(vl),ddt(vl),state(sn),out(size_t(rows)*vl*128),pend(size_t(2)*MAXP*vl*DREC);
    Device<int> accepted(1); Device<unsigned> step(1);
    dc.put(cv); da.put(ab); dA.put(A); ddt.put(dt); accepted.zero(); step.zero(); pend.zero(); state.put(initial);
    // Production prefill recurrence commits all rows at once; CPU also checks final state.
    k_dn_rec_d<<<dim3(vl,16),128>>>(dc.p,ch,kl,vl,p.k0,p.v0,da.p,dA.p,ddt.p,state.p,pend.p,accepted.p,step.p,out.p,rows,true);
    CUDA_CHECK(cudaDeviceSynchronize()); compare(out.get(),expected,1e-4); compare(state.get(),reference_state,1e-4);
    // Deferred decode: commit prior accepted row, alternate pending parity, then commit the last row with m=0.
    state.put(initial); pend.zero(); accepted.zero(); step.zero();
    for(int t=0;t<rows;++t) {
        k_dn_rec_d<<<dim3(vl,16),128>>>(dc.p+size_t(t)*ch,ch,kl,vl,p.k0,p.v0,da.p+size_t(t)*ld,dA.p,ddt.p,state.p,pend.p,accepted.p,step.p,out.p+size_t(t)*vl*128,1);
        k_step_bump<<<1,1>>>(step.p);
        const int one=1; CUDA_CHECK(cudaMemcpy(accepted.p,&one,4,cudaMemcpyHostToDevice));
    }
    k_dn_rec_d<<<dim3(vl,16),128>>>(dc.p,ch,kl,vl,p.k0,p.v0,da.p,dA.p,ddt.p,state.p,pend.p,accepted.p,step.p,out.p,0);
    CUDA_CHECK(cudaDeviceSynchronize()); compare(out.get(),expected,1e-4); compare(state.get(),reference_state,1e-4);
    // Real two-sequence batch with distinct slot states and permuted slot ids, 3 verify rows per sequence.
    constexpr int rps=3,nseg=2;
    Device<float> bs0(sn),bs1(sn),bp0(size_t(2)*MAXP*vl*DREC),bp1(size_t(2)*MAXP*vl*DREC),bout(size_t(nseg)*rps*vl*128);
    Device<float*> states(nseg),pending(nseg); Device<int> slots(nseg*rps),bpn(nseg); Device<unsigned> steps(nseg);
    bs0.put(initial); auto second=initial; for(float& v:second) v*=2; bs1.put(second); bp0.zero(); bp1.zero(); bpn.zero(); steps.zero();
    states.put({bs0.p,bs1.p}); pending.put({bp0.p,bp1.p}); slots.put({1,1,1,0,0,0});
    k_dn_rec_b<<<dim3(vl,16,nseg),128>>>(dc.p,ch,kl,vl,p.k0,p.v0,da.p,dA.p,ddt.p,states.p,pending.p,slots.p,bpn.p,steps.p,bout.p,rps);
    CUDA_CHECK(cudaDeviceSynchronize()); std::vector<float> batch_expected;
    for(int sg=0;sg<nseg;++sg) {
        std::vector<float> scv(cv.begin()+size_t(sg*rps)*ch,cv.begin()+size_t((sg+1)*rps)*ch);
        std::vector<float> sab(ab.begin()+size_t(sg*rps)*ld,ab.begin()+size_t((sg+1)*rps)*ld);
        auto st=sg==0?second:initial; auto want=delta_reference(scv,sab,A,dt,st,rps,kl,vl,p.k0,p.v0);
        batch_expected.insert(batch_expected.end(),want.begin(),want.end());
    }
    compare(bout.get(),batch_expected,1e-4);
    // Actual tensor-core WY chunked API, with explicit value-to-key mapping and stride32 front end.
    Device<float> qk(size_t(rows)*2*kl*128),gb(size_t(rows)*2*vl),chunk_state(2*sn); Device<int> cur(1); cur.zero();
    DeltaShape shape{}; shape.nk=kl; shape.nv=vl; shape.ab_pad=vl; shape.ab_splits=1; shape.ab_rows=rows;
    for(int h=0;h<vl;++h) shape.v2k[h]=(p.v0+h)/3-p.k0;
    std::vector<float> both(2*sn); std::copy(initial.begin(),initial.end(),both.begin()); chunk_state.put(both);
    dn_chunk_init(); dn_chunk_reserve(vl,rows);
    k_dn_qk<<<dim3(rows,kl+1),128>>>(dc.p,ch,kl,vl,p.k0,p.v0,da.p,dA.p,ddt.p,qk.p,gb.p);
    dn_core_chunked(dc.p,ch,shape,qk.p,gb.p,chunk_state.p,0,cur.p,rows,out.p,0,0);
    CUDA_CHECK(cudaDeviceSynchronize()); compare(out.get(),expected,1e-2);
    auto chunk_final=chunk_state.get();
    chunk_final.resize(sn); compare(chunk_final,reference_state,1e-2);
    projection.check(); hist.check(); conv_pending.check(); conv_out.check(); conv_weights.check(); conv_accepted.check();
    dc.check(); da.check(); dA.check(); ddt.check(); state.check(); out.check(); pend.check(); accepted.check(); step.check();
    bs0.check(); bs1.check(); bp0.check(); bp1.check(); bout.check(); states.check(); pending.check(); slots.check(); bpn.check(); steps.check();
    qk.check(); gb.check(); chunk_state.check(); cur.check();
}
int main(int argc,char** argv) {
    int n=5; bool small=false;
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--ranks" && i+1<argc) { char* end=nullptr; const long v=std::strtol(argv[++i],&end,10); if(*end || v<2 || v>5) { std::fprintf(stderr,"--ranks must be 2..5\n"); return 2; } n=int(v); }
        else if(a=="--small") small=true;
        else { std::fprintf(stderr,"unknown/incomplete argument: %s\n",a.c_str()); return 2; }
    }
    try {
        plan_checks();
        int count=0; CUDA_CHECK(cudaGetDeviceCount(&count)); require(count>=n,"not enough distinct CUDA devices for requested ranks");
        for(int r=0;r<n;++r) { CUDA_CHECK(cudaSetDevice(r)); cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,r)); require(prop.canMapHostMemory,"host-mapped communication unavailable"); std::printf("rank %d: CUDA device %d %s\n",r,r,prop.name); }
        collective_checks(n);
        static_assert(XP16==HD/2+2,"TP5 partial FP16 record must contain HD halves + max + sum");
        for(Exchange mode:{Exchange::Q32,Exchange::Q16,Exchange::Local16,Exchange::Partial32,Exchange::Partial16}) exchange_checks(n,small?3:17,mode);
        const auto plans=make_plan(n,std::vector<double>(n,1.0));
        const auto uneven=make_plan(5,std::vector<double>(5,1.0));
        // Chunked workspace is thread-local, just like production's one thread per CUDA worker.
        // Never reuse its cached pointer after switching devices on the same host thread.
        for(int r=0;r<n;++r) {
            std::exception_ptr failure;
            std::thread worker([&] {
                try {
                    deltanet_case(r,plans[r],small);
                    if(n!=5) for(int u=r;u<5;u+=n) deltanet_case(r,uneven[u],small);
                } catch(...) { failure=std::current_exception(); }
            });
            worker.join();
            if(failure) std::rethrow_exception(failure);
        }
        std::printf("flash_tp5_unit: %d real ranks, host LL reduce/gather 70 graph units, FP32/FP16 staged exchanges, DeltaNet passed\n",n);
        std::printf("P2P-specific transport not exercised: host-staged fixture selected\n");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"flash_tp5_unit: %s\n",e.what()); return 1; }
}
