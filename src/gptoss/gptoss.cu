// gpt-oss-120b engine, first version (M2). See gptoss.h and.
#include "gptoss.h"
#include "gptoss_kernels.cuh"
#include "gptoss_sample.cuh"   // [srv-gptoss-capi] sampled decode head
#include "gptoss_dflash.cuh"   // [gptoss-r15-dflash] block drafter kernels

#include <cublas_v2.h>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <array>   // [srv-gptoss-capi]
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>   // [srv-gptoss-capi]

namespace gptoss {

// Every failure is an exception (no abort): the C API turns it into NULL / -1 / 0, gptoss_run prints it.
[[noreturn]] static void fail(const char* file, int line, const char* fmt, ...) {
    char b[1024];
    int n = snprintf(b, sizeof b, "gptoss: ");
    va_list ap;
    va_start(ap, fmt);
    n += vsnprintf(b + n, sizeof b - n, fmt, ap);
    va_end(ap);
    if (n < (int)sizeof b) snprintf(b + n, sizeof b - n, " (%s:%d)", file, line);
    throw std::runtime_error(b);
}
#define GCHECK(cond, ...)                              \
    do {                                               \
        if (!(cond)) fail(__FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e__ = (x);                                                            \
        if (e__ != cudaSuccess) fail(__FILE__, __LINE__, "CUDA error: %s", cudaGetErrorString(e__)); \
    } while (0)
#define CB(x)                                                                         \
    do {                                                                              \
        cublasStatus_t s__ = (x);                                                     \
        if (s__ != CUBLAS_STATUS_SUCCESS) fail(__FILE__, __LINE__, "cuBLAS error %d", (int)s__); \
    } while (0)

static int test_env(const char* k) {   // [gptoss-rel1] test hooks (fault injection); 0 when unset
    const char* v = getenv(k);
    return v && *v ? atoi(v) : 0;
}
// [gptoss-rel3-pin] pinned host memory that may fail under host-RAM pressure: false (error cleared) instead of a throw.
// Test hook GPTOSS_TEST_PIN_FAIL (bit mask): 1 the no-P2P prefill lane staging, 8 that of the last card only (after the
// others were pinned), 2 the first try of the prefill chunk staging, 4 every try of the row / ring copy staging
static bool pin_try(void** p, size_t nb, unsigned flags, int test_bit) {
    *p = nullptr;
    if (test_env("GPTOSS_TEST_PIN_FAIL") & test_bit) return false;
    if (cudaHostAlloc(p, nb, flags) == cudaSuccess) return true;
    cudaGetLastError();
    *p = nullptr;
    return false;
}
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }
static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

// ---------------------------------------------------------------- safetensors index (all shards, mmapped)
struct Tens {
    const uint8_t* p = nullptr;
    size_t n = 0;
    std::string dtype;
    std::vector<long long> shape;
};
struct Files {
    std::map<std::string, Tens> t;
    std::vector<std::pair<void*, size_t>> maps;
    static size_t skip_ws(const std::string& s, size_t i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) i++;
        return i;
    }
    static std::string str_at(const std::string& s, size_t& i) {   // s[i] == '"'
        std::string o;
        for (i++; i < s.size() && s[i] != '"'; i++) {
            if (s[i] == '\\' && i + 1 < s.size()) i++;
            o.push_back(s[i]);
        }
        i++;
        return o;
    }
    static size_t val_of(const std::string& o, const char* key) {
        const std::string k = std::string("\"") + key + "\"";
        size_t i = o.find(k);
        GCHECK(i != std::string::npos, "safetensors header: no %s", key);
        i = skip_ws(o, i + k.size());
        GCHECK(o[i] == ':', "safetensors header");
        return skip_ws(o, i + 1);
    }
    static std::vector<long long> ints_at(const std::string& o, size_t i) {
        std::vector<long long> v;
        GCHECK(o[i] == '[', "safetensors header");
        for (i++; i < o.size() && o[i] != ']';) {
            if (o[i] >= '0' && o[i] <= '9') {
                char* e = nullptr;
                v.push_back(strtoll(o.c_str() + i, &e, 10));
                i = e - o.c_str();
            } else
                i++;
        }
        return v;
    }
    size_t total_bytes() const {   // [gptoss-rel1-cards] the mapped files (the weights as stored)
        size_t b = 0;
        for (const auto& m : maps) b += m.second;
        return b;
    }
    void open_file(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        GCHECK(fd >= 0, "cannot open %s", path.c_str());
        struct stat sb;
        if (fstat(fd, &sb) != 0) {
            ::close(fd);
            GCHECK(false, "fstat %s", path.c_str());
        }
        void* m = sb.st_size ? mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
        ::close(fd);
        GCHECK(m != MAP_FAILED, "mmap %s", path.c_str());
        maps.push_back({m, (size_t)sb.st_size});
        const uint8_t* b = (const uint8_t*)m;
        uint64_t hn = 0;
        if (sb.st_size >= 8) memcpy(&hn, b, 8);
        GCHECK(sb.st_size >= 8 && hn <= (uint64_t)sb.st_size - 8, "bad header in %s", path.c_str());
        const std::string hdr((const char*)b + 8, hn);
        const uint8_t* data = b + 8 + hn;
        size_t i = skip_ws(hdr, 0);
        GCHECK(hdr[i] == '{', "safetensors header");
        i++;
        for (;;) {
            i = skip_ws(hdr, i);
            if (i >= hdr.size() || hdr[i] == '}') break;
            if (hdr[i] == ',') {
                i++;
                continue;
            }
            GCHECK(hdr[i] == '"', "safetensors header key");
            const std::string name = str_at(hdr, i);
            i = skip_ws(hdr, i);
            GCHECK(hdr[i] == ':', "safetensors header");
            i = skip_ws(hdr, i + 1);
            GCHECK(hdr[i] == '{', "safetensors header value");
            size_t j = i;
            int depth = 0;
            for (; j < hdr.size(); j++) {
                if (hdr[j] == '"') {
                    str_at(hdr, j);
                    j--;
                } else if (hdr[j] == '{')
                    depth++;
                else if (hdr[j] == '}' && --depth == 0)
                    break;
            }
            const std::string obj = hdr.substr(i, j - i + 1);
            i = j + 1;
            if (name == "__metadata__") continue;
            Tens x;
            size_t d = val_of(obj, "dtype");
            x.dtype = str_at(obj, d);
            x.shape = ints_at(obj, val_of(obj, "shape"));
            const auto off = ints_at(obj, val_of(obj, "data_offsets"));
            GCHECK(off.size() == 2 && off[0] >= 0 && off[0] <= off[1] && (uint64_t)off[1] <= (uint64_t)sb.st_size - 8 - hn,
                   "bad offsets of %s", name.c_str());
            x.p = data + off[0];
            x.n = off[1] - off[0];
            GCHECK(!t.count(name), "duplicate tensor %s", name.c_str());
            t[name] = x;
        }
    }
    void open_dir(const std::string& dir) {
        std::vector<std::string> fs;
        std::error_code ec;
        GCHECK(std::filesystem::is_directory(dir, ec), "model directory %s not found", dir.c_str());
        for (auto& e : std::filesystem::directory_iterator(dir)) {
            const std::string f = e.path().filename().string();
            if (f.rfind("model-", 0) == 0 && f.size() > 12 && f.substr(f.size() - 12) == ".safetensors") fs.push_back(e.path().string());
        }
        std::sort(fs.begin(), fs.end());
        GCHECK(!fs.empty(), "no model-*.safetensors in %s", dir.c_str());
        for (auto& f : fs) open_file(f);
    }
    // shape: the expected dimensions (their product times the element size must be the byte count)
    const Tens& get(const std::string& name, const char* dtype, size_t bytes, std::initializer_list<long long> shape = {}) const {
        auto it = t.find(name);
        GCHECK(it != t.end(), "tensor %s missing", name.c_str());
        if (shape.size()) GCHECK(it->second.shape == std::vector<long long>(shape), "tensor %s: unexpected shape", name.c_str());
        GCHECK(it->second.dtype == dtype && it->second.n == bytes, "tensor %s: %s %zu bytes, expected %s %zu", name.c_str(),
               it->second.dtype.c_str(), it->second.n, dtype, bytes);
        return it->second;
    }
    ~Files() {
        for (auto& m : maps) munmap(m.first, m.second);
    }
};

// ---------------------------------------------------------------- planner
Plan make_plan(const std::vector<int>& devices, const std::vector<long long>& free_mb) {
    Plan p;
    p.ngpu = (int)devices.size();
    GCHECK(p.ngpu >= 1 && p.ngpu <= 4, "1..4 cards, got %d", p.ngpu);
    const int n = p.ngpu;
    // KV heads: 8 / n each, the remainder to the cards with the most free memory (ties: lower rank)
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return free_mb[a] > free_mb[b]; });
    int kvn[4] = {};
    for (int i = 0; i < n; i++) kvn[i] = NKV / n;
    for (int i = 0; i < NKV % n; i++) kvn[order[i]]++;
    int kv = 0, jb = 0, v = 0;
    for (int i = 0; i < n; i++) {
        p.dev[i] = devices[i];
        p.kv0[i] = kv;
        p.kvn[i] = kvn[i];
        kv += kvn[i];
        p.jb0[i] = jb;
        p.jbn[i] = NBLK / n + (i < NBLK % n ? 1 : 0);
        jb += p.jbn[i];
        p.v0[i] = v;
        p.vn[i] = VOCAB / n + (i < VOCAB % n ? 1 : 0);
        v += p.vn[i];
    }
    return p;
}

// ---------------------------------------------------------------- prefill expert GEMM launch (forms 3 and 4)
// cls: unit class 0..3 = 64 / 32 / 16 / 8 tokens. Form 4 (default): cp.async stages; F(class, 8-token groups, weight
// blocks per tile), picked in the microbench (gptoss_kbench, round 5). The 8-token class of the short down rows
// (30 blocks) runs 5-block tiles (271 vs 296 us per 128 units).
// Round 14 (form 5 = the form-4 kernels with another unit plan): class 4 = units of 65..128 tokens (<16,3>), and the
// gate_up launches may carry the SwiGLU epilogue (ebias != nullptr: `out` is the bf16 activation buffer).
#define G4_FORMS(F) F(0, 8, 5) F(1, 4, 5) F(2, 2, 7) F(3, 1, 7) F(4, 16, 3) F(5, 1, 5)   // 5 = class 3 of the short rows
static void gemm4_attr() {   // per context: the form-4 kernels use more than the 48 KB of static shared memory
#define F(c, n, k) CU(cudaFuncSetAttribute(k_mx_gemm4<n, k, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, g4_smem(n, k, NBLK))); \
    CU(cudaFuncSetAttribute(k_mx_gemm4<n, k, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, g4_smem(n, k, NBLK)));
    G4_FORMS(F)
#undef F
}
static void launch_gemm(int form, int cls, unsigned ny, cudaStream_t st, const uint8_t* cod, int NRW, int KB, const bf16* X, int ldx,
                        const int* un, float* out, const int* cntp, const bf16* ebias = nullptr, const float* ew = nullptr) {
    if (!ny) return;
    const dim3 gr((NRW + 239) / 240, ny);
    if (form >= 4) {
        // a form-4 launch can fail on its configuration (dynamic shared memory above the attribute of this context),
        // and a failed launch would leave stale sums behind without a sign (review r5 L3): ask for this launch's error
        cudaGetLastError();
        switch (cls == 3 && KB <= 32 && !ebias ? 5 : cls) {
#define F(c, n, k) case c: \
            if (ebias) k_mx_gemm4<n, k, 1><<<gr, 480, g4_smem(n, k, KB), st>>>(cod, NRW, KB, X, ldx, un, out, NRW / 2, cntp, ebias, ew); \
            else k_mx_gemm4<n, k, 0><<<gr, 480, g4_smem(n, k, KB), st>>>(cod, NRW, KB, X, ldx, un, out, NRW, cntp); \
            break;
        G4_FORMS(F)
#undef F
        }
        CU(cudaGetLastError());
    } else switch (cls) {
        case 0: k_mx_gemm3<8><<<gr, 480, 0, st>>>(cod, NRW, KB, X, ldx, un, out, NRW, cntp); break;
        case 1: k_mx_gemm3<4><<<gr, 480, 0, st>>>(cod, NRW, KB, X, ldx, un, out, NRW, cntp); break;
        case 2: k_mx_gemm3<2><<<gr, 480, 0, st>>>(cod, NRW, KB, X, ldx, un, out, NRW, cntp); break;
        default: k_mx_gemm3<1><<<gr, 480, 0, st>>>(cod, NRW, KB, X, ldx, un, out, NRW, cntp); break;
        }
}

// ---------------------------------------------------------------- threads
struct Pool {   // one thread per rank; idle workers spin briefly, then sleep-poll (first step after an idle gap: <= 200 us late)
    int n = 0;
    std::vector<std::thread> th;
    std::function<void(int)> fn;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> done{0};
    std::atomic<bool> stop{false};
    std::atomic<bool>* dead = nullptr;   // raised on the first error: barriers stop waiting
    std::mutex mu;
    std::string err;
    void start(int n_, const std::function<void(int)>& init) {
        n = n_;
        for (int r = 0; r < n; r++)
            th.emplace_back([this, r, init] {
                bool ok = true;
                try {
                    init(r);
                } catch (const std::exception& e) {
                    note(e.what());
                    ok = false;
                }
                uint64_t seen = 0;
                for (;;) {
                    long spins = 0;
                    while (gen.load(std::memory_order_acquire) == seen) {
                        if (++spins < 2000000)
                            cpu_relax();
                        else
                            std::this_thread::sleep_for(std::chrono::microseconds(200));
                    }
                    seen = gen.load(std::memory_order_acquire);
                    if (stop.load()) return;
                    try {
                        if (ok) fn(r);
                    } catch (const std::exception& e) {
                        note(e.what());
                    }
                    done.fetch_add(1, std::memory_order_release);
                }
            });
    }
    void note(const char* what) {
        std::lock_guard<std::mutex> g(mu);
        if (err.empty()) err = what;
        if (dead) dead->store(true);
    }
    // [gptoss-rel1-graph] forget a failed run() whose work touched no device state (a failed graph capture): the next
    // run() starts clean; the caller resets the barrier's dead flag
    void clear_error() {
        std::lock_guard<std::mutex> g(mu);
        err.clear();
    }
    void run(const std::function<void(int)>& f) {
        fn = f;
        done.store(0, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_release);
        while (done.load(std::memory_order_acquire) < n) cpu_relax();
        std::lock_guard<std::mutex> g(mu);
        if (!err.empty()) throw std::runtime_error(err);
    }
    void shutdown() {
        if (th.empty()) return;
        stop.store(true);
        gen.fetch_add(1);
        for (auto& t : th) t.join();
        th.clear();
    }
    ~Pool() { shutdown(); }
};
struct Barrier {
    int n = 1;
    std::atomic<int> cnt{0};
    std::atomic<uint64_t> gen{0};
    std::atomic<bool> dead{false};
    // poll: called now and then while waiting (the prefill pass asks its own streams for a device-side error there: a
    // card that only waits makes no CUDA call and would never notice its own fault, review r4 L3); it may throw
    template <class F>
    void wait(F&& poll) {
        const uint64_t g = gen.load(std::memory_order_acquire);
        if (cnt.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            cnt.store(0, std::memory_order_release);
            gen.fetch_add(1, std::memory_order_release);
        } else
            for (unsigned spins = 1; gen.load(std::memory_order_acquire) == g; spins++) {
                if (dead.load(std::memory_order_relaxed)) throw std::runtime_error("gptoss: another card failed during a prefill pass");
                if ((spins & 0xffffu) == 0) poll();
                cpu_relax();
            }
    }
    void wait() {
        wait([] {});
    }
};

// ---------------------------------------------------------------- per-card state
struct LayerW {
    D12 qkv12{}, wo12{};   // 12-bit copies of wqkv and wo (GPTOSS_DENSE=12)
    bf16 *wqkv = nullptr, *bqkv = nullptr, *wo = nullptr, *bo = nullptr, *n1 = nullptr, *n2 = nullptr;
    bf16 *wr = nullptr, *br = nullptr, *bgu = nullptr, *bdn = nullptr;
    float* sinks = nullptr;
    uint8_t *guc = nullptr, *dnc = nullptr;   // expert row records (codes + scales), mx_rs(90) / mx_rs(KB) bytes per row
};

constexpr int MAXLN = 4;   // lanes of a prefill chunk, at most
struct Rank {
    int rank = 0, dev = 0;
    cudaStream_t st = nullptr, st2 = nullptr;   // compute; prefill collectives
    cudaEvent_t evH[MAXLN] = {};   // [gptoss-rel2-hostpf] no peer access: the lane's partial is staged in pinned host memory
    float* hstage = nullptr;       // pinned host rows of this rank [lane][lane_cap][H] fp32: the peers' rows of its partial, then its own rows' sums / normed rows
    cudaEvent_t evP[MAXLN] = {}, evS[MAXLN] = {}, evD[MAXLN] = {};   // per lane: partial ready, own rows summed (row-owner form: and normed), normed rows of all owners ready
    int *rt = nullptr, *roff = nullptr;                  // route records of the two lanes (device routing), expert offsets
    unsigned *rsig = nullptr, *rsig_host = nullptr;      // signature of the pass' routing (compared between the cards)
    cublasHandle_t blas = nullptr;
    int kv0 = 0, kvn = 0, nq = 0, nrows = 0, ncol = 0, jb0 = 0, KB = 0, NR = 0, v0 = 0, vn = 0;
    LayerW L[NL];
    bf16 *nf = nullptr, *lm = nullptr;
    float* invf = nullptr;
    // sequence state
    bf16 *Kf = nullptr, *Vf = nullptr, *Kr = nullptr, *Vr = nullptr;
    // 8-bit store of the full layers (cfg.kv8 == 1, instead of Kf / Vf): [layer][max_ctx + KV8_PAD rows][kvn][64] int8,
    // scales [layer][rows][kvn]; Kq / Vq: bf16 staging rows of a decode step (MAXR rows)
    int8_t *K8 = nullptr, *V8 = nullptr;
    uint32_t* S8 = nullptr;
    uint32_t* K4 = nullptr;   // 12-bit K (cfg.kv8 == 3): nibble plane [layer][rows][kvn][8 words]
    bf16 *Kq = nullptr, *Vq = nullptr;
    int* pos = nullptr;
    unsigned* step = nullptr;
    ArBuf ar{};
    ArAux aux{};
    // two-stage head (plan D5)
    uint8_t* s8 = nullptr;
    float *s8base = nullptr, *lmlo = nullptr, *lmhi = nullptr, *lomax = nullptr, *thr = nullptr, *lbv = nullptr;
    int* lbi = nullptr;
    float* xmax = nullptr;
    ThrBuf tb{};
    Jobs jobs{};
    int *ord = nullptr, *ident = nullptr;
    // work rows (decode uses row 0)
    float *SUM = nullptr, *rinv = nullptr;
    float *Hs = nullptr, *part = nullptr, *PP[3] = {}, *q = nullptr, *apart = nullptr, *rlog = nullptr, *rw = nullptr;
    float *logits = nullptr, *res = nullptr;
    bf16* act = nullptr;
    bf16 *XN = nullptr, *AO = nullptr;
    int* ids = nullptr;
    // prefill scratch
    float *Y = nullptr, *Qf = nullptr, *S = nullptr, *Kg = nullptr, *Vg = nullptr, *Qg = nullptr, *Og = nullptr;
    float *GU = nullptr, *O = nullptr, *pw = nullptr;
    int *pexp = nullptr, *pinv = nullptr, *punits = nullptr;
    bf16 *G = nullptr, *A = nullptr, *Wd = nullptr, *Dd = nullptr, *Ksc = nullptr, *Vsc = nullptr;
    int* prow = nullptr;
    // pinned host
    float *h_host = nullptr, *res_host = nullptr;
    int* pos_host = nullptr;
    cudaGraphExec_t gexec[ASC * (MAXR + 1)] = {};   // decode step graph per (split class, row count)
    // [srv-gptoss-capi] sampled decode head (gptoss_sample.cuh): per-row params, block results, row results, row 0's
    // logits on the host (filtered sampling of a one-row step)
    SampParam *sp = nullptr, *sp_host = nullptr;
    float* pen = nullptr;   // [gptoss-rel2-penalty] [slots][vn] penalty of each token for the slot's reply
    float *sblk = nullptr, *sout = nullptr, *sout_host = nullptr, *slog_host = nullptr;
    cudaGraphExec_t gexec_s[ASC * (MAXR + 1)] = {};   // the sampled step per (split class, row count)
    struct DfRank* df = nullptr;              // [gptoss-r15-dflash] drafter state of this card (gptoss_dflash.inc)
    cudaGraphExec_t gexec_g[ASC * (MAXR + 1)] = {};   // [gptoss-r15-samp2] the sampled step with the certified two-stage head
    unsigned char* stage = nullptr;           // [srv-gptoss-capi] pinned staging of rows_save / rows_load (allocated on first use)
    size_t stage_n = 0;                       // [gptoss-rel3-pin] its bytes (32 MiB, else 4 MiB); 0 with stage_off
    bool stage_off = false;                   // no pinned staging could be had: pageable copies (slower, same bytes)
    bool stage_said = false;                  // the pageable fallback was logged (and not yet left)
    int stage_tries = 0;                      // copies since the last failed try (another try after 256)
    std::map<std::string, size_t> bytes;
    std::vector<void*> dptr, hptr;   // every device / pinned host allocation of this card (freed by ~Impl)
    template <class T>
    T* halloc(size_t bytes_, unsigned flags = cudaHostAllocDefault) {
        void* p = nullptr;
        GCHECK(pin_try(&p, bytes_, flags, 0), "card %d: pinned host memory (%.2f MB) could not be allocated: free host RAM", dev, bytes_ / 1e6);
        hptr.push_back(p);
        return (T*)p;
    }
    size_t free_before = 0, free_mid = 0, free_after = 0, total = 0;

    template <class T>
    T* alloc(size_t n, const char* cat, bool zero = false) {
        T* p = nullptr;
        const size_t b = std::max<size_t>(n * sizeof(T), 16);
        const cudaError_t e = cudaMalloc(&p, b);
        if (e != cudaSuccess) {
            cudaGetLastError();
            fail(__FILE__, __LINE__, "card %d: cannot allocate %.1f MiB of %s (%s)", dev, b / 1048576.0, cat, cudaGetErrorString(e));
        }
        dptr.push_back(p);
        if (zero) CU(cudaMemset(p, 0, b));
        bytes[cat] += b;
        return p;
    }
    template <class T>
    T* up(const void* src, size_t n, const char* cat) {
        T* p = alloc<T>(n, cat);
        CU(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
};

// route record of a lane with T rows (k_route_plan): offsets in ints, class bases and grid bounds of the unit lists.
// A 64-token unit holds >= 33 rows; a remainder class has at most one unit per expert.
// Plan 5 (round 14): class 4 = units of 65..128 rows (the experts' full 128-row units and remainders above 64), class 0
// = a remainder of 33..64 rows; then every class but 4 has at most one unit per expert.
struct RouteLay {
    int row, w, exp, inv, units, ub[5], bound[5], ints;
};
static RouteLay route_lay(int T, int plan) {
    const int n4 = 4 * T;
    RouteLay y;
    y.row = 16;
    y.w = y.row + n4;
    y.exp = y.w + n4;
    y.inv = y.exp + n4;
    y.units = y.inv + n4;
    y.bound[0] = plan == 5 ? std::min(NE, n4 / 33 + 1) : n4 / 33 + 1;
    y.bound[1] = std::min(NE, n4 / 17 + 1);
    y.bound[2] = std::min(NE, n4 / 9 + 1);
    y.bound[3] = std::min(NE, n4);
    y.bound[4] = plan == 5 ? n4 / 65 + 1 : 0;
    y.ub[0] = 0;
    for (int c = 1; c < 5; c++) y.ub[c] = y.ub[c - 1] + y.bound[c - 1];
    y.ints = y.units + 3 * (y.ub[4] + y.bound[4]);
    return y;
}
static size_t route_ints(size_t T) { return (size_t)std::max(route_lay((int)T, 4).ints, route_lay((int)T, 5).ints); }

struct Impl {
    Config cfg;
    Plan plan;
    Files files;
    std::vector<Rank> R;
    Pool pool;
    Barrier bar;
    int ngpu = 0, pos = 0;
    double load_s = 0;
    Dump dump;
    const uint16_t* embed = nullptr;
    float att = 1.f;
    float invf[32];
    // prefill routing shared by rank 0 (pinned, portable)
    int* sh_ids = nullptr;
    float* sh_rw = nullptr;
    int* sh_rows = nullptr;
    float* sh_w = nullptr;
    float* sh_h = nullptr;   // [chunk][H] embedding rows of the chunk
    int sh_off[NE + 1] = {};
    int *sh_exp = nullptr, *sh_inv = nullptr, *sh_units = nullptr;   // fused path: expert per sorted row, sorted index of (t, k), units
    int sh_nunits[MAXLN] = {}, sh_ncls[MAXLN][4] = {};
    bool p2p = true;
    void* ar_host = nullptr;   // host all-reduce words (no peer access)
    size_t ar_host_nb = 0;
    long ar_launches = 0, ar_clears = 0;
    // [gptoss-rel2-tagwrap] before every launch that advances the step counter (a decode step, a drafter pass): see
    // Config::ar_clear. All cards are idle here (the launches are synchronous), and every card's words are zero
    // before any card launches again: a peer's store never meets a later clear. A zero tag is no step's tag
    // (collective + 1 is 1 .. 2 NL + drafter layers < 256, the threshold's tag is odd). One call = one k_begin launch.
    void ar_epoch() {
        static_assert(2 * NL + 2 * DF_L + 2 < 256, "a collective's index + 1 fits the tag's low byte");
        if (++ar_launches < cfg.ar_clear) return;
        ar_launches = 0;
        ar_clears++;
        pool.run([this](int g) {
            Rank& r = R[g];
            if (p2p) {
                CU(cudaMemsetAsync(r.ar.rd, 0, (size_t)AR_SLOTS * ngpu * MAXR * ARW * sizeof(uint4), r.st));
                if (r.tb.rd) CU(cudaMemsetAsync(r.tb.rd, 0, (size_t)4 * MAXR * sizeof(uint2), r.st));
            }
            CU(cudaMemsetAsync(r.aux.ss, 0, 8 * sizeof(uint2), r.st));
            CU(cudaMemsetAsync(r.aux.ready, 0, 8 * sizeof(uint2), r.st));
            CU(cudaMemsetAsync(r.aux.lg, 0, NE * sizeof(uint2), r.st));
            CU(cudaStreamSynchronize(r.st));
        });
        if (!p2p && ar_host) memset(ar_host, 0, ar_host_nb);
    }
    bool graph_ready[ASC * (MAXR + 1)] = {};
    // [srv-gptoss-capi] the bound sequence (Engine::bind: slot, first full-layer row, row limit) and the sampled head:
    // set_sampling for one-row decode steps (samp / spv), srows = the next step_rows runs the sampled head with srow[]
    int b_slot = 0, b_base = 0, lim = 0;
    long long df_tw = 0, df_last = 0;   // [gptoss-r15-dflash] rows written to the tap ring by all steps; first row of the last step
    int df_last_n = 0;
    int df_rs[MAXR] = {}, df_rp[MAXR] = {};   // slot and position of each row of the last step
    void df_clear(int slot, int a, int b);     // (review M3) store rows of the positions [a, b) that sit in the ring: zeroed
    void df_tap(Rank& r, int L, int R);
    static int df_tapi(int L) {
        for (int i = 0; i < DF_NTAP; i++)
            if (DF_TAP_LAYER[i] == L) return i;
        return -1;
    }
    // the prefill's own tap path: row-owner collective on lanes, no dump (a pass with a dump leaves the context to the caller)
    bool df_pf_live() const { return dfh && lanes_on() && cfg.pf_owner && !dump.layers && !dump.hfinal && R[0].df && R[0].df->ptap; }
    bool samp = false, srows = false, graph_s_ready[ASC * (MAXR + 1)] = {}, graph_g_ready[ASC * (MAXR + 1)] = {};
    SampParam spv{}, srow[MAXR] = {};

    void load_rank(Rank& r);
    // [gptoss-r15-dflash] (gptoss_dflash.inc)
    Files dfiles;
    struct DfHost* dfh = nullptr;
    void df_load(Rank& r);
    void df_free(Rank& r);
    void df_launch(Rank& r, int slot, bool host_taps, int ntg);
    void df_pass(int slot, int na, bool host_taps, int B, int bpos, int token, int* out);
    void df_flush(int slot, int B, int token, int* out);
    void alloc_ctx(Rank& r);
    size_t smax = 0;            // prefill score scratch, floats
    int pf_rows = 0;            // sgemm prefill attention: rows of one prefilled sequence its scratch is sized for (review r11 P6)
    std::atomic<bool> broken{false};   // an error left the cards in an unknown state: every later call fails
    long long steps = 0, graph_fails = 0;                 // [gptoss-rel1-graph] decode steps; failed captures
    long long graph_retry[3][MAXR + 1] = {};             // step from which a failed capture is tried again (head mode x rows)
    bool soft_rows = false;            // [gptoss-rel1-rowfault] step_rows: a non-finite row gets -1 instead of throwing
    std::vector<void*> shost;
    ~Impl();
    // smode: 0 greedy head, 1 sampled with the full head, 2 sampled with the certified two-stage head ([gptoss-r15-samp2])
    void launch_step(Rank& r, bool dumping, int R, int smode = 0, int nsg = NSPLIT);
    void prefill_chunk(Rank& r, int pos0, int T, bool last, int call_pos0);
    // ---- lanes of a prefill chunk (round 14). lane_cap = the most rows one lane can hold: the scratch of one compute
    // step is sized by it, the row-indexed buffers by the chunk. One lane under 512 rows, then as few lanes as fit
    // (two up to 2 * lane_cap rows; a chunk of <= 4,096 rows keeps the two-lane split of rounds 3-13).
    int lane_cap = 0;        // rows
    size_t rt_stride = 0;    // ints of one lane's route record
    size_t pp_lane = 0;      // rows of a lane's region in the peers' partial buffers PP (its own rows only)
    bool lanes_on() const { return cfg.pf_lanes && cfg.pf_rs && cfg.pf_fused && (p2p || cfg.pf_hostlanes); }
    int nlanes(int Tc) const {
        if (!lanes_on() || Tc < 512) return 1;
        if (cfg.pf_nlanes > 0) return std::min(cfg.pf_nlanes, MAXLN);
        return std::max(2, (Tc + lane_cap - 1) / lane_cap);
    }
    void ar_lane(Rank& r, const struct Lane& ln, const bf16* nw, int tap = -1);
    void df_prefill_ctx(Rank& r, const struct Lane* lanes, int nl, int pos0c, int Tc);   // [gptoss-r15-dflash]
    void df_prefill_plan(Rank& r, const struct Lane* lanes, int nl, int pos0c, int Tc);
    void ar_rows(Rank& r, int T, const bf16* nw);
    // wait for a stream inside a prefill pass without blocking in the driver: with events a card's streams depend on
    // the other cards' streams, so a card must leave when a peer failed (review pf3 L2) instead of waiting for work
    // that may never complete
    void meet(Rank& r) {
        bar.wait([&] {
            for (cudaStream_t s : {r.st, r.st2}) {
                const cudaError_t e = cudaStreamQuery(s);
                if (e != cudaSuccess && e != cudaErrorNotReady) CU(e);
            }
        });
    }
    void drain(cudaStream_t s) {
        for (long spins = 0;; spins++) {
            const cudaError_t e = cudaStreamQuery(s);
            if (e == cudaSuccess) return;
            if (e != cudaErrorNotReady) CU(e);
            if (bar.dead.load(std::memory_order_relaxed)) throw std::runtime_error("gptoss: another card failed during a prefill pass");
            if (spins < 4000000)
                cpu_relax();
            else
                std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    void embed_row(int tok, float* out) const {
        const uint16_t* e = embed + (size_t)tok * H;
        for (int i = 0; i < H; i++) {
            const uint32_t u = (uint32_t)e[i] << 16;
            memcpy(out + i, &u, 4);
        }
    }
    int merge_res(int row = 0) {
        if (bar.dead.load()) broken.store(true);
        float bv = 0;
        int bi = -1;
        for (int g = 0; g < ngpu; g++) {
            const float v = R[g].res_host[2 * row];
            int i;
            memcpy(&i, R[g].res_host + 2 * row + 1, 4);
            if (bi < 0 || v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        GCHECK(bi >= 0 && bi < VOCAB && std::isfinite(bv) && bv > -1e37f, "the logits are not finite (row %d)", row);
        return bi;
    }
};

static void yarn(float* invf, float* att) {
    const double base = 150000.0, factor = 32.0, orig = 4096.0, dim = HD;
    auto corr = [&](double nrot) { return dim * std::log(orig / (nrot * 2 * M_PI)) / (2 * std::log(base)); };
    const float low = (float)std::max(corr(32.0), 0.0), high = (float)std::min(corr(1.0), dim - 1);
    for (int i = 0; i < HD / 2; i++) {
        const float pf = powf((float)base, (float)(2 * i) / (float)dim);
        const float extra = 1.f / pf, inter = 1.f / ((float)factor * pf);
        const float ramp = std::min(std::max(((float)i - low) / (high - low), 0.f), 1.f);
        const float ef = 1.f - ramp;
        invf[i] = inter * (1.f - ef) + extra * ef;
    }
    *att = (float)(0.1 * std::log(factor) + 1.0);
}

// 12-bit repack of a bf16 matrix [N][K] (layout: gptoss_kernels.cuh D12; tools/gptoss/gptoss_dense12.py checks it)
struct Enc12 {
    std::vector<uint8_t> bp, np;
    std::vector<float> sc;
    std::vector<int> pptr;
    std::vector<uint16_t> pcol, pval;
};
static void encode12(const uint16_t* w, int N, int K, Enc12& o) {
    o.bp.assign((size_t)N * K, 0);
    o.np.assign((size_t)N * K / 2, 0);
    o.sc.resize(N);
    o.pptr.assign(N + 1, 0);
    o.pcol.clear();
    o.pval.clear();
    for (int r = 0; r < N; r++) {
        const uint16_t* wr = w + (size_t)r * K;
        int base = 0;
        for (int i = 0; i < K; i++) base = std::max(base, (wr[i] >> 7) & 0xFF);
        o.sc[r] = std::ldexp(1.f, base - 127);
        uint8_t* b = o.bp.data() + (size_t)r * K;
        uint8_t* n = o.np.data() + (size_t)r * (K / 2);
        for (int i = 0; i < K; i++) {
            const int e = (wr[i] >> 7) & 0xFF, off = base - e;
            const uint8_t sg = (wr[i] >> 8) & 0x80;
            if (e == 0 || off >= 15) {
                b[i] = sg | (15 << 3);
                if (wr[i] & 0x7FFF) {   // a true zero needs no patch
                    o.pcol.push_back((uint16_t)i);
                    o.pval.push_back(wr[i]);
                }
            } else {
                b[i] = sg | (off << 3) | ((wr[i] & 0x7F) >> 4);
                n[i / 2] |= (wr[i] & 0xF) << (4 * (i & 1));
            }
        }
        o.pptr[r + 1] = (int)o.pcol.size();
    }
}
// ---------------------------------------------------------------- load
void Impl::load_rank(Rank& r) {
    const int c = r.rank;
    r.kv0 = plan.kv0[c];
    r.kvn = plan.kvn[c];
    r.nq = r.kvn * (NH / NKV);
    r.nrows = (r.nq + 2 * r.kvn) * HD;
    r.ncol = r.nq * HD;
    r.jb0 = plan.jb0[c];
    r.KB = plan.jbn[c];
    r.NR = 2 * r.KB * 32;
    r.v0 = plan.v0[c];
    r.vn = plan.vn[c];
    GCHECK(r.KB <= 45 && r.kvn >= 1, "plan");
    if (r.KB > 32) cfg.row_union = false;   // the union down kernel keeps the single-row sum order for one block per lane only
    CU(cudaMemGetInfo(&r.free_before, &r.total));
    CU(cudaStreamCreate(&r.st));
    CU(cudaStreamCreate(&r.st2));
    for (int l = 0; l < MAXLN; l++) {
        CU(cudaEventCreateWithFlags(&r.evP[l], cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&r.evH[l], cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&r.evS[l], cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&r.evD[l], cudaEventDisableTiming));
    }
    CB(cublasCreate(&r.blas));
    if (cfg.pf_gemm >= 4) gemm4_attr();   // review r5 L4: nothing of form 4 runs under GPTOSS_PF_GEMM=3
    CB(cublasSetStream(r.blas, r.st));
    const size_t grs = mx_rs(NBLK), drs = mx_rs(r.KB);
    const size_t gun = (size_t)NE * r.NR * grs, dnn = (size_t)NE * H * drs;
    uint8_t* tmp = r.halloc<uint8_t>(std::max(gun, dnn));
    std::vector<uint16_t> small((size_t)std::max(NE * r.NR, H * r.ncol));
    auto bf = [&](const std::string& n, size_t elems) { return (const uint16_t*)files.get(n, "BF16", elems * 2).p; };
    for (int L = 0; L < NL; L++) {
        const std::string p = "model.layers." + std::to_string(L) + ".";
        LayerW& w = r.L[L];
        // attention: [q rows of this card's heads | k rows | v rows]
        const uint16_t* wq = bf(p + "self_attn.q_proj.weight", (size_t)NH * HD * H);
        const uint16_t* wk = bf(p + "self_attn.k_proj.weight", (size_t)NKV * HD * H);
        const uint16_t* wv = bf(p + "self_attn.v_proj.weight", (size_t)NKV * HD * H);
        w.wqkv = r.alloc<bf16>((size_t)r.nrows * H, "attention");
        CU(cudaMemcpy(w.wqkv, wq + (size_t)r.kv0 * 8 * HD * H, (size_t)r.ncol * H * 2, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(w.wqkv + (size_t)r.ncol * H, wk + (size_t)r.kv0 * HD * H, (size_t)r.kvn * HD * H * 2, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(w.wqkv + (size_t)(r.ncol + r.kvn * HD) * H, wv + (size_t)r.kv0 * HD * H, (size_t)r.kvn * HD * H * 2,
                      cudaMemcpyHostToDevice));
        const uint16_t* bq = bf(p + "self_attn.q_proj.bias", NH * HD);
        const uint16_t* bk = bf(p + "self_attn.k_proj.bias", NKV * HD);
        const uint16_t* bv = bf(p + "self_attn.v_proj.bias", NKV * HD);
        memcpy(small.data(), bq + r.kv0 * 8 * HD, r.ncol * 2);
        memcpy(small.data() + r.ncol, bk + r.kv0 * HD, r.kvn * HD * 2);
        memcpy(small.data() + r.ncol + r.kvn * HD, bv + r.kv0 * HD, r.kvn * HD * 2);
        w.bqkv = r.up<bf16>(small.data(), r.nrows, "small");
        const uint16_t* wo = bf(p + "self_attn.o_proj.weight", (size_t)H * NH * HD);
        for (int i = 0; i < H; i++) memcpy(small.data() + (size_t)i * r.ncol, wo + (size_t)i * NH * HD + r.kv0 * 8 * HD, r.ncol * 2);
        w.wo = r.up<bf16>(small.data(), (size_t)H * r.ncol, "attention");
        if (cfg.dense12) {
            Enc12 e;
            auto put = [&](const uint16_t* src, int N, int K, D12& d) {
                encode12(src, N, K, e);
                d.bp = r.up<uint8_t>(e.bp.data(), e.bp.size(), "attention12");
                d.np = r.up<uint8_t>(e.np.data(), e.np.size(), "attention12");
                d.sc = r.up<float>(e.sc.data(), N, "small");
                d.pptr = r.up<int>(e.pptr.data(), N + 1, "small");
                e.pcol.push_back(0);
                e.pval.push_back(0);
                d.pcol = r.up<uint16_t>(e.pcol.data(), e.pcol.size(), "small");
                d.pval = r.up<uint16_t>(e.pval.data(), e.pval.size(), "small");
            };
            std::vector<uint16_t> qkv((size_t)r.nrows * H);
            memcpy(qkv.data(), wq + (size_t)r.kv0 * 8 * HD * H, (size_t)r.ncol * H * 2);
            memcpy(qkv.data() + (size_t)r.ncol * H, wk + (size_t)r.kv0 * HD * H, (size_t)r.kvn * HD * H * 2);
            memcpy(qkv.data() + (size_t)(r.ncol + r.kvn * HD) * H, wv + (size_t)r.kv0 * HD * H, (size_t)r.kvn * HD * H * 2);
            put(qkv.data(), r.nrows, H, w.qkv12);
            put(small.data(), H, r.ncol, w.wo12);   // small still holds this card's o_proj columns
        }
        w.bo = r.up<bf16>(bf(p + "self_attn.o_proj.bias", H), H, "small");
        w.n1 = r.up<bf16>(bf(p + "input_layernorm.weight", H), H, "small");
        w.n2 = r.up<bf16>(bf(p + "post_attention_layernorm.weight", H), H, "small");
        const uint16_t* sk = bf(p + "self_attn.sinks", NH);
        float sf[NH];
        for (int i = 0; i < r.nq; i++) {
            const uint32_t u = (uint32_t)sk[r.kv0 * 8 + i] << 16;
            memcpy(sf + i, &u, 4);
        }
        w.sinks = r.up<float>(sf, r.nq, "small");
        w.wr = r.up<bf16>(bf(p + "mlp.router.weight", (size_t)NE * H), (size_t)NE * H, "router");
        w.br = r.up<bf16>(bf(p + "mlp.router.bias", NE), NE, "small");
        // experts: gate_up rows [2 j0, 2 j1) of every expert, down blocks [jb0, jb0 + KB) of every row
        const uint8_t* gc = files.get(p + "mlp.experts.gate_up_proj_blocks", "U8", (size_t)NE * 2 * H * NBLK * 16, {NE, 2 * H, NBLK, 16}).p;
        const uint8_t* gs = files.get(p + "mlp.experts.gate_up_proj_scales", "U8", (size_t)NE * 2 * H * NBLK).p;
        const uint8_t* dc = files.get(p + "mlp.experts.down_proj_blocks", "U8", (size_t)NE * H * NBLK * 16, {NE, H, NBLK, 16}).p;
        const uint8_t* ds = files.get(p + "mlp.experts.down_proj_scales", "U8", (size_t)NE * H * NBLK).p;
        const size_t r0 = (size_t)2 * r.jb0 * 32;
        // row records: codes, 4-bit scale offsets, the row's smallest scale (gptoss_kernels.cuh, mx_rs)
        auto record = [&](uint8_t* rec, const uint8_t* codes, const uint8_t* sc, int KB) {
            memcpy(rec, codes, (size_t)KB * 16);
            uint8_t lo = 255, hi = 0;
            for (int b2 = 0; b2 < KB; b2++) {
                lo = std::min(lo, sc[b2]);
                hi = std::max(hi, sc[b2]);
            }
            GCHECK(hi - lo <= 15 && lo >= 1, "MXFP4 scale range %d in one row (layer %d): the 4-bit scale offsets cannot hold it", hi - lo, L);
            uint8_t* t = rec + (size_t)KB * 16;
            memset(t, 0, mx_rs(KB) - (size_t)KB * 16);
            for (int b2 = 0; b2 < KB; b2++) t[b2 >> 1] |= (uint8_t)((sc[b2] - lo) << ((b2 & 1) * 4));
            t[(KB + 1) >> 1] = lo;
        };
        for (int e = 0; e < NE; e++)
            for (int row = 0; row < r.NR; row++) {
                const size_t src = (size_t)e * 2 * H + r0 + row;
                record(tmp + ((size_t)e * r.NR + row) * grs, gc + src * 1440, gs + src * 90, NBLK);
            }
        w.guc = r.up<uint8_t>(tmp, gun, "experts");
        for (size_t er = 0; er < (size_t)NE * H; er++) record(tmp + er * drs, dc + (er * NBLK + r.jb0) * 16, ds + er * NBLK + r.jb0, r.KB);
        w.dnc = r.up<uint8_t>(tmp, dnn, "experts");
        const uint16_t* bg = bf(p + "mlp.experts.gate_up_proj_bias", (size_t)NE * 2 * H);
        for (int e = 0; e < NE; e++) memcpy(small.data() + (size_t)e * r.NR, bg + (size_t)e * 2 * H + r0, r.NR * 2);
        w.bgu = r.up<bf16>(small.data(), (size_t)NE * r.NR, "small");
        w.bdn = r.up<bf16>(bf(p + "mlp.experts.down_proj_bias", (size_t)NE * H), (size_t)NE * H, "small");
        if (cfg.verbose && c == 0 && (L % 6 == 5)) fprintf(stderr, "gptoss: loaded layers 0..%d\n", L);
    }
    r.hptr.pop_back();
    CU(cudaFreeHost(tmp));
    r.nf = r.up<bf16>(bf("model.norm.weight", H), H, "small");
    r.lm = r.up<bf16>(bf("lm_head.weight", (size_t)VOCAB * H) + (size_t)r.v0 * H, (size_t)r.vn * H, "lm_head");
    r.invf = r.up<float>(invf, 32, "small");
    // sequence state
    const size_t kvrow = (size_t)r.kvn * HD;
    // ring sets [slot][sliding layer][256 rows] (RING_ROWS rows per set)
    r.Kr = r.alloc<bf16>((size_t)cfg.slots * RING_ROWS * kvrow, "kv", true);
    r.Vr = r.alloc<bf16>((size_t)cfg.slots * RING_ROWS * kvrow, "kv", true);
    r.aux.ss = r.alloc<uint2>(8, "scratch", true);
    r.aux.ready = r.alloc<uint2>(8, "scratch", true);
    r.aux.lg = r.alloc<uint2>(NE, "scratch", true);
    r.ord = r.alloc<int>(4 * MAXR, "scratch", true);     // the step's slots by expert id (k_order)
    r.ident = r.alloc<int>(4 * MAXR, "scratch", true);   // 0, 1, 2, ..: the order of a one-row step
    {
        int id[4 * MAXR];
        for (int i = 0; i < 4 * MAXR; i++) id[i] = i;
        CU(cudaMemcpy(r.ident, id, sizeof id, cudaMemcpyHostToDevice));
    }
    r.jobs.nu = r.alloc<int>(1, "scratch", true);
    r.jobs.ue = r.alloc<int>(4 * MAXR, "scratch", true);
    r.jobs.ucnt = r.alloc<int>(4 * MAXR, "scratch", true);
    r.jobs.uslot = r.alloc<int>(4 * MAXR * MAXR, "scratch", true);
    if (cfg.lm2) {
        // screen plane of this card's lm_head rows (see k_lm_screen)
        const uint16_t* lw = bf("lm_head.weight", (size_t)VOCAB * H) + (size_t)r.v0 * H;
        std::vector<uint8_t> sc((size_t)r.vn * H);
        std::vector<float> sb((size_t)2 * r.vn);
        for (int row = 0; row < r.vn; row++) {
            const uint16_t* wr = lw + (size_t)row * H;
            int base = 0;
            for (int i = 0; i < H; i++) base = std::max(base, (wr[i] >> 7) & 0xFF);
            uint8_t* o = sc.data() + (size_t)row * H;
            int n15 = 0;
            for (int i = 0; i < H; i++) {
                const int e = (wr[i] >> 7) & 0xFF, off = base - e;
                const uint8_t sg = (wr[i] >> 8) & 0x80;
                const bool out = e == 0 || off >= 15;
                o[i] = out ? sg : (uint8_t)(sg | ((15 - off) << 3) | ((wr[i] & 0x7F) >> 4));
                if (out && (wr[i] & 0x7FFF)) n15++;
            }
            sb[2 * row] = std::ldexp(1.f, base - 127);
            // an out-of-range weight is below 2^(base - 14) (15+ exponents under the largest) or a subnormal (below 2^-126)
            sb[2 * row + 1] = (float)n15 * std::ldexp(1.f, std::max(base - 14, 1) - 127);
        }
        r.s8 = r.up<uint8_t>(sc.data(), sc.size(), "lm_head");
        r.s8base = r.up<float>(sb.data(), sb.size(), "small");
        r.xmax = r.alloc<float>(MAXR, "scratch");
        r.lmlo = r.alloc<float>((size_t)MAXR * r.vn, "scratch");
        r.lmhi = r.alloc<float>((size_t)MAXR * r.vn, "scratch");
        r.lomax = r.alloc<float>((size_t)MAXR * ((r.vn + 31) / 32), "scratch");
        r.thr = r.alloc<float>(MAXR, "scratch");
        r.lbv = r.alloc<float>((size_t)MAXR * ((r.vn + 31) / 32 + 8), "scratch");   // rounded up to whole blocks of 8 warps
        r.lbi = r.alloc<int>((size_t)MAXR * ((r.vn + 31) / 32 + 8), "scratch");   // rounded up to whole blocks of 8 warps
        if (p2p) r.tb.rd = r.alloc<uint2>(4 * MAXR, "scratch", true);
    }
    r.pos = r.alloc<int>(3 * MAXR + 4, "scratch", true);   // the step's row table (gptoss_kernels.cuh); [0] = position of row 0
    r.step = r.alloc<unsigned>(1, "scratch", true);
    if (cfg.step0) CU(cudaMemcpy(r.step, &cfg.step0, 4, cudaMemcpyHostToDevice));   // [gptoss-rel2-tagwrap] (tests)
    if (p2p) r.ar.rd = r.alloc<uint4>((size_t)AR_SLOTS * ngpu * MAXR * ARW, "scratch", true);
    // Round 14: the row-indexed buffers (residual, partials, sums, normed rows) hold a chunk, TC rows; everything one
    // compute step uses holds one lane, T rows (lane_cap). The peers' partials are kept for the own rows only (one
    // region per lane; the pull form GPTOSS_PF_AR=pull copies whole partials and keeps full buffers).
    const size_t TC = std::max(cfg.chunk, MAXR);   // rows of a prefill chunk or of a decode step
    const size_t T = std::max(lane_cap, MAXR);     // rows of one lane
    const size_t TP = !cfg.pf_rs ? TC : std::max<size_t>(lanes_on() ? (size_t)nlanes(cfg.chunk) * pp_lane : 0, TC / ngpu + 2);   // a region per lane a chunk can have (review r14 L6)
    r.Hs = r.alloc<float>(TC * H, "scratch");
    r.part = r.alloc<float>(TC * H, "scratch");
    r.SUM = r.alloc<float>(TC * H, "scratch");
    r.rinv = r.alloc<float>(T * 8, "scratch");
    for (int k = 0; k + 1 < ngpu; k++) r.PP[k] = r.alloc<float>(TP * H, "scratch");
    r.XN = r.alloc<bf16>(TC * H, "scratch");
    r.AO = r.alloc<bf16>(T * r.ncol, "scratch");
    r.q = r.alloc<float>((size_t)MAXR * r.ncol, "scratch");
    r.apart = r.alloc<float>((size_t)MAXR * r.nq * NSPLIT * APW, "scratch");
    r.rlog = r.alloc<float>(T * NE, "scratch");
    r.ids = r.alloc<int>(T * 4, "scratch");
    r.rw = r.alloc<float>(T * 4, "scratch");
    r.act = r.alloc<bf16>((size_t)MAXR * 4 * r.NR / 2, "scratch");
    r.logits = r.alloc<float>((size_t)MAXR * r.vn, "scratch");
    r.res = r.alloc<float>(2 * MAXR, "scratch");
    r.Y = r.alloc<float>(T * r.nrows, "scratch");
    r.Qf = r.alloc<float>(T * r.ncol, "scratch");
    r.Qg = r.alloc<float>(T * 512, "scratch");
    r.Og = r.alloc<float>(T * 512, "scratch");
    const size_t TF = cfg.pf_fused ? 4 * T : T;   // fused: all sorted rows of a chunk at once
    // the fp32 gate_up sums (off-switch forms only since round 14) are dead once the activations are made, before the
    // down sums are written: one buffer for both (one card: the gate_up rows, 5,760, are the larger side; review r14 M2)
    r.O = r.alloc<float>(TF * std::max<size_t>(H, r.NR), "scratch");
    r.GU = r.O;
    if (cfg.pf_fused) {
        r.pexp = r.alloc<int>(T * 4, "scratch");
        r.pinv = r.alloc<int>(T * 4, "scratch");
        r.punits = r.alloc<int>((T * 4 / 32 + NE) * 3, "scratch");
    }
    r.rt = r.alloc<int>(MAXLN * rt_stride, "scratch");
    r.roff = r.alloc<int>(NE + 1, "scratch");
    r.rsig = r.alloc<unsigned>(4, "scratch");
    r.rsig_host = r.halloc<unsigned>(4);
    r.pw = r.alloc<float>(T * 4, "scratch");
    r.prow = r.alloc<int>(T * 4, "scratch");
    r.G = r.alloc<bf16>(T * 4 * H, "scratch");
    r.A = r.alloc<bf16>(TF * (r.NR / 2), "scratch");
    r.Wd = r.alloc<bf16>((size_t)r.NR * H, "scratch");
    r.Dd = r.alloc<bf16>((size_t)H * r.KB * 32, "scratch");
    r.Ksc = r.alloc<bf16>((127 + T) * kvrow, "scratch");
    r.Vsc = r.alloc<bf16>((127 + T) * kvrow, "scratch");
    r.h_host = r.halloc<float>((size_t)MAXR * H * sizeof(float));
    r.res_host = r.halloc<float>(8 * MAXR);
    r.pos_host = r.halloc<int>((3 * MAXR + 4) * 4);
    {   // [srv-gptoss-capi]
        const int nb = (r.vn + SAMP_ROWS - 1) / SAMP_ROWS;
        GCHECK(nb * SAMP_K <= 4096, "sampled head: %d logit rows per card exceed k_samp_b's 4096 candidates (3 or more cards)", r.vn);
        r.sp = r.alloc<SampParam>(MAXR, "scratch");
        r.pen = r.alloc<float>((size_t)cfg.slots * r.vn, "scratch", true);
        r.sp_host = r.halloc<SampParam>(MAXR * sizeof(SampParam));
        r.sblk = r.alloc<float>((size_t)MAXR * nb * SAMP_OUT, "scratch");
        r.sout = r.alloc<float>((size_t)MAXR * SAMP_OUT, "scratch");
        r.sout_host = r.halloc<float>((size_t)MAXR * SAMP_OUT * 4);
        r.slog_host = r.halloc<float>((size_t)r.vn * 4, cudaHostAllocMapped | cudaHostAllocPortable);
    }
    if (!cfg.dflash_dir.empty()) df_load(r);   // [gptoss-r15-dflash] before the K/V pool is sized
    CU(cudaMemGetInfo(&r.free_mid, &r.total));
}

// ---- memory plan (review M2). After the weights and the chunk scratch are in, the context is what fits:
// per row and card: K + V of the 18 full layers (bf16); with the sgemm prefill attention (GPTOSS_PF_ATTN=sgemm) also
// the fp32 K / V copy of one head for the prefill scores and 512 floats of score scratch (so a prefill pass always
// has >= 64 query rows per score GEMM). The flash forms read K / V where they are (round 10: 16,384 -> 13,824 B per
// row on a 3-head card, +18.5 % rows).
// 8-bit store (GPTOSS_KV=fp8): 64 + 64 B of codes and 4 B of scales per (row, KV head, full layer): 7,128 B per row on a
// 3-head card instead of 13,824.
constexpr int KV8_PAD = 32;   // k_attn8 reads whole 32-key tiles: rows of slack behind the last row of the pool
// 12-bit K + 8-bit V (GPTOSS_KV=k12): + 32 B of K nibbles: 8,856 B per row on a 3-head card.
static bool kvq(int kv8) { return kv8 == 1 || kv8 == 3; }   // a packed store (int8 planes + scales)
// sgemm = false also for the C API pool (Config.ctx_all): there the sgemm scratch is sized by one conversation and
// is part of ctx_fit's fixed term (review r11 P6).
static size_t ctx_row_bytes(const Rank& r, bool sgemm, int kv8 = 0) {
    if (kvq(kv8)) return (size_t)(NL / 2) * r.kvn * (2 * HD + 4 + (kv8 == 3 ? HD / 2 : 0));
    return (size_t)(NL / 2) * r.kvn * HD * 2 * 2 + (sgemm ? 2 * HD * 4 + 512 * 4 : 0);
}
constexpr size_t SMIN = (size_t)8 * 1024 * (1024 + 127);   // floats: the sliding layers' score scratch
constexpr size_t RESERVE = (size_t)384 << 20;              // cuBLAS workspace, graph, driver
// seq > 0 (sgemm with the C API pool): rows of one conversation; its scratch is a fixed cost, not a cost per pool row.
// chunk: the prefill chunk; alloc_ctx sizes the sgemm scratch Kg / Vg by it (+ 128 rows), never below the 4,096 + 256
// rows budgeted before (review r14 L4: the 10,240-row default chunk took ~3 MB per card from the reserve). An upper bound:
// the chunk can still be cut after the fit (pinned staging fallbacks)
static long long ctx_fit(const Rank& r, bool sgemm, int kv8, size_t extra, int seq, int chunk) {
    const size_t sg_rows = std::max<size_t>(4096 + 256, (size_t)chunk + 128);
    const size_t fixed = RESERVE + extra + (sgemm ? SMIN * 4 + sg_rows * 2 * HD * 4 : 0) + (kvq(kv8) ? (KV8_PAD + 2 * MAXR) * ctx_row_bytes(r, false, kv8) : 0) +
                         (sgemm && seq ? (size_t)seq * (2 * HD * 4 + 512 * 4) : 0);
    return r.free_mid > fixed ? (long long)((r.free_mid - fixed) / ctx_row_bytes(r, sgemm && !seq, kv8)) : 0;
}
void Impl::alloc_ctx(Rank& r) {
    const size_t kvrow = (size_t)r.kvn * HD, T = cfg.chunk;
    if (kvq(cfg.kv8)) {
        const size_t rows = (size_t)cfg.max_ctx + KV8_PAD;
        if (cfg.kv8 == 3) r.K4 = r.alloc<uint32_t>((size_t)(NL / 2) * rows * r.kvn * 8, "kv");
        r.K8 = r.alloc<int8_t>((size_t)(NL / 2) * rows * kvrow, "kv");
        r.V8 = r.alloc<int8_t>((size_t)(NL / 2) * rows * kvrow, "kv");
        r.S8 = r.alloc<uint32_t>((size_t)(NL / 2) * rows * r.kvn, "kv");
        r.Kq = r.alloc<bf16>((size_t)MAXR * kvrow, "scratch");
        r.Vq = r.alloc<bf16>((size_t)MAXR * kvrow, "scratch");
    } else {
        r.Kf = r.alloc<bf16>((size_t)(NL / 2) * cfg.max_ctx * kvrow, "kv");
        r.Vf = r.alloc<bf16>((size_t)(NL / 2) * cfg.max_ctx * kvrow, "kv");
    }
    if (cfg.pf_attn == 0) {   // only the sgemm prefill attention reads these
        r.S = r.alloc<float>(smax, "scratch");
        r.Kg = r.alloc<float>((size_t)(pf_rows + T + 128) * HD, "scratch");
        r.Vg = r.alloc<float>((size_t)(pf_rows + T + 128) * HD, "scratch");
    }
    CU(cudaMemGetInfo(&r.free_after, &r.total));
}

// ---------------------------------------------------------------- decode step (one row), launch order of one card
// R rows of one sequence at positions *pos .. *pos + R - 1 (R > 1: a token plus drafts to verify). Every kernel treats
// a row exactly as the single-row step does, so a row's result does not depend on R.
void Impl::launch_step(Rank& r, bool dumping, int R, int smode, int nsg) {
    const bool sampled = smode != 0;
    GCHECK(R >= 1 && R <= MAXR, "1..%d rows per decode step", MAXR);
    GCHECK(!(sampled && dumping), "a sampled step does not dump");   // [srv-gptoss-capi]
    cudaStream_t st = r.st;
    CU(cudaMemcpyAsync(r.Hs, r.h_host, (size_t)R * H * 4, cudaMemcpyHostToDevice, st));
    CU(cudaMemcpyAsync(r.pos, r.pos_host, (3 * MAXR + 1) * 4, cudaMemcpyHostToDevice, st));   // [3 MAXR]: the drafter's tap ring row of step row 0
    if (sampled) CU(cudaMemcpyAsync(r.sp, r.sp_host, (size_t)R * sizeof(SampParam), cudaMemcpyHostToDevice, st));   // [srv-gptoss-capi]
    k_begin<<<1, 1, 0, st>>>(r.step);
    k_resnorm<<<R, 960, 0, st>>>(r.Hs, nullptr, nullptr, nullptr, nullptr, r.L[0].n1, r.XN, H);
    const size_t kvrow = (size_t)r.kvn * HD;
    const bool art = cfg.ar_trail > 0 && R >= cfg.ar_trail;   // the same on every card: the two forms do not mix
    // push target of the kernel that makes collective idx's partial (n = 0: none; then the collective sends the words)
    auto push_of = [&](int idx) {
        ArPush ps{};
        if (art && (cfg.ar_push >> (idx & 1) & 1) && !(p2p && cfg.ar_split))
            for (int g = 0; g < r.ar.nw; g++)   // (no peer access: the one shared host buffer)
                if (g != r.rank || r.ar.sh) {
                    // (ArPush holds 3 targets: at most 4 cards, make_plan's check; the kernels loop over ps.n: review r9 L6)
                    GCHECK(ps.n < 3, "all-reduce push: more than 3 peers");
                    ps.d[ps.n++] = (float*)((uint4*)r.ar.wr[g] + ((size_t)(idx % AR_SLOTS) * ngpu + r.rank) * MAXR * ARW);
                }
        return ps;
    };
    for (int L = 0; L < NL; L++) {
        const LayerW& w = r.L[L];
        const int sliding = L % 2 == 0, li = L / 2;
        const bool f8 = !sliding && kvq(cfg.kv8);   // 8-bit store: q/k/v write staging rows, k_kv_pack fills the store
        bf16* Kd = sliding ? r.Kr + (size_t)li * 256 * kvrow : f8 ? r.Kq : r.Kf + (size_t)li * cfg.max_ctx * kvrow;
        bf16* Vd = sliding ? r.Vr + (size_t)li * 256 * kvrow : f8 ? r.Vq : r.Vf + (size_t)li * cfg.max_ctx * kvrow;
        const int kvi = f8 ? 2 : sliding;   // row index mode of the q/k/v kernels
        const size_t ls8 = ((size_t)cfg.max_ctx + KV8_PAD) * kvrow;
        const bool dmma = !cfg.dense12 && (cfg.dense_mma == 1 || (cfg.dense_mma == 2 && R > 1));
        const int dw = dm_warps(H), dwo = dm_warps(r.ncol);
        if (cfg.dense12)
            k_qkv12<<<(r.nq + 2 * r.kvn) * 4, 256, 0, st>>>(w.qkv12, w.bqkv, r.XN, r.nq, r.kvn, r.pos, r.invf, att, r.q, Kd, Vd, kvi);
        else if (dmma && R <= 8)
            k_qkv_m<1><<<(r.nq + 2 * r.kvn) * 4, 32 * dw, dm_smem(1, dw), st>>>(w.wqkv, w.bqkv, r.XN, r.nq, r.kvn, r.pos, r.invf, att, r.q, Kd, Vd, kvi, R);
        else if (dmma)
            k_qkv_m<2><<<(r.nq + 2 * r.kvn) * 4, 32 * dw, dm_smem(2, dw), st>>>(w.wqkv, w.bqkv, r.XN, r.nq, r.kvn, r.pos, r.invf, att, r.q, Kd, Vd, kvi, R);
        else
            k_qkv<<<(r.nq + 2 * r.kvn) * 4, 256, 0, st>>>(w.wqkv, w.bqkv, r.XN, r.nq, r.kvn, r.pos, r.invf, att, r.q, Kd, Vd, kvi, R);
        if (!sliding && cfg.kv8 == 2) k_kv_sim<<<(R * r.kvn * 2 + 7) / 8, 256, 0, st>>>(Kd, Vd, r.kvn, r.pos, 0, R, cfg.kv8_sim);
        if (f8) {
            int8_t *K8 = r.K8 + (size_t)li * ls8, *V8 = r.V8 + (size_t)li * ls8;
            uint32_t* S8 = r.S8 + (size_t)li * (ls8 / HD);
            uint32_t* K4 = r.K4 ? r.K4 + (size_t)li * (ls8 / 8) : nullptr;   // 8 words per (row, KV head)
            k_kv_pack<<<(R * (r.nq + 2 * r.kvn) + 7) / 8, 256, 0, st>>>(Kd, Vd, r.q, r.nq, r.kvn, r.pos, 0, 0, R, K8, V8, S8, K4);
            if (K4)
                k_attn8<1><<<dim3(r.kvn, nsg, R), 256, 0, st>>>(r.q, K8, V8, S8, r.kvn, r.pos, r.apart, K4);
            else
                k_attn8<0><<<dim3(r.kvn, nsg, R), 256, 0, st>>>(r.q, K8, V8, S8, r.kvn, r.pos, r.apart);
            k_attn_comb8<<<dim3(r.nq, R), 64, 0, st>>>(r.apart, w.sinks, r.pos, r.AO);
        } else if (sliding && cfg.attn_win) {
            k_attn_win<<<r.kvn, 256, 0, st>>>(r.q, Kd, Vd, r.kvn, r.pos, w.sinks, r.AO);
        } else {
            const dim3 ag(r.kvn, sliding ? SSPLIT : nsg, R), cg(r.nq, R);
            const void* cpf = cfg.tailpf ? (const void*)w.wo : nullptr;
            const size_t cpb = std::min<size_t>(cfg.tailpf, (size_t)H * r.ncol * 2);
            if (cfg.attn_ep) {
                k_attn<1><<<ag, 256, 0, st>>>(r.q, Kd, Vd, r.kvn, r.pos, sliding, r.apart);
                k_attn_comb<1><<<cg, 64, 0, st>>>(r.apart, w.sinks, r.pos, sliding, r.AO, cpf, cpb);
            } else {
                k_attn<0><<<ag, 256, 0, st>>>(r.q, Kd, Vd, r.kvn, r.pos, sliding, r.apart);
                k_attn_comb<0><<<cg, 64, 0, st>>>(r.apart, w.sinks, r.pos, sliding, r.AO, cpf, cpb);
            }
        }
        const ArPush pa = dmma ? push_of(2 * L) : ArPush{};
        // the slot order is made with the top-4 when the new router kernels run and the per-row expert path wants it
        const bool ord_here = cfg.router_r && !(p2p && cfg.ar_split) && R > 1 && cfg.slot_order && !cfg.row_union;
        static_assert(H == 4 * ARP, "a pushed row is the payload of k_ar_t: H floats, 4 per word");
        if (dmma && R <= 8 && pa.n)
            k_gemv_m<1, 2, 1><<<H / 16, 32 * dwo, dm_smem(1, dwo), st>>>(w.wo, H, r.ncol, r.AO, r.ncol, R, r.rank == 0 ? w.bo : nullptr, r.part, H, pa);
        else if (dmma && pa.n)
            k_gemv_m<2, 2, 1><<<H / 16, 32 * dwo, dm_smem(2, dwo), st>>>(w.wo, H, r.ncol, r.AO, r.ncol, R, r.rank == 0 ? w.bo : nullptr, r.part, H, pa);
        else if (dmma && R <= 8)
            k_gemv_m<1><<<H / 16, 32 * dwo, dm_smem(1, dwo), st>>>(w.wo, H, r.ncol, r.AO, r.ncol, R, r.rank == 0 ? w.bo : nullptr, r.part, H, pa);
        else if (dmma)
            k_gemv_m<2><<<H / 16, 32 * dwo, dm_smem(2, dwo), st>>>(w.wo, H, r.ncol, r.AO, r.ncol, R, r.rank == 0 ? w.bo : nullptr, r.part, H, pa);
        else if (cfg.dense12)
            k_gemv12<<<(H + 15) / 16, 256, 0, st>>>(w.wo12, H, r.ncol, r.AO, r.rank == 0 ? w.bo : nullptr, r.part);
        else
            k_gemv<1><<<(H + 7) / 8, 256, 0, st>>>(w.wo, H, r.ncol, r.AO, r.rank == 0 ? w.bo : nullptr, r.part, R, r.ncol, H);
        if (p2p && cfg.ar_split) {
            k_ar1<<<137, 128, 0, st>>>(r.part, r.Hs, w.n2, r.XN, r.ar, r.rank, ngpu, r.step, 2 * L, r.aux, w.wr, w.br, r.ids, r.rw);
        } else {
            if (art)   // (without peer access: the same kernel on the shared host words, [gptoss-rel2-hostar])
                k_ar_t<<<R, 960, 0, st>>>(r.part, r.Hs, w.n2, r.XN, r.ar, r.rank, ngpu, r.step, 2 * L, cfg.tailpf ? (const void*)w.wr : nullptr,
                                          std::min<size_t>(cfg.tailpf, (size_t)NE * H * 2), pa.n > 0);
            else if (p2p)
                k_ar<<<R, 960, 0, st>>>(r.part, r.Hs, w.n2, r.XN, r.ar, r.rank, ngpu, r.step, 2 * L, cfg.tailpf ? (const void*)w.wr : nullptr,
                                        std::min<size_t>(cfg.tailpf, (size_t)NE * H * 2));
            else
                k_ar_host<<<R, 960, 0, st>>>(r.part, r.Hs, w.n2, r.XN, r.ar, r.rank, ngpu, r.step, 2 * L);
            if (cfg.router_r) {
                k_router_r<4><<<NE, 256, 0, st>>>(w.wr, r.XN, w.br, r.rlog, R);
                k_top4_ord<<<1, 32 * R, 0, st>>>(r.rlog, r.ids, r.rw, R, ord_here, r.ord);
            } else {
                k_router<<<dim3(NE, R), 256, 0, st>>>(w.wr, r.XN, w.br, r.rlog);
                k_top4<<<R, 128, 0, st>>>(r.rlog, r.ids, r.rw);
            }
        }
        ArPush pd{};   // set by the down kernel that pushes its sums
        if (R > 1 && cfg.row_union) {   // distinct experts of the step: each read once, every user served from it
            const bf16* db = r.rank == 0 ? w.bdn : nullptr;
            k_jobs<<<1, 64, 0, st>>>(r.ids, R, r.jobs);
            k_gu_u<<<dim3((r.NR + 31) / 32, 4 * R), 256, 0, st>>>(w.guc, w.bgu, r.NR, r.XN, r.jobs, r.rw, r.act);
            if (R <= 4)
                k_dn_u<4><<<(H + 7) / 8, 256, 0, st>>>(w.dnc, r.KB, r.act, r.ids, r.rw, db, r.part, r.jobs, R);
            else
                k_dn_u<MAXR><<<(H + 7) / 8, 256, 0, st>>>(w.dnc, r.KB, r.act, r.ids, r.rw, db, r.part, r.jobs, R);
        } else {
            const bf16* db = r.rank == 0 ? w.bdn : nullptr;
            // several rows: the slots in expert order, so a shared expert's second reader finds its records in L2
            // (round 8; GPTOSS_SLOT_ORDER=0: as routed). A slot's sums do not depend on its place.
            const bool so = R > 1 && cfg.slot_order;
            if (so && !ord_here) k_order<<<1, 64, 0, st>>>(r.ids, 4 * R, r.ord);
            if (R == 1 && cfg.pf_dn)
                k_gu<1><<<dim3((r.NR + 31) / 32, 4 * R), 256, 0, st>>>(w.guc, w.bgu, r.NR, r.XN, r.ids, r.rw, r.act, w.dnc, (size_t)H * mx_rs(r.KB), cfg.pf_dn, r.ident);
            else if (cfg.xreg && R >= 2 && R <= 4)   // inputs held in registers across a warp's two pairs (the same sums)
                k_gu_x<2><<<dim3((r.NR + 31) / 32, 4 * R), 256, 0, st>>>(w.guc, w.bgu, r.NR, r.XN, r.ids, r.rw, r.act, so ? r.ord : r.ident);
            else
                k_gu<0><<<dim3((r.NR + 31) / 32, 4 * R), 256, 0, st>>>(w.guc, w.bgu, r.NR, r.XN, r.ids, r.rw, r.act, nullptr, 0, 0, so ? r.ord : r.ident);
            // unrolled down (round 6, the same sums); GPTOSS_DN_OLD=1 or the tail prefetch take k_dn (load_rank allows KB <= 45
            // only, so k_dn_n<2> covers every plan's KB > 32: the KB > 64 case cannot occur, review r6 L7)
            if (cfg.xreg && R >= 8 && !cfg.dn_old && !cfg.tailpf_dn && r.KB <= 32) {   // activations held across a warp's two rows (the same sums)
                pd = push_of(2 * L + 1);
                if (pd.n)
                    k_dn_x<2, 1><<<dim3(R, (H + 7) / 8), 128, 0, st>>>(w.dnc, r.KB, r.act, r.ids, r.rw, db, r.part, pd);
                else
                    k_dn_x<2><<<dim3(R, (H + 7) / 8), 128, 0, st>>>(w.dnc, r.KB, r.act, r.ids, r.rw, db, r.part, pd);
            }
            else if (!cfg.dn_old && !cfg.tailpf_dn && r.KB <= 32)
                k_dn_n<1><<<dim3(R, (H + 7) / 8), 256, 0, st>>>(w.dnc, r.KB, r.act, r.ids, r.rw, db, r.part);
            else if (!cfg.dn_old && !cfg.tailpf_dn && r.KB <= 64)
                k_dn_n<2><<<dim3(R, (H + 7) / 8), 256, 0, st>>>(w.dnc, r.KB, r.act, r.ids, r.rw, db, r.part);
            else
                k_dn<<<dim3((H + 7) / 8, R), 256, 0, st>>>(w.dnc, r.KB, r.act, r.ids, r.rw, db, r.part,
                                                     cfg.tailpf_dn && L + 1 < NL ? (const void*)r.L[L + 1].wqkv : nullptr, cfg.tailpf_dn);
        }
        const bf16* nxt = L + 1 < NL ? r.L[L + 1].n1 : r.nf;
        if (p2p && cfg.ar_split)
            k_ar2<<<8, 128, 0, st>>>(r.part, r.Hs, nxt, r.XN, r.ar, r.rank, ngpu, r.step, 2 * L + 1, r.aux);
        else if (art)
            k_ar_t<<<R, 960, 0, st>>>(r.part, r.Hs, nxt, r.XN, r.ar, r.rank, ngpu, r.step, 2 * L + 1,
                                      cfg.tailpf ? (L + 1 < NL ? (const void*)r.L[L + 1].wqkv : (const void*)(r.s8 ? (const void*)r.s8 : (const void*)r.lm)) : nullptr,
                                      std::min<size_t>(cfg.tailpf, (size_t)r.nrows * H * 2), pd.n > 0);
        else if (p2p)
            k_ar<<<R, 960, 0, st>>>(r.part, r.Hs, nxt, r.XN, r.ar, r.rank, ngpu, r.step, 2 * L + 1,
                                    cfg.tailpf ? (L + 1 < NL ? (const void*)r.L[L + 1].wqkv : (const void*)(r.s8 ? (const void*)r.s8 : (const void*)r.lm)) : nullptr,
                                    std::min<size_t>(cfg.tailpf, (size_t)r.nrows * H * 2));
        else
            k_ar_host<<<R, 960, 0, st>>>(r.part, r.Hs, nxt, r.XN, r.ar, r.rank, ngpu, r.step, 2 * L + 1);
        if (dumping && r.rank == 0 && dump.layers && dump.rows > 0)
            CU(cudaMemcpyAsync(dump.layers + (size_t)L * dump.rows * H, r.Hs, H * 4, cudaMemcpyDeviceToHost, st));
        if (r.df) df_tap(r, L, R);   // [gptoss-r15-dflash] residual taps of the step's rows
    }
    if (smode == 2) {
        // [gptoss-r15-samp2] screen -> score bounds with each row's noise -> threshold of all cards -> exact logits of the
        // candidates into the logit rows -> the full head's draw kernels over them (gptoss_sample.cuh)
        const int nb = (r.vn + 31) / 32, nbg = (r.vn + 1023) / 1024, nbs = (r.vn + SAMP_ROWS - 1) / SAMP_ROWS;
        const bool scm = R > 1 && cfg.lm_screen_mma;
        k_xmax<<<R, 960, 0, st>>>(r.XN, r.xmax);
        if (R == 1)
            k_lm_screen1<<<nb, 256, 0, st>>>(r.s8, r.s8base, r.XN, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax);
        else if (scm && R <= 8)
            k_lm_screen_m<1><<<(r.vn + 127) / 128, 256, 0, st>>>(r.s8, r.s8base, r.XN, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax, R);
        else if (scm)
            k_lm_screen_m<2><<<(r.vn + 127) / 128, 256, 0, st>>>(r.s8, r.s8base, r.XN, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax, R);
        else
            k_lm_screen<<<nb, 256, 0, st>>>(r.s8, r.s8base, r.XN, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax, R);
        k_lm_gnoise<<<dim3(nbg, R), 1024, 0, st>>>(r.lmlo, r.lmhi, r.vn, r.v0, r.sp, r.pos, r.lomax, r.pen);   // lomax: now of the score bounds
        k_lm_thr<<<R, 1024, 0, st>>>(r.lomax, nbg, r.tb, r.rank, ngpu, r.step, r.thr);
        k_lm_rescore_w<<<dim3((nb + 7) / 8, R), 256, 0, st>>>(r.lm, r.XN, r.lmhi, r.thr, r.vn, r.logits);
        k_samp_a<<<dim3(nbs, R), 1024, 0, st>>>(r.logits, r.vn, r.v0, r.sp, r.pos, r.sblk, r.pen);
        k_samp_b<<<R, 1024, 0, st>>>(r.sblk, nbs, r.sp, r.sout, r.res);
    } else if (sampled) {   // [srv-gptoss-capi] full head, then per row the Gumbel draw (or argmax) + the 64 largest (gptoss_sample.cuh)
        const int nb = (r.vn + SAMP_ROWS - 1) / SAMP_ROWS;
        k_gemv<4><<<(r.vn + 31) / 32, 256, 0, st>>>(r.lm, r.vn, H, r.XN, nullptr, r.logits, R, H, r.vn);
        k_samp_a<<<dim3(nb, R), 1024, 0, st>>>(r.logits, r.vn, r.v0, r.sp, r.pos, r.sblk, r.pen);
        k_samp_b<<<R, 1024, 0, st>>>(r.sblk, nb, r.sp, r.sout, r.res);
        CU(cudaMemcpyAsync(r.sout_host, r.sout, (size_t)R * SAMP_OUT * 4, cudaMemcpyDeviceToHost, st));
        k_samp_copy<<<64, 256, 0, st>>>(r.logits, r.vn, r.sp, r.slog_host, r.v0, r.pen);   // mapped host memory (row 0, cands only)
    } else if (r.s8 && !(dumping && dump.logits)) {
        const int nb = (r.vn + 31) / 32;
        // screen: bounds only (the result is the exact rescoring), so its form may differ between row counts. One row:
        // the scalar screen (234 us on card A); more rows: one pass with the rows as tensor-core columns (~400 us for
        // 2..16 rows; the scalar row loop costs 240 us per row). GPTOSS_LM_SCREEN=scalar keeps the row loop.
        const bool scm = R > 1 && cfg.lm_screen_mma;
        const int nbs = scm ? (r.vn + 127) / 128 : nb;   // blocks of the screen = entries of lomax per row
        k_xmax<<<R, 960, 0, st>>>(r.XN, r.xmax);
        if (R == 1)
            k_lm_screen1<<<nb, 256, 0, st>>>(r.s8, r.s8base, r.XN, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax);
        else if (scm && R <= 8)
            k_lm_screen_m<1><<<nbs, 256, 0, st>>>(r.s8, r.s8base, r.XN, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax, R);
        else if (scm)
            k_lm_screen_m<2><<<nbs, 256, 0, st>>>(r.s8, r.s8base, r.XN, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax, R);
        else
            k_lm_screen<<<nb, 256, 0, st>>>(r.s8, r.s8base, r.XN, r.xmax, r.vn, r.lmlo, r.lmhi, r.lomax, R);
        k_lm_thr<<<R, 1024, 0, st>>>(r.lomax, nbs, r.tb, r.rank, ngpu, r.step, r.thr);
        k_lm_rescore<<<dim3((nb + 7) / 8, R), 256, 0, st>>>(r.lm, r.XN, r.lmhi, r.thr, r.vn, r.lbv, r.lbi);
        k_argbest<<<R, 1024, 0, st>>>(r.lbv, r.lbi, ((nb + 7) / 8) * 8, r.v0, r.res);
    } else {
        k_gemv<4><<<(r.vn + 31) / 32, 256, 0, st>>>(r.lm, r.vn, H, r.XN, nullptr, r.logits, R, H, r.vn);
        k_argmax<<<R, 1024, 0, st>>>(r.logits, r.vn, r.v0, r.res);
    }
    CU(cudaMemcpyAsync(r.res_host, r.res, (size_t)R * 8, cudaMemcpyDeviceToHost, st));
    if (dumping && dump.logits) CU(cudaMemcpyAsync(dump.logits + r.v0, r.logits, (size_t)r.vn * 4, cudaMemcpyDeviceToHost, st));
}

// ---------------------------------------------------------------- prefill
static void gemm_bf16(cublasHandle_t h, const bf16* X, int ldx, const bf16* W, int ldw, float* C, int ldc, int n, int N, int K) {
    // C [n][N] (row-major) = X [n][K] . W [N][K]^T, fp32 accumulation
    const float alpha = 1.f, beta = 0.f;
    CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, n, K, &alpha, W, CUDA_R_16BF, ldw, X, CUDA_R_16BF, ldx, &beta, C,
                    CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
static inline unsigned grid_of(size_t n, int per = 256) { return (unsigned)((n + per - 1) / per); }

// all-reduce of T rows as a reduce-scatter + all-gather over the copy engine (peer to peer), fp32, exact:
// rank g owns rows [T g / n, T (g + 1) / n). (A) every rank pulls the peers' partials of its own rows and sums them
// in rank order; (B) every rank pulls the other owners' sums. 4/3 T rows come in per rank instead of 2 T, and every
// card holds bit-identical sums. Barriers: partials complete / owners' sums complete. A rank reaches the next
// collective's first barrier only after its (B) pulls are done, so SUM and part are never overwritten early.
// GPTOSS_PF_AR=pull: the first form (every rank pulls both peers' full partials; same sums).
void Impl::ar_rows(Rank& r, int T, const bf16* nw) {
    drain(r.st);
    meet(r);
    const float* p[4] = {nullptr, nullptr, nullptr, nullptr};
    if (!cfg.pf_rs) {
        int k = 0;
        for (int g = 0; g < ngpu; g++) {
            if (g == r.rank)
                p[g] = r.part;
            else {
                CU(cudaMemcpyPeerAsync(r.PP[k], r.dev, R[g].part, R[g].dev, (size_t)T * H * 4, r.st));
                p[g] = r.PP[k++];
            }
        }
        k_resnorm<<<T, 960, 0, r.st>>>(r.Hs, p[0], p[1], p[2], p[3], nw, r.XN, H);
        drain(r.st);
        meet(r);
        return;
    }
    auto lo = [&](int g) { return (size_t)((long long)T * g / ngpu); };
    const size_t o0 = lo(r.rank) * H, on = (lo(r.rank + 1) - lo(r.rank)) * H;
    if (on) {
        int k = 0;
        for (int g = 0; g < ngpu; g++) {
            if (g == r.rank)
                p[g] = r.part + o0;
            else {
                CU(cudaMemcpyPeerAsync(r.PP[k], r.dev, R[g].part + o0, R[g].dev, on * 4, r.st));   // own rows only, from the buffer's start
                p[g] = r.PP[k++];
            }
        }
        k_sum_parts<<<grid_of(on), 256, 0, r.st>>>(r.SUM + o0, p[0], p[1], p[2], p[3], on);
    }
    drain(r.st);
    meet(r);
    for (int g = 0; g < ngpu; g++) {
        const size_t go = lo(g) * H, gn = (lo(g + 1) - lo(g)) * H;
        if (g != r.rank && gn) CU(cudaMemcpyPeerAsync(r.SUM + go, r.dev, R[g].SUM + go, R[g].dev, gn * 4, r.st));
    }
    k_resnorm<<<T, 960, 0, r.st>>>(r.Hs, r.SUM, nullptr, nullptr, nullptr, nw, r.XN, H);
}

// One lane = a contiguous part of the chunk's rows that runs through the layers as a unit. With two lanes the
// collective of one lane (peer copies on the card's second stream, ordered by events) runs beside the other lane's
// compute: per layer attn(a), attn(b), experts(a), experts(b) on the compute stream and coll(a) .. coll(b) on the copy
// stream. Lane b's keys include lane a's rows of the same layer, written earlier on the same stream. The row-indexed
// buffers (residual, normed rows, partials, sums) are used at the lane's row offset; the rest of the scratch is used
// by one compute step at a time. Without GPTOSS_PF_LANES: one lane with host-synchronised collectives (first form).
struct Lane {
    int idx, t0, T, pos0;
    bool one = false;   // the call's last row alone (last-layer skip): the events of lane idx, its own staging rows
};

// collective of one lane on the copy stream: (A) wait for every card's partial, pull the peers' partials of the own
// rows, sum in rank order; (B) wait for every owner's sums, pull them, residual add + next norm. The host threads
// only meet at barriers so that every event is recorded before a peer waits on it; no stream is synchronised.
void Impl::ar_lane(Rank& r, const Lane& ln, const bf16* nw, int tap) {
    const int T = ln.T, l = ln.idx;
    float *part = r.part + (size_t)ln.t0 * H, *SUM = r.SUM + (size_t)ln.t0 * H;
    // row-owner form (default): a row's residual lives on its owner only, the owner norms it and the peers pull the
    // normed bf16 row (the same bits every card computed for itself before): 3/4 of the link bytes. The check hooks
    // read the whole residual on card 0, so a pass with a dump runs the first form (sums to every card).
    const bool owner = cfg.pf_owner && !dump.layers && !dump.hfinal;
    auto lo = [&](int g) { return (size_t)((long long)T * g / ngpu); };
    const size_t o0 = lo(r.rank) * H, on = (lo(r.rank + 1) - lo(r.rank)) * H;
    // [gptoss-rel2-hostpf] no peer access: the copy engines move the rows through pinned host memory. A sender stages
    // the peers' rows of its partial (D2H on its copy stream, beside the other lanes' compute), the owners upload them
    // (H2D); then an owner stages its rows' result in the rows of its buffer the partial left free. The same waits as
    // the peer form, one event more (evH: staged). A buffer is written again only after every card's evD of this
    // collective: the sender's next compute step of the lane waits for its own evD, which follows every card's evS (all
    // uploads of the partial queued before it on the readers' streams), and the next collective of the same lane
    // stages its results only after every card's next partial of that lane, so after every card's evD.
    // The one-row lane of the last-layer skip has another row split than lane idx, whose staged rows the peers may
    // still be uploading: it stages in a region of its own, the one after the lanes' (review rel2 H1).
    const bool hst = !p2p;
    const size_t hreg = (size_t)(ln.one ? nlanes(cfg.chunk) : l) * std::max(lane_cap, MAXR) * H;
    float* hs = hst ? r.hstage + hreg : nullptr;
    auto hs_of = [&](int g) { return R[g].hstage + hreg; };
    CU(cudaEventRecord(r.evP[l], r.st));
    if (hst) {
        CU(cudaStreamWaitEvent(r.st2, r.evP[l], 0));
        const size_t e0 = o0 + on, tn = (size_t)T * H;
        if (o0) CU(cudaMemcpyAsync(hs, part, o0 * 4, cudaMemcpyDeviceToHost, r.st2));
        if (tn > e0) CU(cudaMemcpyAsync(hs + e0, part + e0, (tn - e0) * 4, cudaMemcpyDeviceToHost, r.st2));
        CU(cudaEventRecord(r.evH[l], r.st2));
    }
    meet(r);
    for (int g = 0; g < ngpu; g++) CU(cudaStreamWaitEvent(r.st2, hst ? R[g].evH[l] : R[g].evP[l], 0));
    const float* p[4] = {nullptr, nullptr, nullptr, nullptr};
    if (on) {
        int k = 0;
        for (int g = 0; g < ngpu; g++) {
            if (g == r.rank)
                p[g] = part + o0;
            else {
                float* dst = r.PP[k++] + (size_t)l * pp_lane * H;   // the lane's region: its own rows only
                if (hst)
                    CU(cudaMemcpyAsync(dst, hs_of(g) + o0, on * 4, cudaMemcpyHostToDevice, r.st2));
                else
                    CU(cudaMemcpyPeerAsync(dst, r.dev, R[g].part + (size_t)ln.t0 * H + o0, R[g].dev, on * 4, r.st2));
                p[g] = dst;
            }
        }
        if (owner)   // row-owner form: residual add + norm of the own rows only
        {
            k_resnorm<<<(unsigned)(on / H), 960, 0, r.st2>>>(r.Hs + (size_t)ln.t0 * H + o0, p[0], p[1], p[2], p[3], nw, r.XN + (size_t)ln.t0 * H + o0, H);
            // [gptoss-r15-dflash] the drafter's tap of the own rows (their residual lives on this card only)
            if (tap >= 0 && r.df && r.df->ptap)
                k_df_ptap<<<(unsigned)(on / H), 960, 0, r.st2>>>(r.Hs + (size_t)ln.t0 * H + o0, r.df->ptap + (size_t)tap * H, r.df->pbase[l], r.df->pnh[l],
                                                                 r.df->ps0[l], r.df->pcap);
        }
        else
            k_sum_parts<<<grid_of(on), 256, 0, r.st2>>>(SUM + o0, p[0], p[1], p[2], p[3], on);
        if (hst && owner)   // the own rows' normed bf16 rows (half the bytes), else their fp32 sums
            CU(cudaMemcpyAsync(hs + o0, r.XN + (size_t)ln.t0 * H + o0, on * 2, cudaMemcpyDeviceToHost, r.st2));
        else if (hst)
            CU(cudaMemcpyAsync(hs + o0, SUM + o0, on * 4, cudaMemcpyDeviceToHost, r.st2));
    }
    CU(cudaEventRecord(r.evS[l], r.st2));
    meet(r);
    // every card's phase A is done before any card goes on (and later overwrites its partials)
    for (int g = 0; g < ngpu; g++) CU(cudaStreamWaitEvent(r.st2, R[g].evS[l], 0));
    if (owner) {   // (B) pull the other owners' normed rows (bf16: half the bytes of the sums)
        for (int g = 0; g < ngpu; g++) {
            const size_t go = lo(g) * H, gn = (lo(g + 1) - lo(g)) * H;
            if (g == r.rank || !gn) continue;
            if (hst)
                CU(cudaMemcpyAsync(r.XN + (size_t)ln.t0 * H + go, hs_of(g) + go, gn * 2, cudaMemcpyHostToDevice, r.st2));
            else
                CU(cudaMemcpyPeerAsync(r.XN + (size_t)ln.t0 * H + go, r.dev, R[g].XN + (size_t)ln.t0 * H + go, R[g].dev, gn * 2, r.st2));
        }
        CU(cudaEventRecord(r.evD[l], r.st2));
        return;
    }
    for (int g = 0; g < ngpu; g++) {
        const size_t go = lo(g) * H, gn = (lo(g + 1) - lo(g)) * H;
        if (g == r.rank || !gn) continue;
        if (hst)
            CU(cudaMemcpyAsync(SUM + go, hs_of(g) + go, gn * 4, cudaMemcpyHostToDevice, r.st2));
        else
            CU(cudaMemcpyPeerAsync(SUM + go, r.dev, R[g].SUM + (size_t)ln.t0 * H + go, R[g].dev, gn * 4, r.st2));
    }
    k_resnorm<<<T, 960, 0, r.st2>>>(r.Hs + (size_t)ln.t0 * H, SUM, nullptr, nullptr, nullptr, nw, r.XN + (size_t)ln.t0 * H, H);
    CU(cudaEventRecord(r.evD[l], r.st2));
}

void Impl::prefill_chunk(Rank& r, int pos0c, int Tc, bool last, int call_pos0) {
    cudaStream_t st = r.st;
    const size_t kvrow = (size_t)r.kvn * HD;
    const float one = 1.f, zero = 0.f;
    const bool evs = lanes_on();
    const int nl = nlanes(Tc);
    const bool dev_route = evs && cfg.pf_route && cfg.pf_gemm >= 3;
    r.rsig_host[0] = 0;
    Lane lanes[MAXLN];
    for (int l = 0; l < nl; l++) {
        const int a = (int)((long long)Tc * l / nl), b = (int)((long long)Tc * (l + 1) / nl);
        lanes[l] = Lane{l, a, b - a, pos0c + a};
        GCHECK(b - a <= std::max(lane_cap, MAXR), "prefill lane of %d rows, the scratch holds %d", b - a, lane_cap);
    }
    if (r.df && r.df->ptap) df_prefill_plan(r, lanes, nl, pos0c, Tc);   // [gptoss-r15-dflash] which own rows keep their taps
    CU(cudaMemcpyAsync(r.Hs, sh_h, (size_t)Tc * H * 4, cudaMemcpyHostToDevice, st));
    if (dev_route) CU(cudaMemsetAsync(r.rsig, 0, 16, st));
    k_resnorm<<<Tc, 960, 0, st>>>(r.Hs, nullptr, nullptr, nullptr, nullptr, r.L[0].n1, r.XN, H);
    bool pending[MAXLN] = {};   // the lane's collective is on the copy stream: its next compute step waits for it
    auto coll = [&](const Lane& ln, const bf16* nw, int tap = -1) {
        if (!evs) {
            ar_rows(r, ln.T, nw);
            return;
        }
        ar_lane(r, ln, nw, tap);
        pending[ln.idx] = true;
    };
    auto ready = [&](const Lane& ln) {
        if (pending[ln.idx]) CU(cudaStreamWaitEvent(st, r.evD[ln.idx], 0));
        pending[ln.idx] = false;
    };
    // ---- experts, routing on the device: every card routes its own copy of the (bit-identical) normed rows
    // and builds the lane's route record; nothing here waits for a stream or a peer. A peer copy of card
    // 0's record was tried first: it queued behind the other lane's collective copies (10K 8,560 -> 7,000)
    auto experts_dev = [&](const Lane& ln, int L) {
        const LayerW& w = r.L[L];
        const int T = ln.T, l = ln.idx;
        bf16* XN = r.XN + (size_t)ln.t0 * H;
        float* part = r.part + (size_t)ln.t0 * H;
        const int n4 = 4 * T;
        const int plan = cfg.pf_gemm >= 5 ? 5 : 4;
        const RouteLay y = route_lay(T, plan);
        int* rt = r.rt + (size_t)l * rt_stride;
        gemm_bf16(r.blas, XN, H, w.wr, H, r.rlog, NE, T, NE, H);
        k_bias_rows<<<grid_of((size_t)T * NE), 256, 0, st>>>(r.rlog, w.br, T, NE);
        k_top4<<<T, 128, 0, st>>>(r.rlog, r.ids, r.rw);
        k_route_plan<<<1, 1024, 0, st>>>(r.ids, r.rw, n4, rt, r.roff, rt + y.units, y.ub[1], y.ub[2], y.ub[3], r.rsig, plan, y.ub[4]);
        k_route_fill<<<NE, 256, 0, st>>>(r.ids, r.rw, n4, r.roff, rt + y.row, (float*)(rt + y.w), rt + y.exp, rt + y.inv);
        const float* pw = (const float*)(rt + y.w);
        k_gather_rows_p<<<grid_of((size_t)n4 * 360), 256, 0, st>>>(XN, rt + y.row, n4, r.G);
        auto g3 = [&](const uint8_t* cod, int NRW, int KB, const bf16* X, int ldx, float* out, const bf16* ebias = nullptr) {
            const int* un = rt + y.units;
            for (int c : {4, 0, 1, 2, 3}) launch_gemm(cfg.pf_gemm, c, y.bound[c], st, cod, NRW, KB, X, ldx, un + 3 * y.ub[c], out, rt + c, ebias, pw);
        };
        if (plan == 5 && cfg.pf_gu_ep)   // gate_up with the SwiGLU epilogue: the bf16 activations come out of the GEMM
            g3(w.guc, r.NR, NBLK, r.G, H, (float*)r.A, w.bgu);
        else {
            g3(w.guc, r.NR, NBLK, r.G, H, r.GU);
            k_act_all_p<<<grid_of((size_t)n4 * r.NR / 2), 256, 0, st>>>(r.GU, w.bgu, rt + y.exp, pw, n4, r.NR, r.A);
        }
        g3(w.dnc, H, r.KB, r.A, r.NR / 2, r.O);
        k_sum4<<<grid_of((size_t)T * H), 256, 0, st>>>(r.O, rt + y.inv, r.ids, r.rw, r.rank == 0 ? w.bdn : nullptr, T, part);
        coll(ln, L + 1 < NL ? r.L[L + 1].n1 : r.nf, df_tapi(L));
    };
    const size_t ls8 = ((size_t)cfg.max_ctx + KV8_PAD) * kvrow;   // 8-bit store: layer stride (bytes of K or V)
    auto attn_pf = [&](const float* Q, int T, int pos0, int kpos_of_row0, int row_lo, int row_hi, int sliding, const bf16* Kd, const bf16* Vd,
                       const float* sinks, const uint32_t* Sd = nullptr, const uint32_t* K4d = nullptr) {
        if (Sd && K4d) {   // a full layer over the 12-bit K + 8-bit V store
#define APF_ARGS Q, r.nq, Kd, Vd, r.kvn, T, pos0, kpos_of_row0, row_lo, row_hi, sliding, sinks, r.AO, r.ncol, Sd, K4d
            if (cfg.pf_attn == 1) k_attn_pf<1, 0, 0, 2><<<dim3((T + 15) / 16, r.kvn), 256, 0, st>>>(APF_ARGS);
            if (cfg.pf_attn == 2) k_attn_pf<2, 0, 0, 2><<<dim3((T + 31) / 32, r.kvn), 512, 0, st>>>(APF_ARGS);
            if (cfg.pf_attn == 3) k_attn_pf<1, 1, 0, 2><<<dim3((T + 15) / 16, r.kvn), 256, 0, st>>>(APF_ARGS);
            if (cfg.pf_attn == 4) k_attn_pf<1, 1, 1, 2><<<dim3((T + 15) / 16, r.kvn), 256, 0, st>>>(APF_ARGS);
#undef APF_ARGS
            return;
        }
        if (Sd) {   // a full layer over the 8-bit store
#define APF_ARGS Q, r.nq, Kd, Vd, r.kvn, T, pos0, kpos_of_row0, row_lo, row_hi, sliding, sinks, r.AO, r.ncol, Sd
            if (cfg.pf_attn == 1) k_attn_pf<1, 0, 0, 1><<<dim3((T + 15) / 16, r.kvn), 256, 0, st>>>(APF_ARGS);
            if (cfg.pf_attn == 2) k_attn_pf<2, 0, 0, 1><<<dim3((T + 31) / 32, r.kvn), 512, 0, st>>>(APF_ARGS);
            if (cfg.pf_attn == 3) k_attn_pf<1, 1, 0, 1><<<dim3((T + 15) / 16, r.kvn), 256, 0, st>>>(APF_ARGS);
            if (cfg.pf_attn == 4) k_attn_pf<1, 1, 1, 1><<<dim3((T + 15) / 16, r.kvn), 256, 0, st>>>(APF_ARGS);
#undef APF_ARGS
            return;
        }
#define APF_ARGS Q, r.nq, Kd, Vd, r.kvn, T, pos0, kpos_of_row0, row_lo, row_hi, sliding, sinks, r.AO, r.ncol
        if (cfg.pf_attn == 1) k_attn_pf<1, 0, 0><<<dim3((T + 15) / 16, r.kvn), 256, 0, st>>>(APF_ARGS);
        if (cfg.pf_attn == 2) k_attn_pf<2, 0, 0><<<dim3((T + 31) / 32, r.kvn), 512, 0, st>>>(APF_ARGS);
        if (cfg.pf_attn == 3) k_attn_pf<1, 1, 0><<<dim3((T + 15) / 16, r.kvn), 256, 0, st>>>(APF_ARGS);
        if (cfg.pf_attn == 4) k_attn_pf<1, 1, 1><<<dim3((T + 15) / 16, r.kvn), 256, 0, st>>>(APF_ARGS);
#undef APF_ARGS
    };
    // last-layer skip (plan P6): the residual after the last layer is read for one row only, the call's last one (the
    // head), and by the check hooks. Without a dump the last layer computes q / k / v and stores K/V for every row
    // (later rows and decode steps read them), and runs attention, o_proj, experts and both collectives for that one
    // row as a one-row lane; in a chunk that is not the call's last it stops after the K/V store. The last layer is a
    // full-attention layer (no ring).
    static_assert((NL - 1) % 2 == 1, "the last layer must be a full-attention layer for the skip");
    // Row-owner form: the one-row lane's owner is the last card (lo(g) = T g / n with T = 1), which is also the owner
    // of the last row of lane nl - 1, so that row's residual is where the collective looks for it (review r4b L1).
    const bool skip_last = cfg.pf_skip && dev_route && cfg.pf_attn != 0 && !dump.layers && !dump.hfinal;
    // One layer for the lanes in the masks: `ma` = the lanes whose attention step runs, `mx` = the lanes whose expert
    // step runs; `tail` = the part of a skipped last layer behind the lanes' q / k / v (the call's last row).
    auto layer_part = [&](int L, unsigned ma, unsigned mx, bool tail) {
        const LayerW& w = r.L[L];
        const int sliding = L % 2 == 0, li = L / 2;
        if (L == NL - 1 && skip_last) {
            const bool f8 = kvq(cfg.kv8);
            int8_t *K8 = f8 ? r.K8 + (size_t)li * ls8 : nullptr, *V8 = f8 ? r.V8 + (size_t)li * ls8 : nullptr;
            uint32_t* S8 = f8 ? r.S8 + (size_t)li * (ls8 / HD) : nullptr;
            uint32_t* K4 = r.K4 ? r.K4 + (size_t)li * (ls8 / 8) : nullptr;
            const bf16 *Kd = f8 ? (const bf16*)K8 : r.Kf + (size_t)li * cfg.max_ctx * kvrow, *Vd = f8 ? (const bf16*)V8 : r.Vf + (size_t)li * cfg.max_ctx * kvrow;
            for (int l = 0; l < nl; l++) {
                if (!(ma >> l & 1)) continue;
                const Lane& ln = lanes[l];
                ready(ln);
                gemm_bf16(r.blas, r.XN + (size_t)ln.t0 * H, H, w.wqkv, H, r.Y, r.nrows, ln.T, r.nrows, H);
                if (f8) {   // bf16 rows to the chunk scratch, then rotated and packed into the store (q rotated in place)
                    k_qkv_rows<<<grid_of((size_t)ln.T * (r.nq + 2 * r.kvn) * 32), 256, 0, st>>>(r.Y, w.bqkv, ln.T, r.nq, r.kvn, ln.pos0, r.invf, att, r.Qf, r.Ksc, r.Vsc, 0);
                    // K / V only (nq = 0); of the q rows only the call's last one is used here: rotated below (review r11 L6)
                    k_kv_pack<<<grid_of((size_t)ln.T * 2 * r.kvn * 32), 256, 0, st>>>(r.Ksc, r.Vsc, nullptr, 0, r.kvn, nullptr, 0, ln.pos0, ln.T, K8, V8, S8, K4);
                    if (last && l == nl - 1)
                        k_kv_pack<<<grid_of((size_t)r.nq * 32), 256, 0, st>>>(nullptr, nullptr, r.Qf + (size_t)(ln.T - 1) * r.ncol, r.nq, 0, nullptr, 0, 0, 1, nullptr, nullptr, nullptr);
                    continue;
                }
                k_qkv_rows<<<grid_of((size_t)ln.T * (r.nq + 2 * r.kvn) * 32), 256, 0, st>>>(r.Y, w.bqkv, ln.T, r.nq, r.kvn, ln.pos0, r.invf, att, r.Qf, (bf16*)Kd, (bf16*)Vd, ln.pos0);
                if (cfg.kv8 == 2) k_kv_sim<<<(ln.T * r.kvn * 2 + 7) / 8, 256, 0, st>>>((bf16*)Kd, (bf16*)Vd, r.kvn, nullptr, ln.pos0, ln.T, cfg.kv8_sim);
            }
            if (!last || !tail) return;
            // the call's last row: its q is the last lane's last row in the q scratch
            const Lane one{nl - 1, Tc - 1, 1, pos0c + Tc - 1, true};
            attn_pf(r.Qf + (size_t)(lanes[nl - 1].T - 1) * r.ncol, 1, one.pos0, 0, 0, one.pos0 + 1, 0, Kd, Vd, w.sinks, f8 ? S8 : nullptr, K4);
            float* part1 = r.part + (size_t)one.t0 * H;
            gemm_bf16(r.blas, r.AO, r.ncol, w.wo, r.ncol, part1, H, 1, H, r.ncol);
            if (r.rank == 0) k_bias_rows<<<grid_of((size_t)H), 256, 0, st>>>(part1, w.bo, 1, H);
            coll(one, w.n2);
            ready(one);
            experts_dev(one, L);
            return;
        }
        for (int l = 0; l < nl; l++) {
            if (!(ma >> l & 1)) continue;
            const Lane& ln = lanes[l];
            const int T = ln.T, pos0 = ln.pos0;
            bf16* XN = r.XN + (size_t)ln.t0 * H;
            float* part = r.part + (size_t)ln.t0 * H;
            ready(ln);
            // ---- attention
            gemm_bf16(r.blas, XN, H, w.wqkv, H, r.Y, r.nrows, T, r.nrows, H);
            bf16 *Kd, *Vd;
            int krow0, nprev = 0;
            if (sliding) {
                Kd = r.Ksc;
                Vd = r.Vsc;
                krow0 = 127;
                nprev = std::min(127, pos0);
                if (nprev) {
                    bf16* kr = r.Kr + (size_t)li * 256 * kvrow;
                    bf16* vr = r.Vr + (size_t)li * 256 * kvrow;
                    k_ring_copy<<<grid_of((size_t)nprev * r.kvn * 8), 256, 0, st>>>(kr, r.Ksc + (127 - nprev) * kvrow, r.kvn, pos0 - nprev, nprev, 0);
                    k_ring_copy<<<grid_of((size_t)nprev * r.kvn * 8), 256, 0, st>>>(vr, r.Vsc + (127 - nprev) * kvrow, r.kvn, pos0 - nprev, nprev, 0);
                }
            } else if (kvq(cfg.kv8)) {
                Kd = r.Ksc;   // staging; the store is filled by k_kv_pack below
                Vd = r.Vsc;
                krow0 = 0;
            } else {
                Kd = r.Kf + (size_t)li * cfg.max_ctx * kvrow;
                Vd = r.Vf + (size_t)li * cfg.max_ctx * kvrow;
                krow0 = pos0;
            }
            k_qkv_rows<<<grid_of((size_t)T * (r.nq + 2 * r.kvn) * 32), 256, 0, st>>>(r.Y, w.bqkv, T, r.nq, r.kvn, pos0, r.invf, att, r.Qf, Kd, Vd, krow0);
            const uint32_t *S8 = nullptr, *K4 = nullptr;
            if (!sliding && kvq(cfg.kv8)) {
                int8_t *K8 = r.K8 + (size_t)li * ls8, *V8 = r.V8 + (size_t)li * ls8;
                uint32_t* s8 = r.S8 + (size_t)li * (ls8 / HD);
                uint32_t* k4 = r.K4 ? r.K4 + (size_t)li * (ls8 / 8) : nullptr;
                K4 = k4;
                k_kv_pack<<<grid_of((size_t)T * (r.nq + 2 * r.kvn) * 32), 256, 0, st>>>(Kd, Vd, r.Qf, r.nq, r.kvn, nullptr, 0, pos0, T, K8, V8, s8, k4);
                Kd = (bf16*)K8;
                Vd = (bf16*)V8;
                S8 = s8;
            }
            if (!sliding && cfg.kv8 == 2) k_kv_sim<<<(T * r.kvn * 2 + 7) / 8, 256, 0, st>>>(Kd, Vd, r.kvn, nullptr, krow0, T, cfg.kv8_sim);
            const int row_lo = sliding ? 127 - nprev : 0;              // first valid key row (scratch / full index)
            const int row_hi = sliding ? 127 + T : pos0 + T;
            const int kpos_of_row0 = sliding ? pos0 - 127 : 0;         // key position of row index 0
            attn_pf(r.Qf, T, pos0, kpos_of_row0, row_lo, row_hi, sliding, Kd, Vd, w.sinks, S8, K4);
            for (int kh = 0; kh < (cfg.pf_attn ? 0 : r.kvn); kh++) {
                GCHECK((size_t)row_hi <= (size_t)pf_rows + cfg.chunk + 128, "sgemm prefill attention: row %d is beyond the %d rows of one conversation its scratch holds", row_hi, pf_rows);
                k_kv_f32<<<grid_of((size_t)(row_hi - row_lo) * 64), 256, 0, st>>>(Kd + row_lo * kvrow, r.kvn, kh, row_hi - row_lo, r.Kg + (size_t)row_lo * HD);
                k_kv_f32<<<grid_of((size_t)(row_hi - row_lo) * 64), 256, 0, st>>>(Vd + row_lo * kvrow, r.kvn, kh, row_hi - row_lo, r.Vg + (size_t)row_lo * HD);
                const int nt_max = sliding ? std::min(T, 1024) : (int)std::max<size_t>(1, std::min<size_t>(T, smax / 8 / (size_t)(pos0 + T)));
                for (int t0 = 0; t0 < T; t0 += nt_max) {
                    const int nt = std::min(nt_max, T - t0), t1 = t0 + nt;
                    const int j_lo = sliding ? std::max(row_lo, t0) : 0;
                    const int j_hi = sliding ? 127 + t1 : pos0 + t1;
                    const int Lk = j_hi - j_lo;
                    GCHECK((size_t)8 * nt * Lk <= smax, "score scratch");
                    k_gather_q<<<grid_of((size_t)nt * 512), 256, 0, st>>>(r.Qf, r.nq, kh, t0, nt, r.Qg);
                    CB(cublasSgemm(r.blas, CUBLAS_OP_T, CUBLAS_OP_N, Lk, 8 * nt, HD, &one, r.Kg + (size_t)j_lo * HD, HD, r.Qg, HD, &zero, r.S, Lk));
                    k_softmax_rows<<<8 * nt, 256, 0, st>>>(r.S, Lk, kpos_of_row0 + j_lo, pos0 + t0, sliding, w.sinks + kh * 8, r.rinv);
                    CB(cublasSgemm(r.blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, 8 * nt, Lk, &one, r.Vg + (size_t)j_lo * HD, HD, r.S, Lk, &zero, r.Og, HD));
                    k_scatter_ao<<<grid_of((size_t)nt * 512), 256, 0, st>>>(r.Og, r.rinv, r.ncol, kh, t0, nt, r.AO);
                }
            }
            if (sliding) {
                const int nin = std::min(T, 256);
                bf16* kr = r.Kr + (size_t)li * 256 * kvrow;
                bf16* vr = r.Vr + (size_t)li * 256 * kvrow;
                k_ring_copy<<<grid_of((size_t)nin * r.kvn * 8), 256, 0, st>>>(kr, r.Ksc + (size_t)(127 + T - nin) * kvrow, r.kvn, pos0 + T - nin, nin, 1);
                k_ring_copy<<<grid_of((size_t)nin * r.kvn * 8), 256, 0, st>>>(vr, r.Vsc + (size_t)(127 + T - nin) * kvrow, r.kvn, pos0 + T - nin, nin, 1);
            }
            gemm_bf16(r.blas, r.AO, r.ncol, w.wo, r.ncol, part, H, T, H, r.ncol);
            if (r.rank == 0) k_bias_rows<<<grid_of((size_t)T * H), 256, 0, st>>>(part, w.bo, T, H);
            coll(ln, w.n2);
        }
        for (int l = 0; l < nl; l++) {
            if (!(mx >> l & 1)) continue;
            const Lane& ln = lanes[l];
            const int T = ln.T;
            bf16* XN = r.XN + (size_t)ln.t0 * H;
            float* part = r.part + (size_t)ln.t0 * H;
            int *ids_h = sh_ids + 4 * ln.t0, *rows_h = sh_rows + 4 * ln.t0, *exp_h = sh_exp + 4 * ln.t0, *inv_h = sh_inv + 4 * ln.t0;
            float *rw_h = sh_rw + 4 * ln.t0, *w_h = sh_w + 4 * ln.t0;
            int* units = sh_units + (size_t)3 * (cfg.chunk * 4 / 32 + NE) * l;
            int& nun = sh_nunits[l];
            int* ncl = sh_ncls[l];
            ready(ln);
            if (dev_route) {
                experts_dev(ln, L);
                continue;
            }
            // ---- experts: rank 0 routes and shares the row lists (one routing for all cards)
            if (r.rank == 0) {
                gemm_bf16(r.blas, XN, H, w.wr, H, r.rlog, NE, T, NE, H);
                k_bias_rows<<<grid_of((size_t)T * NE), 256, 0, st>>>(r.rlog, w.br, T, NE);
                k_top4<<<T, 128, 0, st>>>(r.rlog, r.ids, r.rw);
                CU(cudaMemcpyAsync(ids_h, r.ids, (size_t)T * 16, cudaMemcpyDeviceToHost, st));
                CU(cudaMemcpyAsync(rw_h, r.rw, (size_t)T * 16, cudaMemcpyDeviceToHost, st));
                drain(st);
                int cnt[NE] = {};
                for (int i = 0; i < 4 * T; i++) {
                    GCHECK(ids_h[i] >= 0 && ids_h[i] < NE, "router index");
                    cnt[ids_h[i]]++;
                }
                sh_off[0] = 0;
                for (int e = 0; e < NE; e++) sh_off[e + 1] = sh_off[e] + cnt[e];
                int fill[NE];
                for (int e = 0; e < NE; e++) fill[e] = sh_off[e];
                for (int t = 0; t < T; t++)
                    for (int k = 0; k < 4; k++) {
                        const int e = ids_h[t * 4 + k], o = fill[e]++;
                        rows_h[o] = t;
                        w_h[o] = rw_h[t * 4 + k];
                        if (cfg.pf_fused) {
                            exp_h[o] = e;
                            inv_h[t * 4 + k] = o;
                        }
                    }
                if (cfg.pf_fused) {
                    nun = 0;
                    const int us = cfg.pf_gemm >= 3 ? 64 : 32;   // form 3: 64-row units, then the remainders of <= 32 rows
                    for (int e = 0; e < NE; e++)
                        for (int o = sh_off[e]; o < sh_off[e + 1]; o += us) {
                            const int n = sh_off[e + 1] - o;
                            if (us == 64 && n <= 32) break;
                            int* u = units + 3 * nun++;
                            u[0] = e;
                            u[1] = o;
                            u[2] = std::min(us, n);
                        }
                    ncl[0] = nun;
                    // form 3: the experts' remainders (<= 32 rows) go to 32- / 16- / 8-token blocks by size
                    for (int c = 1; c < 4; c++) {
                        const int hi = 64 >> c, lo = c == 3 ? 0 : hi / 2;
                        if (us == 64)
                            for (int e = 0; e < NE; e++) {
                                const int n = (sh_off[e + 1] - sh_off[e]) % 64;
                                if (n <= lo || n > hi) continue;
                                int* u = units + 3 * nun++;
                                u[0] = e;
                                u[1] = sh_off[e + 1] - n;
                                u[2] = n;
                            }
                        ncl[c] = nun;
                    }
                }
            }
            meet(r);
            CU(cudaMemcpyAsync(r.prow, rows_h, (size_t)T * 16, cudaMemcpyHostToDevice, st));
            CU(cudaMemcpyAsync(r.pw, w_h, (size_t)T * 16, cudaMemcpyHostToDevice, st));
            if (cfg.pf_fused && cfg.pf_gemm >= 3)
                k_gather_rows_p<<<grid_of((size_t)4 * T * 360), 256, 0, st>>>(XN, r.prow, 4 * T, r.G);
            else
                k_gather_rows<<<grid_of((size_t)4 * T * 360), 256, 0, st>>>(XN, r.prow, 4 * T, r.G);
            if (cfg.pf_fused) {
                const int nu = nun, n4 = 4 * T;
                CU(cudaMemcpyAsync(r.pexp, exp_h, (size_t)T * 16, cudaMemcpyHostToDevice, st));
                CU(cudaMemcpyAsync(r.pinv, inv_h, (size_t)T * 16, cudaMemcpyHostToDevice, st));
                CU(cudaMemcpyAsync(r.punits, units, (size_t)nu * 12, cudaMemcpyHostToDevice, st));
                if (r.rank) {
                    CU(cudaMemcpyAsync(r.ids, ids_h, (size_t)T * 16, cudaMemcpyHostToDevice, st));
                    CU(cudaMemcpyAsync(r.rw, rw_h, (size_t)T * 16, cudaMemcpyHostToDevice, st));
                }
                if (cfg.pf_gemm >= 3) {
                    const int* nc = ncl;   // units by class: [0, nc[0]) 64 tokens, then 32, 16, 8
                    auto g3 = [&](const uint8_t* cod, int NRW, int KB, const bf16* X, int ldx, float* out) {
                        launch_gemm(cfg.pf_gemm, 0, nc[0], st, cod, NRW, KB, X, ldx, r.punits, out, nullptr);
                        for (int c = 1; c < 4; c++) launch_gemm(cfg.pf_gemm, c, nc[c] - nc[c - 1], st, cod, NRW, KB, X, ldx, r.punits + 3 * nc[c - 1], out, nullptr);
                    };
                    g3(w.guc, r.NR, NBLK, r.G, H, r.GU);
                    k_act_all_p<<<grid_of((size_t)n4 * r.NR / 2), 256, 0, st>>>(r.GU, w.bgu, r.pexp, r.pw, n4, r.NR, r.A);
                    g3(w.dnc, H, r.KB, r.A, r.NR / 2, r.O);
                } else if (cfg.pf_gemm == 2) {
                    k_mx_gemm2<<<dim3((r.NR + 127) / 128, nu), 256, 0, st>>>(w.guc, r.NR, NBLK, r.G, H, r.punits, r.GU, r.NR);
                    k_act_all<<<grid_of((size_t)n4 * r.NR / 2), 256, 0, st>>>(r.GU, w.bgu, r.pexp, r.pw, n4, r.NR, r.A);
                    k_mx_gemm2<<<dim3((H + 127) / 128, nu), 256, 0, st>>>(w.dnc, H, r.KB, r.A, r.NR / 2, r.punits, r.O, H);
                } else {
                    k_mx_gemm<<<dim3((r.NR + 127) / 128, nu), 256, 0, st>>>(w.guc, r.NR, NBLK, r.G, H, r.punits, r.GU, r.NR);
                    k_act_all<<<grid_of((size_t)n4 * r.NR / 2), 256, 0, st>>>(r.GU, w.bgu, r.pexp, r.pw, n4, r.NR, r.A);
                    k_mx_gemm<<<dim3((H + 127) / 128, nu), 256, 0, st>>>(w.dnc, H, r.KB, r.A, r.NR / 2, r.punits, r.O, H);
                }
                k_sum4<<<grid_of((size_t)T * H), 256, 0, st>>>(r.O, r.pinv, r.ids, r.rw, r.rank == 0 ? w.bdn : nullptr, T, part);
            } else {
                CU(cudaMemsetAsync(part, 0, (size_t)T * H * 4, st));
                for (int e = 0; e < NE; e++) {
                    const int o = sh_off[e], n = sh_off[e + 1] - o;
                    if (!n) continue;
                    k_deq<<<grid_of((size_t)r.NR * NBLK), 256, 0, st>>>(w.guc + (size_t)e * r.NR * mx_rs(NBLK), r.NR, NBLK, r.Wd);
                    gemm_bf16(r.blas, r.G + (size_t)o * H, H, r.Wd, H, r.GU, r.NR, n, r.NR, H);
                    k_act_rows<<<grid_of((size_t)n * r.NR / 2), 256, 0, st>>>(r.GU, w.bgu + (size_t)e * r.NR, r.pw + o, n, r.NR, r.A);
                    k_deq<<<grid_of((size_t)H * r.KB), 256, 0, st>>>(w.dnc + (size_t)e * H * mx_rs(r.KB), H, r.KB, r.Dd);
                    gemm_bf16(r.blas, r.A, r.NR / 2, r.Dd, r.KB * 32, r.O, H, n, H, r.NR / 2);
                    k_scatter_add<<<grid_of((size_t)n * H), 256, 0, st>>>(r.O, r.prow + o, r.pw + o, r.rank == 0 ? w.bdn + (size_t)e * H : nullptr, n, part);
                }
            }
            coll(ln, L + 1 < NL ? r.L[L + 1].n1 : r.nf, df_tapi(L));
        }
        const int pos0 = pos0c, T = Tc;
        if (r.rank == 0 && dump.layers && pos0 - call_pos0 < dump.rows) {   // every chunk's rows, up to dump.rows of the call
            for (int l = 0; l < nl; l++) ready(lanes[l]);
            if (evs) drain(r.st2);
            const int off = pos0 - call_pos0;
            int di = L;
            if (dump.taps) {   // [gptoss-r15-dflash] only the drafter's 5 tap layers, as [5][rows][H]
                di = -1;
                for (int i = 0; i < DF_NTAP; i++)
                    if (DF_TAP_LAYER[i] == L) di = i;
            }
            if (di >= 0)
                CU(cudaMemcpy(dump.layers + ((size_t)di * dump.rows + off) * H, r.Hs, (size_t)std::min(T, dump.rows - off) * H * 4, cudaMemcpyDeviceToHost));
        }
    };
    // Step order. Up to two lanes (and every pass with a layer dump): layer by layer, attention of every lane, then
    // the experts of every lane (rounds 3-13). Three or more lanes: lane l runs one step (attention, experts,
    // attention, ...) per round and is l steps behind lane 0, so between two steps of a lane the other lanes run an
    // attention step AND an expert step: the lane's collective (0.9 ms of copies per 2,048 rows) always has an expert
    // step beside it, also in the sliding layers whose attention step is ~0.5 ms. A lane's attention at layer L still
    // runs after the earlier lanes' attention at L (their K/V rows, the sliding ring).
    const unsigned all = (1u << nl) - 1;
    if (nl <= 2 || cfg.pf_stagger == 0 || dump.layers) {
        for (int L = 0; L < NL; L++) layer_part(L, all, all, true);
    } else {
        for (int k = 0; k < 2 * NL + nl - 1; k++)
            for (int l = 0; l < nl; l++) {
                const int sidx = k - l;
                if (sidx < 0 || sidx >= 2 * NL) continue;
                if (sidx & 1) layer_part(sidx / 2, 0, 1u << l, false);
                else layer_part(sidx / 2, 1u << l, 0, false);
            }
        if (skip_last) layer_part(NL - 1, 0, 0, true);   // the call's last row (the last lane's q is still in the scratch)
    }
    const int pos0 = pos0c, T = Tc;
    for (int l = 0; l < nl; l++) ready(lanes[l]);
    if (evs) drain(r.st2);
    if (r.rank == 0 && dump.hfinal)
        CU(cudaMemcpy(dump.hfinal + (size_t)(pos0 - call_pos0) * H, r.Hs, (size_t)T * H * 4, cudaMemcpyDeviceToHost));
    if (dev_route) CU(cudaMemcpyAsync(r.rsig_host, r.rsig, 4, cudaMemcpyDeviceToHost, st));
    if (last) {
        k_gemv<4><<<(r.vn + 31) / 32, 256, 0, st>>>(r.lm, r.vn, H, r.XN + (size_t)(T - 1) * H, nullptr, r.logits, 1, H, r.vn);
        k_argmax<<<1, 1024, 0, st>>>(r.logits, r.vn, r.v0, r.res);
        CU(cudaMemcpyAsync(r.res_host, r.res, 8, cudaMemcpyDeviceToHost, st));
        if (dump.logits) CU(cudaMemcpyAsync(dump.logits + r.v0, r.logits, (size_t)r.vn * 4, cudaMemcpyDeviceToHost, st));
    }
    if (df_pf_live()) df_prefill_ctx(r, lanes, nl, pos0c, Tc);   // [gptoss-r15-dflash] the chunk's rows into the drafter context
    drain(st);
}


// ---------------------------------------------------------------- Engine
static void build(Impl& m, const Config& cfg);
Engine::Engine(const Config& cfg) : p_(new Impl) {
    try {
        build(*p_, cfg);
    } catch (...) {
        delete p_;
        throw;
    }
    if (p_->cfg.verbose) fprintf(stderr, "gptoss: ready in %.1f s, context %d rows\n%s", p_->load_s, p_->cfg.max_ctx, vram_table().c_str());
}
static void build(Impl& m, const Config& cfg) {
    const auto t0 = clk::now();
    m.cfg = cfg;
    if (const char* v = getenv("GPTOSS_PF_GEMM")) m.cfg.pf_gemm = std::max(1, std::min(5, atoi(v)));
    if (const char* v = getenv("GPTOSS_PF_GU_EP")) m.cfg.pf_gu_ep = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_PF_NLANES")) m.cfg.pf_nlanes = std::max(0, std::min(MAXLN, atoi(v)));
    if (const char* v = getenv("GPTOSS_PF_STAGGER")) m.cfg.pf_stagger = atoi(v);
    if (const char* v = getenv("GPTOSS_TAILPF_DN_MB")) m.cfg.tailpf_dn = (size_t)(atof(v) * 1048576.0);
    if (const char* v = getenv("GPTOSS_SLOTS"); v && *v) m.cfg.slots = std::max(1, std::min(16, atoi(v)));   // (review r6 L4: as the C API, empty = the caller's count)
    if (const char* v = getenv("GPTOSS_ROW_UNION")) m.cfg.row_union = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_SLOT_ORDER")) m.cfg.slot_order = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_AR_TRAIL")) m.cfg.ar_trail = atoi(v);
    if (const char* v = getenv("GPTOSS_ATTN_GRID")) m.cfg.attn_grid = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_ATTN_EP")) m.cfg.attn_ep = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_ROUTER_R")) m.cfg.router_r = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_AR_PUSH")) m.cfg.ar_push = atoi(v);
    if (const char* v = getenv("GPTOSS_XREG")) m.cfg.xreg = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_PF_DN_KB")) m.cfg.pf_dn = (size_t)atoi(v) * 1024;
    if (const char* v = getenv("GPTOSS_DN_OLD")) m.cfg.dn_old = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_TAILPF_MB")) m.cfg.tailpf = (size_t)(atof(v) * 1048576.0);
    if (const char* v = getenv("GPTOSS_LM_SCREEN")) m.cfg.lm_screen_mma = std::string(v) != "scalar";
    if (const char* v = getenv("GPTOSS_DENSE_MMA")) m.cfg.dense_mma = std::max(0, std::min(2, atoi(v)));
    if (const char* v = getenv("GPTOSS_KV")) {
        const std::string a = v;
        GCHECK(a == "fp8" || a == "k12" || a == "fp8sim" || a == "bf16" || a.empty(), "GPTOSS_KV=%s: bf16 (default), k12, fp8 or fp8sim", v);   // review r11 L2
        m.cfg.kv8 = a == "fp8" ? 1 : a == "fp8sim" ? 2 : a == "k12" ? 3 : 0;
    }
    if (const char* v = getenv("GPTOSS_KV_SIM")) m.cfg.kv8_sim = atoi(v);
    if (const char* v = getenv("GPTOSS_ATTN")) m.cfg.attn_win = std::string(v) == "win";
    if (const char* v = getenv("GPTOSS_PF_MOE")) m.cfg.pf_fused = std::string(v) != "blas";
    if (const char* v = getenv("GPTOSS_DENSE")) m.cfg.dense12 = std::string(v) == "12";
    if (const char* v = getenv("GPTOSS_LM")) m.cfg.lm2 = std::string(v) != "full";
    if (const char* v = getenv("GPTOSS_SAMP")) m.cfg.samp2 = std::string(v) != "full";   // [gptoss-r15-samp2]
    if (const char* v = getenv("GPTOSS_AR")) m.cfg.ar_split = std::string(v) == "split";
    if (const char* v = getenv("GPTOSS_PF_ATTN")) {
        const std::string a = v;
        m.cfg.pf_attn = a == "sgemm" || a == "blas" ? 0 : a == "flash" ? 1 : a == "flash2" ? 2 : a == "flashq" ? 3 : a == "flashx" ? 4 : -1;
        GCHECK(m.cfg.pf_attn >= 0, "GPTOSS_PF_ATTN=%s: use sgemm, flash, flash2, flashq or flashx", v);
    }
    if (const char* v = getenv("GPTOSS_PF_ROUTE")) m.cfg.pf_route = std::string(v) != "host";
    if (const char* v = getenv("GPTOSS_PF_SKIP")) m.cfg.pf_skip = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_PF_LANES")) m.cfg.pf_lanes = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_PF_AR")) {
        m.cfg.pf_rs = std::string(v) != "pull";
        m.cfg.pf_owner = std::string(v) != "pull" && std::string(v) != "rs";
    }
    if (const char* v = getenv("GPTOSS_AR_CLEAR")) m.cfg.ar_clear = std::max(1L, atol(v));
    m.cfg.ar_clear = std::min(1L << 23, std::max(1L, m.cfg.ar_clear));   // (a word must stay younger than 2^24 launches, also when set in code: second review L1)
    if (const char* v = getenv("GPTOSS_STEP0")) m.cfg.step0 = (unsigned)strtoul(v, nullptr, 10);
    if (const char* v = getenv("GPTOSS_PF_HOSTLANES")) m.cfg.pf_hostlanes = atoi(v) != 0;
    if (const char* v = getenv("GPTOSS_COMM")) m.cfg.comm = std::string(v) == "p2p" ? 1 : std::string(v) == "host" ? 2 : 0;
    m.ngpu = (int)cfg.devices.size();
    GCHECK(m.cfg.pf_attn >= 0 && m.cfg.pf_attn <= 4 && m.cfg.pf_gemm >= 1 && m.cfg.pf_gemm <= 5, "pf_attn 0..4, pf_gemm 1..5");
    GCHECK(cfg.chunk >= 1 && cfg.chunk <= 10240 && cfg.max_ctx >= 256, "chunk 1..10240, max_ctx >= 256");
    GCHECK(!cfg.devices.empty() && cfg.devices.size() <= 4, "1..4 cards");
    for (size_t i = 0; i < cfg.devices.size(); i++)
        for (size_t j = 0; j < i; j++) GCHECK(cfg.devices[i] != cfg.devices[j], "card %d listed twice", cfg.devices[i]);
    // the files first: a missing or broken model fails before any card is touched
    m.files.open_dir(cfg.model_dir);
    m.embed = (const uint16_t*)m.files.get("model.embed_tokens.weight", "BF16", (size_t)VOCAB * H * 2, {VOCAB, H}).p;
    if (const char* v = getenv("GPTOSS_DFLASH")) m.cfg.dflash_dir = v;   // [gptoss-r15-dflash]
    if (const char* v = getenv("GPTOSS_DF_CTX")) m.cfg.df_ctx = atoi(v);
    if (!m.cfg.dflash_dir.empty()) {   // (review L7) a drafter that cannot be used leaves the engine without it, never fails the load
        std::string why;
        std::ifstream cf(m.cfg.dflash_dir + "/config.json");
        std::string js((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>()), c;
        for (char ch : js)
            if (!isspace((unsigned char)ch)) c += ch;
        if (c.empty()) why = "no config.json";
        for (const char* k : {"\"num_hidden_layers\":8", "\"hidden_size\":2880", "\"intermediate_size\":7680", "\"num_attention_heads\":64",
                              "\"num_key_value_heads\":8", "\"head_dim\":64", "\"mask_token_id\":200000", "\"target_layer_ids\":[1,9,17,25,33]",
                              "\"rope_theta\":150000", "\"factor\":32.0", "\"original_max_position_embeddings\":4096"})
            if (why.empty() && c.find(k) == std::string::npos) why = std::string("config.json lacks ") + k;
        if (why.empty()) {
            try {
                m.dfiles.open_file(m.cfg.dflash_dir + "/model.safetensors");
            } catch (const std::exception& ex) {
                why = ex.what();
            }
        }
        if (!why.empty()) {
            fprintf(stderr, "gptoss: drafter %s not loaded: %s\n", m.cfg.dflash_dir.c_str(), why.c_str());
            m.cfg.dflash_dir.clear();
        }
    }
    int ndev = 0;
    CU(cudaGetDeviceCount(&ndev));
    // device routing needs bit-identical router logits (cuBLAS) on every card: only promised for the same GPU model
    // (review r4 M1); mixed cards route on the host
    if (m.cfg.pf_route) {
        cudaDeviceProp p0{};
        for (size_t i = 0; i < cfg.devices.size(); i++) {
            GCHECK(cfg.devices[i] >= 0 && cfg.devices[i] < ndev, "device %d not visible (%d devices; is CUDA_VISIBLE_DEVICES set?)", cfg.devices[i], ndev);
            cudaDeviceProp pi{};
            CU(cudaGetDeviceProperties(&pi, cfg.devices[i]));
            if (i == 0)
                p0 = pi;
            else if (strcmp(pi.name, p0.name) != 0 || pi.multiProcessorCount != p0.multiProcessorCount || pi.major != p0.major || pi.minor != p0.minor)
                m.cfg.pf_route = false;
        }
        if (!m.cfg.pf_route && m.cfg.verbose) fprintf(stderr, "gptoss: the cards are not one model: prefill routing on the host\n");
    }
    {   // [gptoss-rel1-arch] the build has sm_86 code only (8.0 cards cannot run it), and the certified two-stage head's
        // 1 % rounding margin was measured on RTX 3090 tensor cores: 8.7 / 8.9 cards run the sm_86 code with the full
        // head (exact by construction), untested; other generations are not supported in 1.5 (review rel1 M5)
        bool other = false;
        for (int d : cfg.devices) {
            GCHECK(d >= 0 && d < ndev, "device %d not visible (%d devices; is CUDA_VISIBLE_DEVICES set?)", d, ndev);
            cudaDeviceProp pi{};
            CU(cudaGetDeviceProperties(&pi, d));
            GCHECK(pi.major == 8 && pi.minor >= 6, "card %d (%s) is compute capability %d.%d: gpt-oss-120B supports "
                   "compute capability 8.6 (RTX 3090 class; 8.9 runs untested) in this release", d, pi.name, pi.major, pi.minor);
            other = other || pi.minor != 6;
        }
        if (other) {
            const bool lm_set = getenv("GPTOSS_LM") != nullptr, samp_set = getenv("GPTOSS_SAMP") != nullptr;
            if (!lm_set) m.cfg.lm2 = false;
            if (!samp_set) m.cfg.samp2 = false;
            if (!getenv("GPTOSS_LM_SCREEN")) m.cfg.lm_screen_mma = false;
            fprintf(stderr, "gptoss: not every card is compute capability 8.6 (RTX 3090 class): the output head runs in full "
                    "(%s), untested on these cards\n", lm_set || samp_set ? "except where GPTOSS_LM / GPTOSS_SAMP say otherwise"
                    : "the two-stage head's margin was measured on RTX 3090s only");
        }
    }
    // [gptoss-rel1-cards] the weights split over 2..4 cards (one card cannot hold its share: 90 MLP blocks per card > 45);
    // tested on exactly three 24 GB cards
    GCHECK(cfg.devices.size() >= 2, "gpt-oss-120B needs 2 to 4 cards with together more than %.0f GB free (the weights) "
           "plus room for the context (tested: three 24 GB RTX 3090s); 1 card given", m.files.total_bytes() / 1e9);
    if (cfg.devices.size() != 3)
        fprintf(stderr, "gptoss: WARNING %zu cards: accepted, but only three cards are tested\n", cfg.devices.size());
    std::vector<long long> free_mb;
    for (int d : cfg.devices) {
        GCHECK(d >= 0 && d < ndev, "device %d not visible (%d devices; is CUDA_VISIBLE_DEVICES set?)", d, ndev);
        // flags before the context exists (review L8); a context another engine of this process made keeps its own
        if (cudaInitDevice(d, cudaDeviceScheduleSpin | cudaDeviceMapHost, cudaInitDeviceFlagsAreValid) != cudaSuccess) {
            cudaGetLastError();
            if (cfg.verbose) fprintf(stderr, "gptoss: card %d keeps its existing scheduling flags\n", d);
        }
        CU(cudaSetDevice(d));
        size_t f = 0, t = 0;
        CU(cudaMemGetInfo(&f, &t));
        free_mb.push_back((long long)(f >> 20));
    }
    bool can = true;
    for (int a : cfg.devices)
        for (int b : cfg.devices) {
            int ok = 1;
            if (a != b) CU(cudaDeviceCanAccessPeer(&ok, a, b));
            can = can && ok;
        }
    GCHECK(can || m.cfg.comm != 1, "GPTOSS_COMM=p2p, but not every pair of cards has peer access");
    m.p2p = can && m.cfg.comm != 2;
    if (!m.p2p) m.cfg.ar_split = false;   // (the split all-reduce exists over P2P only: review rel2 L2)
    {   // lanes of a prefill chunk (round 14): the scratch of one compute step holds lane_cap rows
        if (!m.lanes_on()) m.cfg.chunk = std::min(m.cfg.chunk, 4096);   // one lane per chunk: its scratch is the chunk's
        m.cfg.pf_nlanes = std::max(0, std::min(MAXLN, m.cfg.pf_nlanes));   // also when set in code (review r14 L3)
        const int C = m.cfg.chunk, fn = m.cfg.pf_nlanes;
        m.lane_cap = !m.lanes_on() ? C : fn > 0 ? std::max(std::min(C, 511), (C + fn - 1) / fn) : std::max({std::min(C, 511), (C + 2) / 3, std::min((C + 1) / 2, 2048)});
        m.rt_stride = route_ints((size_t)m.lane_cap);
        m.pp_lane = (size_t)m.lane_cap / m.ngpu + 2;
    }
    if (m.p2p)
        for (int a : cfg.devices)
            for (int b : cfg.devices) {
                if (a == b) continue;
                CU(cudaSetDevice(a));
                const cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
                GCHECK(e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled, "peer access %d -> %d: %s", a, b, cudaGetErrorString(e));
            }
    if (m.cfg.verbose)
        fprintf(stderr, "gptoss: decode all-reduce over %s, prefill experts %s\n", m.p2p ? "P2P (peer memory)" : "pinned host memory (no peer access or GPTOSS_COMM=host; prefill collectives staged by the copy engines)",
                m.cfg.pf_fused ? "fused MXFP4 GEMM" : "dequant + cuBLAS");
    if (!m.cfg.dflash_dir.empty() && !(m.cfg.ar_trail == 1 && !m.cfg.ar_split)) {
        fprintf(stderr, "gptoss: the drafter needs the trailer all-reduce (GPTOSS_AR_TRAIL=1, no GPTOSS_AR_SPLIT); not loaded\n");
        m.cfg.dflash_dir.clear();
    }
    if (!m.cfg.dflash_dir.empty()) {
        m.dfh = new DfHost;
        m.dfh->cap = std::max(256, m.cfg.df_ctx / DF_TK * DF_TK);
        m.dfh->head = m.dfh->cap / 4;
        m.dfh->slot.resize(std::max(1, m.cfg.slots));
    }
    m.plan = make_plan(cfg.devices, free_mb);
    yarn(m.invf, &m.att);
    m.R.resize(m.ngpu);
    m.bar.n = m.ngpu;
    m.pool.dead = &m.bar.dead;
    for (int g = 0; g < m.ngpu; g++) {
        m.R[g].rank = g;
        m.R[g].dev = cfg.devices[g];
    }
    GCHECK(!(kvq(m.cfg.kv8) && m.cfg.pf_attn == 0), "GPTOSS_KV=fp8 / k12 needs a flash prefill attention (GPTOSS_PF_ATTN=sgemm reads bf16 K/V)");
    m.pool.start(m.ngpu, [&m](int r) { CU(cudaSetDevice(m.R[r].dev)); });
    try {
        m.pool.run([&m](int r) { m.load_rank(m.R[r]); });
    } catch (const std::exception& ex) {
        // [gptoss-rel1-cards] a card without room for its share of the weights: say what is needed (2 x 24 GB cannot
        // hold the ~65 GB)
        if (!strstr(ex.what(), "cannot allocate")) throw;   // (a device allocation, not pinned host memory: review rel1 L13)
        std::string fr;
        for (size_t i = 0; i < free_mb.size(); i++) fr += (i ? ", " : "") + std::to_string(free_mb[i] / 1024) + " GiB";
        GCHECK(false, "the weights do not fit: each of the %d cards needs more than about %.1f GiB free for its share "
               "of the %.0f GB of weights plus context; free: %s (tested: three 24 GB RTX 3090s). First error: %s", m.ngpu,
               m.files.total_bytes() / (double)m.ngpu / (1 << 30), m.files.total_bytes() / 1e9, fr.c_str(), ex.what());
    }
    const int seq_asked = m.cfg.ctx_all ? std::max(256, m.cfg.max_ctx) : 0;   // C API: max_ctx came in as the rows of one conversation
    if (m.cfg.ctx_all) m.cfg.max_ctx = 0x7fffff00;   // the pool: whatever fits (no "reduced" warning: nothing was asked for)
    long long fit = m.cfg.max_ctx;
    for (const Rank& r : m.R) fit = std::min(fit, ctx_fit(r, m.cfg.pf_attn == 0, m.cfg.kv8, m.cfg.reserve_extra, seq_asked, m.cfg.chunk));
    if (m.cfg.ctx_all && fit >= 256) m.cfg.max_ctx = (int)fit;
    GCHECK(fit >= 256, "the weights are loaded but only %lld context rows fit in the free VRAM (need >= 256); free memory on the cards or lower --chunk", fit);
    if (fit < m.cfg.max_ctx) {
        fprintf(stderr, "gptoss: WARNING context reduced from %d to %lld rows: that is what fits in the free VRAM\n", m.cfg.max_ctx, fit);
        m.cfg.max_ctx = (int)fit;
    }
    m.pf_rows = seq_asked ? std::min(seq_asked, m.cfg.max_ctx) : m.cfg.max_ctx;
    m.smax = std::max(SMIN, (size_t)512 * m.pf_rows);
    m.pool.run([&m](int r) { m.alloc_ctx(m.R[r]); });
    m.lim = m.cfg.max_ctx;   // [srv-gptoss-capi] (bind narrows it to a slot's rows)
    CU(cudaSetDevice(cfg.devices[0]));
    if (m.p2p) {
        for (int g = 0; g < m.ngpu; g++) {
            m.R[g].ar.nw = m.ngpu;
            m.R[g].tb.nw = m.ngpu;
            for (int k = 0; k < m.ngpu; k++) m.R[g].tb.wr[k] = m.R[k].tb.rd;
            for (int k = 0; k < m.ngpu; k++) m.R[g].ar.wr[k] = m.R[k].ar.rd;
        }
    } else {
        // one pinned buffer mapped on every card (unified addressing: the host pointer is the device pointer): the
        // all-reduce words (k_ar_host's layout is the larger one: 3 ARW 8 B words per row against ARW 16 B words of
        // k_ar_t), then the head threshold words of every rank
        const size_t nb = (size_t)AR_SLOTS * m.ngpu * MAXR * 3 * ARW * 8, nt = (size_t)4 * MAXR * sizeof(uint2);
        // (the cards poll these words: no pageable or smaller form exists; a few MB, allocated before the larger staging)
        GCHECK(pin_try((void**)&m.ar_host, nb + nt, cudaHostAllocPortable | cudaHostAllocMapped, 0),
               "pinned host memory for the all-reduce words without P2P (%.1f MB) could not be allocated: free host RAM", (nb + nt) / 1e6);
        m.shost.push_back(m.ar_host);
        memset(m.ar_host, 0, nb + nt);
        m.ar_host_nb = nb + nt;
        for (int g = 0; g < m.ngpu; g++) {
            m.R[g].ar.nw = 1;
            m.R[g].ar.sh = 1;
            m.R[g].ar.wr[0] = m.ar_host;
            m.R[g].ar.rd = m.ar_host;
            m.R[g].tb.nw = 1;   // [gptoss-rel2-hostar] the two-stage head's threshold exchange: one store, the ranks poll
            m.R[g].tb.wr[0] = m.R[g].tb.rd = (uint2*)((char*)m.ar_host + nb);
        }
    }
    // [gptoss-rel3-pin] host staging of the prefill under host-RAM pressure: the no-P2P lane staging falls back to the
    // one-lane host form (one lane per chunk, chunk <= lane_cap), the chunk staging (embedding rows, host routing)
    // halves the chunk; every scratch was sized for the larger chunk. The lane staging goes first, then the chunk
    // staging at the final chunk (nothing pinned for a size given up); a halving pins the lane staging again at the
    // new size.
    auto free_lanes = [&m] {
        for (int g = 0; g < m.ngpu; g++)
            if (m.R[g].hstage) {
                cudaFreeHost(m.R[g].hstage);
                m.shost.erase(std::find(m.shost.begin(), m.shost.end(), (void*)m.R[g].hstage));
                m.R[g].hstage = nullptr;
            }
    };
    for (bool first = true;; first = false) {
        if (!m.p2p && m.lanes_on()) {   // [gptoss-rel2-hostpf] staging rows of every rank's lanes
            bool ok = true;
            for (int g = 0; g < m.ngpu && ok; g++) {
                const size_t nb = ((size_t)m.nlanes(m.cfg.chunk) * std::max(m.lane_cap, MAXR) + MAXR) * H * 4;   // + the one-row lane's region (second review L2: a few rows)
                void* hp = nullptr;
                ok = pin_try(&hp, nb, cudaHostAllocPortable, g == m.ngpu - 1 ? 1 | 8 : 1);
                if (ok) {
                    m.shost.push_back(hp);
                    m.R[g].hstage = (float*)hp;
                }
            }
            if (!ok) {   // (review rel2 L3) the one-lane host form of rel1: no staging rows
                free_lanes();
                m.cfg.pf_hostlanes = false;
                const int ck = std::min(m.cfg.chunk, m.lane_cap);   // (one lane holds the chunk: its scratch is lane_cap rows)
                fprintf(stderr, "gptoss: WARNING pinned host memory for the prefill lane staging failed (host RAM low): the one-lane "
                        "host form (as GPTOSS_PF_HOSTLANES=0), prefill chunk %d -> %d rows (slower prefill)\n", m.cfg.chunk, ck);
                m.cfg.chunk = ck;
                bool df = false;
                for (Rank& r : m.R)
                    if (r.df && r.df->ptap) r.df->ptap = nullptr, df = true;   // (its taps come from the lanes only)
                if (df) fprintf(stderr, "gptoss: the drafter's prefill taps are off with one lane (its prompt rows come from the caller)\n");
            }
        }
        const size_t t = (size_t)m.cfg.chunk;   // the clamped chunk (review r14 L5)
        void* q[8];
        const size_t nb[8] = {t * 16, t * 16, (size_t)MAXLN * (t * 4 / 32 + NE) * 12, t * 16, t * 16, t * 16, t * 16, t * H * 4};
        int k = 0;
        while (k < 8 && pin_try(&q[k], nb[k], cudaHostAllocPortable, first ? 2 : 0)) k++;
        if (k == 8) {
            m.sh_exp = (int*)q[0], m.sh_inv = (int*)q[1], m.sh_units = (int*)q[2], m.sh_ids = (int*)q[3];
            m.sh_rw = (float*)q[4], m.sh_rows = (int*)q[5], m.sh_w = (float*)q[6], m.sh_h = (float*)q[7];
            for (void* x : q) m.shost.push_back(x);
            break;
        }
        for (int j = 0; j < k; j++) cudaFreeHost(q[j]);
        size_t tot = 0;
        for (size_t x : nb) tot += x;
        GCHECK(m.cfg.chunk > 256, "pinned host memory for the prefill staging (%.1f MB at a %d-row chunk) could not be allocated: free host RAM", tot / 1e6, m.cfg.chunk);
        const int ck = std::max(256, m.cfg.chunk / 2);
        fprintf(stderr, "gptoss: WARNING pinned host memory for the prefill staging failed (host RAM low): prefill chunk %d -> %d rows\n", m.cfg.chunk, ck);
        m.cfg.chunk = ck;
        free_lanes();   // (pinned again at the new size, if the lanes are on)
    }
    m.load_s = secs(t0);
}
Engine::~Engine() { delete p_; }
// everything goes back in reverse order (review M4): a second engine in the same process must find the memory free
Impl::~Impl() {
    pool.shutdown();
    // every stream of every card first (review L1): a peer copy queued on one card's copy stream reads another
    // card's buffers, so nothing is freed while any stream still has work
    // No blocking synchronise (review r4 L3): with events and tag waits a card's streams depend on the other cards'
    // streams, and a dead peer must not hang the shutdown. The streams are polled up to a deadline (a healthy engine
    // is idle here; a tag wait traps after AR_WAIT_NS, so a failed step ends by itself well inside it). If work is
    // still queued after the deadline nothing of the cards is freed or destroyed: queued work may still read the
    // buffers, and a leak at the end of a broken engine is better than a hang or a use after free.
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    bool quiet = false;
    for (long spins = 0; !quiet; spins++) {
        quiet = true;
        for (Rank& r : R) {
            if (cudaSetDevice(r.dev) != cudaSuccess) continue;
            for (cudaStream_t s : {r.st, r.st2})
                if (s && cudaStreamQuery(s) == cudaErrorNotReady) quiet = false;   // an error state counts as finished
        }
        if (quiet || std::chrono::steady_clock::now() > t_end) break;
        if (spins > 1000) std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (!quiet) {
        fprintf(stderr, "gptoss: streams still busy 30 s after the last step (a card stopped answering); the engine's device memory is left to the process exit: restart the process before loading a model on these cards again\n");
        cudaGetLastError();
        return;
    }
#ifdef GPTOSS_AR_TIME   // (probe builds; after the bounded poll, so no blocking synchronise: review r9 L5)
    for (Rank& r : R) {
        if (!r.ar.rd || cudaSetDevice(r.dev) != cudaSuccess) continue;
        unsigned long long acc[8] = {0}, t[8];
        for (int s = 0; s < AR_SLOTS; s++) {
            cudaMemcpy(t, (uint4*)r.ar.rd + (((size_t)s * ngpu + r.rank) * MAXR) * ARW + 800, sizeof t, cudaMemcpyDeviceToHost);
            for (int k = 0; k < 8; k++) acc[k] += t[k];
        }
        const double n = acc[0] ? (double)acc[0] : 1.;
        fprintf(stderr, "AR_TIME card %d: %llu calls; ns since entry: stores issued %.0f, trailer + residual %.0f, valid %.0f, end %.0f; %.3f extra checks\n",
                r.rank, acc[0], acc[1] / n, acc[2] / n, acc[4] / n, acc[5] / n, acc[6] / n);
    }
#endif
    for (Rank& r : R) {
        if (cudaSetDevice(r.dev) != cudaSuccess) continue;
        for (cudaGraphExec_t g : r.gexec)
            if (g) cudaGraphExecDestroy(g);
        for (cudaGraphExec_t g : r.gexec_s)   // [srv-gptoss-capi]
            if (g) cudaGraphExecDestroy(g);
        for (cudaGraphExec_t g : r.gexec_g)   // [gptoss-r15-samp2]
            if (g) cudaGraphExecDestroy(g);
        if (r.blas) cublasDestroy(r.blas);
        if (r.st2) cudaStreamDestroy(r.st2);
        for (int l = 0; l < MAXLN; l++) {
            if (r.evP[l]) cudaEventDestroy(r.evP[l]);
            if (r.evH[l]) cudaEventDestroy(r.evH[l]);
            if (r.evS[l]) cudaEventDestroy(r.evS[l]);
            if (r.evD[l]) cudaEventDestroy(r.evD[l]);
        }
        if (r.st) cudaStreamDestroy(r.st);
        df_free(r);   // [gptoss-r15-dflash]
        for (void* p : r.dptr) cudaFree(p);
        for (void* p : r.hptr) cudaFreeHost(p);
    }
    for (void* p : shost) cudaFreeHost(p);
    // peer access stays enabled: it is a property of the process' contexts, and another live engine may be using it
    cudaGetLastError();
}
int Engine::max_ctx() const { return p_->cfg.max_ctx; }
int Engine::chunk() const { return p_->cfg.chunk; }
int Engine::pos() const { return p_->pos; }
double Engine::load_seconds() const { return p_->load_s; }
const Plan& Engine::plan() const { return p_->plan; }
bool Engine::p2p() const { return p_->p2p; }
void Engine::set_dump(const Dump& d) { p_->dump = d; }
void Engine::reset() { p_->pos = 0; }
void Engine::truncate(int n) {
    GCHECK(n >= 0 && n <= p_->pos && (n == 0 || p_->pos - n <= 128), "truncate %d of %d", n, p_->pos);
    p_->pos = n;
}

// ---------------------------------------------------------------- [srv-gptoss-capi] server hooks (see gptoss.h)
void Engine::set_sampling(float temperature, unsigned long long seed, bool cands, bool draw) {
    Impl& m = *p_;
    m.samp = temperature > 0.f;
    m.spv = SampParam{m.samp ? 1.f / temperature : 1.f, (unsigned)seed, (unsigned)(seed >> 32), cands ? 1 : 0, draw ? 1 : 0, 0, -1, 0.f};
}
// [gptoss-rel2-penalty]
static void pen_param(Impl& m, SampParam& sp, int slot, int id, float val) {
    GCHECK(slot < m.cfg.slots, "penalty row of slot %d of %d", slot, m.cfg.slots);
    GCHECK(slot < 0 || m.R[0].pen, "penalties need the sampled head's buffers");
    sp.pen = slot < 0 ? 0 : slot + 1;
    sp.pen_id = id;
    sp.pen_val = val;
}
void Engine::set_sampling_penalty(int slot, int id, float val) { pen_param(*p_, p_->spv, slot, id, val); }
void Engine::pen_reset(int slot, const int* ids, const float* vals, int n) {
    Impl& m = *p_;
    GCHECK(!m.broken.load(), "the engine stopped after an earlier error");
    GCHECK(slot >= 0 && slot < m.cfg.slots && m.R[0].pen, "pen_reset: slot %d of %d", slot, m.cfg.slots);
    for (int i = 0; i < n; i++) GCHECK(ids[i] >= 0 && ids[i] < VOCAB, "pen_reset: token id %d", ids[i]);
    m.pool.run([&](int g) {
        Rank& r = m.R[g];
        std::vector<float> row((size_t)r.vn, 0.f);
        for (int i = 0; i < n; i++)
            if (ids[i] >= r.v0 && ids[i] < r.v0 + r.vn) row[ids[i] - r.v0] = vals[i];
        CU(cudaMemcpyAsync(r.pen + (size_t)slot * r.vn, row.data(), (size_t)r.vn * 4, cudaMemcpyHostToDevice, r.st));
        CU(cudaStreamSynchronize(r.st));   // (the host row is this call's)
    });
}
SampleCard Engine::sample_card(int g, int row) const {
    const Rank& r = p_->R[g];
    return SampleCard{r.sout_host + (size_t)row * SAMP_OUT, row == 0 ? r.slog_host : nullptr, r.v0, r.vn};
}
int Engine::num_slots() const { return p_->cfg.slots; }
bool Engine::broken() const { return p_->broken.load(); }
void Engine::bind(int slot, int base, int pos, int lim) {
    Impl& m = *p_;
    GCHECK(slot >= 0 && slot < m.cfg.slots && base >= 0 && lim >= 1 && (long long)base + lim <= m.cfg.max_ctx && pos >= 0 && pos <= lim,
           "bind slot %d base %d pos %d lim %d", slot, base, pos, lim);
    m.b_slot = slot;
    m.b_base = base;
    m.pos = pos;
    m.lim = lim;
}
// the arrays that hold a full layer's rows on a card: bf16 K and V, or the 8-bit store's K, V and scales (row-major
// each, so rows move as bytes in either format)
struct FullArr {
    unsigned char* p;   // row 0 of the layer
    size_t rb;          // bytes per row
};
static std::vector<FullArr> full_arrs(const Impl& m, const Rank& r, int li) {
    const size_t kvrow = (size_t)r.kvn * HD;
    if (kvq(m.cfg.kv8)) {
        const size_t rows = (size_t)m.cfg.max_ctx + KV8_PAD;
        std::vector<FullArr> a{{(unsigned char*)r.K8 + (size_t)li * rows * kvrow, kvrow},
                               {(unsigned char*)r.V8 + (size_t)li * rows * kvrow, kvrow},
                               {(unsigned char*)(r.S8 + (size_t)li * rows * r.kvn), (size_t)r.kvn * 4}};
        if (r.K4) a.push_back({(unsigned char*)(r.K4 + (size_t)li * rows * r.kvn * 8), (size_t)r.kvn * 32});
        return a;
    }
    const size_t ls = (size_t)m.cfg.max_ctx * kvrow;
    return {{(unsigned char*)(r.Kf + li * ls), kvrow * 2}, {(unsigned char*)(r.Vf + li * ls), kvrow * 2}};
}
static size_t full_row_bytes(const Impl& m, const Rank& r) {   // one row of every full layer on this card
    size_t b = 0;
    for (const FullArr& a : full_arrs(m, r, 0)) b += a.rb;
    return b * (NL / 2);
}
// full layers' rows [src, src + n) -> [dst, dst + n) of every layer (non-overlapping), on every card
void Engine::copy_rows(int src, int dst, int n) {
    Impl& m = *p_;
    GCHECK(!m.broken.load(), "the engine stopped after an earlier error");
    GCHECK(n >= 0 && src >= 0 && dst >= 0 && src + n <= m.cfg.max_ctx && dst + n <= m.cfg.max_ctx &&
           (src + n <= dst || dst + n <= src), "copy_rows %d -> %d of %d", src, dst, n);
    if (n == 0) return;
    for (Rank& r : m.R) {
        CU(cudaSetDevice(r.dev));
        CU(cudaStreamSynchronize(r.st));
        for (int li = 0; li < NL / 2; li++)
            for (const FullArr& a : full_arrs(m, r, li))
                CU(cudaMemcpyAsync(a.p + (size_t)dst * a.rb, a.p + (size_t)src * a.rb, (size_t)n * a.rb, cudaMemcpyDeviceToDevice, r.st));
        CU(cudaStreamSynchronize(r.st));
    }
}
// [gptoss-rel3-pin] b bytes between the card and a host blob through the card's pinned staging buffer (made on first
// use: 32 MiB, else 4 MiB; host-RAM pressure: pageable copies with one log line, slower, the same bytes)
static void stage_copy(Rank& r, unsigned char* dv, unsigned char* hv, size_t b, bool save) {
    if (r.stage_off && ++r.stage_tries >= 256) r.stage_off = false, r.stage_tries = 0;   // (pressure passes: try pinning again)
    if (!r.stage && !r.stage_off) {
        for (size_t n : {(size_t)32 << 20, (size_t)4 << 20}) {
            void* p = nullptr;
            if (pin_try(&p, n, cudaHostAllocDefault, 4)) {
                r.hptr.push_back(p);
                r.stage = (unsigned char*)p;
                r.stage_n = n;
                break;
            }
        }
        if (!r.stage) {
            if (!r.stage_said) fprintf(stderr, "gptoss: WARNING card %d: no pinned host memory for the row copies (host RAM low): pageable copies (slower; pinning is tried again every 256 copies)\n", r.rank);
            r.stage_off = r.stage_said = true;
        } else if (r.stage_said) {
            fprintf(stderr, "gptoss: card %d: pinned staging for the row copies again (%zu MiB)\n", r.rank, r.stage_n >> 20);
            r.stage_said = false;
        }
    }
    if (r.stage_off) {   // (synchronous on a pageable buffer; the stream is idle here)
        if (save) CU(cudaMemcpyAsync(hv, dv, b, cudaMemcpyDeviceToHost, r.st));
        else CU(cudaMemcpyAsync(dv, hv, b, cudaMemcpyHostToDevice, r.st));
        CU(cudaStreamSynchronize(r.st));
        return;
    }
    for (size_t o = 0; o < b; o += r.stage_n) {
        const size_t k = std::min(r.stage_n, b - o);
        if (save) {
            CU(cudaMemcpyAsync(r.stage, dv + o, k, cudaMemcpyDeviceToHost, r.st));
            CU(cudaStreamSynchronize(r.st));
            memcpy(hv + o, r.stage, k);
        } else {
            memcpy(r.stage, hv + o, k);
            CU(cudaMemcpyAsync(dv + o, r.stage, k, cudaMemcpyHostToDevice, r.st));
            CU(cudaStreamSynchronize(r.st));
        }
    }
}
// the bound slot's ring set (18 sliding layers x 256 rows) as a host blob, and back (every card in parallel through its
// pinned staging buffer, as rows_save)
static void ring_copy(Impl& m, unsigned char* host, bool save) {
    std::vector<size_t> off(m.R.size(), 0);
    for (size_t g = 1; g < m.R.size(); g++) off[g] = off[g - 1] + 2 * (size_t)RING_ROWS * m.R[g - 1].kvn * HD * sizeof(bf16);
    m.pool.run([&](int g) {
        Rank& r = m.R[g];
        CU(cudaSetDevice(r.dev));
        CU(cudaStreamSynchronize(r.st));
        const size_t kvrow = (size_t)r.kvn * HD, b = (size_t)RING_ROWS * kvrow * sizeof(bf16), o = (size_t)m.b_slot * RING_ROWS * kvrow;
        for (int kv = 0; kv < 2; kv++)
            stage_copy(r, (unsigned char*)((kv ? r.Vr : r.Kr) + o), host + off[g] + kv * b, b, save);
    });
}
void Engine::ring_save(std::vector<unsigned char>& out) const {
    Impl& m = *p_;
    size_t total = 0;
    for (const Rank& r : m.R) total += 2 * (size_t)RING_ROWS * r.kvn * HD * sizeof(bf16);
    out.resize(total);
    ring_copy(m, out.data(), true);
}
void Engine::ring_load(const std::vector<unsigned char>& in, int n) {
    Impl& m = *p_;
    GCHECK(!m.broken.load(), "the engine stopped after an earlier error");
    GCHECK(n >= 0 && n <= m.lim, "ring_load at %d", n);
    size_t total = 0;
    for (const Rank& r : m.R) total += 2 * (size_t)RING_ROWS * r.kvn * HD * sizeof(bf16);
    GCHECK(in.size() == total, "ring_load: the blob is not this engine's");
    ring_copy(m, const_cast<unsigned char*>(in.data()), false);
    m.pos = n;
}
// full-layer rows [src, src + n) of every layer and card as a host blob (card, layer, K then V), and back to [dst, dst + n)
size_t Engine::row_bytes() const {
    size_t b = 0;
    for (const Rank& r : p_->R) b += full_row_bytes(*p_, r);
    return b;
}
// every card in parallel on its rank thread, through its pinned staging buffer (pageable copies ran at ~2.5 GB/s)
static void rows_copy(Impl& m, int dev_row, int n, unsigned char* host, int host_rows, bool save) {
    std::vector<size_t> off(m.R.size(), 0);
    for (size_t g = 1; g < m.R.size(); g++)
        off[g] = off[g - 1] + (size_t)host_rows * full_row_bytes(m, m.R[g - 1]);
    m.pool.run([&](int g) {
        Rank& r = m.R[g];
        CU(cudaSetDevice(r.dev));
        CU(cudaStreamSynchronize(r.st));
        unsigned char* hv = host + off[g];   // blob of a card: layer by layer, each array's host_rows rows
        for (int li = 0; li < NL / 2; li++)
            for (const FullArr& a : full_arrs(m, r, li)) {
                stage_copy(r, a.p + (size_t)dev_row * a.rb, hv, (size_t)n * a.rb, save);
                hv += (size_t)host_rows * a.rb;
            }
    });
}
void Engine::rows_save(int src, int n, std::vector<unsigned char>& out) const {
    Impl& m = *p_;
    GCHECK(n >= 0 && src >= 0 && src + n <= m.cfg.max_ctx, "rows_save %d + %d", src, n);
    // a fresh blob's pages are touched here for the first time: with 4 KiB pages that bound the copy (2.7 GB/s for
    // 4.8 GB); ask for huge pages before the first touch (THP in madvise mode)
    const size_t sz = row_bytes() * (size_t)n;
    if (out.capacity() < sz) {
        out.reserve(sz);
        const uintptr_t a = ((uintptr_t)out.data() + 4095) & ~(uintptr_t)4095, e = ((uintptr_t)out.data() + out.capacity()) & ~(uintptr_t)4095;
        if (e > a + (2u << 20)) madvise((void*)a, e - a, MADV_HUGEPAGE);
    }
    out.resize(sz);
    if (n) rows_copy(m, src, n, out.data(), n, true);
}
void Engine::rows_load(const std::vector<unsigned char>& in, int dst, int n) {
    Impl& m = *p_;
    GCHECK(!m.broken.load(), "the engine stopped after an earlier error");
    GCHECK(n >= 0 && dst >= 0 && dst + n <= m.cfg.max_ctx && in.size() >= row_bytes() * (size_t)n, "rows_load %d + %d", dst, n);
    // the blob may hold more rows than n (its own row count sets the layout)
    if (n) rows_copy(m, dst, n, const_cast<unsigned char*>(in.data()), (int)(in.size() / row_bytes()), false);
}
void Engine::step_rows_sampled(const StepRow* rows, const RowSamp* rs, int n, int* out) {
    Impl& m = *p_;
    GCHECK(n >= 1 && n <= MAXR, "1..%d rows per decode step", MAXR);
    for (int i = 0; i < n; i++) {
        const bool draw = rs[i].temperature > 0.f;
        m.srow[i] = SampParam{draw ? 1.f / rs[i].temperature : 1.f, (unsigned)rs[i].seed, (unsigned)(rs[i].seed >> 32), 0, draw ? 1 : 0, 0, -1, 0.f};
        if (rs[i].pen) pen_param(m, m.srow[i], rows[i].slot, rs[i].pen_id, rs[i].pen_val);   // [gptoss-rel2-penalty]
    }
    for (int i = 0; i < n; i++)   // (a penalty row has one owner thread per token: one row per sequence)
        for (int j = 0; j < i; j++) GCHECK(!(rs[i].pen && rs[j].pen && rows[i].slot == rows[j].slot), "two penalised rows of slot %d in one step", rows[i].slot);
    m.srows = true;
    step_rows(rows, n, out);
}

#include "gptoss_dflash.inc"   // [gptoss-r15-dflash] loader, passes, Engine::draft_*

std::string Engine::vram_table() const {
    const Impl& m = *p_;
    std::string s = "card  dev  kv-heads  blocks   experts attention  lm_head   router    small       kv  scratch | free before -> after (MiB)\n";
    char b[512];
    for (const Rank& r : m.R) {
        auto mb = [&](const char* c) {
            auto it = r.bytes.find(c);
            return it == r.bytes.end() ? 0.0 : it->second / 1048576.0;
        };
        snprintf(b, sizeof b, "%4d %4d  %d..%d      %2d..%2d %9.0f %9.0f %8.0f %8.0f %8.0f %8.0f %8.0f | %6.0f -> %6.0f\n", r.rank, r.dev, r.kv0,
                 r.kv0 + r.kvn - 1, r.jb0, r.jb0 + r.KB - 1, mb("experts"), mb("attention"), mb("lm_head"), mb("router"), mb("small"),
                 mb("kv"), mb("scratch"), r.free_before / 1048576.0, r.free_after / 1048576.0);
        s += b;
    }
    return s;
}

int Engine::decode(int token) {
    int out = -1;
    decode_rows(&token, 1, &out);
    return out;
}

void Engine::decode_rows(const int* tokens, int R, int* out) {
    Impl& m = *p_;
    GCHECK(R >= 1 && R <= MAXR, "1..%d rows per decode step", MAXR);
    GCHECK(m.pos + R <= m.lim, "context full (%d)", m.lim);   // [srv-gptoss-capi] the bound slot's rows
    StepRow rows[MAXR];
    for (int i = 0; i < R; i++) rows[i] = StepRow{m.b_slot, m.b_base, m.pos + i, tokens[i]};
    if (m.samp && R == 1) {   // [srv-gptoss-capi] a sampled one-row step (set_sampling)
        GCHECK(!m.spv.pen || m.spv.pen - 1 == m.b_slot, "the penalty row is slot %d's, the bound slot is %d", m.spv.pen - 1, m.b_slot);
        m.srow[0] = m.spv;
        m.srows = true;
    }
    step_rows(rows, R, out);
    m.pos += R;
}

void Engine::step_rows(const StepRow* rows, int R, int* out) {
    Impl& m = *p_;
    const bool want_s = std::exchange(m.srows, false);   // [srv-gptoss-capi] (cleared whatever the checks below say)
    GCHECK(!m.broken.load(), "the engine stopped after an earlier error");
    GCHECK(R >= 1 && R <= MAXR, "1..%d rows per decode step", MAXR);
    for (int i = 0; i < R; i++) {
        const StepRow& q = rows[i];
        GCHECK(q.token >= 0 && q.token < VOCAB, "token id %d outside the vocabulary", q.token);
        GCHECK(q.slot >= 0 && q.slot < m.cfg.slots, "row %d: slot %d of %d", i, q.slot, m.cfg.slots);
        GCHECK(q.base >= 0 && q.pos >= 0 && (long long)q.base + q.pos < m.cfg.max_ctx, "row %d: K/V row %lld outside the context %d", i,
               (long long)q.base + q.pos, m.cfg.max_ctx);
    }
    const bool dumping = m.dump.layers || m.dump.logits;
    const bool sampled = want_s && !dumping;   // [srv-gptoss-capi] the sampled head, own graphs
    GCHECK(!sampled || R == 1 || !m.srow[0].cands, "candidates (filtered sampling) take one row per step");
    // [gptoss-r15-samp2] plain temperature rows take the certified two-stage head (the same tokens as the full head);
    // a step that wants candidates (top_k / top_p / min_p, constraints: one row) keeps the full head
    const int smode = !sampled ? 0 : (m.cfg.samp2 && m.R[0].s8 && !m.srow[0].cands) ? 2 : 1;
    if (sampled)
        for (int g = 0; g < m.ngpu; g++) memcpy(m.R[g].sp_host, m.srow, (size_t)R * sizeof(SampParam));
    {   // bad arguments must not leave the engine broken: checked before anything is launched
        const bool one = R == 1 && rows[0].slot == 0 && rows[0].base == 0;
        GCHECK(one || !(m.cfg.dense12 || m.cfg.attn_win || m.cfg.ar_split || dumping),
               "the opt-in kernels (12-bit dense, window attention, split all-reduce) and the dump path take one row of slot 0 per step");
    }
    struct Guard {   // an exception below leaves the cards mid-step
        Impl& m;
        bool ok = false;
        ~Guard() { if (!ok) m.broken.store(true); }
    } guard{m};
    m.ar_epoch();   // [gptoss-rel2-tagwrap]
    for (int i = 0; i < R; i++) m.embed_row(rows[i].token, m.R[0].h_host + (size_t)i * H);
    for (int g = 0; g < m.ngpu; g++) {
        if (g) memcpy(m.R[g].h_host, m.R[0].h_host, (size_t)R * H * 4);
        int* rt = m.R[g].pos_host;
        for (int i = 0; i < MAXR; i++) {
            const StepRow& q = rows[i < R ? i : 0];
            rt[i] = q.pos;
            rt[MAXR + i] = q.base;
            rt[2 * MAXR + i] = q.slot;
        }
        rt[3 * MAXR] = (int)(m.df_tw % DF_RING);   // [gptoss-r15-dflash]
    }
    m.df_last = m.df_tw;
    m.df_last_n = R;
    for (int i = 0; i < R; i++) m.df_rs[i] = rows[i].slot, m.df_rp[i] = rows[i].pos;   // (review L5: draft_commit checks them)
    m.df_tw += R;
    // Split class of the step (round 9): the full layers launch only as many key splits as the longest row of the
    // step can use (AS_GRID: 8 / 32 / 128 splits = up to 256 / 1,024 / any keys), not 128 mostly empty ones at a
    // short context. A split's sums and the split rule (context length only) do not change; one graph per class.
    int cls = ASC - 1;
    if (m.cfg.attn_grid) {
        int nsmax = 1;
        // the 8-bit store's kernel (k_attn8) splits by 8 tiles of 32 keys: 8 / 32 / 128 splits = up to 2,048 / 8,192 / any keys (review r11 L1)
        const int smin = kvq(m.cfg.kv8) ? 8 * 32 : SPLIT_MIN;
        for (int i = 0; i < R; i++) nsmax = std::max(nsmax, std::min(NSPLIT, (rows[i].pos + 1 + smin - 1) / smin));
        cls = 0;
        while (AS_GRID[cls] < nsmax) cls++;
    }
    const int nsg = AS_GRID[cls], gi = cls * (MAXR + 1) + R;
    bool* const readyv = smode == 2 ? m.graph_g_ready : smode ? m.graph_s_ready : m.graph_ready;
    bool& ready = readyv[gi];
    if (m.cfg.graph && !dumping && !ready && m.graph_retry[smode][R] <= m.steps) {
        // every class of this row count at once (review r9 M2): no capture inside a later step when the context
        // crosses 256 or 1,024 keys
        try {
            m.pool.run([&m, R, smode](int g) {
                Rank& r = m.R[g];
                for (int c = 0; c < ASC; c++) {
                    const int gc = c * (MAXR + 1) + R;
                    cudaGraphExec_t& ge = smode == 2 ? r.gexec_g[gc] : smode ? r.gexec_s[gc] : r.gexec[gc];
                    if (ge) continue;
                    cudaGraph_t gr = nullptr;
                    CU(cudaStreamBeginCapture(r.st, cudaStreamCaptureModeRelaxed));
                    m.launch_step(r, false, R, smode, AS_GRID[c]);
                    CU(cudaStreamEndCapture(r.st, &gr));
                    // (test hook: GPTOSS_TEST_CAPTURE_FAIL=R fails the instantiation of R-row graphs on card 0 once)
                    static std::atomic<int> tfail{test_env("GPTOSS_TEST_CAPTURE_FAIL")};
                    const cudaError_t e = g == 0 && R == tfail.load() && tfail.exchange(0) == R ? cudaErrorMemoryAllocation
                                                                                                : cudaGraphInstantiate(&ge, gr, 0);
                    cudaGraphDestroy(gr);
                    if (e != cudaSuccess) ge = nullptr;
                    CU(e);
                }
            });
            for (int c = 0; c < ASC; c++) readyv[c * (MAXR + 1) + R] = true;
        } catch (const std::exception& ex) {
            // [gptoss-rel1-graph] a capture / instantiate failure (no VRAM for the graph under memory pressure) runs
            // nothing on the cards: undo the capture, run this step without a graph and try again later, instead of
            // the guard marking the engine broken. An error the cards report afterwards is real: it fails as before.
            int dev0 = 0;
            cudaGetDevice(&dev0);
            for (Rank& r : m.R) {
                cudaSetDevice(r.dev);
                cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
                if (cudaStreamIsCapturing(r.st, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
                    cudaGraph_t gr = nullptr;
                    cudaStreamEndCapture(r.st, &gr);
                    if (gr) cudaGraphDestroy(gr);
                }
                cudaGetLastError();
                for (int c = 0; c < ASC; c++) {   // (the row count's graphs on every card, or none: one form per step)
                    cudaGraphExec_t& ge = smode == 2 ? r.gexec_g[c * (MAXR + 1) + R] : smode ? r.gexec_s[c * (MAXR + 1) + R]
                                                                                             : r.gexec[c * (MAXR + 1) + R];
                    if (ge) cudaGraphExecDestroy(ge);
                    ge = nullptr;
                }
                const cudaError_t q = cudaStreamQuery(r.st);   // (a sticky error: thrown, the guard marks the engine broken)
                if (q != cudaErrorNotReady) CU(q);
            }
            cudaSetDevice(dev0);
            m.pool.clear_error();
            m.bar.dead.store(false);
            m.graph_fails++;
            m.graph_retry[smode][R] = m.steps + 256;
            fprintf(stderr, "gptoss: WARNING decode graph capture for %d row(s) failed (%s): steps run without a graph, "
                    "next try in 256 steps\n", R, ex.what());
        }
    }
    m.steps++;
    // (test hook: GPTOSS_TEST_FAULT_STEP=N throws inside the N-th decode step, after the guard: the engine is broken,
    // as after a device fault; the C API then ends the process with code 75)
    static const long long tfault = test_env("GPTOSS_TEST_FAULT_STEP");
    GCHECK(tfault <= 0 || m.steps != tfault, "test fault injected in decode step %lld (GPTOSS_TEST_FAULT_STEP)", m.steps);
    const bool use_graph = m.cfg.graph && !dumping && readyv[gi];
    m.pool.run([&m, use_graph, dumping, R, smode, nsg, gi](int g) {
        Rank& r = m.R[g];
        if (use_graph)
            CU(cudaGraphLaunch(smode == 2 ? r.gexec_g[gi] : smode ? r.gexec_s[gi] : r.gexec[gi], r.st));
        else
            m.launch_step(r, dumping, R, smode, nsg);
        CU(cudaStreamSynchronize(r.st));
    });
    guard.ok = true;
    if (!m.soft_rows || R == 1) {
        for (int i = 0; i < R; i++) out[i] = m.merge_res(i);
        return;
    }
    // [gptoss-rel1-rowfault] (soft_rows: the C API's joint step) a row whose logits are not finite gets -1, the other
    // rows their tokens: one bad sequence must not fail every conversation of the step. A failed card still throws.
    std::string first;
    int bad = 0;
    for (int i = 0; i < R; i++) {
        try {
            out[i] = m.merge_res(i);
        } catch (const std::exception& ex) {
            if (m.broken.load()) throw;
            out[i] = -1;
            if (!bad++) first = ex.what();
        }
    }
    GCHECK(bad < R, "%s", first.c_str());
}
void Engine::set_soft_rows(bool on) { p_->soft_rows = on; }

int Engine::prefill(const int* ids, int n) {
    Impl& m = *p_;
    GCHECK(!m.broken.load(), "the engine stopped after an earlier error");
    GCHECK(n >= 1 && m.pos + n <= m.lim, "prefill of %d rows at %d exceeds the context %d", n, m.pos, m.lim);
    // refused here, before any layer runs (review r12 L1): the sgemm prefill attention's scratch holds one conversation
    GCHECK(m.cfg.pf_attn != 0 || m.pos + n <= m.pf_rows, "prefill of %d rows at %d exceeds the %d rows the sgemm prefill attention is sized for", n, m.pos, m.pf_rows);
    for (int i = 0; i < n; i++) GCHECK(ids[i] >= 0 && ids[i] < VOCAB, "token id %d outside the vocabulary (row %d)", ids[i], i);
    struct Guard {
        Impl& m;
        bool ok = false;
        ~Guard() { if (!ok) m.broken.store(true); }
    } guard{m};
    // [srv-gptoss-capi] the prefill kernels address ring set 0 and full-layer row 0: point them at the bound slot's
    // ring set and rows for this call (restored on every exit)
    struct Bound {
        Impl& m;
        std::vector<std::array<bf16*, 4>> keep;
        std::vector<std::tuple<int8_t*, int8_t*, uint32_t*, uint32_t*>> keep8;
        explicit Bound(Impl& mm) : m(mm) {
            for (Rank& r : m.R) {
                keep.push_back({r.Kr, r.Vr, r.Kf, r.Vf});
                keep8.push_back({r.K8, r.V8, r.S8, r.K4});
                const size_t kvrow = (size_t)r.kvn * HD, ro = (size_t)m.b_slot * RING_ROWS * kvrow, fo = (size_t)m.b_base * kvrow;
                r.Kr += ro;
                r.Vr += ro;
                if (kvq(m.cfg.kv8)) {
                    r.K8 += fo;
                    r.V8 += fo;
                    r.S8 += (size_t)m.b_base * r.kvn;
                    if (r.K4) r.K4 += (size_t)m.b_base * r.kvn * 8;
                } else {
                    r.Kf += fo;
                    r.Vf += fo;
                }
            }
        }
        ~Bound() {
            for (size_t g = 0; g < m.R.size(); g++) {
                m.R[g].Kr = keep[g][0];
                m.R[g].Vr = keep[g][1];
                m.R[g].Kf = keep[g][2];
                m.R[g].Vf = keep[g][3];
                std::tie(m.R[g].K8, m.R[g].V8, m.R[g].S8, m.R[g].K4) = keep8[g];
            }
        }
    } bound{m};
    const int call_pos0 = m.pos;
    // chunks of equal size (round 14): the fewest chunks that hold the call, n / nch rows each (+ 1 in the first
    // n % nch): a short last chunk would pay the per-lane costs for few rows
    const int nch = (n + m.cfg.chunk - 1) / m.cfg.chunk;
    for (int o = 0, ci = 0; o < n; ci++) {
        const int T = n / nch + (ci < n % nch ? 1 : 0);
        for (int t = 0; t < T; t++) m.embed_row(ids[o + t], m.sh_h + (size_t)t * H);
        const int pos0 = m.pos;
        const bool last = o + T == n;
        m.pool.run([&m, pos0, T, last, call_pos0](int g) { m.prefill_chunk(m.R[g], pos0, T, last, call_pos0); });
        if (m.cfg.pf_route)   // device routing: every card routed for itself; one routing for all cards is the contract
            for (int g = 1; g < m.ngpu; g++)
                GCHECK(m.R[g].rsig_host[0] == m.R[0].rsig_host[0], "prefill routing differs between cards %d and 0", g);
        if (m.dfh) {   // [gptoss-r15-dflash] the chunk's rows are in the slot's drafter context (or missing: a pass with a dump)
            DfSlot& ds = m.dfh->slot[m.b_slot];
            if (m.df_pf_live()) {
                if (ds.cnt > pos0) m.df_clear(m.b_slot, pos0, ds.cnt);   // (review L5) a longer earlier sequence: its later rows leave the ring
                if (ds.cnt != pos0) ds.pend.clear();   // the context was not in step with the sequence: its rows before this chunk may be missing
                ds.cnt = pos0 + T;
            }
        }
        m.pos += T;
        o += T;
    }
    guard.ok = true;
    return m.merge_res();
}

}  // namespace gptoss
