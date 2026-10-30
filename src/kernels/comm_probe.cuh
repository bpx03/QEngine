// Topology, startup interconnect probe and the prefill transfer path (both engines). Included once, by comm.cu.
//
// On machines where every GPU pair sits behind the CPU's root complex (nvidia-smi topo "SYS", e.g. one GPU per NUMA node
// of a multi-die EPYC), stores from SMs into a peer GPU's memory can be far slower than the copy engine or than copies
// staged through pinned host memory. The probe measures the three paths per GPU pair at startup and the prefill
// collectives switch to the fastest one when SM stores are markedly slower (comm_select_path).
#include "../env.h"
#include "comm_rule.h"
#include <dirent.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace {

__global__ void probe_store_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}
// P2P data check (issue #2): an address- and round-dependent pattern, written on the sending card, moved by one transport
// (SM loads by the receiver, SM stores by the sender, copy engine) and verified on the receiving card. Some PCIe root
// complexes forward peer writes but not peer reads; a failed read completes as all-ones (0xFFFFFFFF: two bf16 0xFFFF,
// both NaN), so the check also counts all-ones words.
__host__ __device__ __forceinline__ unsigned probe_pat(size_t k, unsigned salt) {
    unsigned x = (unsigned)k * 2654435761u ^ salt;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    return x == 0xFFFFFFFFu ? 0x7FFFFFFFu : x;  // never all-ones (that is what a failed read returns)
}
__global__ void probe_fill_kernel(unsigned* __restrict__ a, size_t n, unsigned salt) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a[i] = probe_pat(i, salt);
}
__global__ void probe_load_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16) {  // pull (volatile loads)
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) dst[i] = __ldcv(src + i);
}
struct ProbeChk {
    unsigned bad, ones;        // mismatching words, of them all-ones
    unsigned long long first;  // (index << 32) | value of the first mismatching word (~0: none)
};
__global__ void probe_verify_kernel(const unsigned* __restrict__ a, size_t n, unsigned salt, ProbeChk* __restrict__ res) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const unsigned v = a[i];
        if (v != probe_pat(i, salt)) {
            atomicAdd(&res->bad, 1u);
            if (v == 0xFFFFFFFFu) atomicAdd(&res->ones, 1u);
            atomicMin(&res->first, ((unsigned long long)i << 32) | v);
        }
    }
}

// co-run tax probe (prefill-r4): a local memory-bound compute proxy (how much a GEMM / MoE kernel is slowed by a
// transport running next to it), timed alone and next to each transport's traffic on a side stream. SM stores / loads
// over a slow link (SYS / cross-NUMA) saturate the memory system and stall every concurrent kernel; the copy engine
// does not (tests/bench_ar_occupy.py, mb_gdn.cu). This is the signal that picks the prefill transport, not raw
// bandwidth (on SYS P2P and CE have similar GB/s but P2P stalls compute ~2x).
__global__ void probe_compute_kernel(const float4* __restrict__ a, size_t n16, int reps, float* __restrict__ sink) {
    float acc = 0.f;
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
            const float4 v = __ldcg(a + i);
            acc += v.x + v.y + v.z + v.w;
        }
    if (acc == -1.f) sink[threadIdx.x & 255] = acc;  // keep the loop live (never taken)
}
// dst[i] = src[i] until *stop (SM transfer: store to a peer when dst is the peer, load from a peer when src is)
__global__ void probe_spin_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16,
                                  const int* __restrict__ stop) {
    while (!*reinterpret_cast<const volatile int*>(stop))
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
            dst[i] = __ldcv(src + i);
}

struct ProbeBw {
    float sm = 0.f, ce = 0.f, host = 0.f;  // GB/s, directed pair i -> j (i's rows into j's memory)
    float ld = 0.f;                        // GB/s, SM loads: j reads i's rows from i's memory ("pull")
    // co-run tax: local compute slowdown (x) while this pair's traffic runs on a side stream of the transferring card.
    // tax_st on card i (push: i stores to j), tax_ld on card j (pull: j loads from i), tax_ce on card i (copy engine)
    float tax_st = 1.f, tax_ld = 1.f, tax_ce = 1.f;
    // spread (min / max) of the co-run tax over the probe runs (the fields above hold the medians)
    float st_lo = 1.f, st_hi = 1.f, ld_lo = 1.f, ld_hi = 1.f, ce_lo = 1.f, ce_hi = 1.f;
    // P2P data check (QE_COMM_CHECK, default on): mismatching words per transport, all rounds (0: the data arrived)
    unsigned bad_ld = 0, bad_st = 0, bad_ce = 0;
};

// best of `reps` timed runs (after one warm-up) of fn() on stream st of the current device, in ms
template <class F>
float probe_ms(cudaStream_t st, F fn, int reps = 3) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    fn();
    CUDA_CHECK(cudaStreamSynchronize(st));
    float best = 1e30f;
    for (int i = 0; i < reps; i++) {
        CUDA_CHECK(cudaEventRecord(a, st));
        fn();
        CUDA_CHECK(cudaEventRecord(b, st));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, ms);
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return best;
}

std::string read_line(const std::string& path) {
    std::string s;
    if (FILE* f = fopen(path.c_str(), "r")) {
        char buf[256];
        if (fgets(buf, sizeof(buf), f)) s = buf;
        fclose(f);
    }
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

}  // namespace

// sysfs directory of a GPU's PCI function ("" if unknown)
static std::string gpu_sysfs(int device) {
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), device) != cudaSuccess) {
        cudaGetLastError();
        return "";
    }
    std::string id(bus);
    for (char& ch : id) ch = (char)tolower(ch);
    // sysfs names are dddd:bb:dd.f; the CUDA id may carry an 8-digit domain
    if (id.size() > 12 && id.find(':') == 8) id = id.substr(4);
    return "/sys/bus/pci/devices/" + id;
}

int gpu_numa_node(int device) {
    const std::string d = gpu_sysfs(device);
    const std::string v = d.empty() ? "" : read_line(d + "/numa_node");
    return v.empty() ? -1 : atoi(v.c_str());
}

// negotiated PCIe link width of a GPU (lanes; -1 unknown). The width holds at idle, the speed does not (idle cards
// drop to 2.5 GT/s), so only the width is read.
int gpu_link_width(int device) {
    const std::string d = gpu_sysfs(device);
    const std::string v = d.empty() ? "" : read_line(d + "/current_link_width");
    const int w = v.empty() ? -1 : atoi(v.c_str());
    return w > 0 ? w : -1;
}

int numa_nodes() {
    int n = 0;
    if (DIR* d = opendir("/sys/devices/system/node")) {
        while (dirent* e = readdir(d))
            if (!strncmp(e->d_name, "node", 4) && e->d_name[4] >= '0' && e->d_name[4] <= '9') n++;
        closedir(d);
    }
    return std::max(1, n);
}

void numa_place(void* p, size_t n, const std::vector<int>& nodes) {
    if (nodes.empty() || numa_nodes() < 2 || !p || !n) return;
    unsigned long mask[2] = {0, 0};
    for (int x : nodes)
        if (x >= 0 && x < 128) mask[x / 64] |= 1ul << (x % 64);
    if (!mask[0] && !mask[1]) return;
    constexpr int kPreferred = 1, kInterleave = 3;  // MPOL_PREFERRED, MPOL_INTERLEAVE (linux/mempolicy.h)
    const int mode = nodes.size() == 1 ? kPreferred : kInterleave;
    const long pg = sysconf(_SC_PAGESIZE);
    const uintptr_t a = (uintptr_t)p & ~(uintptr_t)(pg - 1), e = (uintptr_t)p + n;
    if (syscall(SYS_mbind, (void*)a, (unsigned long)(e - a), mode, mask, 129ul, 0u) != 0)
        fprintf(stderr, "[comm] NUMA: mbind of %.1f MB failed (%s), default placement\n", n / 1e6, strerror(errno));
}

const char* comm_path_name(int xfer) {
    return xfer == kXferCE ? "copy engine" : xfer == kXferHost ? "host-staged" : "P2P SM loads / stores";
}

int comm_select_path(Comm& c, const int* devices, const char* env, bool staged) {
    const char* ev = env_str(env);
    std::string want = ev ? ev : "auto";
    if (want != "auto" && want != "p2p" && want != "pull" && want != "push" && want != "ce" && want != "host") {
        fprintf(stderr, "[comm] %s=%s: expected auto, p2p, pull, push, ce or host; using auto\n", env, want.c_str());
        want = "auto";
    }
    const int ng = c.ngpu;
    int prev = 0;
    CUDA_CHECK(cudaGetDevice(&prev));
    // topology: NUMA node per GPU, P2P performance rank per pair
    std::string topo = "numa";
    bool cross_numa = false;  // the GPUs sit on different NUMA nodes (SYS: P2P over the CPU interconnect stalls compute)
    for (int i = 1; i < ng; i++) if (gpu_numa_node(devices[i]) != gpu_numa_node(devices[0])) cross_numa = true;
    for (int i = 0; i < ng; i++) topo += (i ? "," : " ") + std::to_string(gpu_numa_node(devices[i]));
    topo += " (" + std::to_string(numa_nodes()) + " node" + (numa_nodes() > 1 ? "s" : "") + ")";
    int min_lanes = 0;  // narrowest PCIe link of the set (0: unknown), the second signal of the 1 NUMA node tax rule
    topo += ", PCIe";
    for (int i = 0; i < ng; i++) {
        const int w = gpu_link_width(devices[i]);
        topo += (i ? ",x" : " x") + (w > 0 ? std::to_string(w) : std::string("?"));
        if (w > 0) min_lanes = min_lanes ? std::min(min_lanes, w) : w;
    }
    if (ng > 1) {
        topo += ", P2P rank";
        for (int i = 0; i < ng; i++)
            for (int j = i + 1; j < ng; j++) {
                int acc = 0, perf = -1;
                if (cudaDeviceGetP2PAttribute(&acc, cudaDevP2PAttrAccessSupported, devices[i], devices[j]) != cudaSuccess) acc = 0;
                if (!acc || cudaDeviceGetP2PAttribute(&perf, cudaDevP2PAttrPerformanceRank, devices[i], devices[j]) != cudaSuccess)
                    perf = -1;
                cudaGetLastError();
                topo += " " + std::to_string(i) + "-" + std::to_string(j) + ":" + (acc ? std::to_string(perf) : "none");
            }
    }
    fprintf(stderr, "[comm] topology: %s\n", topo.c_str());
    // probe: every directed pair, SM stores / CE peer copy / D2H + H2D through pinned host memory
    std::vector<ProbeBw> bw((size_t)ng * ng);
    ProbeBw mn{1e30f, 1e30f, 1e30f};
    const bool probe = ng > 1 && c.p2p && env_int("QE_COMM_PROBE", 1);
    // P2P data check (issue #2): QE_COMM_CHECK rounds (default 4, 0: off) per directed pair and transport over the probe
    // buffers, then one round over each of the comm's P2P receive buffers; a transport that corrupts data on a pair is
    // not used there (comm_check_pull / comm_check_pick below). Runs with QE_COMM_PROBE=0 too.
    const int chk = ng > 1 && c.p2p ? std::max(0, env_int("QE_COMM_CHECK", 4)) : 0;
    int tax_runs = 0;  // co-run tax runs done (the medians' sample count)
    if (probe || chk > 0) {
        const size_t bytes = (size_t)std::max(1, env_int("QE_COMM_PROBE_MB", 16)) << 20;
        std::vector<void*> src(ng), dst(ng);
        std::vector<cudaStream_t> st(ng);
        void* hbuf = nullptr;
        CUDA_CHECK(cudaHostAlloc(&hbuf, bytes, cudaHostAllocPortable));
        for (int i = 0; i < ng; i++) {
            CUDA_CHECK(cudaSetDevice(devices[i]));
            CUDA_CHECK(cudaMalloc(&src[i], bytes));
            CUDA_CHECK(cudaMalloc(&dst[i], bytes));
            CUDA_CHECK(cudaMemset(src[i], 1, bytes));
            CUDA_CHECK(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking));
        }
        for (int i = 0; i < ng && probe; i++)
            for (int j = 0; j < ng; j++) {
                if (i == j) continue;
                ProbeBw& b = bw[(size_t)i * ng + j];
                CUDA_CHECK(cudaSetDevice(devices[i]));
                const float t_sm = probe_ms(st[i], [&] {
                    probe_store_kernel<<<32, 512, 0, st[i]>>>((uint4*)dst[j], (const uint4*)src[i], bytes / 16);
                });
                const float t_ce = probe_ms(st[i], [&] {
                    CUDA_CHECK(cudaMemcpyPeerAsync(dst[j], devices[j], src[i], devices[i], bytes, st[i]));
                });
                CUDA_CHECK(cudaSetDevice(devices[j]));  // j pulls i's rows (SM loads from i's memory)
                const float t_ld = probe_ms(st[j], [&] {
                    probe_store_kernel<<<32, 512, 0, st[j]>>>((uint4*)dst[j], (const uint4*)src[i], bytes / 16);
                });
                CUDA_CHECK(cudaSetDevice(devices[i]));
                const float t_d2h = probe_ms(st[i], [&] {
                    CUDA_CHECK(cudaMemcpyAsync(hbuf, src[i], bytes, cudaMemcpyDeviceToHost, st[i]));
                });
                CUDA_CHECK(cudaSetDevice(devices[j]));
                const float t_h2d = probe_ms(st[j], [&] {
                    CUDA_CHECK(cudaMemcpyAsync(dst[j], hbuf, bytes, cudaMemcpyHostToDevice, st[j]));
                });
                b.sm = bytes / (t_sm * 1e6f);
                b.ld = bytes / (t_ld * 1e6f);
                b.ce = bytes / (t_ce * 1e6f);
                b.host = bytes / ((t_d2h + t_h2d) * 1e6f);  // the staged path runs D2H, then H2D
                mn.sm = std::min(mn.sm, b.sm);
                mn.ce = std::min(mn.ce, b.ce);
                mn.host = std::min(mn.host, b.host);
            }
        if (chk > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            // test hook: QE_COMM_CHECK_TEST=ld|st|ce[:i-j] writes 64 all-ones bytes (a failed PCIe read) into the received
            // rows of that transport (every pair, or directed pair i -> j only) before the verify
            int tmode = -1, ti = -1, tj = -1;
            if (const char* e = env_str("QE_COMM_CHECK_TEST")) {
                tmode = !strncmp(e, "ld", 2) ? 0 : !strncmp(e, "st", 2) ? 1 : !strncmp(e, "ce", 2) ? 2 : -1;
                if (const char* q = strchr(e, ':')) sscanf(q + 1, "%d-%d", &ti, &tj);
                fprintf(stderr, "[comm] QE_COMM_CHECK_TEST=%s: injecting a fault (%s)\n", e,
                        tmode < 0 ? "unknown transport, nothing injected" : "test only");
            }
            std::vector<ProbeChk*> res(ng);
            for (int d = 0; d < ng; d++) {
                CUDA_CHECK(cudaSetDevice(devices[d]));
                CUDA_CHECK(cudaMalloc(&res[d], sizeof(ProbeChk)));
            }
            static const char* const kModeName[3] = {"SM load", "SM store", "copy engine"};
            // per directed pair and transport: failing transfers, transfers run, and the first failure's details
            std::vector<int> nfail((size_t)ng * ng * 3, 0), nrun((size_t)ng * ng * 3, 0);
            std::vector<std::string> first((size_t)ng * ng * 3);
            // one transfer of n bytes: i's `s` (pattern) -> j's `d` by transport `mode`, verified on j
            auto check1 = [&](int i, int j, int mode, void* d, void* s, size_t n, unsigned salt, const char* where) {
                const size_t nw = n / 4;
                CUDA_CHECK(cudaSetDevice(devices[i]));
                probe_fill_kernel<<<256, 512, 0, st[i]>>>((unsigned*)s, nw, salt);
                CUDA_CHECK(cudaStreamSynchronize(st[i]));
                CUDA_CHECK(cudaSetDevice(devices[j]));
                CUDA_CHECK(cudaMemsetAsync(d, 0, n, st[j]));
                const ProbeChk init{0u, 0u, ~0ull};
                CUDA_CHECK(cudaMemcpyAsync(res[j], &init, sizeof init, cudaMemcpyHostToDevice, st[j]));
                CUDA_CHECK(cudaStreamSynchronize(st[j]));
                if (mode == 0) {  // j pulls i's rows
                    probe_load_kernel<<<32, 512, 0, st[j]>>>((uint4*)d, (const uint4*)s, n / 16);
                    CUDA_CHECK(cudaStreamSynchronize(st[j]));
                } else {
                    CUDA_CHECK(cudaSetDevice(devices[i]));
                    if (mode == 1) probe_store_kernel<<<32, 512, 0, st[i]>>>((uint4*)d, (const uint4*)s, n / 16);
                    else CUDA_CHECK(cudaMemcpyPeerAsync(d, devices[j], s, devices[i], n, st[i]));
                    CUDA_CHECK(cudaStreamSynchronize(st[i]));
                    CUDA_CHECK(cudaSetDevice(devices[j]));
                }
                if (mode == tmode && (ti < 0 || (ti == i && tj == j)) && n >= 8192)
                    CUDA_CHECK(cudaMemsetAsync((char*)d + 4096, 0xFF, 64, st[j]));
                probe_verify_kernel<<<256, 512, 0, st[j]>>>((const unsigned*)d, nw, salt, res[j]);
                ProbeChk h;
                CUDA_CHECK(cudaMemcpyAsync(&h, res[j], sizeof h, cudaMemcpyDeviceToHost, st[j]));
                CUDA_CHECK(cudaStreamSynchronize(st[j]));
                const size_t slot = ((size_t)i * ng + j) * 3 + mode;
                nrun[slot]++;
                if (!h.bad) return;
                ProbeBw& b = bw[(size_t)i * ng + j];
                (mode == 0 ? b.bad_ld : mode == 1 ? b.bad_st : b.bad_ce) += h.bad;
                if (nfail[slot]++) return;  // the log details the first failure of each pair and transport
                const size_t k = (size_t)(h.first >> 32);
                const unsigned v = (unsigned)h.first;
                char m[320];
                snprintf(m, sizeof m, "first in the %s: %u of %zu words wrong, %u all-ones, first at byte %zu: 0x%08x, want 0x%08x%s",
                         where, h.bad, nw, h.ones, k * 4, v, probe_pat(k, salt),
                         v == 0xFFFFFFFFu ? " (a failed PCIe read, NaN in bf16)" : "");
                first[slot] = m;
            };
            // the comm's own P2P receive buffers (decode LL16 words, prefill reduce-scatter rows and sums, CE staging):
            // the same transports through the address ranges the collectives use (contents are reset afterwards)
            struct Real { const char* name; void** p; size_t n; };
            const size_t hw = (size_t)c.ll_hidden;
            std::vector<Real> real;
            if (c.pll[0]) real.push_back({"LL16 receive buffer", (void**)c.pll,
                                          (size_t)4 * ng * kMaxBatchRows * pll_row_words(c.ll_hidden) * sizeof(uint4)});
            if (c.prs1[0]) real.push_back({"reduce-scatter rows", (void**)c.prs1, (size_t)ng * c.rs_own * hw * 2});
            if (c.prs2[0]) real.push_back({"all-reduce sums", (void**)c.prs2, (size_t)c.rs_rows * hw * 2});
            if (c.pstage[0]) real.push_back({"CE staging rows", (void**)c.pstage, (size_t)c.rs_rows * hw * 2});
            size_t real_mb = 0;
            for (int i = 0; i < ng; i++)
                for (int j = 0; j < ng; j++) {
                    if (i == j) continue;
                    for (int mode = 0; mode < 3; mode++) {
                        for (int r = 0; r < chk; r++)
                            check1(i, j, mode, dst[j], src[i], bytes,
                                   0x9E3779B9u * (unsigned)(1 + i + 4 * j + 16 * mode + 64 * r), "probe buffer");
                        for (size_t q = 0; q < real.size(); q++) {
                            const size_t n = std::min(real[q].n, (size_t)64 << 20) / 16 * 16;
                            if (n < 16) continue;
                            check1(i, j, mode, real[q].p[j], real[q].p[i], n,
                                   0x85EBCA6Bu * (unsigned)(1 + i + 4 * j + 16 * mode + 64 * (int)q), real[q].name);
                            if (i == 0 && j == 1 && mode == 0) real_mb += n;
                        }
                    }
                }
            for (const Real& q : real)  // back to their start state (zeros)
                for (int d = 0; d < ng; d++) {
                    CUDA_CHECK(cudaSetDevice(devices[d]));
                    CUDA_CHECK(cudaMemset(q.p[d], 0, q.n));
                }
            for (int d = 0; d < ng; d++) {
                CUDA_CHECK(cudaSetDevice(devices[d]));
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaFree(res[d]));
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::string bad;
            int nbad = 0;
            for (int i = 0; i < ng; i++)
                for (int j = 0; j < ng; j++)
                    for (int mode = 0; mode < 3 && i != j; mode++) {
                        const size_t slot = ((size_t)i * ng + j) * 3 + mode;
                        if (!nfail[slot]) continue;
                        nbad += nfail[slot];
                        char m[480];
                        snprintf(m, sizeof m, "%s%d->%d %s: %d of %d transfers wrong (%s)", bad.empty() ? "" : "; ", i, j,
                                 kModeName[mode], nfail[slot], nrun[slot], first[slot].c_str());
                        bad += m;
                    }
            if (bad.empty())
                fprintf(stderr, "[comm] P2P data check: SM loads, SM stores and copy engine ok on every pair (%d x %zu MB "
                                "probe rounds + %zu MB of the comm buffers per pair and transport, %.0f ms)\n",
                        chk, bytes >> 20, real_mb >> 20, ms);
            else
                fprintf(stderr, "[comm] P2P data check FAILED (%d transfers; auto does not use a failing transport, "
                                "QE_COMM_CHECK=0: off): %s\n", nbad, bad.c_str());
        }
        // co-run tax: a local compute proxy timed alone and next to each transport's traffic (QE_COMM_CORUN=0: off)
        // (not with CUDA_LAUNCH_BLOCKING=1: the spinning side-stream kernel would block the host launch forever).
        // Repeated QE_COMM_CORUN_RUNS times (default 5; at least 3 within QE_COMM_CORUN_MS, default 1500 ms): the
        // medians decide (one run on our box read 3.69 against 1.33-1.54 and flipped auto to CE, issue #2).
        if (probe && env_int("QE_COMM_CORUN", 1) && !env_int("CUDA_LAUNCH_BLOCKING", 0)) {
            const auto t0 = std::chrono::steady_clock::now();
            const size_t n16 = bytes / 16;
            const int reps = std::max(1, env_int("QE_COMM_CORUN_REPS", 8));
            const int runs = std::max(1, env_int("QE_COMM_CORUN_RUNS", 5));
            const double budget = env_double("QE_COMM_CORUN_MS", 1500.0);
            std::vector<cudaStream_t> stB(ng);
            std::vector<float*> sink(ng);
            int* stop = nullptr;  // host-mapped: one flag stops every spin kernel
            CUDA_CHECK(cudaHostAlloc((void**)&stop, 4, cudaHostAllocMapped | cudaHostAllocPortable));
            *stop = 0;
            for (int i = 0; i < ng; i++) {
                CUDA_CHECK(cudaSetDevice(devices[i]));
                CUDA_CHECK(cudaStreamCreateWithFlags(&stB[i], cudaStreamNonBlocking));
                CUDA_CHECK(cudaMalloc(&sink[i], 256 * 4));
            }
            // compute proxy on device d (reads src[d]); each co-run time is divided by an alone time measured right
            // before it (clock or load changes between the two do not enter the ratio)
            auto alone = [&](int d) {
                CUDA_CHECK(cudaSetDevice(devices[d]));
                return probe_ms(st[d], [&] {
                    probe_compute_kernel<<<160, 256, 0, st[d]>>>((const float4*)src[d], n16, reps, sink[d]);
                });
            };
            auto spin_tax = [&](int d, uint4* sd, const uint4* ss) {
                const float ta = alone(d);
                *stop = 0;
                probe_spin_kernel<<<16, 512, 0, stB[d]>>>(sd, ss, n16, stop);
                const float t = probe_ms(st[d], [&] {
                    probe_compute_kernel<<<160, 256, 0, st[d]>>>((const float4*)src[d], n16, reps, sink[d]);
                });
                *stop = 1;
                CUDA_CHECK(cudaStreamSynchronize(stB[d]));
                return t / ta;
            };
            std::vector<std::vector<float>> vst((size_t)ng * ng), vld((size_t)ng * ng), vce((size_t)ng * ng);
            for (int run = 0; run < runs; run++) {
                if (run >= 3 && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > budget)
                    break;
                for (int i = 0; i < ng; i++)
                    for (int j = 0; j < ng; j++) {
                        if (i == j) continue;
                        const size_t p = (size_t)i * ng + j;
                        vst[p].push_back(spin_tax(i, (uint4*)dst[j], (const uint4*)src[i]));  // i pushes to j
                        vld[p].push_back(spin_tax(j, (uint4*)dst[j], (const uint4*)src[i]));  // j pulls from i
                        // copy engine on i: keep it busy with enough peer copies to outlast the timed compute
                        const float ta = alone(i);
                        for (int k = 0; k < 12; k++)  // outlast the timed compute (warm + 3 reps)
                            CUDA_CHECK(cudaMemcpyPeerAsync(dst[j], devices[j], src[i], devices[i], bytes, stB[i]));
                        vce[p].push_back(probe_ms(st[i], [&] {
                            probe_compute_kernel<<<160, 256, 0, st[i]>>>((const float4*)src[i], n16, reps, sink[i]);
                        }) / ta);
                        CUDA_CHECK(cudaStreamSynchronize(stB[i]));
                    }
                tax_runs = run + 1;
            }
            for (int i = 0; i < ng; i++)
                for (int j = 0; j < ng; j++) {
                    if (i == j) continue;
                    const size_t p = (size_t)i * ng + j;
                    ProbeBw& b = bw[p];
                    const CommStat a = comm_median(vst[p]), l = comm_median(vld[p]), e = comm_median(vce[p]);
                    b.tax_st = a.med, b.st_lo = a.lo, b.st_hi = a.hi;
                    b.tax_ld = l.med, b.ld_lo = l.lo, b.ld_hi = l.hi;
                    b.tax_ce = e.med, b.ce_lo = e.lo, b.ce_hi = e.hi;
                }
            for (int i = 0; i < ng; i++) {
                CUDA_CHECK(cudaSetDevice(devices[i]));
                CUDA_CHECK(cudaStreamDestroy(stB[i]));
                CUDA_CHECK(cudaFree(sink[i]));
            }
            CUDA_CHECK(cudaFreeHost(stop));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::string line;
            for (int i = 0; i < ng; i++)
                for (int j = 0; j < ng; j++)
                    if (i != j) {
                        const ProbeBw& b = bw[(size_t)i * ng + j];
                        char s[160];
                        snprintf(s, sizeof(s), "%s%d->%d %.2f-%.2f/%.2f-%.2f/%.2f-%.2f", line.empty() ? "" : ", ", i, j,
                                 b.st_lo, b.st_hi, b.ld_lo, b.ld_hi, b.ce_lo, b.ce_hi);
                        line += s;
                    }
            fprintf(stderr, "[comm] co-run tax: median of %d runs (%.0f ms), spread min-max store/load/ce: %s\n", tax_runs,
                    ms, line.c_str());
        }
        for (int i = 0; i < ng; i++) {
            CUDA_CHECK(cudaSetDevice(devices[i]));
            CUDA_CHECK(cudaStreamDestroy(st[i]));
            CUDA_CHECK(cudaFree(src[i]));
            CUDA_CHECK(cudaFree(dst[i]));
        }
        CUDA_CHECK(cudaFreeHost(hbuf));
        if (probe) {
            std::string line;
            for (int i = 0; i < ng; i++)
                for (int j = 0; j < ng; j++)
                    if (i != j) {
                        const ProbeBw& b = bw[(size_t)i * ng + j];
                        char s[160];
                        snprintf(s, sizeof(s), "%s%d->%d %.1f/%.1f/%.1f/%.1f (tax %.2f/%.2f/%.2f)", line.empty() ? "" : ", ", i, j,
                                 b.sm, b.ld, b.ce, b.host, b.tax_st, b.tax_ld, b.tax_ce);
                        line += s;
                    }
            fprintf(stderr, "[comm] probe GB/s (SM store / SM load by receiver / copy engine / host-staged; tax = local compute slowdown store/load/ce, median): %s\n", line.c_str());
        }
    }
    bool bad_st = false, bad_ld = false, bad_ce = false;  // the data check failed on some pair
    for (const ProbeBw& b : bw) bad_st |= b.bad_st != 0, bad_ld |= b.bad_ld != 0, bad_ce |= b.bad_ce != 0;
    // P2P SM path, per directed pair (staged engines only, i.e. Flash-Next): the receiver pulls the rows by SM loads
    // unless they are markedly slower than the sender's SM stores (QE_COMM_PULL_RATIO, default 0.7): SM stores over
    // PCIe stall the sender's memory system for every kernel next to them, loads of the same bytes barely (prefill-r4:
    // W4A8 GEMM 4.3x vs +12 %), so equal bandwidth favours pull. "pull" / "push" force one mode on every pair.
    unsigned pm = 0;
    std::string pairs;
    float mn_p2p = 1e30f, tax_p2p = 1.f, tax_ce = 1.f;  // tax of the chosen P2P mode / of the copy engine (worst pair)
    float tax_lo = 1.f, tax_hi = 1.f;                    // spread of tax_p2p over the probe runs
    std::string p2p_bad;                                 // pairs whose chosen P2P mode failed the data check
    int moved = 0;                                       // pairs the data check moved to the other direction
    if (staged && ng > 1 && c.p2p) {
        static const float pr = (float)env_double("QE_COMM_PULL_RATIO", 0.7f);
        for (int i = 0; i < ng; i++)
            for (int j = 0; j < ng; j++) {
                if (i == j) continue;
                const ProbeBw& b = bw[(size_t)i * ng + j];  // i's rows to j: j pulls or i pushes
                // pull unless its load bandwidth is markedly below the store bandwidth or its tax is markedly worse
                bool pl = want == "pull" ||
                          (want != "push" && (!probe || (b.ld >= pr * b.sm && b.tax_ld <= b.tax_st * 1.1f + 0.05f)));
                // the data check: a direction that corrupts data is not used while the other one passed (forced
                // pull / push stay as asked, with a warning below)
                const bool pl0 = pl;
                if (want != "pull" && want != "push") pl = comm_check_pull(pl, b.bad_ld != 0, b.bad_st != 0);
                moved += pl != pl0;
                if (pl) pm |= 1u << (4 * j + i);
                if (probe) {
                    mn_p2p = std::min(mn_p2p, pl ? b.ld : b.sm);
                    const float t = pl ? b.tax_ld : b.tax_st;
                    if (t > tax_p2p) tax_p2p = t, tax_lo = pl ? b.ld_lo : b.st_lo, tax_hi = pl ? b.ld_hi : b.st_hi;
                }
                tax_ce = std::max(tax_ce, b.tax_ce);
                pairs += (pairs.empty() ? "" : " ") + std::to_string(j) + "<-" + std::to_string(i) + (pl ? ":pull" : ":push") +
                         (pl != pl0 ? " (data check)" : "");
                if (pl ? b.bad_ld : b.bad_st) {
                    char w[200];
                    snprintf(w, sizeof w, "%s%d<-%d %s", p2p_bad.empty() ? "" : ", ", j, i, pl ? "SM loads (pull)" : "SM stores (push)");
                    p2p_bad += w;
                }
            }
        if (probe) mn.sm = mn_p2p;  // the P2P path's bandwidth is the chosen mode's per pair
    }
    // the choice: SM stores unless they are markedly slower than the best alternative on some pair (the slowest pair
    // bounds every collective); QE_COMM_RATIO: the factor that counts as markedly (default 0.5)
    int x = kXferP2P;
    const char* why = "";
    if (ng < 2) {
        why = "one GPU";
    } else if (!c.p2p) {
        x = kXferHost;  // no peer access: every collective goes through host memory already
        why = "no peer access";
    } else if (want == "p2p" || want == "pull" || want == "push") {
        x = kXferP2P, why = "forced";
    } else if (want == "ce") {
        x = kXferCE, why = "forced";
    } else if (want == "host") {
        x = kXferHost, why = "forced";
    } else if (probe) {
        static const float ratio = (float)env_double("QE_COMM_RATIO", 0.5f);
        if (staged) {
            // P2P (pull/push) keeps compute on the memory system it shares with the transfer, so a slow link stalls it
            // (SYS / cross-NUMA): switch the prefill collectives to the copy engine when the P2P co-run tax is high and
            // markedly worse than the copy engine's, or when P2P bandwidth is far below the alternatives. The copy
            // engine leaves the SMs alone (its tax ~1x) and was more stable than host on the SYS tester, so prefer it
            // over host unless host bandwidth is far higher. Our 1-NUMA PCIe4 box (pull tax ~1.1) stays on P2P.
            static const float taxcap = (float)env_double("QE_COMM_TAX", 1.30f);
            const float alt = std::max(mn.ce, mn.host);
            // The tax rule (cap, single-NUMA-node cap) is comm_auto_rule (comm_rule.h); the cross-NUMA
            // tester: P2P ~970 vs CE ~2030 tok/s. QE_COMM_XNUMA=0 lets the cap decide on one node too.
            static const bool xnuma = env_int("QE_COMM_XNUMA", 1);
            // 1 NUMA node: a high tax counts only on a narrow link (issue #2, comm_rule.h): our x16 box reads up to
            // 2.17 on ~1 % of boots, the reporter (one card at gen4 x4) 2.11-2.17
            static const float severe = (float)env_double("QE_COMM_TAX_SEVERE", 3.0);
            static const float cap1 = (float)env_double("QE_COMM_TAX_1NODE", 1.90);
            static const int narrow_x = env_int("QE_COMM_NARROW_X", 8);
            static const float narrow_gbps = (float)env_double("QE_COMM_NARROW_GBPS", 10.0);
            const bool narrow = comm_narrow_link(min_lanes, mn.sm, narrow_x, narrow_gbps);
            const CommRule rule = comm_auto_rule(tax_p2p, tax_ce, taxcap, cap1, severe, cross_numa, xnuma, narrow,
                                                 mn.sm < ratio * alt);
            if (rule.leave_p2p) x = mn.ce >= 0.8f * mn.host ? kXferCE : kXferHost;
            const char* dest = !rule.leave_p2p ? "" : x == kXferCE ? "-> copy engine (" : "-> host-staged (";
            char w[512];
            snprintf(w, sizeof(w),
                     "auto: P2P tax %.2f (median of %d runs, %.2f-%.2f; cap %.2f, 1 NUMA node cap %.2f on a narrow link, "
                     "severe %.2f, %s, narrowest link x%d%s), CE tax %.2f, P2P %.0f CE %.0f host %.0f GB/s: %s%s%s",
                     tax_p2p, tax_runs, tax_lo, tax_hi, taxcap, cap1, severe, cross_numa ? "cross-NUMA" : "1 NUMA node",
                     min_lanes, narrow ? " = narrow" : "", tax_ce,
                     mn.sm, mn.ce, mn.host, dest, rule.why, rule.leave_p2p ? ")" : "");
            static std::string ws;  // keep the string alive for the final log line
            ws = w;
            if (moved)
                ws += "; P2P data check: " + std::to_string(moved) + " pair(s) moved off a corrupt " +
                      (bad_ld ? "SM load (pull) direction" : "SM store (push) direction");
            why = ws.c_str();
        } else {  // engines with a copy-engine prefill all-reduce by default: host staging only if it is markedly faster
            x = std::max(mn.sm, mn.ce) < ratio * mn.host ? kXferHost : kXferCE;
            why = "auto";
        }
    } else {
        x = staged || ng < 2 ? kXferP2P : kXferCE;  // the engine's default
        why = ng > 1 ? "auto, probe off" : "one GPU";
    }
    // the P2P data check (issue #2): auto never keeps a transport that corrupted data; a forced mode only warns, loudly
    if (ng > 1 && c.p2p && chk > 0) {
        // every P2P collective stores into the peers (decode LL16 included); the 27B engine (not staged) runs its
        // prefill all-reduce on the copy engine whatever x says
        const bool ce_used = x == kXferCE || !staged;
        if (want == "auto") {
            const CommCheckPick k = comm_check_pick(ce_used && !staged ? 1 : x, bad_st, bad_ce && ce_used);
            if (k.why) {
                x = k.x;
                static std::string ws;
                ws = std::string(why) + "; " + k.why;
                why = ws.c_str();
                if (k.p2p_off) {
                    comm_free_p2p(c, devices);  // every collective through host memory (the QE_P2P=0 path)
                    pairs.clear();
                    pm = 0;
                }
            }
        } else {
            std::string what;  // the failing transports this forced mode uses
            auto add = [&](const std::string& w) { what += (what.empty() ? "" : ", ") + w; };
            if (bad_st) add("SM stores");
            if (!p2p_bad.empty() && x == kXferP2P) add("pairs " + p2p_bad);
            if (bad_ce && (x == kXferCE || !staged)) add("copy engine");
            if (!what.empty())
                fprintf(stderr, "[comm] WARNING: %s=%s is forced, but the P2P data check found corrupt transfers on this "
                                "machine (%s): outputs may be wrong (NaN faults). Use %s=auto.\n",
                        env, want.c_str(), what.c_str(), env);
        }
    }
    if (c.p2p && x != kXferP2P && staged) {
        for (int i = 0; i < ng; i++) {
            CUDA_CHECK(cudaSetDevice(devices[i]));
            const size_t fb = (size_t)2 * ng * sizeof(unsigned);
            if (x == kXferHost) {
                unsigned* h = nullptr;
                CUDA_CHECK(cudaHostAlloc((void**)&h, fb, cudaHostAllocMapped | cudaHostAllocPortable));
                memset(h, 0, fb);
                CUDA_CHECK(cudaHostGetDevicePointer((void**)&c.xfl[i], h, 0));
                const size_t rb = (size_t)c.rs_rows * c.ll_hidden * 2, pg = 2u << 20, cap = (rb + pg - 1) / pg * pg;
                for (uint2** q : {&c.hrs[i], &c.hag[i]}) {
                    void* p = nullptr;
                    CHECK(posix_memalign(&p, pg, cap) == 0 && p, "comm: host staging buffers");
                    numa_place(p, cap, {gpu_numa_node(devices[i])});  // the rank's node: its D2H writes stay local
                    memset(p, 0, cap);
                    CUDA_CHECK(cudaHostRegister(p, cap, cudaHostRegisterPortable));
                    *q = (uint2*)p;
                }
            } else {
                CUDA_CHECK(cudaMalloc(&c.xfl[i], fb));
                CUDA_CHECK(cudaMemset(c.xfl[i], 0, fb));
                CUDA_CHECK(cudaMalloc(&c.xsw[i], 16));
                CUDA_CHECK(cudaMemset(c.xsw[i], 0, 16));
            }
            CUDA_CHECK(cudaMalloc(&c.xcnt[i], sizeof(unsigned)));
            CUDA_CHECK(cudaMemset(c.xcnt[i], 0, sizeof(unsigned)));
            if (!c.pstage[i]) CUDA_CHECK(cudaMalloc(&c.pstage[i], (size_t)c.rs_rows * c.ll_hidden * 2));
        }
        c.xfer = x;
    } else if (c.p2p) {
        c.xfer = x;  // the engine decides what the path means for it (27B: CE all-reduce / no peer access)
    }
    if (const double gb = env_double("FLASH_TEST_LINK_GBPS", 0.0); gb > 0.0) {  // test: slow-link emulation (CE path)
        if (c.xfer == kXferCE && staged) {
            for (int i = 0; i < ng; i++) {
                CUDA_CHECK(cudaSetDevice(devices[i]));
                CUDA_CHECK(cudaMalloc(&c.lbusy[i], 8));
                CUDA_CHECK(cudaMemset(c.lbusy[i], 0, 8));
            }
            c.lemu = (float)(1.0 / gb);
            c.lreal = (float)(1.0 / env_double("FLASH_TEST_LINK_REAL_GBPS", 26.3));
            fprintf(stderr, "[comm] TEST: slow-link emulation, %.2f GB/s egress per card (FLASH_TEST_LINK_GBPS; copy-engine "
                            "copies + a spin per transfer)\n", gb);
        } else {
            fprintf(stderr, "[comm] TEST: FLASH_TEST_LINK_GBPS ignored (works on the copy-engine path only: FLASH_COMM=ce)\n");
        }
    }
    if (const char* e = env_str("QE_COMM_PULLM")) pm = (unsigned)strtoul(e, nullptr, 16), pairs += " (QE_COMM_PULLM override)";  // test: mixed pairs
    c.pullm = x == kXferP2P && staged ? pm : 0u;
    CUDA_CHECK(cudaSetDevice(prev));
    if (c.pullm || (x == kXferP2P && staged && ng > 1 && c.p2p))
        fprintf(stderr, "[comm] prefill transfers: %s, pairs %s (%s=%s, %s)\n", comm_path_name(x), pairs.c_str(), env, want.c_str(), why);
    else
    fprintf(stderr, "[comm] prefill transfers: %s (%s=%s, %s)\n", comm_path_name(x), env, want.c_str(), why);
    return x;
}
