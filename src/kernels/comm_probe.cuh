// Topology, startup interconnect probe and the prefill transfer path (both engines). Included once, by comm.cu.
//
// On machines where every GPU pair sits behind the CPU's root complex (nvidia-smi topo "SYS", e.g. one GPU per NUMA node
// of a multi-die EPYC), stores from SMs into a peer GPU's memory can be far slower than the copy engine or than copies
// staged through pinned host memory. The probe measures the three paths per GPU pair at startup and the prefill
// collectives switch to the fastest one when SM stores are markedly slower (comm_select_path).
#include <dirent.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

namespace {

__global__ void probe_store_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
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

int gpu_numa_node(int device) {
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), device) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    std::string id(bus);
    for (char& ch : id) ch = (char)tolower(ch);
    // sysfs names are dddd:bb:dd.f; the CUDA id may carry an 8-digit domain
    if (id.size() > 12 && id.find(':') == 8) id = id.substr(4);
    const std::string v = read_line("/sys/bus/pci/devices/" + id + "/numa_node");
    return v.empty() ? -1 : atoi(v.c_str());
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
    const char* ev = getenv(env);
    std::string want = ev && *ev ? ev : "auto";
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
    if (probe) {
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
        for (int i = 0; i < ng; i++)
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
        // co-run tax: a local compute proxy timed alone and next to each transport's traffic (QE_COMM_CORUN=0: off)
        // (not with CUDA_LAUNCH_BLOCKING=1: the spinning side-stream kernel would block the host launch forever)
        if (env_int("QE_COMM_CORUN", 1) && !env_int("CUDA_LAUNCH_BLOCKING", 0)) {
            const size_t n16 = bytes / 16;
            const int reps = std::max(1, env_int("QE_COMM_CORUN_REPS", 8));
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
            // compute proxy on device d (reads src[d]), timed while `setup` keeps a transport busy on stB[d]; `teardown`
            // stops it. Returns the slowdown vs t_alone[d].
            std::vector<float> t_alone(ng);
            for (int d = 0; d < ng; d++) {
                CUDA_CHECK(cudaSetDevice(devices[d]));
                t_alone[d] = probe_ms(st[d], [&] {
                    probe_compute_kernel<<<160, 256, 0, st[d]>>>((const float4*)src[d], n16, reps, sink[d]);
                });
            }
            auto spin_tax = [&](int d, uint4* sd, const uint4* ss) {
                CUDA_CHECK(cudaSetDevice(devices[d]));
                *stop = 0;
                probe_spin_kernel<<<16, 512, 0, stB[d]>>>(sd, ss, n16, stop);
                const float t = probe_ms(st[d], [&] {
                    probe_compute_kernel<<<160, 256, 0, st[d]>>>((const float4*)src[d], n16, reps, sink[d]);
                });
                *stop = 1;
                CUDA_CHECK(cudaStreamSynchronize(stB[d]));
                return t / t_alone[d];
            };
            for (int i = 0; i < ng; i++)
                for (int j = 0; j < ng; j++) {
                    if (i == j) continue;
                    ProbeBw& b = bw[(size_t)i * ng + j];
                    b.tax_st = spin_tax(i, (uint4*)dst[j], (const uint4*)src[i]);  // i pushes to j
                    b.tax_ld = spin_tax(j, (uint4*)dst[j], (const uint4*)src[i]);  // j pulls from i (reads peer i, writes local)
                    // copy engine on i: keep it busy with enough peer copies to outlast the timed compute
                    CUDA_CHECK(cudaSetDevice(devices[i]));
                    for (int k = 0; k < 12; k++)  // outlast the timed compute (warm + 3 reps)
                        CUDA_CHECK(cudaMemcpyPeerAsync(dst[j], devices[j], src[i], devices[i], bytes, stB[i]));
                    b.tax_ce = probe_ms(st[i], [&] {
                        probe_compute_kernel<<<160, 256, 0, st[i]>>>((const float4*)src[i], n16, reps, sink[i]);
                    }) / t_alone[i];
                    CUDA_CHECK(cudaStreamSynchronize(stB[i]));
                }
            for (int i = 0; i < ng; i++) {
                CUDA_CHECK(cudaSetDevice(devices[i]));
                CUDA_CHECK(cudaStreamDestroy(stB[i]));
                CUDA_CHECK(cudaFree(sink[i]));
            }
            CUDA_CHECK(cudaFreeHost(stop));
        }
        for (int i = 0; i < ng; i++) {
            CUDA_CHECK(cudaSetDevice(devices[i]));
            CUDA_CHECK(cudaStreamDestroy(st[i]));
            CUDA_CHECK(cudaFree(src[i]));
            CUDA_CHECK(cudaFree(dst[i]));
        }
        CUDA_CHECK(cudaFreeHost(hbuf));
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
        fprintf(stderr, "[comm] probe GB/s (SM store / SM load by receiver / copy engine / host-staged; tax = local compute slowdown store/load/ce): %s\n", line.c_str());
    }
    // P2P SM path, per directed pair (staged engines only, i.e. Flash-Next): the receiver pulls the rows by SM loads
    // unless they are markedly slower than the sender's SM stores (QE_COMM_PULL_RATIO, default 0.7): SM stores over
    // PCIe stall the sender's memory system for every kernel next to them, loads of the same bytes barely (prefill-r4:
    // W4A8 GEMM 4.3x vs +12 %), so equal bandwidth favours pull. "pull" / "push" force one mode on every pair.
    unsigned pm = 0;
    std::string pairs;
    float mn_p2p = 1e30f, tax_p2p = 1.f, tax_ce = 1.f;  // tax of the chosen P2P mode / of the copy engine (worst pair)
    if (staged && ng > 1 && c.p2p) {
        static const float pr = getenv("QE_COMM_PULL_RATIO") ? (float)atof(getenv("QE_COMM_PULL_RATIO")) : 0.7f;
        for (int i = 0; i < ng; i++)
            for (int j = 0; j < ng; j++) {
                if (i == j) continue;
                const ProbeBw& b = bw[(size_t)i * ng + j];  // i's rows to j: j pulls or i pushes
                // pull unless its load bandwidth is markedly below the store bandwidth or its tax is markedly worse
                const bool pl = want == "pull" ||
                                (want != "push" && (!probe || (b.ld >= pr * b.sm && b.tax_ld <= b.tax_st * 1.1f + 0.05f)));
                if (pl) pm |= 1u << (4 * j + i);
                if (probe) { mn_p2p = std::min(mn_p2p, pl ? b.ld : b.sm); tax_p2p = std::max(tax_p2p, pl ? b.tax_ld : b.tax_st); }
                tax_ce = std::max(tax_ce, b.tax_ce);
                pairs += (pairs.empty() ? "" : " ") + std::to_string(j) + "<-" + std::to_string(i) + (pl ? ":pull" : ":push");
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
        static const float ratio = getenv("QE_COMM_RATIO") ? (float)atof(getenv("QE_COMM_RATIO")) : 0.5f;
        if (staged) {
            // P2P (pull/push) keeps compute on the memory system it shares with the transfer, so a slow link stalls it
            // (SYS / cross-NUMA): switch the prefill collectives to the copy engine when the P2P co-run tax is high and
            // markedly worse than the copy engine's, or when P2P bandwidth is far below the alternatives. The copy
            // engine leaves the SMs alone (its tax ~1x) and was more stable than host on the SYS tester, so prefer it
            // over host unless host bandwidth is far higher. Our 1-NUMA PCIe4 box (pull tax ~1.1) stays on P2P.
            static const float taxcap = getenv("QE_COMM_TAX") ? (float)atof(getenv("QE_COMM_TAX")) : 1.30f;
            const float alt = std::max(mn.ce, mn.host);
            // The co-run tax proxy (a memory-bound kernel) over-reads the P2P load tax on a fast single-NUMA PCIe link
            // (~1.4x here vs the real GEMM's ~1.1x), where P2P pull is still the fastest path end to end. The tax only
            // decides against P2P on a cross-NUMA box (SYS): there the transfer rides the CPU interconnect and genuinely
            // stalls compute (the tester: P2P ~970 vs CE ~2030 tok/s). QE_COMM_XNUMA=0 lets the tax decide anywhere.
            static const bool xnuma = !getenv("QE_COMM_XNUMA") || atoi(getenv("QE_COMM_XNUMA"));
            const bool tax_bad = tax_p2p >= taxcap && tax_p2p > tax_ce * 1.15f && (!xnuma || cross_numa);
            const bool bw_bad = mn.sm < ratio * alt;
            if (tax_bad || bw_bad) x = mn.ce >= 0.8f * mn.host ? kXferCE : kXferHost;
            char w[128];
            snprintf(w, sizeof(w), "auto: P2P tax %.2f (cap %.2f, %s), CE tax %.2f, P2P %.0f CE %.0f host %.0f GB/s",
                     tax_p2p, taxcap, cross_numa ? "cross-NUMA" : "1 NUMA node", tax_ce, mn.sm, mn.ce, mn.host);
            static std::string ws;  // keep the string alive for the final log line
            ws = w;
            why = ws.c_str();
        } else {  // engines with a copy-engine prefill all-reduce by default: host staging only if it is markedly faster
            x = std::max(mn.sm, mn.ce) < ratio * mn.host ? kXferHost : kXferCE;
            why = "auto";
        }
    } else {
        x = staged || ng < 2 ? kXferP2P : kXferCE;  // the engine's default
        why = ng > 1 ? "auto, probe off" : "one GPU";
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
    if (const char* e = getenv("QE_COMM_PULLM")) pm = (unsigned)strtoul(e, nullptr, 16), pairs += " (QE_COMM_PULLM override)";  // test: mixed pairs
    c.pullm = x == kXferP2P && staged ? pm : 0u;
    CUDA_CHECK(cudaSetDevice(prev));
    if (c.pullm || (x == kXferP2P && staged && ng > 1 && c.p2p))
        fprintf(stderr, "[comm] prefill transfers: %s, pairs %s (%s=%s, %s)\n", comm_path_name(x), pairs.c_str(), env, want.c_str(), why);
    else
    fprintf(stderr, "[comm] prefill transfers: %s (%s=%s, %s)\n", comm_path_name(x), env, want.c_str(), why);
    return x;
}
