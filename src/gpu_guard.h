#pragma once
// One QEngine process per GPU (both engines, every tool that creates an engine): at qe_create the process takes an
// advisory lock per physical GPU, ~/.cache/qengine/locks/gpu-<uuid>.lock, held until it exits (the kernel drops it
// then, also after a crash or SIGKILL). A second engine on a card that one already uses fails at once with the
// holder's pid and command line, instead of both running out of GPU / host memory later (10-27 evening: a steering
// tool's engine next to the llama-swap server, 15+ aborts of both and the global OOM killer).
// QE_ALLOW_SHARED_GPU=1 skips the check (experts: e.g. a small test engine next to a server on purpose).
// The bench lock (.gpu.lock) is a separate, coarse lock of the test scripts; this one is taken only by the engine
// process itself, so a script that holds .gpu.lock and starts a server is unaffected.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace qe_gpu_guard {

inline std::string uuid_of(int dev) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        cudaGetLastError();
        return "";
    }
    char b[64];
    const unsigned char* u = (const unsigned char*)p.uuid.bytes;
    snprintf(b, sizeof b, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2], u[3],
             u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return b;
}

inline std::string self_cmdline() {
    std::string s;
    if (FILE* f = fopen("/proc/self/cmdline", "r")) {
        int c;
        while ((c = fgetc(f)) != EOF && s.size() < 400) s += c ? (char)c : ' ';
        fclose(f);
    }
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

inline void mkdirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/') mkdir(path.substr(0, i).c_str(), 0755);
}

// Locks every CUDA device in `devs` (ordinals within CUDA_VISIBLE_DEVICES); "" when the process may use them, else the
// error message (nothing stays locked then). Locks this process already holds are kept (a second engine in-process).
inline std::string acquire(const std::vector<int>& devs) {
    if (getenv("QE_ALLOW_SHARED_GPU") && atoi(getenv("QE_ALLOW_SHARED_GPU"))) return "";
    const char* home = getenv("HOME");
    const std::string dir = std::string(home && *home ? home : "/tmp") + "/.cache/qengine/locks";
    mkdirs(dir);
    static std::vector<std::pair<std::string, int>> held;  // uuid -> fd, for the process lifetime
    std::vector<int> mine;
    for (int d : devs) {
        const std::string id = uuid_of(d);
        if (id.empty()) continue;  // the engine reports a bad device itself
        bool have = false;
        for (auto& h : held) have |= h.first == id;
        if (have) continue;
        const std::string path = dir + "/gpu-" + id + ".lock";
        const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) continue;  // no lock directory: no guard (never stop a start for it)
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            char buf[512] = {0};
            const ssize_t n = pread(fd, buf, sizeof buf - 1, 0);
            close(fd);
            std::string who = n > 0 ? std::string(buf, n) : "pid ?";
            const size_t nl = who.find('\n');
            std::string pid = nl == std::string::npos ? who : who.substr(0, nl);
            std::string cmd = nl == std::string::npos ? "" : who.substr(nl + 1);
            while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == ' ')) cmd.pop_back();
            for (int fd2 : mine) {  // give back what this call took
                for (size_t i = 0; i < held.size(); i++)
                    if (held[i].second == fd2) { close(fd2); held.erase(held.begin() + i); break; }
            }
            return "GPU " + std::to_string(d) + " (" + id + ") is used by QEngine pid " + pid + " (" + cmd +
                   "); stop it first or pick other GPUs (QE_DEVICES / CUDA_VISIBLE_DEVICES); QE_ALLOW_SHARED_GPU=1 "
                   "skips this check";
        }
        const std::string me = std::to_string(getpid()) + "\n" + self_cmdline() + "\n";
        if (ftruncate(fd, 0) == 0 && pwrite(fd, me.data(), me.size(), 0) < 0) {}
        held.push_back({id, fd});
        mine.push_back(fd);
    }
    return "";
}

}  // namespace qe_gpu_guard
