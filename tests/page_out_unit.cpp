// CPU-only unit test of SafeTensors::page_out (issue #6): two small .safetensors files on a disk filesystem (argv[1]
// or $TMPDIR; not tmpfs, whose pages cannot leave memory), both read through the mapping; page_out drops the cache of
// the one not kept, the kept one stays cached, and both still read back the same bytes. argv[2] (optional): a readable
// file the process neither owns nor may write (MADV_PAGEOUT skips those silently): drop_file_cache must still drop it.
// g++-15 -std=c++20 -I src tests/page_out_unit.cpp src/safetensors.cpp -o /tmp/pou && /tmp/pou <dir on disk> [file]
#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define EXPECT(c)                                                                         \
    do {                                                                                  \
        if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

static const size_t N = (size_t)64 << 20;  // 64 MB of data per file

static void write_file(const std::string& path, const char* name, unsigned char seed) {
    const std::string hdr = std::string("{\"") + name + "\":{\"dtype\":\"I32\",\"shape\":[" + std::to_string(N / 4) +
                            "],\"data_offsets\":[0," + std::to_string(N) + "]}}";
    FILE* f = fopen(path.c_str(), "wb");
    const uint64_t hl = hdr.size();
    fwrite(&hl, 8, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    std::vector<unsigned char> buf(1 << 20);
    for (size_t o = 0; o < N; o += buf.size()) {
        for (size_t i = 0; i < buf.size(); i++) buf[i] = (unsigned char)((o + i) * 31 + seed);
        fwrite(buf.data(), 1, buf.size(), f);
    }
    fclose(f);
}

static unsigned long sum(const TensorInfo& t) {
    unsigned long s = 0;
    for (size_t i = 0; i < t.nbytes; i += 64) s += (unsigned char)t.data[i];
    return s;
}

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : getenv("TMPDIR");
    if (!dir) { fprintf(stderr, "usage: page_out_unit <dir on a disk filesystem>\n"); return 2; }
    const std::string a = std::string(dir) + "/pou_a.safetensors", b = std::string(dir) + "/pou_b.safetensors";
    write_file(a, "weights", 1);
    write_file(b, "ngram", 7);
    sync();
    unsigned long sa, sb;
    PageOut out;
    {
        SafeTensors st;
        st.open(a);
        st.open(b);
        sa = sum(st.get("weights"));
        sb = sum(st.get("ngram"));
        const char* keep_p = st.get("ngram").data;
        out = st.page_out([&](const void* base, size_t n) { return keep_p >= (const char*)base && keep_p < (const char*)base + n; });
        // the kept file is still resident
        const TensorInfo& tb = st.get("ngram");
        const size_t pg = sysconf(_SC_PAGESIZE);
        const uintptr_t lo = (uintptr_t)tb.data & ~(uintptr_t)(pg - 1);
        std::vector<unsigned char> v((tb.nbytes + pg) / pg + 1);
        EXPECT(mincore((void*)lo, (uintptr_t)tb.data + tb.nbytes - lo, v.data()) == 0);
        size_t res = 0;
        for (unsigned char c : v) res += c & 1;
        fprintf(stderr, "paged out %.1f MB of %.1f MB; kept file resident %.1f MB\n", out.dropped / 1e6, N / 1e6, res * pg / 1e6);
        EXPECT(out.measured);
        EXPECT(out.dropped >= N * 9 / 10 && out.dropped <= N + (1 << 20));
        EXPECT(res * pg >= N * 9 / 10);
        EXPECT(sum(st.get("weights")) == sa);  // read back from the file
        EXPECT(sum(st.get("ngram")) == sb);
        EXPECT(st.page_out([](const void*, size_t) { return true; }).requested == 0);  // everything kept: nothing happens
    }
    unlink(a.c_str());
    unlink(b.c_str());
    if (argc > 2) {  // a foreign read-only file (not ours, not writable)
        const int fd = open(argv[2], O_RDONLY);
        struct stat stt;
        EXPECT(fd >= 0 && fstat(fd, &stt) == 0);
        if (fd >= 0 && stt.st_uid != geteuid() && access(argv[2], W_OK) != 0) {
            const size_t n = stt.st_size;
            void* p = mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
            EXPECT(p != MAP_FAILED);
            volatile unsigned long s2 = 0;
            for (size_t i = 0; i < n; i += 4096) s2 += ((const unsigned char*)p)[i];
            PageOut o;
            const long long c0 = meminfo_cached();
            drop_file_cache(argv[2], p, n, &o);
            const long long c1 = meminfo_cached();
            // the kernel hides the cache state of such a file (mincore / cachestat): the host's Cached must fall by
            // about the file (other activity adds noise; pick a file nothing else maps)
            fprintf(stderr, "foreign file %s: %.1f MB, measured %d, host Cached %+.1f MB\n", argv[2], n / 1e6, (int)o.measured,
                    (c1 - c0) / 1e6);
            EXPECT(!o.measured || o.dropped >= n / 2);
            EXPECT(c0 - c1 >= (long long)n / 2);
            unsigned long s3 = 0;
            for (size_t i = 0; i < n; i += 4096) s3 += ((const unsigned char*)p)[i];
            EXPECT(s3 == s2);
            munmap(p, n);
        } else {
            fprintf(stderr, "foreign file case skipped (%s is ours or writable)\n", argv[2]);
        }
        if (fd >= 0) close(fd);
    }
    if (fails) return 1;
    printf("page_out_unit OK\n");
    return 0;
}
