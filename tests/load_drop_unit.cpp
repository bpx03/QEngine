// CPU unit test of the load-time page-cache drop (src/safetensors.h LoadDrop, SafeTensors::drop_prefix): a group's file
// pages go only once every loader is ready and done with it; other groups (also a prefix that extends the name, layers.1.
// vs layers.10.) keep theirs; off = nothing dropped. Residency is read with mincore (the file is our own, so the kernel
// answers truthfully; a dropped group is checked as < half resident: pages still in a per-CPU LRU batch are skipped by
// POSIX_FADV_DONTNEED). ctest
#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define EXPECT(c)                                                       \
    do {                                                                \
        if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); fails++; } \
    } while (0)

static const size_t kT = (size_t)4 << 20;  // bytes per tensor
static const char* kNames[] = {"model.layers.1.a", "model.layers.1.b", "model.layers.10.a", "model.layers.2.a"};

static std::string make_file() {
    // in the working directory (ctest: the build directory), not /tmp: a tmpfs keeps its pages (no page cache to drop)
    char path[] = "qe_load_drop_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); exit(2); }
    std::string h = "{";
    for (int i = 0; i < 4; i++)
        h += std::string(i ? "," : "") + "\"" + kNames[i] + "\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(kT) +
             "],\"data_offsets\":[" + std::to_string(i * kT) + "," + std::to_string((i + 1) * kT) + "]}";
    h += "}";
    while ((h.size() + 8) % 4096) h += ' ';  // tensors start page aligned
    uint64_t n = h.size();
    std::vector<char> data(4 * kT);
    for (size_t i = 0; i < data.size(); i++) data[i] = (char)(i * 131 + 7);
    if (write(fd, &n, 8) != 8 || write(fd, h.data(), n) != (ssize_t)n || write(fd, data.data(), data.size()) != (ssize_t)data.size()) {
        perror("write");
        exit(2);
    }
    fsync(fd);  // clean pages: POSIX_FADV_DONTNEED drops only those
    close(fd);
    return path;
}

// share of a tensor's pages in the page cache (0..1)
static double resident(const SafeTensors& st, const char* name) {
    const TensorInfo& t = st.get(name);
    const long pg = sysconf(_SC_PAGESIZE);
    const uintptr_t a = ((uintptr_t)t.data + pg - 1) / pg * pg, e = ((uintptr_t)t.data + t.nbytes) / pg * pg;
    std::vector<unsigned char> v((e - a) / pg);
    if (mincore((void*)a, e - a, v.data()) != 0) return -1;
    size_t r = 0;
    for (unsigned char c : v) r += c & 1;
    return (double)r / v.size();
}

static void touch(const SafeTensors& st) {
    volatile unsigned s = 0;
    for (const char* nm : kNames) {
        const TensorInfo& t = st.get(nm);
        for (size_t i = 0; i < t.nbytes; i += 512) s += (unsigned char)t.data[i];
    }
}

int main() {
    const std::string path = make_file();
    {
        SafeTensors st;
        st.open(path);
        touch(st);
        for (const char* nm : kNames) EXPECT(resident(st, nm) > 0.99);
        EXPECT(st.mapped_bytes() >= 4 * kT);

        LoadDrop d;
        d.begin(2, true);
        const std::string l1 = "model.layers.1.", l2 = "model.layers.2.";
        d.expect(&st, l1);  // loader A
        d.expect(&st, l2);
        d.ready();
        d.done(&st, l1);
        EXPECT(resident(st, "model.layers.1.a") > 0.99);  // loader B not ready yet
        d.expect(&st, l1);  // loader B
        d.expect(&st, l2);
        d.ready();
        EXPECT(resident(st, "model.layers.1.a") > 0.99);  // B still reads layer 1
        d.done(&st, l1);
        EXPECT(resident(st, "model.layers.1.a") < 0.5 && resident(st, "model.layers.1.b") < 0.5);
        EXPECT(resident(st, "model.layers.10.a") > 0.99);  // another layer whose name extends the prefix
        EXPECT(resident(st, "model.layers.2.a") > 0.99);
        EXPECT(d.dropped() == 2 * kT);
        d.done(&st, l2);
        EXPECT(resident(st, "model.layers.2.a") > 0.99);  // one of two loaders done
        d.done(&st, l2);
        EXPECT(resident(st, "model.layers.2.a") < 0.5);
        EXPECT(d.dropped() == 3 * kT);
        // the mapping stays valid: the bytes read back from the file
        const TensorInfo& t = st.get("model.layers.1.b");
        bool same = true;
        for (size_t i = 0; i < t.nbytes; i += 4093) same &= t.data[i] == (char)((kT + i) * 131 + 7);
        EXPECT(same);

        // off: nothing dropped, counts ignored
        touch(st);
        d.begin(1, false);
        d.expect(&st, l1);
        d.ready();
        d.done(&st, l1);
        EXPECT(resident(st, "model.layers.1.a") > 0.99 && d.dropped() == 0);
        // keep: a name part that stays cached (Flash-Next: the n-gram table under layer 1's prefix)
        d.begin(1, true, {".b"});
        d.expect(&st, l1);
        d.ready();
        d.done(&st, l1);
        EXPECT(resident(st, "model.layers.1.a") < 0.5 && resident(st, "model.layers.1.b") > 0.99 && d.dropped() == kT);
        touch(st);
        // a done for a group nobody declared: ignored
        d.begin(1, true);
        d.ready();
        d.done(&st, l1);
        EXPECT(resident(st, "model.layers.1.a") > 0.99 && d.dropped() == 0);
    }
    unlink(path.c_str());
    if (fails) return 1;
    printf("load_drop_unit: ok\n");
    return 0;
}
