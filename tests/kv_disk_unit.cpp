// CPU test of the Flash-Next disk cache file format and writer (src/flash/kv_disk.h): round trip, every refusal, write failures,
// replaced and cancelled jobs. No GPU. A file that fails any of these must never reach a restore.
#include "flash/kv_disk.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

using namespace kvd;
static int fails = 0;
#define EXPECT(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

static Job make_job(const std::string& path, const std::vector<std::vector<char>>& data, long long key = 1) {
    Job j;
    j.key = key;
    j.path = path;
    j.m.fp = "fp-a";
    j.m.toks = {1, 2, 3, 4, 5, 1 << 30};
    j.m.kv_n = 6;
    j.m.ng = 2;
    BlobMeta s; s.kind = KV_SEG; s.a = 0; s.b = 6; s.size = {data[0].size(), data[1].size()};
    BlobMeta e; e.kind = END_STATE; e.a = 6; e.b = 3; e.pend = 1; e.st_kind = 4; e.size = {data[2].size(), data[3].size()};
    j.m.blobs = {s, e};
    j.ptr = {{data[0].data(), data[1].data()}, {data[2].data(), data[3].data()}};
    return j;
}
static std::vector<char> bytes(size_t n, unsigned seed) {
    std::vector<char> v(n);
    unsigned x = seed;
    for (auto& c : v) { x = x * 1103515245u + 12345u; c = (char)(x >> 16); }
    return v;
}
static void patch(const std::string& p, long off, char xr) {
    FILE* f = fopen(p.c_str(), "r+b");
    fseek(f, off, SEEK_SET);
    int c = fgetc(f);
    fseek(f, off, SEEK_SET);
    fputc(c ^ xr, f);
    fclose(f);
}
static bool exists(const std::string& p) { struct stat s; return stat(p.c_str(), &s) == 0; }

int main() {
    char tmpl[] = "/tmp/kv_disk_XXXXXX";
    const char* base = getenv("KV_DISK_TEST_DIR");
    std::string dir;
    if (base) { std::string t = std::string(base) + "/kv_disk_XXXXXX"; std::vector<char> b(t.begin(), t.end()); b.push_back(0); dir = mkdtemp(b.data()); }
    else dir = mkdtemp(tmpl);
    std::vector<std::vector<char>> d = {bytes(1 << 20, 1), bytes((1 << 20) + 7, 2), bytes(4097, 3), bytes(33, 4)};
    const std::string p = dir + "/a.qfk";

    {   // round trip
        Job j = make_job(p, d);
        int err = 0;
        EXPECT(write_file(j, nullptr, err));
        EXPECT(exists(p) && !exists(p + ".part"));
        Meta m;
        EXPECT(read_header(p, "fp-a", m) == RD_OK);
        EXPECT(m.toks == j.m.toks && m.kv_n == 6 && m.ng == 2 && m.blobs.size() == 2);
        EXPECT(m.blobs[1].kind == END_STATE && m.blobs[1].a == 6 && m.blobs[1].b == 3 && m.blobs[1].pend == 1 && m.blobs[1].st_kind == 4);
        const int fd = open(p.c_str(), O_RDONLY);
        for (size_t i = 0; i < 2; i++)
            for (int r = 0; r < 2; r++) {
                std::vector<char> got(m.blobs[i].size[r]);
                EXPECT(read_blob(fd, m, i, r, got.data()));
                EXPECT(got == d[i * 2 + r]);
            }
        close(fd);
        // another configuration: refused without reading on
        Meta o;
        EXPECT(read_header(p, "fp-b", o) == RD_OTHER);
    }
    {   // the steering of a file: kept apart from the fingerprint, round trip; a flipped byte in it is a damaged file; every part of the id counts
        const std::string q = dir + "/s.qfk";
        const std::string id = steer_id("strata_l8_33", 0x1234, 1, 8, 33, 1.f);
        Job j = make_job(q, d);
        j.m.steer = id;
        int err = 0;
        EXPECT(write_file(j, nullptr, err));
        Meta m;
        EXPECT(read_header(q, "fp-a", m) == RD_OK && m.steer == id && m.toks == j.m.toks);
        Meta o;
        EXPECT(read_header(q, "fp-b", o) == RD_OTHER);
        Meta none;
        EXPECT(read_header(p, "fp-a", none) == RD_OK && none.steer.empty());  // (unsteered)
        const size_t at = 12 + 4 + j.m.fp.size() + 4 + 3;  // a byte of the id
        patch(q, (long)at, 1);
        Meta bad;
        EXPECT(read_header(q, "fp-a", bad) == RD_BAD);
        unlink(q.c_str());
        EXPECT(steer_id("a", 1, 1, 8, 33, 1.f) == steer_id("a", 1, 1, 8, 33, 1.f));
        for (const std::string& x : {steer_id("b", 1, 1, 8, 33, 1.f), steer_id("a", 2, 1, 8, 33, 1.f), steer_id("a", 1, 0, 8, 33, 1.f),
                                     steer_id("a", 1, 1, 9, 33, 1.f), steer_id("a", 1, 1, 8, 32, 1.f), steer_id("a", 1, 1, 8, 33, 0.5f)})
            EXPECT(x != steer_id("a", 1, 1, 8, 33, 1.f));
    }
    const off_t full = [&] { struct stat s; stat(p.c_str(), &s); return s.st_size; }();
    {   // truncated anywhere: refused, also in the header and at the last byte
        const std::string q = dir + "/t.qfk";
        for (off_t cut : {(off_t)0, (off_t)7, (off_t)8, (off_t)11, (off_t)40, (off_t)200, full / 2, full - 1}) {
            std::string c = "cp " + p + " " + q;
            EXPECT(system(c.c_str()) == 0);
            EXPECT(truncate(q.c_str(), cut) == 0);
            Meta m;
            const Read v = read_header(q, "fp-a", m);
            EXPECT(cut < 8 ? v == RD_NOT_OURS : v == RD_BAD);
        }
        // a longer file too
        std::string c = "cp " + p + " " + q + " && echo x >> " + q;
        EXPECT(system(c.c_str()) == 0);
        Meta m;
        EXPECT(read_header(q, "fp-a", m) == RD_BAD);
        // not our magic
        patch(q, 0, 1);
        Meta m2;
        EXPECT(read_header(q, "fp-a", m2) == RD_NOT_OURS);
        unlink(q.c_str());
    }
    {   // a flipped bit in the header: refused; in the data: the header passes, the read of that blob fails
        const std::string q = dir + "/f.qfk";
        Meta m0;
        read_header(p, "fp-a", m0);
        for (long off : {12L, 20L, (long)m0.data_off - 30, (long)m0.data_off - 1}) {
            std::string c = "cp " + p + " " + q;
            EXPECT(system(c.c_str()) == 0);
            patch(q, off, 4);
            Meta m;
            EXPECT(read_header(q, "fp-a", m) == RD_BAD);
        }
        std::string c = "cp " + p + " " + q;
        EXPECT(system(c.c_str()) == 0);
        patch(q, full - 1, 1);
        Meta m;
        EXPECT(read_header(q, "fp-a", m) == RD_OK);
        const int fd = open(q.c_str(), O_RDONLY);
        std::vector<char> a(m.blobs[0].size[0]), z(m.blobs[1].size[1]);
        EXPECT(read_blob(fd, m, 0, 0, a.data()));   // the other blobs still read
        EXPECT(!read_blob(fd, m, 1, 1, z.data()));  // the damaged one does not
        close(fd);
        unlink(q.c_str());
    }
    {   // disk full on a real descriptor; a file size limit mid-write: failure, no file and no .part left
        const int fd = open("/dev/full", O_WRONLY);
        if (fd >= 0) {
            errno = 0;
            EXPECT(!write_all(fd, d[0].data(), d[0].size()));
            EXPECT(errno == ENOSPC);
            close(fd);
        }
        signal(SIGXFSZ, SIG_IGN);
        rlimit old, lim;
        getrlimit(RLIMIT_FSIZE, &old);
        lim = old;
        lim.rlim_cur = 1 << 16;
        setrlimit(RLIMIT_FSIZE, &lim);
        const std::string q = dir + "/full.qfk";
        Job j = make_job(q, d);
        int err = 0;
        EXPECT(!write_file(j, nullptr, err));
        EXPECT(err != 0 && !exists(q) && !exists(q + ".part"));
        setrlimit(RLIMIT_FSIZE, &old);
        // an unwritable place
        Job j2 = make_job(dir + "/no/such/dir/x.qfk", d);
        err = 0;
        EXPECT(!write_file(j2, nullptr, err) && err == ENOENT);
    }
    {   // the writer: a queued job of the same conversation is replaced, the result says so; the held blobs are listed
        // (a big first job keeps the writer busy while the others are pushed)
        std::vector<std::vector<char>> big = {bytes(64 << 20, 5), bytes(64 << 20, 6), bytes(16, 7), bytes(16, 8)};
        Writer w;
        Job first = make_job(dir + "/w0.qfk", big, 7);
        auto keep = std::make_shared<int>(1);
        first.hold.push_back(keep);
        w.push(std::move(first), 10);
        Job second = make_job(dir + "/w1.qfk", d, 8);
        Job third = make_job(dir + "/w2.qfk", d, 8);
        w.push(std::move(second), 20);
        w.push(std::move(third), 30);
        std::vector<const void*> held;
        w.held(held);
        EXPECT(w.wait(60));
        std::vector<Done> dn = w.take_done();
        EXPECT(dn.size() == 3);
        uint64_t res = 0;
        int ok = 0, canc = 0;
        for (auto& x : dn) { res += x.reserved; ok += x.ok; canc += x.cancelled; }
        EXPECT(res == 60 && ok == 2 && canc == 1);
        EXPECT(exists(dir + "/w0.qfk") && exists(dir + "/w2.qfk") && !exists(dir + "/w1.qfk"));
        EXPECT(keep.use_count() == 1);  // the writer let go of the blobs
        // cancelled while running: nothing stays
        Job a = make_job(dir + "/c0.qfk", big, 9);
        Job b = make_job(dir + "/c1.qfk", d, 9);
        w.push(std::move(a), 1);
        w.push(std::move(b), 2);
        EXPECT(w.wait(60));
        EXPECT(!exists(dir + "/c0.qfk.part") && exists(dir + "/c1.qfk"));
        // a failed job reports its error
        w.take_done();
        w.push(make_job(dir + "/no/such/x.qfk", d, 10), 1);
        EXPECT(w.wait(60));
        dn = w.take_done();
        EXPECT(dn.size() == 1 && !dn[0].ok && !dn[0].cancelled && dn[0].err == ENOENT);
        w.stop();
    }
    std::string rm = "rm -rf " + dir;
    EXPECT(system(rm.c_str()) == 0);
    printf(fails ? "kv_disk_unit: %d FAILED\n" : "kv_disk_unit: ok\n", fails);
    return fails ? 1 : 0;
}
