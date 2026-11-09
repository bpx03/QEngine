#pragma once
// Flash-Next disk cache (--disk-cache): the file format and the background writer.
// One file per conversation under the host-RAM archive: its KV segments and recurrent states, as the archive holds them.
//
// A file serves only the engine that wrote it, or one equal in everything that shapes its bytes (the fingerprint: build,
// weights, layout, see FE::dk_fingerprint), and a damaged file is refused, never restored:
//   header (magic, length, fingerprint, tokens, one entry per blob with its size and hash per card), hash of the header, data.
// Written to <path>.part, synced, then renamed: a crash leaves a .part (removed at the next start) or a whole file.
//
// Header only and no GPU types, so that tests/kv_disk_unit.cpp runs all of this without a card.
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace kvd {

constexpr char kMagic[8] = {'Q', 'F', 'K', 'V', '0', '0', '0', '2'};
constexpr uint32_t kMaxHeader = 1u << 28;
constexpr size_t kChunk = (size_t)64 << 20;

// four 64-bit lanes, a multiply and a rotate per word (several GB/s on one core): finds torn and flipped data, not an attacker
inline uint64_t hash_bytes(const void* data, size_t n) {
    constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full;
    auto rot = [](uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
    uint64_t h[4] = {P1 + P2, P2, 0, 0 - P1};
    const char* p = (const char*)data;
    size_t i = 0;
    for (; i + 32 <= n; i += 32)
        for (int k = 0; k < 4; k++) {
            uint64_t w;
            memcpy(&w, p + i + 8 * k, 8);
            h[k] = rot(h[k] + w * P2, 31) * P1;
        }
    uint64_t t = (uint64_t)n;
    for (int k = 0; k < 4; k++) t = rot(t ^ h[k], 27) * P1 + P2;
    for (; i < n; i++) t = rot(t ^ (uint64_t)(uint8_t)p[i] * P1, 11) * P2;
    t ^= t >> 29;
    t *= P2;
    return t ^ (t >> 32);
}

// A steering key that means the same after a restart. The engine's key holds the vector's index in the loaded list, which changes
// with QE_STEER_DIR; this holds its name, a hash of what it applies (every layer and mode, sidecar scale and strengths folded in) and
// the request's mode, layers and scale. A vector changed or gone yields no matching id, so its files are never loaded as a match.
inline std::string steer_id(const std::string& name, uint64_t vec_hash, int mode, int lo, int hi, float scale) {
    uint32_t bits;
    memcpy(&bits, &scale, 4);
    char b[96];
    snprintf(b, sizeof b, ":%016llx|%d|%d|%d|%08x", (unsigned long long)vec_hash, mode, lo, hi, bits);
    return name + b;
}

enum BlobKind { KV_SEG = 0, END_STATE = 1, CK_STATE = 2 };
struct BlobMeta {
    int kind = KV_SEG;
    int a = 0, b = 0;       // KV_SEG: positions [a, b). A state: position a, prev_m b
    int pend = 0, st_kind = 0;
    std::vector<uint64_t> size, hash;  // per card
};
struct Meta {
    std::string fp;
    std::string steer;  // the steering it was computed with (steer_id), empty: none. Per file: another vector set keeps the unsteered files
    std::vector<int> toks;
    int kv_n = 0, ng = 0;
    std::vector<BlobMeta> blobs;
    uint64_t data_off = 0, total = 0;  // file offset of the first blob, bytes of all blobs
};
inline uint64_t data_bytes(const Meta& m) {
    uint64_t t = 0;
    for (auto& b : m.blobs)
        for (uint64_t s : b.size) t += s;
    return t;
}

template <class T>
void put(std::vector<char>& v, const T& x) {
    const char* p = (const char*)&x;
    v.insert(v.end(), p, p + sizeof(T));
}
// the whole header; sets m.data_off, m.total
inline std::vector<char> encode_header(Meta& m) {
    std::vector<char> body;
    put(body, (uint32_t)m.fp.size());
    body.insert(body.end(), m.fp.begin(), m.fp.end());
    put(body, (uint32_t)m.steer.size());
    body.insert(body.end(), m.steer.begin(), m.steer.end());
    put(body, (int32_t)m.toks.size());
    put(body, (int32_t)m.kv_n);
    put(body, (int32_t)m.ng);
    put(body, (uint32_t)m.blobs.size());
    for (auto& b : m.blobs) {
        for (int x : {b.kind, b.a, b.b, b.pend, b.st_kind}) put(body, (int32_t)x);
        for (int r = 0; r < m.ng; r++) {
            put(body, (uint64_t)b.size[r]);
            put(body, (uint64_t)b.hash[r]);
        }
    }
    const char* t = (const char*)m.toks.data();
    body.insert(body.end(), t, t + m.toks.size() * 4);
    std::vector<char> out(kMagic, kMagic + 8);
    put(out, (uint32_t)body.size());
    out.insert(out.end(), body.begin(), body.end());
    put(out, hash_bytes(body.data(), body.size()));
    m.data_off = out.size();
    m.total = data_bytes(m);
    return out;
}

inline bool read_all(int fd, void* dst, size_t n, uint64_t off) {
    char* d = (char*)dst;
    while (n) {
        const ssize_t r = pread(fd, d, std::min(n, kChunk), (off_t)off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        d += r;
        off += (uint64_t)r;
        n -= (size_t)r;
    }
    return true;
}
inline bool write_all(int fd, const void* src, size_t n, const std::atomic<bool>* cancel = nullptr) {
    const char* s = (const char*)src;
    while (n) {
        if (cancel && cancel->load()) return false;
        const ssize_t w = write(fd, s, std::min(n, kChunk));
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) {
            if (w == 0) errno = ENOSPC;
            return false;
        }
        s += w;
        n -= (size_t)w;
    }
    return true;
}

enum Read {
    RD_OK,
    RD_NOT_OURS,  // no magic: not touched
    RD_OTHER,     // a valid file of another engine or configuration (fingerprint): refused, not touched
    RD_BAD,       // truncated or damaged
};
// the header of a file; the file's length is checked too. A file of another fingerprint: only that verdict
inline Read read_header(const std::string& path, const std::string& fp, Meta& m, const char** why = nullptr) {
    auto bad = [&](const char* w) { if (why) *why = w; return RD_BAD; };
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return bad("cannot open");
    struct Close { int fd; ~Close() { close(fd); } } cl{fd};
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) return bad("not a regular file");
    char mg[8];
    uint32_t len = 0;
    if ((size_t)st.st_size < 8) return RD_NOT_OURS;
    if (!read_all(fd, mg, 8, 0)) return bad("cannot read");
    if (memcmp(mg, kMagic, 8) != 0) return RD_NOT_OURS;
    if ((size_t)st.st_size < 12 || !read_all(fd, &len, 4, 8)) return bad("truncated header");
    if (len > kMaxHeader || (uint64_t)st.st_size < 12ull + len + 8) return bad("truncated header");
    std::vector<char> body(len);
    uint64_t h = 0;
    if (!read_all(fd, body.data(), len, 12) || !read_all(fd, &h, 8, 12ull + len)) return bad("cannot read the header");
    if (h != hash_bytes(body.data(), len)) return bad("header checksum");
    size_t off = 0;
    auto get = [&](void* d, size_t n) {
        if (off + n > len) return false;
        memcpy(d, body.data() + off, n);
        off += n;
        return true;
    };
    uint32_t fl = 0, nb = 0;
    int32_t n = 0, kv = 0, ng = 0;
    if (!get(&fl, 4) || fl > len - off) return bad("header fields");
    m.fp.assign(body.data() + off, fl);
    off += fl;
    if (m.fp != fp) return RD_OTHER;
    uint32_t sl = 0;
    if (!get(&sl, 4) || sl > len - off) return bad("header fields");
    m.steer.assign(body.data() + off, sl);
    off += sl;
    if (!get(&n, 4) || !get(&kv, 4) || !get(&ng, 4) || !get(&nb, 4)) return bad("header fields");
    if (n <= 0 || kv < 0 || kv > n || ng <= 0 || ng > 64 || nb > 1u << 16) return bad("header fields");
    m.kv_n = kv;
    m.ng = ng;
    m.blobs.resize(nb);
    for (auto& b : m.blobs) {
        int32_t f[5];
        if (!get(f, sizeof f)) return bad("header fields");
        b.kind = f[0]; b.a = f[1]; b.b = f[2]; b.pend = f[3]; b.st_kind = f[4];
        b.size.resize(ng);
        b.hash.resize(ng);
        for (int r = 0; r < ng; r++)
            if (!get(&b.size[r], 8) || !get(&b.hash[r], 8)) return bad("header fields");
        if (b.kind < KV_SEG || b.kind > CK_STATE || b.a < 0) return bad("header fields");
        if (b.kind == KV_SEG ? b.b <= b.a || b.a >= n : b.a > n) return bad("header fields");
    }
    m.toks.resize(n);
    if (!get(m.toks.data(), (size_t)n * 4) || off != len) return bad("header fields");
    m.data_off = 12ull + len + 8;
    m.total = data_bytes(m);
    if ((uint64_t)st.st_size != m.data_off + m.total) return bad("truncated data");
    return RD_OK;
}

// offset of blob i, card r
inline uint64_t blob_off(const Meta& m, size_t i, int r) {
    uint64_t o = m.data_off;
    for (size_t k = 0; k < i; k++)
        for (uint64_t s : m.blobs[k].size) o += s;
    for (int q = 0; q < r; q++) o += m.blobs[i].size[q];
    return o;
}
// one card's bytes of a blob into memory, checked
inline bool read_blob(int fd, const Meta& m, size_t i, int r, void* dst) {
    const uint64_t n = m.blobs[i].size[r];
    return read_all(fd, dst, n, blob_off(m, i, r)) && hash_bytes(dst, n) == m.blobs[i].hash[r];
}

// A job: the conversation's blobs (ptr[blob][card], held until written: the archive never changes a blob once it is filled)
struct Job {
    long long key = 0;           // the conversation: a newer job of the same one replaces this one
    std::string path, old_path;  // old_path: the file this one replaces, for the caller
    Meta m;
    std::vector<std::vector<const char*>> ptr;
    std::vector<std::shared_ptr<const void>> hold;
};
struct Done {
    long long key = 0;
    std::string path, old_path;
    Meta m;
    uint64_t reserved = 0;  // what the caller counted for it
    bool ok = false, cancelled = false;
    int err = 0;
    double ms = 0;
};
// The whole write: hashes, <path>.part, sync, rename. A cancel between chunks stops it. false: nothing stays behind
inline bool write_file(Job& j, const std::atomic<bool>* cancel, int& err) {
    Meta& m = j.m;
    for (size_t i = 0; i < m.blobs.size(); i++) {
        m.blobs[i].hash.resize(m.ng);
        for (int r = 0; r < m.ng; r++) {
            if (cancel && cancel->load()) return false;
            m.blobs[i].hash[r] = hash_bytes(j.ptr[i][r], m.blobs[i].size[r]);
        }
    }
    const std::vector<char> h = encode_header(m);
    const std::string tmp = j.path + ".part";
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) { err = errno; return false; }
    bool ok = write_all(fd, h.data(), h.size(), cancel);
    for (size_t i = 0; ok && i < m.blobs.size(); i++)
        for (int r = 0; ok && r < m.ng; r++) ok = write_all(fd, j.ptr[i][r], m.blobs[i].size[r], cancel);
    if (ok) ok = fdatasync(fd) == 0;
    if (!ok) err = errno;
#ifdef POSIX_FADV_DONTNEED
    if (ok) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // no gigabytes of page cache in front of the n-gram table (disk mode)
#endif
    if (close(fd) != 0 && ok) { err = errno; ok = false; }
    if (ok && rename(tmp.c_str(), j.path.c_str()) != 0) { err = errno; ok = false; }
    if (!ok) unlink(tmp.c_str());
    return ok;
}

// One background thread; the caller's thread pushes jobs and collects results, the writer touches nothing else.
class Writer {
public:
    Writer() : th_([this] { loop(); }) {}
    ~Writer() { stop(); }
    // A job of the same conversation still queued is replaced, one being written is cancelled (it stops within a chunk)
    void push(Job&& j, uint64_t reserved) {
        std::lock_guard<std::mutex> l(mu_);
        for (auto it = q_.begin(); it != q_.end();) {
            if (it->j.key == j.key) {
                finish_locked(it->j, it->reserved, false, true, 0, 0);
                it = q_.erase(it);
            } else {
                ++it;
            }
        }
        if (busy_ && cur_key_ == j.key) cancel_ = true;
        q_.push_back(Item{std::move(j), reserved});
        cv_.notify_all();
    }
    size_t queued() const { std::lock_guard<std::mutex> l(mu_); return q_.size() + (busy_ ? 1 : 0); }
    // blobs held by queued and running jobs: an eviction of their conversation frees nothing until they are written
    void held(std::vector<const void*>& out) const {
        std::lock_guard<std::mutex> l(mu_);
        for (auto& x : q_)
            for (auto& h : x.j.hold) out.push_back(h.get());
        for (const void* p : cur_hold_) out.push_back(p);
    }
    std::vector<Done> take_done() {
        std::lock_guard<std::mutex> l(mu_);
        std::vector<Done> d;
        d.swap(done_);
        return d;
    }
    // until nothing is queued or running; false: the time ran out (a hung disk: the caller goes on)
    bool wait(double seconds) {
        std::unique_lock<std::mutex> l(mu_);
        return cv_.wait_for(l, std::chrono::duration<double>(seconds), [&] { return q_.empty() && !busy_; });
    }
    void stop() {  // the queue is finished first
        {
            std::lock_guard<std::mutex> l(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

private:
    struct Item { Job j; uint64_t reserved; };
    void finish_locked(Job& j, uint64_t reserved, bool ok, bool cancelled, int err, double ms) {
        Done d;
        d.key = j.key;
        d.path = std::move(j.path);
        d.old_path = std::move(j.old_path);
        d.m = std::move(j.m);
        d.reserved = reserved;
        d.ok = ok;
        d.cancelled = cancelled;
        d.err = err;
        d.ms = ms;
        done_.push_back(std::move(d));
    }
    void loop() {
        std::unique_lock<std::mutex> l(mu_);
        while (true) {
            cv_.wait(l, [&] { return stop_ || !q_.empty(); });
            if (q_.empty()) return;
            Item it = std::move(q_.front());
            q_.pop_front();
            busy_ = true;
            cancel_ = false;
            cur_key_ = it.j.key;
            cur_hold_.clear();
            for (auto& h : it.j.hold) cur_hold_.push_back(h.get());
            l.unlock();
            const auto t0 = std::chrono::steady_clock::now();
            int err = 0;
            const bool ok = write_file(it.j, &cancel_, err);
            it.j.hold.clear();  // (the blobs of an evicted conversation go back to the archive's memory here)
            it.j.ptr.clear();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            l.lock();
            finish_locked(it.j, it.reserved, ok, !ok && cancel_.load(), err, ms);
            busy_ = false;
            cur_hold_.clear();
            cv_.notify_all();
        }
    }
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> q_;
    std::vector<Done> done_;
    std::vector<const void*> cur_hold_;
    long long cur_key_ = 0;
    bool busy_ = false, stop_ = false;
    std::atomic<bool> cancel_{false};
    std::thread th_;
};

}  // namespace kvd
