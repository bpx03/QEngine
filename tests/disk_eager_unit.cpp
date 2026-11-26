// 27B disk cache, --disk-eager (src/disk_eager.h) on a model of Engine::drop_covered / disk_enqueue / disk_forget / the writer /
// persist: no GPU.: the cap, the skip, a running write kept, persist writing the skipped
// archives, superseded files deleted only after the newer file of the same conversation is written.
#include "disk_eager.h"
#include <cstdio>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

#define EXPECT(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)
static int fails = 0;
using qe_disk::Cover;

struct Job { std::string path; std::vector<std::pair<std::string, size_t>> covered; bool cancel = false; };
struct Ent { std::string conv, path; int n = 0; bool done = false, ok = false, cancel = false, skip = false; };

// one conversation per key, one archive entry per turn; the writer does one job at a time
struct Model {
    bool eager = true;
    size_t cap = 8;
    bool persisting = false;
    std::deque<Job> q;
    std::string running;           // path being written
    std::set<std::string> drive;   // files on the drive
    std::vector<Ent> ents;
    std::vector<std::pair<std::string, size_t>> covered_files;
    size_t live() const { size_t n = 0; for (auto& j : q) n += !j.cancel; return n; }
    bool full() const { return eager && !persisting && qe_disk::full(live(), cap); }
    Ent* ent(const std::string& path) { for (auto& e : ents) if (e.path == path) return &e; return nullptr; }
    void forget(Ent& e) {  // disk_forget
        if (e.path.empty()) return;
        const std::string p = e.path;
        if (qe_disk::take_queued(q, [&](const Job& j) { return j.path == p; }, covered_files)) e.done = true, e.ok = false, e.cancel = true;
        else if (e.done) drive.erase(p);
        else e.cancel = true;
        for (auto& j : q) if (j.path == p) j.cancel = true;
        if (running == p && e.cancel) running_cancel = true;
        e.path.clear();
    }
    bool running_cancel = false;
    void enqueue(Ent& e) {  // disk_enqueue
        if (full()) { e.skip = true; return; }
        e.skip = false;
        e.path = e.conv + "_" + std::to_string(e.n);
        Job j{e.path, covered_files};
        covered_files.clear();
        q.push_back(j);
    }
    void reply(const std::string& conv, int n) {  // bg_finish / archive_slot: drop_covered, then the enqueue
        const bool keep = full();
        for (auto& e : ents) {
            if (e.conv != conv || e.n >= n || e.path.empty()) continue;
            Cover c = qe_disk::cover(true, keep, e.done, e.ok, e.cancel, running == e.path, eager);
            if (c == Cover::KEEP) continue;
            if (c == Cover::CARRY) { covered_files.push_back({e.path, 1}); e.path.clear(); continue; }
            forget(e);
        }
        ents.push_back(Ent{conv, "", n});
        enqueue(ents.back());
        if (!keep || ents.back().skip) {  // disk_flush_covered: what no new file took over goes now
            for (auto& [p, s] : covered_files) drive.erase(p);
            covered_files.clear();
        }
    }
    std::vector<std::pair<std::string, size_t>> wcov;  // the writer's carried list
    void start() {  // the writer pops the next job
        if (!running.empty() || q.empty()) return;
        Job j = q.front();
        q.pop_front();
        wcov.insert(wcov.end(), j.covered.begin(), j.covered.end());
        if (j.cancel) { return; }
        running = j.path;
        running_cancel = false;
    }
    void finish() {  // ... and lands it (rename), then drops what it carried
        if (running.empty()) return;
        Ent* e = ent(running);
        const bool ok = !running_cancel;
        if (ok) drive.insert(running);
        if (e) e->done = true, e->ok = ok;
        if (ok) { for (auto& [p, s] : wcov) if (p != running) drive.erase(p); wcov.clear(); }
        running.clear();
    }
    void persist() {
        persisting = true;
        for (auto& e : ents) if (e.skip && e.path.empty()) enqueue(e);
        persisting = false;
        while (!q.empty() || !running.empty()) { start(); finish(); }
    }
    bool conv_on_drive(const std::string& c) const { for (auto& f : drive) if (f.rfind(c + "_", 0) == 0) return true; return false; }
};

int main() {
    // the rule itself: master's (eager off) is unchanged
    EXPECT(qe_disk::cover(false, false, true, true, false, false, false) == Cover::NONE);
    EXPECT(qe_disk::cover(true, true, true, true, false, false, false) == Cover::KEEP);
    EXPECT(qe_disk::cover(true, true, false, false, false, true, false) == Cover::DROP);
    EXPECT(qe_disk::cover(true, false, true, true, false, false, false) == Cover::CARRY);
    EXPECT(qe_disk::cover(true, false, false, false, false, true, false) == Cover::DROP);   // master cancels a running write
    EXPECT(qe_disk::cover(true, false, true, false, false, false, true) == Cover::DROP);    // a failed file is no copy
    // eager: a running write finishes; one in flight is kept when the covering archive gets no file
    EXPECT(qe_disk::cover(true, false, false, false, false, true, true) == Cover::CARRY);
    EXPECT(qe_disk::cover(true, false, false, false, false, false, true) == Cover::DROP);    // queued: replaced
    EXPECT(qe_disk::cover(true, true, false, false, false, false, true) == Cover::KEEP);
    EXPECT(qe_disk::cover(true, true, false, false, true, true, true) == Cover::DROP);      // (cancelled already)
    EXPECT(!qe_disk::full(7, 8) && qe_disk::full(8, 8) && !qe_disk::full(100, 0));

    {   // short gaps: every reply while the last one is still written; the running write lands, the queued one is replaced
        Model m;
        m.reply("x", 1); m.start();               // x_1 running
        m.reply("x", 2);                           // x_2 queued, carries x_1
        m.reply("x", 3);                           // x_2 replaced (not started): x_3 queued, carries x_1 (M1)
        EXPECT(m.q.size() == 1 && m.q[0].path == "x_3");
        m.finish();                                // x_1 lands
        EXPECT(m.drive.count("x_1"));
        m.start(); m.finish();                     // x_3 lands, x_1 goes after it
        EXPECT(m.drive.count("x_3") && !m.drive.count("x_1") && m.drive.size() == 1);
    }
    {   // M1: a replaced queued job's superseded file goes with THIS conversation's newer file, not with another's write
        Model m;
        m.reply("x", 1); m.start(); m.finish();   // x_1 on the drive
        m.reply("z", 1); m.start();                // z_1 running
        m.reply("x", 2);                           // x_2 queued, carries x_1
        m.reply("x", 3);                           // x_2 out of the queue: x_1 moves to x_3's job
        m.reply("y", 1);                           // y_1 queued after x_3
        m.finish();                                // z_1 lands: x_1 must stay (x_3 not written yet)
        EXPECT(m.drive.count("x_1"));
        m.start(); m.finish();                     // x_3 lands: x_1 goes
        EXPECT(m.drive.count("x_3") && !m.drive.count("x_1"));
    }
    {   // H1: the cap is full: the covered file stays (and the running write is kept), the new archive is skipped, persist writes it
        Model m;
        m.cap = 2;
        m.reply("x", 1); m.start(); m.finish();   // x_1 on the drive
        m.reply("a", 1); m.start();                // a_1 running
        m.reply("b", 1); m.reply("c", 1);          // 2 queued: full
        EXPECT(m.full());
        m.reply("x", 2);                           // skipped: x_1 must stay
        EXPECT(m.ents.back().skip && m.drive.count("x_1"));
        m.reply("a", 2);                           // a_1 running, cap full: kept (KEEP), a_2 skipped
        EXPECT(m.running == "a_1" && m.ents.back().skip);
        m.finish();
        EXPECT(m.drive.count("a_1"));              // the running write landed, nothing deleted it
        EXPECT(m.conv_on_drive("x") && m.conv_on_drive("a"));
        m.persist();                               // the skipped ones are written
        EXPECT(m.drive.count("x_2") && m.drive.count("a_2") && m.drive.count("b_1") && m.drive.count("c_1"));
    }
    {   // invariant over a long random-ish run: a conversation with a file never has none after a reply
        Model m;
        m.cap = 3;
        const char* cs[] = {"p", "q", "r", "s", "t"};
        int step = 0;
        for (int t = 1; t <= 40; t++)
            for (int k = 0; k < 5; k++) {
                const std::string c = cs[k];
                const bool had = m.conv_on_drive(c);
                m.reply(c, t);
                EXPECT(!had || m.conv_on_drive(c));
                if (++step % 3 == 0) { m.start(); }
                if (step % 4 == 0) { m.finish(); }
            }
        m.persist();
        for (const char* c : cs) EXPECT(m.drive.count(std::string(c) + "_40"));
    }
    printf("disk_eager_unit: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
