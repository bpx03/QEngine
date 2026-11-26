#pragma once
// 27B disk cache: what happens to the file of an archive that a newer archive of the same conversation covers
// (Engine::drop_covered), and the removal of a write that has not started (Engine::disk_forget). Header only, no GPU types,
// so tests/disk_eager_unit.cpp checks it without a card. [disk-eager] (--disk-eager, QE_DISK_EAGER) review rounds
//.
#include <cstddef>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace qe_disk {

enum class Cover {
    NONE,   // no file (nor a write) to decide about
    KEEP,   // the covered entry stays, on disk only: the covering archive gets no file now (keep_files)
    CARRY,  // its file goes once the covering archive's file is written (the writer's drop_carried)
    DROP,   // forgotten now: a finished file is deleted, a write in flight cancelled, a queued one removed
};

// keep_files: the covering archive gets no file (no file plan, or the eager queue is full); done / ok / cancel: the covered
// file's write state; running: its write is the one the writer is doing now; eager: QE_DISK_EAGER.
// Without eager this is the rule of master: a finished file is kept or carried, anything still being written dropped.
// Eager keeps a write in flight (queued or running) when the covering archive is not written (H1: never lose the last good
// file before its replacement is on the drive), and lets a running one finish when it is (M3: in agent loops with short
// gaps, cancelling it every turn left nothing on the drive).
inline Cover cover(bool has_file, bool keep_files, bool done, bool ok, bool cancel, bool running, bool eager) {
    if (!has_file) return Cover::NONE;
    const bool good = done && ok, in_flight = !done && !cancel;
    if (keep_files) return good || (eager && in_flight) ? Cover::KEEP : Cover::DROP;
    if (good || (eager && running && in_flight)) return Cover::CARRY;
    return Cover::DROP;
}

// the eager queue cap: live (not cancelled) queued writes
inline bool full(size_t live, size_t cap) { return cap > 0 && live >= cap; }

// M1: a write that has not started leaves the queue (its memory copy goes with it); the files it superseded move to
// `covered`, so the archive that covers this one carries them (deleted only once that archive's file is written).
// is_it(job) picks the job. true: it was queued (not started).
template <typename Job, typename F>
bool take_queued(std::deque<Job>& q, F&& is_it, std::vector<std::pair<std::string, size_t>>& covered) {
    for (auto it = q.begin(); it != q.end(); ++it)
        if (is_it(*it)) {
            covered.insert(covered.end(), it->covered.begin(), it->covered.end());
            q.erase(it);
            return true;
        }
    return false;
}

}  // namespace qe_disk
