#!/usr/bin/env python3
"""Archive list checks on the engine API (GPU, one engine load per part, no disk cache).

  supersede: the archive budget holds one archive, so making room for A2 drops A, the archive it supersedes.
  window:    an archive of a prefix made after bg_prepare keeps its entry when the background archive finishes.
  pinned:    the state at the system-prompt end has aged out of the snapshots: the file takes it from the pinned prefix.

usage: flock .gpu.lock python3 tests/test_archive_races.py <libqengine.so> [devices=0,1,2]
A part runs in its own process with freed memory overwritten (glibc perturb), so a use after free shows as a crash.
A part whose path was not reached fails too."""
import os, shutil, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
MODEL = os.environ.get("QE_MODEL", os.path.expanduser("~/GIT/QEngine/models/Qwen3.8-27B-INT4"))
P = [100 + i % 5000 for i in range(7000)]


def gen(e, ids, k=0, n=8):
    t, _, slot = e.start(ids, k)
    out = [t]
    while len(out) < n:
        out += e.step()
    return out[:n], slot


def part(name, lib, devs):
    from engine import Engine
    e = Engine(MODEL, devices=",".join(str(i) for i in range(len(devs.split(",")))), max_t=65536, lib_path=lib)
    e.set_concurrent(True)
    e.set_sampling(0)
    if name == "supersede":
        out, s = gen(e, P)
        e.release(s)
        e.persist()
        out2, s = gen(e, P + out + [7, 8, 9])
        e.release(s)
        e.persist()
        print("archives", e.num_archives(), flush=True)
        assert e.num_archives() == 1, "A2 alone must remain"
        return
    if name == "pinned":
        hint = 7000
        x = P + list(range(200, 400))
        t, _, s = e.start(x, 0, 0, hint)
        e.release(s)
        for i in range(4):                # other cold prompts age the kind 0 snapshot at the hint out of the store
            o = [6000 + i + j % 3000 for j in range(7000)]
            t, _, sl = e.start(o, 0, 0, 3000)
            e.release(sl)
        e.persist()
        d = os.environ["QE_DISK_CACHE"]
        got = {}
        for nm in os.listdir(d):
            if nm.endswith(".qkv"):
                with open(os.path.join(d, nm), "rb") as f:
                    f.read(8)
                    f.read(struct.unpack("<I", f.read(4))[0])
                    n, ns = struct.unpack("<iI", f.read(8))
                    got[n] = struct.unpack(f"<{ns}i", f.read(4 * ns))
        print("files", sorted(got.items()), flush=True)
        assert len(x) in got, "the conversation has no file"
        assert hint in got[len(x)], "the system-prompt state must come from the pinned prefix"
        return
    long = P + list(range(200, 400))
    _, t = gen(e, P, n=1)                # a prefix, in the lower slot: persist reaches it first
    _, s = gen(e, long, n=1)             # the longest idle conversation: the background archiver takes it
    assert t != s, "same slot"
    e.release(s)
    e.release(t)
    assert not e.archive_more(1 << 20)   # bg_prepare + the first pieces of `long`, not complete
    e.persist()                          # archives the prefix, then completes `long`: the prefix is newer than bg_prepare
    print("slots", s, t, "archives", e.num_archives(), flush=True)
    if t > s:
        print("not reached: the prefix was archived after the long one", flush=True)
        sys.exit(2)
    assert e.num_archives() == 2, "the archive made after bg_prepare must stay"


def main():
    lib, devs = os.path.abspath(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "0,1,2"
    if len(sys.argv) > 3:
        return part(sys.argv[3], lib, devs)
    rc = 0
    tmp = tempfile.mkdtemp(prefix="qe_races_")
    for name, gb, extra in (("supersede", "1.2", {}), ("window", "32", {}), ("pinned", "32", {"QE_SNAPSHOTS": "4", "QE_DISK_CACHE": tmp})):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devs, QE_POOL_LOG="1", QE_KV_ARCHIVE_GB=gb,
                   GLIBC_TUNABLES="glibc.malloc.perturb=165")
        env.pop("QE_DISK_CACHE", None)
        env.update(extra)
        r = subprocess.run([sys.executable, __file__, lib, devs, name], env=env, capture_output=True, text=True)
        keep = [x for x in r.stdout.splitlines() if x.startswith(("archive: slot", "archive: drop", "archives", "slots", "not reached", "files"))]
        code = r.returncode
        if name == "supersede" and code == 0 and not any(x.startswith("archive: drop") for x in keep):
            keep.append("not reached: no archive was dropped to make room")
            code = 2
        print(f"{name}: exit {code}\n  " + "\n  ".join(keep), flush=True)
        if code:
            print("  " + "\n  ".join(r.stderr.splitlines()[-3:]), flush=True)
            rc = 1
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if not rc else "FAIL", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
