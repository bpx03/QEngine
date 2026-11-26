#!/usr/bin/env python3
"""Issue #47 (GPU, 27B server, HTTP): a conversation attached to a pinned shared system prompt (QE_SHARE_PREFIX=1, the
default) must reach the disk cache and come back after a graceful restart.

  run A (--disk-cache): X (creates the pinned prefix), Y (attaches to it), idle persist: both on disk; disk accounting =
         the payload bytes of the files; SIGTERM
  run B (same cache): Y's next turn resumes from its own file (cached ~ all of Y), Z (new, same system prompt) resumes at
         the system end and pins it again, W attaches to the re-pinned prefix
  run C (new lib, no --disk-cache): X, Y, Y's next turn in RAM (reference); X's reply = the default path's
  run M (reference lib, no --disk-cache): X (the default path's md5 must equal C's)

  overflow (env QE_TEST_OVERFLOW=1, one start, QE_MAX_SEQS=2, small QE_KV_ARCHIVE_GB): Y's memory copy is pushed out by
         other conversations after its file is written; Y's next turn restores from the file (the entry took the file's
         flat layout) and must equal the reference reply (QE_TEST_Y2_MD5, from the main run's run C)

usage: flock .gpu.lock python3 tests/test_disk_shared.py <libqengine.so> <reference libqengine.so> [port=8181] [devices=0]
Exit 0: pass, 1: fail. Takes one GPU for four server starts (past the 10 s of the quick suites); the cache is removed at the end."""
import hashlib, json, os, re, shutil, signal, struct, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(sys.argv[1])
REF = os.path.abspath(sys.argv[2])
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8181
DEVS = sys.argv[4] if len(sys.argv) > 4 else "0"
MPORT = 18861
MODEL = os.environ.get("QE_MODEL", os.path.expanduser("~/GIT/QEngine/models/Qwen3.8-27B-INT4"))
TMP = tempfile.mkdtemp(prefix="qe_disk_shared_")
CACHE = os.path.join(TMP, "cache")
LOG = os.path.join(TMP, "server.log")
SYS = " ".join(f"Rule {i}: answer record {i * 7919 % 10007} with the field {i * 104729 % 1009} first." for i in range(330))
PIDS = []


def start(lib, disk, persist_idle=3, extra_env=None):
    # QE_MAX_SEQS 2 (QE_TEST_SEQS): with the one-card default the shared region found no room next to X's prefill
    # reservation, so Y copied the prefix instead of attaching (gate run 11-25 23:38); 2 slots leave the room
    env = dict(os.environ, QE_POOL_LOG="1", QE_WARMUP="0", CUDA_VISIBLE_DEVICES=DEVS,
               QE_MAX_SEQS=os.environ.get("QE_TEST_SEQS", "2"))
    env.update(extra_env or {})
    out = open(LOG, "ab")
    out.write(f"\n===== start lib={lib} disk={disk}\n".encode())
    out.flush()
    args = [sys.executable, os.path.join(HERE, "..", "server", "app.py"), "--model", MODEL, "--engine-lib", lib,
            "--devices", ",".join(str(i) for i in range(len(DEVS.split(",")))), "--port", str(PORT),
            "--metrics-port", str(MPORT), "--log-file", "", "--stats-file", os.path.join(TMP, "stats.json")]
    if disk:
        args += ["--disk-cache", CACHE, "--disk-cache-gb", "40", "--persist-idle", str(persist_idle)]
    p = subprocess.Popen(args, env=env, stdout=out, stderr=subprocess.STDOUT)
    PIDS.append(p.pid)
    print(f"server pid {p.pid} ({'disk' if disk else 'no disk'}, {os.path.basename(os.path.dirname(os.path.dirname(lib)))})", flush=True)
    for _ in range(2400):
        if p.poll() is not None:
            raise SystemExit(f"server exited with {p.returncode}: see {LOG}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
            return p
        except OSError:
            time.sleep(0.25)
    raise SystemExit("server did not come up")


def stop(p):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def chat(msgs, n=60):
    body = {"model": "m", "messages": msgs, "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    rq = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(rq, timeout=600))
    u = r["usage"]
    return r["choices"][0]["message"]["content"], u["prompt_tokens"], u.get("prompt_tokens_details", {}).get("cached_tokens", 0)


def log_lines():
    return open(LOG, errors="replace").read().splitlines()


def wait_count(pat, n0, tmo=180):
    for _ in range(int(tmo / 0.1)):
        if sum(bool(re.search(pat, x)) for x in log_lines()) > n0:
            return True
        time.sleep(0.1)
    return False


def count(pat, since=0):
    return sum(bool(re.search(pat, x)) for x in log_lines()[since:])


def files():
    res = []
    for nm in sorted(os.listdir(CACHE)):
        if nm.endswith(".qkv"):
            path = os.path.join(CACHE, nm)
            with open(path, "rb") as f:
                f.read(8)
                f.read(struct.unpack("<I", f.read(4))[0])
                n, ns = struct.unpack("<iI", f.read(8))
                subs = list(struct.unpack(f"<{ns}i", f.read(4 * ns)))
                ng = struct.unpack("<I", f.read(4))[0]
                sizes = struct.unpack(f"<{ng}Q", f.read(8 * ng))
            res.append({"n": n, "subs": subs, "payload": sum(sizes), "bytes": os.path.getsize(path)})
    return sorted(res, key=lambda x: x["n"])


def stats_disk_bytes():
    j = json.load(urllib.request.urlopen(f"http://127.0.0.1:{MPORT}/api/stats", timeout=10))
    found = []

    def walk(x):
        if isinstance(x, dict):
            if "disk_bytes" in x:
                found.append(x["disk_bytes"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(j)
    return found[0] if found else None


XLONG = "List rules 1 to 80, one line each, with their record and field."


def xy(s, qx, qy, need_pin=True):
    """X (long reply, so it keeps its slot) and, while X decodes, Y: Y must take another slot and attach to the prefix
    X pinned (a sequential Y reuses X's own slot and never attaches). Returns X, (Y, prompt, cached), Y's archive GB"""
    import threading
    u = lambda q: {"role": "user", "content": q}
    pins = count(r"archive: pinned prefix")
    box = {}
    t = threading.Thread(target=lambda: box.update(x=chat([s, u(qx)], 400)))
    t.start()
    if need_pin and not wait_count(r"archive: pinned prefix", pins, 120):
        raise SystemExit("X did not pin its system prompt")
    time.sleep(0.5 if need_pin else 3.0)  # (X decodes: 400 tokens take ~1.5 s at ~260 tok/s, its prefill ~3 s)
    y = chat([s, u(qy)])
    t.join()
    return box["x"][0], y


def split_archives(since):
    """(tokens, GB) of the archives written since line `since` that are attached (pool log: 'split' marker absent, so by size)"""
    out = []
    for x in log_lines()[since:]:
        m = re.search(r"archive: slot \d+ (\d+) tokens .* ([\d.]+) GB \(used", x)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()[:8]


def overflow():
    u = lambda q: {"role": "user", "content": q}
    a_ = lambda q: {"role": "assistant", "content": q}
    s = {"role": "system", "content": SYS}
    p = start(LIB, True, 3, {"QE_MAX_SEQS": "2", "QE_KV_ARCHIVE_GB": os.environ.get("QE_TEST_ARCH_GB", "1.2")})
    try:
        persisted = count(r"persisted slots")
        _, (y, ny, cy) = xy(s, XLONG, "What is rule 2?")
        print(f"overflow: Y prompt {ny} cached {cy}; archives {split_archives(0)}", flush=True)
        if not wait_count(r"persisted slots", persisted):
            raise SystemExit("no idle persist")
        drops0 = count(r"archive: drop")
        for i in range(3):  # other conversations (another system prompt): they take the slots and the archive's room
            other = " ".join(f"Item {k}: colour {k * 31 % 97} weight {k * 17 % 89}." for k in range(400 + 40 * i))
            chat([{"role": "system", "content": other}, u(f"What is item {i}?")])
        drops = count(r"archive: drop") - drops0
        y2, n2, c2 = chat([s, u("What is rule 2?"), a_(y), u("And rule 3? Explain it in detail.")], 200)
        want = os.environ.get("QE_TEST_Y2_MD5", "")
        ok = drops > 0 and n2 - c2 < 64 and md5(y2) == want
        print(f"{'PASS' if ok else 'FAIL'} overflow: archive drops {drops}, Y2 prompt {n2} cached {c2}, md5 {md5(y2)} "
              f"(reference {want})", flush=True)
        return 0 if ok else 1
    finally:
        stop(p)
        shutil.rmtree(CACHE, ignore_errors=True)
        print("server log:", LOG, "pids:", PIDS, flush=True)


def main():
    if os.environ.get("QE_TEST_OVERFLOW") == "1":
        return overflow()
    u = lambda q: {"role": "user", "content": q}
    a_ = lambda q: {"role": "assistant", "content": q}
    s = {"role": "system", "content": SYS}
    qx, qy, qy2, qz, qw = "What is rule 1?", "What is rule 2?", "And rule 3? Explain it in detail.", "What is rule 4?", "What is rule 5?"
    res, ok = {}, True
    p = None
    try:
        # ---- run A
        p = start(LIB, True)
        persisted = count(r"persisted slots")
        _, (y, ny, cy) = xy(s, XLONG, qy)
        x = chat([s, u(qx)])[0]  # (sequential, after: the default path's request in a disk run)
        nx = cx = 0
        print(f"A: X prompt {nx} cached {cx}; Y prompt {ny} cached {cy} (attached: {count(r'attached')})", flush=True)
        if not wait_count(r"persisted slots", persisted):
            raise SystemExit("no idle persist in run A")
        time.sleep(1)
        fa = files()
        db = stats_disk_bytes()
        print(f"A: files {fa}; /api/stats disk_bytes {db}", flush=True)
        arch = split_archives(0)
        res["A"] = {"X": [nx, cx], "Y": [ny, cy], "files": fa, "disk_bytes": db, "ram_archives": arch,
                    "attached": len(arch) >= 2 and min(g for _, g in arch) < 0.7 * max(g for _, g in arch)}
        stop(p)
        p = None
        hint = cy  # Y attached at the system end: its cached tokens = the pinned prefix's length
        y_file = [f for f in fa if hint in f["subs"] and f["n"] > hint]
        acct = db is not None and abs(db - sum(f["payload"] for f in fa)) < 1
        res["A"]["checks"] = {"files_ge_2": len(fa) >= 2, "state_at_hint_in_every_file": all(hint in f["subs"] for f in fa),
                              "accounting_equals_files": acct}
        ok &= len(fa) >= 2 and acct and bool(y_file)
        # ---- run B
        at = len(log_lines())
        p = start(LIB, True, 300)
        y2d, ny2, cy2 = chat([s, u(qy), a_(y), u(qy2)], 200)
        z, nz, cz = chat([s, u(qz)])
        pins = count(r"pinned prefix", at)
        w, nw, cw = chat([s, u(qw)])
        print(f"B: Y2 prompt {ny2} cached {cy2}; Z prompt {nz} cached {cz}; pinned lines {pins}; W prompt {nw} cached {cw}", flush=True)
        stop(p)
        p = None
        res["B"] = {"Y2": [ny2, cy2, md5(y2d)], "Z": [nz, cz], "pins": pins, "W": [nw, cw]}
        # ---- run C: the same turns in RAM, no disk cache
        p = start(LIB, False)
        x_c, _, _ = chat([s, u(qx)])  # sequential: the default path's md5 (run M the same)
        chat([u("Clear the slot state.")], 1)
        _, (y_c, _, cyc) = xy(s, XLONG, qy, need_pin=False)
        y2r, ny2r, cy2r = chat([s, u(qy), a_(y_c), u(qy2)], 200)
        stop(p)
        p = None
        res["C"] = {"X": md5(x_c), "Y": md5(y_c), "Y2": [ny2r, cy2r, md5(y2r)]}
        # ---- run M: the reference lib, default path
        p = start(REF, False)
        x_m, _, _ = chat([s, u(qx)])
        stop(p)
        p = None
        res["M"] = {"X": md5(x_m)}
        checks = {
            "Y2 resumed from its own file (recomputed < 64)": ny2 - cy2 < 64,
            "Y2 from disk == Y2 in RAM (200 tokens)": y2d == y2r,
            "Y (run A) == Y (run C)": y == y_c,
            "Z resumed at the system end": cz == hint,
            "re-pinned in run B": pins >= 1,
            "W attached / cached at the system end": cw == hint,
            "default path X md5 == reference": x_c == x_m,
            "Y attached in run A (its RAM archive smaller than X's)": res["A"]["attached"],
        }
        res["checks"] = checks
        for k, v in checks.items():
            print(f"{'PASS' if v else 'FAIL'} {k}", flush=True)
        ok &= all(checks.values())
        print(json.dumps(res, indent=1), flush=True)
        print("PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 1
    finally:
        if p is not None and p.poll() is None:
            p.kill()
            p.wait()
        shutil.rmtree(CACHE, ignore_errors=True)
        print("server log:", LOG, "pids:", PIDS, flush=True)


if __name__ == "__main__":
    sys.exit(main())
