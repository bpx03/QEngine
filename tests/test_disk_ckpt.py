#!/usr/bin/env python3
"""Issue #14 (GPU, 27B server, HTTP): an archive written in the background must keep the checkpoints of the archive it
replaces, so the system-prompt state survives on disk.

  A cold -> restart -> B and A2 from the archive, short requests so that A2 goes through the background archiver
  -> restart -> D (a new conversation with the same system prompt) must resume from the end of the system prompt.

usage: flock .gpu.lock python3 tests/test_disk_ckpt.py <libqengine.so> [port=8181] [devices=0,1,2]
Exit 0: pass, 1: fail, 2: the background path was not reached. Takes the GPUs for four server starts, past the 10 s of the quick suites; the cache directory is removed at the end."""
import json, os, re, shutil, signal, struct, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(sys.argv[1])
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8181
DEVS = sys.argv[3] if len(sys.argv) > 3 else "0,1,2"
MODEL = os.environ.get("QE_MODEL", os.path.expanduser("~/GIT/QEngine/models/Qwen3.8-27B-INT4"))
TMP = tempfile.mkdtemp(prefix="qe_disk_ckpt_")
CACHE = os.path.join(TMP, "cache")
LOG = os.path.join(TMP, "server.log")
TURNS = 4
SYS = " ".join(f"Rule {i}: answer record {i * 7919 % 10007} with the field {i * 104729 % 1009} first." for i in range(330))


def start():
    # QE_COMM_CACHE=0: every start runs the decode calibration, whose resets must keep the disk cache
    env = dict(os.environ, QE_POOL_LOG="1", QE_WARMUP="0", QE_COMM_CACHE="0", CUDA_VISIBLE_DEVICES=DEVS)
    out = open(LOG, "ab")
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "..", "server", "app.py"), "--model", MODEL, "--engine-lib", LIB,
                          "--devices", ",".join(str(i) for i in range(len(DEVS.split(",")))), "--port", str(PORT),
                          "--metrics-port", "0", "--log-file", "", "--stats-file", os.path.join(TMP, "stats.json"),
                          "--disk-cache", CACHE, "--disk-cache-gb", "20", "--persist-idle", "3"],
                         env=env, stdout=out, stderr=subprocess.STDOUT)
    for _ in range(1200):
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
    p.wait(timeout=300)


def chat(msgs, n=60):
    body = {"model": "m", "messages": msgs, "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    rq = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(rq, timeout=600))
    u = r["usage"]
    return r["choices"][0]["message"]["content"], u["prompt_tokens"], u.get("prompt_tokens_details", {}).get("cached_tokens", 0)


def log_lines():
    return open(LOG, errors="replace").read().splitlines()


def wait_log(pat, since, tmo=120):
    for _ in range(int(tmo / 0.1)):
        ls = log_lines()[since:]
        if any(re.search(pat, x) for x in ls):
            return
        time.sleep(0.1)
    raise SystemExit(f"no log line /{pat}/: see {LOG}")


def files():
    res = []
    for nm in sorted(os.listdir(CACHE)):
        if nm.endswith(".qkv"):
            with open(os.path.join(CACHE, nm), "rb") as f:
                f.read(8)
                f.read(struct.unpack("<I", f.read(4))[0])
                n, ns = struct.unpack("<iI", f.read(8))
                res.append((n, sorted(struct.unpack(f"<{ns}i", f.read(4 * ns)), reverse=True)))
    return sorted(res)


def main():
    u = lambda q: {"role": "user", "content": q}
    s = {"role": "system", "content": SYS}
    q1, q2, q3, q4 = "What is rule 1?", "And rule 2?", "What is rule 3?", "What is rule 4?"
    p = start()
    try:
        a, n, c = chat([s, u(q1)])
        print(f"A   prompt {n} cached {c}", flush=True)
        wait_log(r"persisted slots", 0)
        print("after A        ", files(), flush=True)
        stop(p)

        at = len(log_lines())
        p = start()
        _, n, c = chat([s, u(q3)])
        print(f"B   prompt {n} cached {c}", flush=True)
        n_b = n
        if c == 0:
            print("FAIL: B found nothing in the disk cache after the restart", flush=True)
            return 1
        # A2 and TURNS more turns, on both save paths in turn. A reply of one token ends without a decode step, which re-arms
        # the background archiver; a normal reply is saved by the idle persist (archive_slot).
        msgs, last = [s, u(q1), {"role": "assistant", "content": a}], ""
        for t in range(TURNS + 1):
            bg = t % 2 == 0
            mark = "in the background" if bg else "persisted slots"
            seen = sum(mark in x for x in log_lines()[at:])
            msgs.append(u(q2 if t == 0 else f"And rule {t + 4}?"))
            last, n, c = chat(msgs, 1 if bg else 60)
            msgs.append({"role": "assistant", "content": last})
            for _ in range(200):
                if sum(mark in x for x in log_lines()[at:]) > seen:
                    break
                time.sleep(0.1)
            else:
                print(f"INVALID: {mark!r} did not appear after turn {t + 2}", flush=True)
                return 2
            # the old file stays until the new one is written (B's own file is the one with a state at B's prompt end)
            if not [f for f in files() if n_b not in f[1]]:
                print(f"FAIL: after turn {t + 2} the conversation has no file: {files()}", flush=True)
                return 1
            print(f"A{t + 2}  prompt {n} cached {c}, saved {'in the background' if bg else 'by the idle persist'}: {files()}", flush=True)
        # the last turn's file: its idle persist waits for the write
        seen = sum("persisted slots" in x for x in log_lines()[at:])
        for _ in range(300):
            if sum("persisted slots" in x for x in log_lines()[at:]) > seen:
                break
            time.sleep(0.1)
        last_files = files()
        print("after the turns", last_files, flush=True)
        stop(p)

        p = start()
        d, n, c = chat([s, u(q4)])
        print(f"D   prompt {n} cached {c}", flush=True)
        msgs.append(u("And the last rule?"))  # the conversation itself, from the end state of a background archive
        e, ne, ce = chat(msgs)
        print(f"E   prompt {ne} cached {ce}", flush=True)
        stop(p)
        shutil.rmtree(CACHE)
        p = start()  # the same requests cold: the restored states must give the same replies
        d0, _, c0 = chat([s, u(q4)])
        chat([u("Clear the slot state.")], 1)
        e0, _, _ = chat(msgs)
        stop(p)
        p = None
        # D recomputes only its question; every file kept the system-prompt state D resumed from
        all_kept = all(c in sub for _, sub in last_files)
        ok = n - c < 64 and c0 == 0 and d == d0 and ce >= c and e == e0 and all_kept
        print(f"{'PASS' if ok else 'FAIL'}: D recomputed {n - c} of {n}, reply {'=' if d == d0 else '!='} cold; "
              f"E recomputed {ne - ce} of {ne}, reply {'=' if e == e0 else '!='} cold; "
              f"state {c} in every file: {all_kept}", flush=True)
        return 0 if ok else 1
    finally:
        if p is not None and p.poll() is None:
            p.kill()
            p.wait()
        shutil.rmtree(CACHE, ignore_errors=True)
        print("server log:", LOG, flush=True)


if __name__ == "__main__":
    sys.exit(main())
