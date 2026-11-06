#!/usr/bin/env python3
"""27B archive eviction order (GPU, server, HTTP; fix-27b-evict). A small archive, 1 slot (every new conversation sends the last one to the archive), a main thread forced with
--main-agent-re, several other conversations:
  A  an active main stays while the others' old writes go; its next turn resumes from the archive without a re-prefill;
  B  once it is stale (FLASH_MAIN_STALE_S) it is an ordinary old conversation: it goes first, without the main's log line,
     and the new write is archived (before the fix it refused all other writes for good);
  C  a large active main that leaves no room: the others go first, then it goes too, with a log line of its tokens and
     bytes; the writer is archived; no crash.
usage: flock .gpu.lock python3 tests/test_archive_evict.py <libqengine.so> [port=8181] [devices=0,1,2]
  env: AE_ARCHIVE_GB (default 2.2), AE_STALE_S (default 40). Exit 0: pass. Past the 10 s of the quick suites."""
import json, os, random, re, signal, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(sys.argv[1])
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8181
DEVS = sys.argv[3] if len(sys.argv) > 3 else "0,1,2"
MODEL = os.environ.get("QE_MODEL", os.path.expanduser("~/GIT/QEngine/models/Qwen3.8-27B-INT4"))
ARCHIVE_GB = os.environ.get("AE_ARCHIVE_GB", "2.2")
STALE_S = float(os.environ.get("AE_STALE_S", "40"))
TMP = tempfile.mkdtemp(prefix="qe_arch_evict_")
LOG = os.path.join(TMP, "server.log")
W = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo".split()
fails = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)


def text(seed, words):
    r = random.Random(seed)
    return " ".join(r.choice(W) for _ in range(words))


def chat(msgs, n=8):
    body = {"model": "m", "messages": msgs, "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    u = r["usage"]
    return r["choices"][0]["message"]["content"], u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def conv(name, words, main=False):
    tag = "MAINTHREADMARK " if main else ""
    return [{"role": "system", "content": f"{tag}Conversation {name}. Notes: {text(name, 300)}"},
            {"role": "user", "content": f"{name}: {text(name + 'u', words)}. Say ok."}]


def log():
    return open(LOG, errors="replace").read()


env = dict(os.environ, QE_POOL_LOG="1", QE_WARMUP="0", QE_MAX_SEQS="1", QE_KV_ARCHIVE_GB=ARCHIVE_GB, FLASH_MAIN_STALE_S=str(STALE_S),
           CUDA_VISIBLE_DEVICES=DEVS)
out = open(LOG, "ab")
p = subprocess.Popen([sys.executable, os.path.join(HERE, "..", "server", "app.py"), "--model", MODEL, "--engine-lib", LIB,
                      "--devices", ",".join(str(i) for i in range(len(DEVS.split(",")))), "--port", str(PORT), "--metrics-port", "0",
                      "--log-file", "", "--stats-file", os.path.join(TMP, "stats.json"), "--main-agent-re", "MAINTHREADMARK"],
                     env=env, stdout=out, stderr=subprocess.STDOUT)
print(f"server pid {p.pid}, log {LOG}", flush=True)
try:
    for _ in range(1200):
        if p.poll() is not None:
            raise SystemExit(f"server exited with {p.returncode}: see {LOG}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
            break
        except OSError:
            time.sleep(0.25)
    else:
        raise SystemExit("server did not come up")
    time.sleep(1)
    SMALL, BIG = 1500, 20000

    # A: an active main keeps its place while the others' old writes go
    m = conv("main", SMALL, True)
    reply, n0, _ = chat(m)
    m += [{"role": "assistant", "content": reply}, {"role": "user", "content": "One more short answer, please."}]
    for i in range(6):
        chat(conv(f"sub{i}", SMALL))
    l = log()
    print(*[x for x in l.splitlines() if x.startswith("archive:")][-14:], sep="\n", flush=True)
    check(re.search(r"archive: drop \d+ tokens prio [01]\b", l) is not None, "A: other conversations' old writes evicted")
    check(re.search(r"archive: drop \d+ tokens prio 2", l) is None and "active main evicted" not in l, "A: the active main not evicted")
    reply, n1, c1 = chat(m)
    check(c1 >= n0 * 0.9, f"A: active main restored without a re-prefill: {c1} of {n1} tokens cached, first turn {n0}")
    m += [{"role": "assistant", "content": reply}, {"role": "user", "content": "And one more, please."}]

    # B: stale: the same main is an ordinary old conversation
    for i in range(2):
        chat(conv(f"bsub{i}", SMALL))  # it leaves its slot, into the archive with its last use
    t0 = len(log())
    time.sleep(STALE_S + 2)
    for i in range(4):
        chat(conv(f"csub{i}", SMALL))
    l = log()[t0:]
    print(*[x for x in l.splitlines() if x.startswith("archive:")], sep="\n", flush=True)
    d = re.findall(r"archive: drop (\d+) tokens prio (\d)", l)
    # (what is older than its last turn goes before it, then the main, before the conversations used after it)
    check([x[1] for x in d[:2]].count("2") == 1, f"B: the stale main goes in LRU order, without protection: {d}")
    check("active main evicted" not in l, "B: no active-main log line for a stale one")
    check(re.search(r"archive: slot \d+ \d+ tokens", l) is not None and "archive: no room" not in l, "B: new writes archived, none refused")

    # C: a large active main: the others first, then it, with the log line
    t0 = len(log())
    big = conv("bigmain", BIG, True)
    _, nb, _ = chat(big)
    chat(conv("dsub0", SMALL))
    chat(conv("dsub1", SMALL))  # the main leaves its slot: required write, the older ones go
    l = log()[t0:]
    check(any(int(t) >= nb for t in re.findall(r"archive: slot \d+ (\d+) tokens prio 2", l)), f"C: the main ({nb} tokens) archived")
    chat(conv("dsub2", SMALL))
    chat(conv("dsub3", SMALL))
    l = log()[t0:]
    print(*[x for x in l.splitlines() if x.startswith("archive:")], sep="\n", flush=True)
    mm = re.search(r"archive: active main evicted, nothing else left: (\d+) tokens, ([\d.]+) GB", l)
    check(mm is not None and int(mm.group(1)) >= nb, f"C: log line names the main's tokens and bytes: {mm.group(0) if mm else None}")
    if mm:
        after = l[mm.end():]
        check(re.search(r"archive: slot \d+ \d+ tokens prio [01]", after) is not None, "C: the writer archived after it")
        before = l[:mm.start()]
        check(before.count("archive: drop") >= 1, "C: others went before it")
    check(p.poll() is None, "server alive")
    reply, n2, c2 = chat(big + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "Short answer."}])
    check(n2 > nb and c2 < nb * 0.5, f"C: the evicted main still answers, from a re-prefill: {c2} of {n2} cached")
finally:
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            rc = p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
    else:
        rc = p.returncode
    print("server exit", rc, flush=True)
print(f"log kept: {LOG}")
if rc != 0:
    fails.append(f"exit {rc}")
print("FAILED: " + "; ".join(fails) if fails else "pass")
sys.exit(1 if fails else 0)
