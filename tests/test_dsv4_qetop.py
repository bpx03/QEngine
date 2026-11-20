"""[dsv4-qetop] CPU test of the DSV4 server's /api/stats for qetop, with a fake engine (no GPU, no model files).

  python3 tests/test_dsv4_qetop.py

Checks: the server answers chat requests through the real worker / time-sliced scheduler with a fake engine; while
requests are in service /api/stats has every key qetop reads and qetop.view renders it (queued, prefill and decode
rows, the DSV4 engine panel); POST /api/reset; a taken metrics port leaves monitoring off and the API serving;
--metrics-port 0 = no monitoring; a failing stats hook never reaches the request path; the replies are the same with
monitoring on and off.
"""
import asyncio
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import types
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

import dsv4_app  # noqa: E402
import dsv4_stats  # noqa: E402
import qetop  # noqa: E402

EOS_ID = 1


class FakeTok:
    """byte-level tokenizer: id = byte + 10; EOS = 1"""

    def token_to_id(self, t):
        return EOS_ID

    def encode(self, text, add_special_tokens=False):
        return types.SimpleNamespace(ids=[b + 10 for b in text.encode()])

    def decode(self, ids, skip_special_tokens=False):
        return bytes(i - 10 for i in ids if i >= 10).decode(errors="replace")


class FakeL:
    def __init__(self, e):
        self.e = e

    def dsv4_pos(self, h):
        return len(self.e.toks)

    def dsv4_set_nocache(self, h):
        pass


class FakeEngine:
    """the Engine interface of dsv4_app with a deterministic token stream: "a".."z" repeated, 2 tokens per pass,
    STEP_S seconds per pass (released GIL like a ctypes call), PF_S per prompt call"""
    STEP_S, PF_S = 0.01, 0.15

    def __init__(self, lib, args):
        self.max_ctx = 4096
        self.has_sched, self.has_pfx, self.has_kvhost = True, True, True
        self.h = 1
        self.L = FakeL(self)
        self.toks = []
        self.thread = None

    def _check(self):   # every engine call on one thread (the worker), as the real library needs
        t = threading.current_thread()
        if self.thread is None and t.name != "MainThread":
            self.thread = t
        assert t is self.thread or t.name == "MainThread", f"engine called from {t.name}"

    def _next(self):
        return 10 + ord("a") + len(self.toks) % 26

    def set_sampling(self, *a):
        self._check()

    def start(self, ids, cache=True):
        self._check()
        time.sleep(self.PF_S)
        reuse = 0
        while reuse < min(len(ids), len(self.toks)) and ids[reuse] == self.toks[reuse]:
            reuse += 1
        self.toks = list(ids)
        t = self._next()
        self.toks.append(t)
        return t, reuse

    def prefill_part(self, ids):
        self._check()
        time.sleep(self.PF_S / 3)
        self.toks = list(ids)
        return 0

    def resume(self, ids):
        self._check()
        self.toks = list(ids)
        return True

    def reusable(self, ids):
        self._check()
        return 0

    def step(self, cap=32):
        self._check()
        time.sleep(self.STEP_S)
        out = []
        for _ in range(min(2, cap)):
            t = self._next()
            self.toks.append(t)
            out.append(t)
        return out

    def stats(self):
        self._check()
        return None

    def prefix_hint(self, n):
        self._check()

    def prefix_info(self):
        self._check()
        return 2, 5

    def set_live(self, jid):
        self._check()

    def end_live(self, jid):
        self._check()

    def archive_info(self):
        self._check()
        return 3, 1.5e9, "archive", [4, 2, 1, 0]

    def kvpool_info(self):
        self._check()
        return 1, 120.0, 30.0, [5, 1, 2, 0, 0]

    def kvhost_info(self):
        self._check()
        return 64.0, 32.0, [3, 1, 0]

    def archive_pressure(self):
        self._check()

    def end_request(self):
        self._check()
        return 0.0

    def trim(self, keep):
        self._check()
        return True


def free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def make_server(argv):
    """dsv4_app.main up to uvicorn.run (captured): the Server with the fake engine"""
    dsv4_app.G.load_tokenizer = lambda model, cache_dir: (FakeTok(), None, {})
    dsv4_app.Engine = FakeEngine
    got = {}
    fake_uv = types.ModuleType("uvicorn")
    fake_uv.run = lambda app, **k: got.update(app=app)
    sys.modules["uvicorn"] = fake_uv
    orig = dsv4_app.Server

    class Cap(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            got["server"] = self
    dsv4_app.Server = Cap
    try:
        dsv4_app.main(argv, lambda lib: ("9.9.9", "abc1234"))
    finally:
        dsv4_app.Server = orig
    return got["server"], got["app"]


def get(url):
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.load(r)


async def run_requests(s, prompts, max_tokens):
    """submit the prompts at once through Server.submit (the path /v1/chat/completions takes); -> replies"""
    async def one(p, delay):
        await asyncio.sleep(delay)
        aq, cancel = s.submit(s.tok.encode(p).ids, dict(temperature=0, top_p=1, top_k=0, min_p=0, seed=1), max_tokens)
        out, info = [], None
        while True:
            ev, val = await aq.get()
            if ev == "tokens":
                out += val
            elif ev == "done":
                info = val
                break
            else:
                raise RuntimeError(val)
        return s.tok.decode(out), info
    return await asyncio.gather(*(one(p, 0.02 * i) for i, p in enumerate(prompts)))


def render(snap, lang="en"):
    from rich.console import Console
    qetop.LANG = lang
    c = Console(file=io.StringIO(), width=160, height=50, color_system=None, force_terminal=False)
    c.print(qetop.view(snap, 160))
    return c.file.getvalue()


def test_live(tmp):
    port = free_port()
    s, app = make_server(["--model", "/nonexistent", "--metrics-port", str(port), "--max-active", "2",
                          "--slice-passes", "2", "--prefill-chunk", "64",
                          "--stats-file", os.path.join(tmp, "stats.json")])
    assert isinstance(s.st, dsv4_stats.Stats)
    url = f"http://127.0.0.1:{port}"
    snaps, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            try:
                snaps.append(get(url + "/api/stats"))
            except Exception as ex:
                snaps.append({"error": repr(ex)})
            time.sleep(0.02)
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    prompts = ["x" * 200, "hello", "short one", "y" * 30]
    res = asyncio.run(run_requests(s, prompts, 60))
    time.sleep(0.1)
    stop.set()
    th.join()
    assert not [x for x in snaps if "error" in x], [x for x in snaps if "error" in x][:2]
    states = {j["state"] for x in snaps for j in x["jobs"]}
    assert states == {"queued", "prefill", "decode"}, states
    assert any(len([j for j in x["jobs"] if j["state"] != "queued"]) == 2 for x in snaps), "2 in service at once"
    assert all(len([j for j in x["jobs"] if j["slot"] == 0]) <= 1 for x in snaps)
    for x in snaps:   # every snapshot renders, in both languages
        render(x, "en")
    mid = max(snaps, key=lambda x: len(x["jobs"]))
    txt = render(mid, "pl")
    assert "czeka" in txt or "prefill" in txt or "dekoduje" in txt, txt
    fin = get(url + "/api/stats")
    for _, info in res:
        assert info["completion"] == 60 and info["finish"] == "length", info
    se = fin["session"]
    assert se["requests"] == 4 and se["gen_tokens"] == 240, se
    assert se["prompt_tokens"] == sum(len(p) for p in prompts), se
    assert se["ttft_n"] == 4 and fin["engine"]["dsv4"]["avg_ttft_s"] > 0
    assert 1.9 < fin["rate"]["tokens_per_step"] <= 2.0, fin["rate"]
    assert fin["version"] == "9.9.9" and fin["build"] == "abc1234" and fin["max_ctx"] == 4096
    d = fin["engine"]["dsv4"]
    assert d["archive"]["entries"] == 3 and d["kvpool"]["parked_vram"] == 1 and d["prefix"] == {"made": 2, "used": 5}
    assert len(fin["recent"]) == 4 and all(r["ttft"] is not None for r in fin["recent"])
    assert len(fin["ended"]) == 4 and not fin["jobs"]
    txt = render(fin, "en")
    for w in ("archive", "shared prefixes", "TTFT", "ctx 4 096", "9.9.9", "finished"):
        assert w in txt, (w, txt)
    print(txt[:400])
    # reset
    req = urllib.request.Request(url + "/api/reset", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=3) as r:
        assert json.load(r) == {"ok": True}
    after = get(url + "/api/stats")
    assert after["session"]["gen_tokens"] == 0 and after["global"]["gen_tokens"] == 240, after["global"]
    assert qetop.reset_stats(url)[1] == "bold green"
    s.st.save()
    saved = json.load(open(os.path.join(tmp, "stats.json")))
    assert saved["gen_tokens"] == 240 and saved["runs"] == 1
    # unknown paths: 404, not a crash
    try:
        get(url + "/nope")
        raise AssertionError("404 expected")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    return [r[0] for r in res]


def test_port_taken(tmp, ref):
    so = socket.socket()
    so.bind(("127.0.0.1", 0))
    so.listen(1)
    port = so.getsockname()[1]
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        s, app = make_server(["--model", "/nonexistent", "--metrics-port", str(port), "--max-active", "2",
                              "--slice-passes", "2", "--prefill-chunk", "64",
                              "--stats-file", os.path.join(tmp, "stats2.json")])
    finally:
        sys.stdout = old
    so.close()
    assert "unavailable" in buf.getvalue() and "monitoring off" in buf.getvalue(), buf.getvalue()
    res = asyncio.run(run_requests(s, ["x" * 200, "hello", "short one", "y" * 30], 60))
    assert [r[0] for r in res] == ref   # the API serves on


def test_off(tmp, ref):
    s, app = make_server(["--model", "/nonexistent", "--metrics-port", "0", "--max-active", "2",
                          "--slice-passes", "2", "--prefill-chunk", "64"])
    assert isinstance(s.st, dsv4_stats.Off)
    res = asyncio.run(run_requests(s, ["x" * 200, "hello", "short one", "y" * 30], 60))
    assert [r[0] for r in res] == ref, "replies differ with monitoring off"


def test_bad_port(tmp):
    """review L8: an out-of-range port turns monitoring off, the start goes on"""
    s, app = make_server(["--model", "/nonexistent", "--metrics-port", "70000"])
    assert isinstance(s.st, dsv4_stats.Off)


def test_hook_failure(tmp, ref):
    s, app = make_server(["--model", "/nonexistent", "--metrics-port", str(free_port()), "--max-active", "2",
                          "--slice-passes", "2", "--prefill-chunk", "64", "--stats-file", os.path.join(tmp, "s3.json")])
    s.st.jobs = None   # every hook that touches the jobs now raises inside
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        res = asyncio.run(run_requests(s, ["x" * 200, "hello", "short one", "y" * 30], 60))
    finally:
        sys.stdout = old
    assert [r[0] for r in res] == ref
    assert "stats: queued failed" in buf.getvalue(), buf.getvalue()


def test_missing_fields():
    """qetop on a minimal DSV4 snapshot (no GPUs, nothing in service, no engine counters yet)"""
    st = dsv4_stats.Stats(os.path.join(tempfile.mkdtemp(), "s.json"), "qe-dsv4", 262144, nvml=False)
    snap = json.loads(json.dumps(st.snapshot()))
    render(snap, "pl")
    render(snap, "en")
    snap["engine"] = {}
    render(snap, "en")


if __name__ == "__main__":
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        ref = test_live(tmp)
        test_port_taken(tmp, ref)
        test_off(tmp, ref)
        test_bad_port(tmp)
        test_hook_failure(tmp, ref)
        test_missing_fields()
    print(f"test_dsv4_qetop: all passed ({time.time() - t0:.1f} s)")
