"""CPU test of the bounded wait for a queued request that fits nowhere while nothing runs (review M4): on the mock engine
with can_admit false (a KV pool shrunk under VRAM pressure, the memory not coming back) the request ends with a clear error
after QE_NO_ROOM_S, naming another process only when the pool did shrink; if the room returns in time it is served; the
server keeps serving afterwards. A request waiting behind a running sequence is not under that deadline (QE_START_RETRY_S
for a refused start stays 60 s). No GPU.

  python tests/test_admit_deadline.py
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
_nv = types.ModuleType("pynvml")


def _no_nvml():
    raise RuntimeError("NVML disabled in the mock test")


_nv.nvmlInit = _no_nvml
sys.modules["pynvml"] = _nv
MODEL = os.environ.get("QE_MOCK_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
TMP = tempfile.mkdtemp(prefix="qe_admit_")
os.environ.update(QE_DIAG_LOG=os.path.join(TMP, "diag.log"), QE_WARMUP="0")
for v in ("QE_NO_ROOM_S", "QE_START_RETRY_S"):
    os.environ.pop(v, None)
import app  # noqa: E402
import engine  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

assert app.NO_ROOM_S == 10 and app.START_RETRY_S == 60, (app.NO_ROOM_S, app.START_RETRY_S)  # the defaults
app.NO_ROOM_S = 1.0  # (read at run time; the test stays short)

so = os.path.join(TMP, "mock.so")
subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "mock_engine.c")])
script = os.path.join(TMP, "script.txt")
args = app.parse_args(["--model", MODEL, "--engine-lib", so, "--metrics-port", "0", "--stats-file", os.path.join(TMP, "stats.json")])
tok, _, _ = args.profile.tokenizer()
with open(script, "w") as f:
    f.write(",".join(map(str, tok.encode("Fine.<|im_end|>", add_special_tokens=False).ids)) + "\n")
os.environ.update(QE_MOCK_SCRIPT=script, QE_MOCK_LOG=os.path.join(TMP, "log.txt"))
gen = app.Generator(args)
c = TestClient(app.build_app(gen, args))
name = c.get("/v1/models").json()["data"][0]["id"]
body = {"model": name, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}

# the room does not come back, the pool as at start: a clear error about at the deadline, no claim about another process
os.environ["QE_MOCK_NO_ADMIT"] = "1"
t0 = time.time()
r = c.post("/v1/chat/completions", json=body)
dt = time.time() - t0
assert r.status_code == 500 and "no room in the KV pool: this request doesn't fit" in r.text, (r.status_code, r.text[:300])
assert "another process took" not in r.text and "requested " in r.text, r.text[:300]
assert 0.8 < dt < 4, dt
assert not gen.queue

# the pool is smaller than at start: it says so, with the sizes and the free VRAM per card when known
pool0 = gen.engine.pool_tokens
os.environ["QE_MOCK_POOL_NOW"] = "1000"
fake = [{"i": 0, "mem": 23 << 30, "mem_total": 24 << 30}, {"i": 2, "mem": 22 << 30, "mem_total": 24 << 30}]
type(gen.metrics).gpus = property(lambda self: fake, lambda self, v: None)  # (the sampler without NVML writes an empty list)
t0 = time.time()
r = c.post("/v1/chat/completions", json=body)
dt = time.time() - t0
assert r.status_code == 500 and "another process took the GPU memory and this request can't be finished" in r.text, r.text[:300]
assert f"KV pool now 1000 tokens (at start {pool0})" in r.text and "GPU0 1.0 GB, GPU2 2.0 GB" in r.text, r.text[:400]
assert 0.8 < dt < 4, dt
# ... but not for a request that would not have fit at the start size either
assert "this request doesn't fit" in gen.no_room_msg(pool0 + 1, 3, 1.0)
del type(gen.metrics).gpus
del os.environ["QE_MOCK_POOL_NOW"]

# the room comes back within the deadline: served
t = threading.Timer(0.5, lambda: os.environ.pop("QE_MOCK_NO_ADMIT"))
os.environ["QE_MOCK_NO_ADMIT"] = "1"
t.start()
r = c.post("/v1/chat/completions", json=body)
t.join()
assert r.status_code == 200 and "Fine" in r.text, (r.status_code, r.text[:300])

# and the server goes on
r = c.post("/v1/chat/completions", json=body)
assert r.status_code == 200, r.text[:300]


# a long reply runs: requests behind it are not under the no-room deadline
def post_async(b):
    out = {}

    def run():
        t0 = time.time()
        out["r"] = c.post("/v1/chat/completions", json=b)
        out["dt"] = time.time() - t0

    th = threading.Thread(target=run)
    th.start()
    return th, out


long_body = dict(body, max_tokens=240, ignore_eos=True)  # 3 tokens per step, 20 ms a step: about 1.6 s
os.environ["QE_MOCK_STEP_MS"] = "20"
ta, a = post_async(long_body)
for _ in range(100):
    if gen.jobs_ref and gen.jobs_ref[0]:
        break
    time.sleep(0.02)
assert gen.jobs_ref[0], "the long reply did not start"

# can_admit false behind it, for longer than the deadline: still queued; the room is back when the reply ends: served
os.environ["QE_MOCK_NO_ADMIT"] = "1"
tb, b = post_async(body)
ta.join(30)
assert a["r"].status_code == 200 and a["dt"] > 1.0 + 0.3, (a["r"].status_code, a["dt"])
assert "r" not in b or b["dt"] > a["dt"] - 0.5, "failed behind the running reply"
del os.environ["QE_MOCK_NO_ADMIT"]
tb.join(30)
assert b["r"].status_code == 200, b["r"].text[:300]

# a start refused for lack of room while the long reply runs: requeued past the no-room deadline (QE_START_RETRY_S, 60 s),
# not failed at it; when nothing runs any more the next refusal is final, with the engine's message
ta, a = post_async(long_body)
for _ in range(100):
    if gen.jobs_ref and gen.jobs_ref[0]:
        break
    time.sleep(0.02)
assert gen.jobs_ref[0], "the long reply did not start"
engine._TEST_REFUSE[:] = [engine.StartRefused.NO_ROOM] * 1000
tb, b = post_async(body)
ta.join(30)
tb.join(30)
del engine._TEST_REFUSE[:]
assert a["r"].status_code == 200, a["r"].text[:300]
assert a["dt"] > 1.0 + 0.3 and b["dt"] > 1.0 + 0.1, (a["dt"], b["dt"])
assert b["r"].status_code == 500 and "KV pool full" in b["r"].text, b["r"].text[:300]
os.environ.pop("QE_MOCK_STEP_MS")

# and the next request is served
r = c.post("/v1/chat/completions", json=body)
assert r.status_code == 200 and "Fine" in r.text, r.text[:300]
print("test_admit_deadline: ok", flush=True)
os._exit(0)
