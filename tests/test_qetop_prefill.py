"""[qetop-pf] CPU test of the live prefill progress in /api/stats and qetop (no GPU, no model files).

  python3 tests/test_qetop_prefill.py

Checks: Generator.pf_progress writes rows done / to compute / live rate into the prefill job's metrics entry after each
chunk (cached rows at the start excluded), never touches a decoding job, never raises (engine error, job gone);
prefill_done overwrites it with the final numbers; the metrics snapshot answers while a fake engine call is in flight
on another thread; qetop.view shows the running prefill in the top bar and its live rate in the headline (and the
last finished rate when none runs).
"""
import io
import os
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
_nv = types.ModuleType("pynvml")  # no NVML in this test


def _no_nvml():
    raise RuntimeError("NVML disabled")


_nv.nvmlInit = _no_nvml
sys.modules["pynvml"] = _nv

import app  # noqa: E402
import qetop  # noqa: E402
from metrics import Metrics  # noqa: E402
from rich.console import Console  # noqa: E402

FAILS = []


def check(what, ok, got=""):
    print(f"  {'ok ' if ok else 'FAIL'} {what}" + (f": {got}" if got != "" else ""))
    if not ok:
        FAILS.append(what)


class FakeEngine:
    def __init__(self):
        self.pos = {}
        self.boom = False

    def prefill_pos(self, slot):
        if self.boom:
            raise RuntimeError("engine gone")
        return self.pos.get(slot, -1)


def render(snap, width=160):
    c = Console(file=io.StringIO(), width=width, height=60, record=True)
    c.print(qetop.view(snap, width))
    return c.export_text()


def main():
    m = Metrics(os.path.join(tempfile.mkdtemp(prefix="qetop_pf_"), "stats.json"), "test")
    e = FakeEngine()
    g = types.SimpleNamespace(metrics=m, engine=e)
    prog = lambda jobs, others=(): app.Generator.pf_progress(g, jobs, others)
    long_ = types.SimpleNamespace(ids=list(range(60000)), slot=1, pf0=10000, reused=10000, src="")
    dec = types.SimpleNamespace(ids=list(range(500)), slot=2, pf0=0, reused=0, src="")
    m.queued(long_, "long agent")
    m.queued(dec, "decoder")
    m.prefill_start(dec)
    m.prefill_done(dec)
    m.prefill_start(long_)
    m.jobs[id(long_)]["t_p"] = time.time() - 2.0  # began 2 s ago

    prog([long_])  # nothing in flight yet (-1): no fields
    x = m.snapshot()["jobs"][0]
    check("no progress before the first chunk", "prefill_tok" not in x, x.get("prefill_tok"))
    e.pos[1] = 30000
    prog([long_, dec])
    jobs = {j["label"]: j for j in m.snapshot()["jobs"]}
    lj = jobs["long agent"]
    check("rows done exclude the cached start", lj.get("prefill_tok") == 20000, lj.get("prefill_tok"))
    check("rows to compute", lj.get("prefill_total") == 50000, lj.get("prefill_total"))
    check("live rate ~10K tok/s", 9000 < (lj.get("prefill_tps") or 0) < 10100, lj.get("prefill_tps"))
    check("decoding job untouched", "prefill_total" not in jobs["decoder"])
    e.boom = True
    try:
        prog([long_])
        check("engine error swallowed", True)
    except Exception as ex:
        check("engine error swallowed", False, ex)
    e.boom = False
    prog([types.SimpleNamespace(ids=[1], slot=1, pf0=0)])  # a job metrics does not know
    check("unknown job ignored", True)

    # a waiting prompt's rate decays (others, no engine read); a prompt whose rows do not move yet writes nothing
    short = types.SimpleNamespace(ids=list(range(3000)), slot=3, pf0=0, reused=0, src="")
    m.queued(short, "cut-in")
    m.prefill_start(short)
    e.pos[3] = 0  # (Flash-Next mode B: rows move only at the finish)
    t_before = m.jobs[id(long_)]["prefill_tps"]
    m.jobs[id(long_)]["t_p"] -= 2.0  # 2 s more without a chunk of its own
    calls = []
    real = e.prefill_pos
    e.prefill_pos = lambda s: (calls.append(s), real(s))[1]
    prog([short], [long_, short])
    e.prefill_pos = real
    jobs = {j["label"]: j for j in m.snapshot()["jobs"]}
    check("waiting prompt's rate decays", abs(jobs["long agent"]["prefill_tps"] - t_before / 2) < 100,
          jobs["long agent"]["prefill_tps"])
    check("no engine read for the waiting prompt", calls == [3], calls)
    check("no rows yet: no progress fields", "prefill_total" not in jobs["cut-in"])
    check("rows done unchanged while waiting", jobs["long agent"]["prefill_tok"] == 20000)
    narrow = render(m.snapshot(), 80)
    check("KPI row kept at width 80 with two prefill lines", "aktywnych" in narrow and "W na GPU" in narrow and narrow.count("▶ prefill") == 2,
          narrow.count("▶ prefill"))
    m.finished(short)
    m.jobs[id(long_)]["t_p"] += 2.0
    prog([long_])

    # qetop: top bar + headline from the live numbers
    out = render(m.snapshot())
    check("top bar shows the running prefill", "▶ prefill long agent: 20.0 k / 50.0 k (40 %)" in out,
          [ln for ln in out.splitlines() if "▶" in ln][:2])
    check("headline marks a live rate", "prefill tok/s ▶" in out)
    check("headline is the live rate, not the decoder's", "10 000" in out or "9 9" in out)
    qetop.LANG = "en"
    check("renders in EN", "▶ prefill long agent" in render(m.snapshot()))
    qetop.LANG = "pl"

    # the snapshot answers while an engine call runs on the scheduler thread (ctypes releases the GIL; time.sleep too)
    busy = threading.Event()

    def engine_call():
        busy.set()
        time.sleep(0.5)

    th = threading.Thread(target=engine_call)
    th.start()
    busy.wait()
    t0 = time.time()
    m.snapshot()
    check("snapshot during an engine call < 50 ms", time.time() - t0 < 0.05, f"{(time.time() - t0) * 1e3:.1f} ms")
    th.join()

    # the end: prefill_done's final numbers replace the live ones; no live mark any more
    m.prefill_done(long_)
    lj = m.snapshot()["jobs"]
    lj = next(j for j in lj if j["label"] == "long agent")
    check("final rows after prefill_done", lj["state"] == "decode" and lj["prefill_tok"] == 50000, lj.get("prefill_tok"))
    out = render(m.snapshot())
    check("no top-bar line once decoding", "▶ prefill" not in out and "prefill tok/s ▶" not in out)
    print("FAILED: " + ", ".join(FAILS) if FAILS else "all ok")
    return 1 if FAILS else 0


def test_qetop_prefill():  # pytest
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
