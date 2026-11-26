"""[rel160] CPU test: on Inkling (as on gpt-oss) an engine "KV pool full" slot fault ends the reply with finish_reason
length instead of an HTTP 500 (1.6.0 stress gate: busy replies next to a 557K-token prompt got 500s); another fault
still fails the request. Generator.faults on fake jobs and a fake engine; no GPU; well under a second.

  python3 tests/test_inkling_pool_full.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
sys.argv = ["app"]
import app  # noqa: E402

fails = 0


def check(name, ok, info=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name + (f": {info}" if info and not ok else ""))
    fails += 0 if ok else 1


class Job:
    def __init__(self, slot):
        self.slot, self.ids, self.n, self.finished, self.finish, self.max_new, self.err = slot, [1, 2, 3], 5, False, None, 99, None

    def feed(self, toks):
        pass

    def fail(self, e):
        self.err = e
        self.finished = True


for fam in ("inkling", "gptoss", "qwen38-27b"):
    msgs = {0: "KV pool full: no room for this sequence to grow next to the running ones", 1: "non-finite values in the step"}
    eng = types.SimpleNamespace(fault_count=lambda: 2, slot_fault=lambda s: msgs.get(s))
    g = types.SimpleNamespace(engine=eng, n_faults=0,
                              profile=types.SimpleNamespace(is_gptoss=fam == "gptoss", is_inkling=fam == "inkling"))
    a, b = Job(0), Job(1)
    hit = app.Generator.faults(g, [a, b])
    if fam in ("inkling", "gptoss"):
        check(f"{fam}: pool full -> finish length, no error", a.finish == "length" and a.err is None and a.max_new == 5, vars(a))
    else:
        check(f"{fam}: pool full stays an error (unchanged)", a.err is not None, vars(a))
    check(f"{fam}: another fault still fails the request", b.err is not None and hit, vars(b))
print(f"{fails} failed")
sys.exit(1 if fails else 0)
