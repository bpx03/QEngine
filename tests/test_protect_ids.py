"""[ink-archprot] archprot_review2 L3 / L1: Engine.protect_ids (server/engine.py) on a fake library: no call without
qe_protect_ids, one ctypes array per waiting prompt (reused across blocked admission passes), a new one for another
prompt. No GPU; well under a second.

  python3 tests/test_protect_ids.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
sys.modules.setdefault("pynvml", types.ModuleType("pynvml"))
import engine  # noqa: E402

FAILS = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + str(info) if info else ''}", flush=True)
    if not ok:
        FAILS.append(name)


class Lib:
    def __init__(self):
        self.calls = []

    def qe_protect_ids(self, h, arr, n):
        self.calls.append((arr, n, list(arr[:n]) if arr is not None else []))


def main():
    e = object.__new__(engine.Engine)
    e.lib, e.h = Lib(), None
    e.fns = set()
    e.protect_ids([1, 2, 3])
    check("no qe_protect_ids: nothing called", e.lib.calls == [])
    e.fns = {"qe_protect_ids"}
    ja, jb = types.SimpleNamespace(ids=list(range(100000))), types.SimpleNamespace(ids=[7, 8, 9])
    for _ in range(3):   # two waiting requests alternate (review3 L2: the cache lives on each request)
        e.protect_ids(ja.ids, ja)
        e.protect_ids(jb.ids, jb)
    c = e.lib.calls
    check("two prompts alternating: one array each", len(c) == 6 and c[0][0] is c[2][0] is c[4][0] and
          c[1][0] is c[3][0] is c[5][0] and c[0][1] == 100000 and c[0][2][:3] == [0, 1, 2] and c[1][2] == [7, 8, 9])
    e.protect_ids([7, 8, 9])
    check("no owner: a fresh array, the ids passed", e.lib.calls[-1][0] is not c[1][0] and e.lib.calls[-1][2] == [7, 8, 9])
    e.protect_ids([])
    check("empty prompt: no call", len(e.lib.calls) == 7)
    e.protect_clear()
    check("protect_clear: one call with no ids (a pending prompt dropped)", e.lib.calls[-1][1] == 0 and len(e.lib.calls) == 8)
    print(f"{len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
