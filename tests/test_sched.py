"""Scheduler admission (server/app.py Admission) with a fake engine, no GPU: main-thread precedence with aging,
backfill past a request that does not fit + its starvation guard, pausing lower-prio sequences for the main
thread (never another main thread), main-thread cut-in on a long prefill. Run: python tests/test_sched.py"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import app  # noqa: E402


class FakeEngine:
    """Pool of `pool` tokens and `nslots` slots; active = {slot: (tokens, prio, prefilling)}."""

    def __init__(self, pool, nslots=4):
        self.pool, self.nslots, self.active, self.paused, self.calls = pool, nslots, {}, set(), []

    def set_steering(self, *a):
        pass

    def probe_cached(self, ids):
        return 0

    def prefill_pos(self, slot):
        return 0

    def used(self, skip=()):
        return sum(t for s, (t, p, pf) in self.active.items() if s not in self.paused and s not in skip)

    def can_admit(self, n):
        paused = sum(self.active[s][0] for s in self.paused)  # paused ones resume first
        return len(self.active) < self.nslots and self.used() + paused + n <= self.pool

    def pause_for(self, n, prio):
        self.calls.append((n, prio))
        if len(self.active) >= self.nslots:
            return False
        vs = sorted((s for s, (t, p, pf) in self.active.items() if p < prio and not pf and s not in self.paused),
                    key=lambda s: self.active[s][1])
        left, k = self.used(), 0
        while k < len(vs) and left + n > self.pool:
            left -= self.active[vs[k]][0]
            k += 1
        if left + n > self.pool:
            return False
        self.paused |= set(vs[:k])
        return True


def job(n, prio=1, waited=0.0, now=1000.0):
    ids = app.Ids(range(n))
    return types.SimpleNamespace(ids=ids, prio=prio, t_enq=now - waited, est=None, t_est=0.0, p={}, slot=-1,
                                 cut_at=None)


def gen(engine, queue, **kw):
    a = types.SimpleNamespace(prefill_aging=4000, preempt_min=8192, preempt_max=1024, main_bonus=200000,
                              backfill_wait=30, mtp_k=3)
    for k, v in kw.items():
        setattr(a, k, v)
    g = types.SimpleNamespace(engine=engine, args=a, queue=queue, sched={"main_pauses": 0, "main_cutins": 0,
                                                                         "backfills": 0}, cur_slot=0)
    return g, app.Admission(g)


FAILS = []


def check(what, got, want):
    ok = got == want
    print(f"  {'ok ' if ok else 'FAIL'} {what}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        FAILS.append(what)


NOW = 1000.0
print("queue order: main first, aging still lets a long-waiting subagent through")
e = FakeEngine(10 ** 6)
q = [job(1000), job(50000, prio=2)]
g, adm = gen(e, q)
check("main (50K) before a fresh subagent (1K)", adm.admit_next(NOW, []), 1)
q = [job(1000, waited=60), job(50000, prio=2)]
g, adm = gen(e, q)
check("subagent waiting 60 s (> 200000/4000 = 50 s) goes first", adm.admit_next(NOW, []), 0)
q = [job(1000), job(50000, prio=2)]
g, adm = gen(e, q, main_bonus=0)
check("--main-bonus 0: shortest first", adm.admit_next(NOW, []), 0)

print("backfill: a request that does not fit no longer blocks the queue")
e = FakeEngine(100000)
e.active = {0: (60000, 1, False)}
q = [job(30000, waited=5), job(50000, waited=10)]  # the 50K one waited longer but est order puts 30K first
g, adm = gen(e, q, prefill_aging=0)
check("30K fits, 50K would not: 30K", adm.admit_next(NOW, []), 0)
# aging 4000 tok/s: the 50K one waiting 10 s (key 10000) is ahead of a fresh 20K one (key 20000)
q = [job(50000, waited=10), job(20000)]
g, adm = gen(e, q, main_bonus=0)
check("50K first in order but does not fit: 20K backfills", adm.admit_next(NOW, []), 1)
check("backfill counted", g.sched["backfills"], 1)
q = [job(50000, waited=40), job(20000)]
g, adm = gen(e, q, main_bonus=0)
check("starvation guard: 50K waited 40 s > 30 s, 20K may not pass", adm.admit_next(NOW, []), None)
q = [job(50000, waited=40), job(20000, prio=2)]
g, adm = gen(e, q, main_bonus=0)
check("... but a main-thread request (higher class) still may", adm.admit_next(NOW, []), 1)
g, adm = gen(e, [job(50000, waited=10), job(20000)], main_bonus=0, backfill_wait=0)
check("--backfill-wait 0: strict order", adm.admit_next(NOW, []), None)

print("main thread vs a pool full of subagents: pause the lowest-prio decoding sequences")
e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (40000, 1, False), 2: (15000, 0, False)}
g, adm = gen(e, [job(30000, prio=2)])
check("main admitted", adm.admit_next(NOW, []), 0)
check("paused: the finished (prio 0) one and one subagent", sorted(e.paused), [0, 2])
check("pauses counted", g.sched["main_pauses"], 1)
e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (75000, 2, False)}
g, adm = gen(e, [job(30000, prio=2)])
check("another main thread is never paused for a main thread", adm.admit_next(NOW, []), None)
check("nothing paused", e.paused, set())
e = FakeEngine(100000)
e.active = {0: (40000, 1, True), 1: (50000, 1, True)}
g, adm = gen(e, [job(30000, prio=2)])
check("prompts being prefilled are never paused", adm.admit_next(NOW, []), None)
e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (40000, 1, False)}
g, adm = gen(e, [job(30000)])
check("a subagent never pauses anyone", (adm.admit_next(NOW, []), e.calls), (None, []))

print("cut-in on a long prefill in progress")
e = FakeEngine(10 ** 6)
long_sub = types.SimpleNamespace(ids=list(range(100000)), slot=0, cut_at=None, prio=1)
g, adm = gen(e, [job(30000, prio=2)])
check("main of any size cuts in on a long subagent prefill", adm.admit_next(NOW, [long_sub]), 0)
long_main = types.SimpleNamespace(ids=list(range(100000)), slot=0, cut_at=None, prio=2)
g, adm = gen(e, [job(30000, prio=2)])
check("not on another main prefill", adm.admit_next(NOW, [long_main]), None)
g, adm = gen(e, [job(30000)])
check("a large subagent does not cut in", adm.admit_next(NOW, [long_sub]), None)
g, adm = gen(e, [job(500)])
check("a short one still does", adm.admit_next(NOW, [long_sub]), 0)

print("FAIL: " + ", ".join(FAILS) if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
