"""Scheduler admission (server/app.py Admission) with a fake engine, no GPU: order by size and age (no main-thread
rule), backfill past a request that does not fit + its starvation guard, pausing decoding sequences by the pool order
for any request, cut-ins on a long prefill, shared batches. Run: python tests/test_sched.py"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import app  # noqa: E402


class FakeEngine:
    """Pool of `pool` tokens and `nslots` slots; active = {slot: (tokens, admitted at, prefilling)}. pause_for pauses as
    qe_pause_for does (src/pool_order.h): decoding sequences, never a prefill, the most recently admitted first,
    whatever their class; nothing when that cannot make the request fit."""

    def __init__(self, pool, nslots=4):
        self.pool, self.nslots, self.active, self.paused, self.calls = pool, nslots, {}, set(), []
        self.npause = 0

    def pause_state(self):  # (pause events so far, paused now), as qe_pool_counters c[3] / c[5]
        return self.npause, len(self.paused)

    def set_steering(self, *a):
        pass

    def probe_cached(self, ids):
        return 0

    def prefill_pos(self, slot):
        return 0

    def used(self, skip=()):
        return sum(t for s, (t, a, pf) in self.active.items() if s not in self.paused and s not in skip)

    def can_admit(self, n):
        paused = sum(self.active[s][0] for s in self.paused)  # paused ones resume first
        return len(self.active) < self.nslots and self.used() + paused + n <= self.pool

    def pause_for(self, n, prio):
        self.calls.append((n, prio))
        if len(self.active) >= self.nslots:
            return False
        vs = sorted((s for s, (t, a, pf) in self.active.items() if not pf and s not in self.paused),
                    key=lambda s: -self.active[s][1])
        left, k = self.used(), 0
        while k < len(vs) and left + n > self.pool:
            left -= self.active[vs[k]][0]
            k += 1
        if left + n > self.pool:
            return False
        self.paused |= set(vs[:k])
        self.npause += k
        return True


def job(n, prio=1, waited=0.0, now=1000.0):
    ids = app.Ids(range(n))
    return types.SimpleNamespace(ids=ids, prio=prio, t_enq=now - waited, est=None, t_est=0.0, p={}, slot=-1,
                                 cut_at=None)


def gen(engine, queue, **kw):
    a = types.SimpleNamespace(prefill_aging=4000, preempt_min=8192, preempt_max=1024, backfill_wait=30, mtp_k=3)
    for k, v in kw.items():
        setattr(a, k, v)
    g = types.SimpleNamespace(engine=engine, args=a, queue=queue, sched={"pauses": 0, "backfills": 0}, cur_slot=0)
    return g, app.Admission(g)


FAILS = []


def check(what, got, want):
    ok = got == want
    print(f"  {'ok ' if ok else 'FAIL'} {what}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        FAILS.append(what)


NOW = 1000.0
print("queue order: least left first, aged; no main-thread rule (user decision 2026-11-12)")
e = FakeEngine(10 ** 6)
q = [job(1000), job(50000, prio=2)]
g, adm = gen(e, q)
check("a main-thread 50K prompt does not pass a fresh 1K one", adm.admit_next(NOW, []), 0)
q = [job(50000, waited=12), job(1000)]
g, adm = gen(e, q)
check("aging: 50K waiting 12 s (key 2000) before a fresh 1K one (key 1000)? no: 1K", adm.admit_next(NOW, []), 1)
q = [job(5000, waited=2), job(1000)]
g, adm = gen(e, q)
check("aging: 5K waiting 2 s (key -3000) before a fresh 1K", adm.admit_next(NOW, []), 0)

print("backfill: a request that does not fit no longer blocks the queue")
e = FakeEngine(100000)
e.active = {0: (60000, 1, False)}
q = [job(30000, waited=5), job(50000, waited=10)]  # the 50K one waited longer but est order puts 30K first
g, adm = gen(e, q, prefill_aging=0)
check("30K fits, 50K would not: 30K", adm.admit_next(NOW, []), 0)
# aging 4000 tok/s: the 50K one waiting 10 s (key 10000) is ahead of a fresh 20K one (key 20000)
e.active = {0: (60000, 1, True)}  # (a prefill: nothing can be paused for the 50K one)
q = [job(50000, waited=10), job(20000)]
g, adm = gen(e, q)
check("50K first in order but does not fit: 20K backfills", adm.admit_next(NOW, []), 1)
check("backfill counted", g.sched["backfills"], 1)
q = [job(50000, waited=40), job(20000)]
g, adm = gen(e, q)
check("starvation guard: 50K waited 40 s > 30 s, 20K may not pass", adm.admit_next(NOW, []), None)
q = [job(50000, waited=40), job(20000, prio=2)]
g, adm = gen(e, q)
check("... a main-thread request may not pass it either (no class)", adm.admit_next(NOW, []), None)
g, adm = gen(e, [job(50000, waited=10), job(20000)], backfill_wait=0)
check("--backfill-wait 0: strict order", adm.admit_next(NOW, []), None)

print("no room: any request may pause decoding sequences, the most recently admitted first")
e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (40000, 3, False), 2: (15000, 2, False)}  # slot 1 admitted last
g, adm = gen(e, [job(30000, waited=5)])
check("admitted", adm.admit_next(NOW, []), 0)
check("paused: the most recently admitted one only (slot 1, 40K)", sorted(e.paused), [1])
check("pauses counted", g.sched["pauses"], 1)
e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (75000, 2, False)}
g, adm = gen(e, [job(30000, prio=2, waited=5)])
check("a main-thread request pauses another main thread like any sequence", (adm.admit_next(NOW, []), sorted(e.paused)), (0, [1]))
e = FakeEngine(100000)
e.active = {0: (40000, 1, True), 1: (50000, 2, True)}
g, adm = gen(e, [job(30000, prio=2, waited=5)])
check("prompts being prefilled are never paused", (adm.admit_next(NOW, []), e.paused), (None, set()))

print("pauses: never while one is paused (review H1), only after pause_wait (M1), counted only when done")
e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (40000, 2, False)}
e.paused = {1}  # slot 1 paused at 40K: it waits for the room can_admit keeps for it
g, adm = gen(e, [job(30000, waited=10)])
check("a 30K request does not take the paused one's room", (adm.admit_next(NOW, []), g.sched["pauses"]), (None, 0))
g, adm = gen(e, [job(25000, waited=10), job(30000, waited=20)])
check("nor a stream of later arrivals", adm.admit_next(NOW, []), None)
e.paused = set()  # it resumed: now a request may pause again
g, adm = gen(e, [job(30000, waited=10)])
check("once it resumed, a waiting request may pause", (adm.admit_next(NOW, []), g.sched["pauses"], sorted(e.paused)), (0, 1, [1]))
e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (40000, 2, False)}
g, adm = gen(e, [job(30000, waited=1)])
check("a request that waited 1 s (< --pause-wait 3) does not pause", (adm.admit_next(NOW, []), e.paused), (None, set()))
e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (40000, 2, False), 2: (15000, 3, False)}
n_adm, events = 0, 0
q = []
for k in range(20):  # a 20K arrival every second against a full pool: pauses stay bounded
    q.append(job(20000, now=NOW + k))
    g, adm = gen(e, q)
    p0 = e.npause
    i = adm.admit_next(NOW + k, [])
    events += e.npause > p0
    if i is not None:
        x = q.pop(i)
        n_adm += 1
        e.active[10 + k] = (20000, 10 + k, False)
check("20 arrivals, 1 s apart, full pool: one pause event (then the paused ones block further pauses)", (events, n_adm), (1, 1))
e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (40000, 2, False)}
e.pause_for = lambda n, prio: True  # fits now, nothing paused (the engine's answer when no room is missing)
g, adm = gen(e, [job(30000, waited=5)])
check("pause_for that paused nothing: admitted, not counted", (adm.admit_next(NOW, []), g.sched["pauses"]), (0, 0))

e = FakeEngine(100000)
e.active = {0: (40000, 1, False), 1: (40000, 2, False), 2: (15000, 3, False)}
def half(n, prio):  # pause_slot fails halfway: one sequence paused, then False (review N1)
    e.paused.add(2)
    e.npause += 1
    return False
e.pause_for = half
g, adm = gen(e, [job(30000, waited=5)])
g.cur_slot = 2
check("a pause that failed halfway: not admitted, but counted and the current slot reset",
      (adm.admit_next(NOW, []), g.sched["pauses"], g.cur_slot), (None, 1, -1))

print("cut-in on a long prefill in progress (--long-share 0): by size only")
e = FakeEngine(10 ** 6)
long_sub = types.SimpleNamespace(ids=list(range(100000)), slot=0, cut_at=None, prio=1)
g, adm = gen(e, [job(30000, prio=2)])
check("a 30K main-thread prompt does not cut in on a long subagent prefill", adm.admit_next(NOW, [long_sub]), None)
g, adm = gen(e, [job(30000)])
check("a large subagent does not cut in", adm.admit_next(NOW, [long_sub]), None)
g, adm = gen(e, [job(500)])
check("a short one still does", adm.admit_next(NOW, [long_sub]), 0)
g, adm = gen(e, [job(500, prio=2)])
check("a short main-thread one the same", adm.admit_next(NOW, [long_sub]), 0)

print("shared prefill batches (Flash-Next): cut-ins at any time, the longest prompt keeps its share")


class FakeMulti(FakeEngine):
    def __init__(self, pool, bound=True):
        super().__init__(pool, nslots=16)
        self.pos, self.bound = {}, bound

    def can_prefill_multi(self):
        return True

    def prefill_pass_bound(self):
        return self.bound

    def prefill_pos(self, slot):
        return self.pos.get(slot, -1)


def inflight(n, slot, pos, prio=2, waited=0.0):
    return types.SimpleNamespace(ids=list(range(n)), slot=slot, cut_at=None, prio=prio, t_enq=NOW - waited, round_tok=0)


def batch(e, order, T=4096, PFM=512, caps=None):
    """qe_prefill_multi's row picker (pass-bound): in order, a piece that ends within the batch takes it all, else whole
    PFM chunks of what is left; the first prompt that cannot end takes the rest and the batch is full. caps (id -> rows,
    Admission.caps): at most that many rows of the prompt (whole chunks). Returns {slot: rows}."""
    used, got = 0, {}
    for j in order:
        want = len(j.ids) - e.pos[j.slot]
        if caps and caps.get(id(j)):
            want = min(want, max(PFM, caps[id(j)] // PFM * PFM))
        m = want if used + want <= T else (T - used) // PFM * PFM
        if m <= 0:
            break
        e.pos[j.slot] += m
        got[j.slot] = m
        used += m
        if used >= T:
            break
    return got


def run(e, adm, jobs, until, T=4096, cap=400, dt=0.65, rows=None):
    """Batches until until() (or cap); returns the sequence of batch leaders (slot ids) and the batch count. rows: a list
    that gets each batch's {slot: rows}."""
    seq, n = [], 0
    while not until() and n < cap:
        live = [j for j in jobs if e.pos[j.slot] < len(j.ids)]
        order = adm.batch_order(NOW + n * dt, live, T) if len(live) >= 2 else live
        seq.append(order[0].slot)
        got = batch(e, order, T, caps=adm.caps if len(live) >= 2 else None)
        if rows is not None:
            rows.append(got)
        if len(live) >= 2:
            adm.account(live, T)
        n += 1
    return seq, n


K = dict(preempt_max=8192, long_share=0.5, prefill_chunk=4096, prefill_chunk_idle=4096)  # the credit rule (default)
e = FakeMulti(10 ** 7)
e.pos = {0: 1000}
long_ = inflight(50000, 0, 1000)
long_.cut_at, long_.round_tok = 1000, 4000  # a cut-in round just ended: before, the next waited preempt_min tokens
for n, want in ((3000, 0), (8000, 0), (9000, None)):
    g, adm = gen(e, [job(n, prio=2)], **K)
    check(f"{n} tokens join at once on a 50K prompt", adm.admit_next(NOW, [long_]), want)
g, adm = gen(e, [job(3000, prio=2)], preempt_max=1024, long_share=0.0, prefill_chunk=2048, prefill_chunk_idle=4096)
check("--long-share 0: the old rule (3000 above preempt-max waits)", adm.admit_next(NOW, [long_]), None)

print("--fill-passes 0 (the credit rule): owed rows first, else last (H1); by size and age only (no main-thread rule)")
L, C = inflight(50000, 0, 0), inflight(3000, 1, 0)
e.pos = {0: 1000, 1: 0}
g, adm = gen(e, [], **dict(K, prefill_chunk=2048))
seq, n = run(e, adm, [L, C], lambda: e.pos[1] >= 3000, T=2048)
check("a 3000 cut-in and the long prompt alternate", seq, [1, 0, 1])
L, C = inflight(200000, 0, 0, prio=1, waited=40), inflight(6000, 1, 0, prio=1)  # aged 40 s, cut-in above join_cap
e.pos = {0: 40000, 1: 0}
g, adm = gen(e, [], **K)
seq, n = run(e, adm, [L, C], lambda: e.pos[1] >= 6000)
check("an aged long prompt does not hold a 6K cut-in: done within 3 batches", (n <= 3, seq[0]), (True, 1))
g, adm = gen(e, [job(30000, prio=2)], **K)
check("a 30K main-thread prompt does not join a 100K subagent prompt (no main rule on the new path)",
      adm.pick_next(NOW, [inflight(100000, 0, 0, prio=1)]), [])
L, M = inflight(100000, 0, 0, prio=2), inflight(30000, 1, 0, prio=2)  # both main threads: the share applies
e.pos = {0: 0, 1: 0}
g, adm = gen(e, [], **K)
seq, n = run(e, adm, [L, M], lambda: e.pos[1] >= 30000)
check("a 30K prompt next to a 100K one (same class): the long one keeps about half the batches", 13 <= n <= 17 and seq[:4] == [1, 0, 1, 0], True)

L = inflight(60000, 0, 0, prio=1)  # a subagent's long prompt next to main-thread tool results (each up to preempt_max)
e.pos = {0: 0}
g, adm = gen(e, [], **K)
k = 0
while e.pos[0] < 60000 and k < 200:
    C = inflight(3000, 10 + k, 0, prio=2)
    C.est = 3000
    e.pos[C.slot] = 0
    batch(e, adm.batch_order(NOW, [L, C]))
    adm.account([L, C], 4096)
    k += 1
check("a subagent's 60K next to a main cut-in every pass keeps its share: <= 2x its 15 passes", k <= 30, True)

print("nothing starves behind a stream of short cut-ins (H2)")
for what, big in (("a 20K prompt", job(20000, prio=1, waited=300.0)), ("an image prompt", job(500, prio=1, waited=300.0))):
    if what.startswith("an image"):
        big.ids.mm = object()
    e.pos = {1: 0}
    g, adm = gen(e, [big, job(500, prio=1)], **K)
    check(f"{what} waiting 300 s: nothing joins, the batch drains", adm.pick_next(NOW, [inflight(6000, 1, 0, prio=1)]), [])
    big.t_enq = NOW - 5.0
    big.est = None
    g, adm = gen(e, [big, job(500, prio=1)], **K)
    check(f"{what} waiting 5 s (< backfill_wait): the short one still joins",
          adm.pick_next(NOW, [inflight(6000, 1, 0, prio=1)]), [1])
g, adm = gen(e, [job(20000, prio=1, waited=300.0)], **K)
check("with nothing in flight it starts", adm.pick_next(NOW, []), [0])

print("accounting edges")
L, C = inflight(50000, 0, 0), inflight(2000, 1, 0)
e.pos = {0: 0, 1: 0}
g, adm = gen(e, [], **K)
adm.batch_order(NOW, [L, C])
e.pos = {0: 0, 1: -1}  # the cut-in ended in this batch (its prefill is gone): its 2000 rows count
adm.account([L, C], 4096)
check("a prompt done in the batch counts its rows (the long one got none of the pass)", adm.credit, 0.5)
adm.batch_order(NOW, [L, C])
adm.credit = 0.8
e.pos = {0: 0, 1: 2000}
adm.account([L, C], 4096)
check("credit bounded by one pass", adm.credit, 1.0)
print("a stream of small cut-in batches without the long prompt: it still leads every other pass")
L = inflight(60000, 0, 0)
e.pos = {0: 0}
g, adm = gen(e, [], **K)
seq, k = [], 0
while e.pos[0] < 60000 and k < 200:  # each pass: a fresh 500-token cut-in that the batch carries alone unless L leads
    C = inflight(500, 10 + k, 0)
    e.pos[C.slot] = 0
    order = adm.batch_order(NOW, [L, C])
    seq.append(order[0] is L)
    if order[0] is L:
        batch(e, order)
    else:
        e.pos[C.slot] = 500  # (the engine dropped the long prompt from this batch: segment cap / layout)
    adm.account([L, C], 4096)
    k += 1
check("passes for 60K next to a cut-in every pass: about 2x its 15 alone", 28 <= k <= 32, True)
e.pos = {0: 0, 1: 0}
L2 = inflight(80000, 2, 0)
e.pos[2] = 0
adm.batch_order(NOW, [L, C, L2])
check("a new longest prompt starts at no credit", (adm.long_job is L2, adm.credit), (True, 0.0))
e.pos[2] = -1
adm.account([L, C, L2], 4096)
check("a long prompt that ended is not kept", adm.long_job, None)

print("full passes (--fill-passes 1, opt-in): the others first by age, the long prompt fills every pass, keeps its share")
KF = dict(K, fill_passes=1)
L, C = inflight(50000, 0, 0), inflight(3000, 1, 0)
e.pos = {0: 1000, 1: 0}
g, adm = gen(e, [], **KF)
rows = []
seq, n = run(e, adm, [L, C], lambda: e.pos[1] >= 3000, rows=rows)
check("a 3000 cut-in next to a 50K prompt: done in 2 passes, the long one keeps 2048 of the first and fills the second",
      (n, rows[0], rows[1]), (2, {0: 2048, 1: 2048}, {1: 952, 0: 3072}))  # (a cut part takes whole 512-row chunks)
L, C = inflight(200000, 0, 0, prio=1, waited=40), inflight(6000, 1, 0, prio=1)  # aged 40 s, cut-in above join_cap (H1)
e.pos = {0: 40000, 1: 0}
g, adm = gen(e, [], **KF)
rows = []
seq, n = run(e, adm, [L, C], lambda: e.pos[1] >= 6000, rows=rows)
check("an aged long prompt does not hold a 6K cut-in: done in 3 passes, each pass full (within a chunk), the long one >= 2048 rows each",
      (n, all(sum(r.values()) > 4096 - 512 and r.get(0, 0) >= 2048 for r in rows)), (3, True))
L = inflight(60000, 0, 0)
e.pos = {0: 0}
g, adm = gen(e, [], **KF)
k, full = 0, True
while e.pos[0] < 60000 and k < 200:  # a fresh 500-token cut-in every pass
    C = inflight(500, 10 + k, 0)
    e.pos[C.slot] = 0
    order = adm.batch_order(NOW, [L, C], 4096)
    got = batch(e, order, caps=adm.caps)
    full = full and (sum(got.values()) > 4096 - 512 or e.pos[0] >= 60000)
    k += 1
check("60K next to a 500-token cut-in every pass: 3584 rows of it per pass, 17 passes, all full (within a chunk)", (k, full), (17, True))
L, A, B2, C = inflight(80000, 0, 0), inflight(3000, 1, 0, waited=3), inflight(3000, 2, 0, waited=9), inflight(3000, 3, 0, waited=1)
e.pos = {0: 0, 1: 0, 2: 0, 3: 0}
g, adm = gen(e, [], **KF)
order = adm.batch_order(NOW, [L, A, B2, C], 4096)
check("several cut-ins wanting more than half: the long prompt leads capped to 2048, then the others oldest first",
      ([j.slot for j in order], adm.caps.get(id(L))), ([0, 2, 1, 3], 2048))
got = batch(e, order, caps=adm.caps)
check("... the oldest cut-in gets the rest of the pass", got, {0: 2048, 2: 2048})
L, M = inflight(100000, 0, 0, prio=2), inflight(30000, 1, 0, prio=2)  # two prompts above one batch: the longer keeps half
e.pos = {0: 0, 1: 0}
g, adm = gen(e, [], **KF)
rows = []
seq, n = run(e, adm, [L, M], lambda: e.pos[1] >= 30000, rows=rows)
check("a 30K prompt next to a 100K one: 2048 rows each per pass (15 passes), the 100K one never below half",
      (n, all(r.get(0, 0) >= 2048 for r in rows), rows[0]), (15, True, {0: 2048, 1: 2048}))
g, adm = gen(e, [], **dict(KF, long_share=0.25))
L, C = inflight(50000, 0, 0), inflight(8000, 1, 0)
e.pos = {0: 0, 1: 0}
order = adm.batch_order(NOW, [L, C], 4096)
check("--long-share 0.25: the long prompt leads capped to 1024 rows", (order[0] is L, adm.caps.get(id(L))), (True, 1024))
g, adm = gen(e, [], **KF)
L, C = inflight(50000, 0, 0), inflight(1500, 1, 0)
e.pos = {0: 0, 1: 0}
order = adm.batch_order(NOW, [L, C], 4096)
check("a cut-in within (1 - share) of the pass: it goes first, the long prompt fills the rest uncapped",
      ([j.slot for j in order], adm.caps), ([1, 0], {}))
g, adm = gen(e, [], **KF)
adm.batch_order(NOW, [L, C], 4096)
adm.account([L, C], 4096)
check("no credit is kept with full passes", (adm.long_job, adm.credit), (None, 0.0))

g, adm = gen(e, [], **KF)
L, C = inflight(50000, 0, 0), inflight(8000, 1, 0)
e.pos = {0: 0, 1: 0}
order = adm.batch_order(NOW, [L, C], 16384)
check("full passes with a budget above the batch: the share of the engine's 4096-row batch (2048), not of 16384",
      adm.caps.get(id(L)), 2048)

print("conversation list (pool_info): LRU first, fields; internal callers without it (review M3)")
import ctypes
from engine import Engine


class FakeLib:
    def qe_conv_list(self, h, buf, m):
        rows = [(7, 70000, 2, 12500, 3, 0), (9, 300, -1, -1, 5, 0), (4, 1200, 0, 0, 9, 1)]
        for k, r in enumerate(rows):
            for f, v in enumerate(r):
                buf[6 * k + f] = v
        return len(rows)


fe = types.SimpleNamespace(lib=FakeLib(), h=None, fns={"qe_conv_list": 1}, slots=[], pool_tokens=1, max_t=1, _active=set())
pi = Engine.pool_info(fe)
check("rows in engine order with fields", [(c["id"], c["tokens"], c["where"], c["idle_s"], c["generating"]) for c in pi["conversations"]],
      [(7, 70000, "slot 2", 12.5, False), (9, 300, "archive", None, False), (4, 1200, "slot 0", 0.0, True)])
check("convs=False: no list", Engine.pool_info(fe, convs=False)["conversations"], [])

print("defaults resolved at start (M1, M2, L3)")
prof = types.SimpleNamespace(defaults={"preempt_max": 8192})


def fresh(g, **kw):  # (gen() already built an Admission, which resolved the defaults into args: start again with a profile)
    for k, v in kw.items():
        setattr(g.args, k, v)
    g.profile = prof
    return app.Admission(g)


for bound, want in ((True, (0.5, 8192, 4096)), (False, (0.0, 1024, 2048))):
    g, adm = gen(FakeMulti(10 ** 7, bound=bound), [], preempt_max=None, long_share=None, prefill_chunk=2048,
                 prefill_chunk_auto=True, prefill_chunk_idle=4096)
    adm = fresh(g, preempt_max=None, prefill_chunk=2048)
    check(f"{'pass-bound' if bound else 'all-VRAM'}: long-share, preempt-max, chunk", (adm.long_share, adm.pre_max,
          g.args.prefill_chunk), want)
g, adm = gen(FakeMulti(10 ** 7), [], preempt_max=None, long_share=0.0, prefill_chunk=2048, prefill_chunk_auto=True,
             prefill_chunk_idle=4096)
check("--long-share 0 restores --preempt-max 1024", (adm.long_share, adm.pre_max), (0.0, 1024))
g, adm = gen(FakeMulti(10 ** 7, bound=False), [], preempt_max=None, long_share=0.5, prefill_chunk=2048,
             prefill_chunk_auto=True, prefill_chunk_idle=4096)
adm = fresh(g, preempt_max=None)
check("an explicit --long-share applies on an all-VRAM server", (adm.long_share, adm.pre_max), (0.5, 8192))
g, adm = gen(FakeMulti(10 ** 7), [], preempt_max=None, long_share=None, prefill_chunk=2048, prefill_chunk_auto=False,
             prefill_chunk_idle=4096)
check("an explicit --prefill-chunk 2048 stays", g.args.prefill_chunk, 2048)
g, adm = gen(FakeMulti(10 ** 7), [], preempt_max=None, long_share=None, prefill_chunk=2048, prefill_chunk_auto=True,
             prefill_chunk_idle=8192)
check("--prefill-chunk-idle 8192: the chunk while decoding stays at the measured 4096", g.args.prefill_chunk, 4096)
# [gptoss-pf90-idle] review pfidle M1: an idle chunk without shared batches (gpt-oss 10240) leaves the cut-in gap alone
g, adm = gen(FakeEngine(10 ** 6), [], prefill_chunk=2048, prefill_chunk_auto=True, prefill_chunk_idle=10240)
check("no shared batches: join_cap ignores the idle chunk", (adm.multi, adm.join_cap(), g.args.prefill_chunk), (False, 2560, 2048))
g, adm = gen(FakeMulti(10 ** 7, bound=False), [], prefill_chunk=2048, prefill_chunk_auto=True, prefill_chunk_idle=4096)
check("shared batches: join_cap keeps the idle batch", adm.join_cap(), 4608)
check("gpt-oss profile: idle chunk = the engine chunk", app.FAMILIES["gptoss"]["prefill_chunk_idle"] if hasattr(app, "FAMILIES") else
      __import__("model_profile").FAMILIES["gptoss"]["prefill_chunk_idle"], 10240)
# [gptoss-lows] review pfidle L2 / L1: the idle chunk of a lone long prompt (Generator.idle_budget)
lp = job(90000)
lp.slot = 0
for gpo, pf_pass, qn, want, what in ((True, 10240, None, 10240, "nothing queued: the idle chunk"),
                                     (True, 10240, 5000, 10240, "a queued 5K request that cannot cut in keeps the idle chunk"),
                                     (True, 10240, 500, 2048, "a queued 500-token request that can cut in: --prefill-chunk"),
                                     (True, 4096, None, 8192, "engine chunk 4,096: the idle chunk is a multiple of it"),
                                     (True, 0, None, 10240, "engine chunk not reported: the idle chunk"),
                                     (False, 0, 5000, 2048, "other families: any queued request -> --prefill-chunk (unchanged)")):
    g, adm = gen(FakeEngine(10 ** 6), [job(qn)] if qn else [], prefill_chunk=2048, prefill_chunk_auto=True,
                 prefill_chunk_idle=10240)
    g.gptoss, g.pf_pass = gpo, pf_pass
    check(f"idle budget: {what}", app.Generator.idle_budget(g, 2048, False, [lp], adm), want)
check("idle budget: a decoding slot -> --prefill-chunk", app.Generator.idle_budget(g, 2048, True, [lp], adm), 2048)
# review lows M1: a request that may cut in later (the fairness gap after a cut-in) keeps --prefill-chunk
lp.cut_at, lp.round_tok = 0, 600
g, adm = gen(FakeEngine(10 ** 6), [job(500)], prefill_chunk=2048, prefill_chunk_auto=True, prefill_chunk_idle=10240)
g.gptoss, g.pf_pass = True, 10240
check("idle budget: in the fairness gap pick_next admits none", adm.pick_next(NOW, [lp]), [])
check("idle budget: a 500-token request waiting out the fairness gap -> --prefill-chunk", app.Generator.idle_budget(g, 2048, False, [lp], adm), 2048)
lp.cut_at = None
FLASH = os.environ.get("QE_MOCK_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-GDN8"))
if os.path.exists(os.path.join(FLASH, "config.json")):
    a = app.parse_args(["--model", FLASH])
    check("parse_args: chunk auto, preempt-max and long-share left to the scheduler",
          (a.prefill_chunk, a.prefill_chunk_auto, a.preempt_max, a.long_share), (2048, True, None, None))
    a = app.parse_args(["--model", FLASH, "--prefill-chunk", "2048", "--preempt-max", "1024"])
    check("parse_args: explicit values equal to the defaults stay explicit", (a.prefill_chunk_auto, a.preempt_max),
          (False, 1024))

print("queue-aware split (QaSplit, Admission.qa)")
prof27 = types.SimpleNamespace(defaults={}, family="qwen38-27b")
g, adm = gen(FakeEngine(10 ** 7), [], prefill_share=0.5, prefill_chunk=2048)
check("off without the 27B profile (share-min defaults to --prefill-share)", adm.qa, False)
g.profile = prof27
adm = app.Admission(g)
check("27B default: on, floor 0.10, short 4096", (adm.qa, adm.share_min, adm.short), (True, 0.10, 4096))
g.args.prefill_share_min = 0.5
check("--prefill-share-min = --prefill-share: off (the fixed split)", app.Admission(g).qa, False)
g.args.prefill_share_min = None


class MultiEng(FakeEngine):  # shared prefill batches (Flash-Next)
    def can_prefill_multi(self):
        return True

    def prefill_pass_bound(self):
        return False


gq, admq = gen(MultiEng(10 ** 7), [], prefill_share=0.5, prefill_chunk=2048)
gq.profile = types.SimpleNamespace(defaults={}, family="qwen38-flash-next")
admq = app.Admission(gq)
check("Flash-Next default (3 cards): the time split on (floor 0.10), the one-prompt cut-in rule off",
      (admq.multi, admq.qa, admq.qa_cut, admq.share_min), (True, True, False, 0.10))


class BoundEng(MultiEng):
    def prefill_pass_bound(self):
        return True


gb, admb = gen(BoundEng(10 ** 7), [], prefill_share=0.5, prefill_chunk=2048)
gb.profile = gq.profile
check("Flash-Next, passes bound by expert copies (2 cards): the fixed split as before", app.Admission(gb).qa, False)


class PpEng(MultiEng):
    par = {"mode": "pp"}


gp, _ = gen(PpEng(10 ** 7), [], prefill_share=0.5, prefill_chunk=2048)
gp.profile = gq.profile
check("Flash-Next in mode B (pp): the fixed split as before", app.Admission(gp).qa, False)
qm = app.QaSplit(0.15, 0.5, 0.15)
actq = [types.SimpleNamespace(finished=False) for _ in range(4)]
qm.add(1, 1.0)  # decoders had their share recently
admq.e = types.SimpleNamespace(prefill_pos=lambda slot: 0)
jq = job(800)
jq.slot = 0
check("shared batches, short prompt next: a fixed window of the floor share after the pass",
      round(qm.window(0.2, admq, [jq], [], actq), 4), round(0.2 * 0.15 / 0.85, 4))
qm.add(0, 30.0)  # decoders far below their floor recently: still no catch-up window while short prompts are next
check("shared batches, short prompt next: no catch-up window beyond it",
      round(qm.window(0.2, admq, [jq], [], actq), 4), round(0.2 * 0.15 / 0.85, 4))


class PosEng(FakeEngine):
    def __init__(self, pos):
        super().__init__(10 ** 7)
        self.pos = pos

    def prefill_pos(self, slot):
        return self.pos.get(slot, 0)


def pj(n, slot):
    j = job(n)
    j.slot = slot
    return j


act = [types.SimpleNamespace(finished=False) for _ in range(8)]
adm.e = PosEng({0: 0})
qa = app.QaSplit(0.15, 0.5)
qa.add(1, 0.5)  # decoders had their share recently
check("short prompt next: no window", qa.window(0.2, adm, [pj(1000, 0)], [], act), 0.0)
adm.e = PosEng({0: 0})
w = qa.window(0.2, adm, [pj(20000, 0)], [job(10000)], act)
check("long prompts only: share A / (A + W) capped at --prefill-share (8 / 10 -> 0.5: window = tc)", round(w, 6), 0.2)
w = qa.window(0.2, adm, [pj(20000, 0)], [job(10000)] * 60, act)
check("many waiting: share floor 0.15", round(w, 6), round(0.2 * 0.15 / 0.85, 6))
w = qa.window(0.2, adm, [pj(20000, 0)], [job(100)], act)
check("a short one queued: no window", w, 0.0)
qa2 = app.QaSplit(0.15, 0.5)
qa2.add(0, 3.0)
w = qa2.window(0.2, adm, [pj(1000, 0)], [], act)
check("floor: decoders far below 0.15 of the recent time get a window", w > 0.4, True)
adm.e = PosEng({0: 5000})
long_ = pj(20000, 0)
long_.cut_at = None
g.queue[:] = [job(3000), job(6000)]
check("cut-in up to --prefill-short rows into a long prompt", adm.pick_next(1000.0, [long_]), [0])
long_.cut_at = 5000
check("no second cut-in before the long prompt had a pass", adm.pick_next(1000.0, [long_]), [])
g.queue[:] = []

print("FAIL: " + ", ".join(FAILS) if FAILS else "all passed")


def test_sched():  # pytest: the checks above ran at import
    assert not FAILS, FAILS


if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
