"""Shared prefix KV (branch cache-priority, stage 4): agents of one kind share the KV rows of
their system part (pinned prefix) - one region in the pool read by every attached slot, each
slot holding only its own rows. Checks: greedy outputs of attached agents decoding together
(batched steps) and alone against the same prompts computed alone from scratch (unattached);
decode step time attached vs unattached; an attached conversation's next turn after an
eviction (archive holds only its own rows, re-attached) against the same turn on the live
state; a new agent attaching to the shared region after that. Run with QE_SHARE_PREFIX=0 for
the copy path (must pass too). usage: python3 tests/test_shared_prefix.py

Attached vs alone from scratch is gated like test_prefill_sweep (B), not on identical tokens: an
attached slot prefills only its own rows, i.e. other chunk sizes than a full prefill (other stream-K
splits of the prefill GEMM's last wave, attention / delta-net chunk splits), so near-tie argmaxes flip. A divergence passes
when, teacher-forced on the alone tokens, both variants have a top1 - top2 margin < 0.1 logit at the
first divergent position and the mean KL(alone || attached) over the N positions is <= 1e-3; the
number of identical tokens is reported. Hard checks: reuse counts (attached / restored / re-attached),
the restored turn identical to the live one (same rows from the archive), no failure."""
import ctypes, os, sys, time, random
import numpy as np
os.environ.setdefault("QE_POOL_TOKENS", "98304")
os.environ.setdefault("QE_MAX_SEQS", "6")
os.environ.setdefault("QE_POOL_HEADROOM", "256")
os.environ.setdefault("QE_POOL_GROW_FRAC", "0")
os.environ.setdefault("QE_MTP_ADAPT", "0")
os.environ.setdefault("QE_POOL_LOG", "1")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine
from tokenizers import Tokenizer

MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
text = open(os.path.join(ROOT, "tests", "data", "cache_text.txt")).read()
W = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
SHARE = os.environ.get("QE_SHARE_PREFIX", "1") != "0"


def filler(seed, n):
    r = random.Random(seed)
    return " ".join(r.choice(W) for _ in range(n))


def chat(system, user):
    return tok.encode(f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                      f"<|im_start|>assistant\n<think>\n\n</think>\n\n").ids


def sys_end(ids):
    pat = tok.encode("<|im_start|>user", add_special_tokens=False).ids
    for i in range(len(ids) - len(pat) + 1):
        if ids[i:i + len(pat)] == pat:
            return i
    return 0


SYS = "You are a research subagent. Tools and rules:\n" + text[:16000]  # ~5.7K tokens of shared prefix
agents = [chat(SYS, filler(10 + i, 1800) + f"\nTask {i}: explain the code above in two sentences.") for i in range(3)]
pin = sys_end(agents[0])
e = Engine(MODEL, max_t=262144)
e.set_sampling(0)
e.set_concurrent(True)
N = 32
print(f"pool {e.pool_tokens} tokens; agents {[len(a) for a in agents]}, system part {pin} (pinned); "
      f"sharing {'on' if SHARE else 'off'}", flush=True)


MARGIN, KL_MAX = 0.1, 1e-3
pending = []  # (label, ids, attached-variant setup prompts, d) checked teacher-forced at the end


def forced_logits(ids, seq, creator=None):
    """Prompt-end logits of ids + seq[:j], j < N, on a fresh engine state where ids is prefilled from
    scratch (creator None) or attached to the shared prefix created by `creator`; the continuations
    reuse those rows and prefill only seq[:j] (k = 0: no drafts, the same rows path for both)."""
    e.reset()
    if creator is not None:
        e.start(creator, 3, 0x2, sys_end(creator))  # stays live: holds the pinned prefix region
    t, r, s = e.start(ids, 3, 0x2, sys_end(ids))
    assert creator is None or r >= pin, (r, pin)
    e.release(s)
    lg = np.zeros((N, e.vocab()), dtype=np.float32)
    for j in range(N):
        t, r, s = e.start(ids + seq[:j], 0, 0x2, sys_end(ids))
        e.lib.qe_logits(ctypes.c_void_p(e.h), lg[j].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        e.release(s)
    return lg


def near_tie(label, ids, seq, d, creator):
    la, lt = forced_logits(ids, seq), forced_logits(ids, seq, creator)
    lsm = lambda x: x - x.max(-1, keepdims=True) - np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True))
    pa, pt = lsm(la.astype(np.float64)), lsm(lt.astype(np.float64))
    kl = float((np.exp(pa) * (pa - pt)).sum(-1).mean())
    ma, mt = [float(np.diff(np.sort(x[d])[-2:])[0]) for x in (la, lt)]
    good = ma < MARGIN and mt < MARGIN and kl <= KL_MAX
    print(f"{label}: first divergence at {d}: top1 - top2 margin alone {ma:.4f} attached {mt:.4f} "
          f"(< {MARGIN}), teacher-forced mean KL(alone || attached) {kl:.2e} (<= {KL_MAX:g}): "
          f"{'near-tie, ok' if good else 'NOT a near-tie'}")
    return good


def alone(ids, n=N):
    t, r, s = e.start(ids, 3, 0x2, sys_end(ids))
    out = [t]
    while len(out) < n:
        out += e.step()
    e.release(s)
    return out, r, s


# references: every prompt from scratch (reset forgets prefixes; the first prefill of a kind is
# the creator of the pinned prefix and never attached)
refs = []
for ids in agents:
    e.reset()
    out, r, s = alone(ids)
    refs.append(out)
e.reset()

ok = True
# 1. three agents of one kind: the first creates the prefix, the others attach; batched decode
live = []
for i, ids in enumerate(agents):
    t, r, s = e.start(ids, 3, 0x2, sys_end(ids))
    print(f"agent {i}: slot {s} reused {r} copied {e.last_copied()} from archive {e.last_from_archive()}", flush=True)
    live.append(([t], s))
    if i > 0:
        ok &= r >= pin
while min(len(o) for o, _ in live) < N:
    for (o, _), r in zip(live, e.step_batch([s for _, s in live])):
        o += r
for i, (o, _) in enumerate(live):
    d = next((j for j in range(N) if o[j] != refs[i][j]), N)
    print(f"agent {i} batched (attached: {'yes' if i and SHARE else 'no'}): {d}/{N} tokens identical to alone from scratch")
    if d < N:  # gated teacher-forced at the end (needs fresh engine states)
        pending.append((f"agent {i} batched", agents[i], refs[i], d, agents[0] if i else None))
info = e.pool_info()
print("slot regions (logical capacity):", [(x["cap"], x["pos"]) for x in info["slots"] if x["cap"]],
      f"evictions {info['events']['evict']}")
# decode step time: attached (agents 1, 2) vs unattached creator (agent 0), single-sequence
# (30 warm-up steps build the slot's graphs; outputs kept so the live states stay known; the
# archive pool must be pinned first - registering a slab stalls the GPUs ~0.2 s)
budget = (float(os.environ.get("QE_KV_ARCHIVE_GB", "32")) + float(os.environ.get("QE_PREFIX_GB", "8"))) * (1 << 30)
t0 = time.time()
while (os.environ.get("QE_ARCHIVE_PINNED", "1") != "0" and e.pool_info()["archives"]["pinned_bytes"] < budget * 0.95
       and time.time() - t0 < 120):
    time.sleep(0.5)
for i in (0, 1, 2):
    o, s = live[i]
    e.resume(s)
    for _ in range(30):
        o += e.step()
    t0 = time.time()
    for _ in range(60):
        o += e.step()
    print(f"agent {i} ({'attached' if i and SHARE else 'own rows'}, {len(agents[i]) + len(o)} tokens): "
          f"{(time.time() - t0) / 60 * 1e3:.2f} ms/step", flush=True)
for o, s in live:
    e.release(s)

# 2. agent 1 (attached) gets a next turn on its live state (reference), then a big prompt evicts
# every agent (and the unreferenced prefix region); the same turn again is restored from an
# archive that holds only its own rows (+ the prompt-end snapshot), re-attached to the shared
# region, and must decode the same tokens
a1_out, a1_slot = live[1]
cont = agents[1] + a1_out + tok.encode("<|im_end|>\n<|im_start|>user\nAnd one more sentence.<|im_end|>\n"
                                        "<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
ref_c, rc0, _ = alone(cont)
print(f"agent 1 next turn on the live state: reused {rc0}/{len(cont)} (live end {len(agents[1]) + len(a1_out) - 1})")
ok &= rc0 >= len(agents[1]) + len(a1_out) - 1
big = chat("You are a summarizer.", filler(77, 80000))[:94000]  # evicts every agent and the unreferenced prefix region
t, r, s = e.start(big, 3, 0x9, 0)
e.step()
e.release(s)
probe = e.probe_cached(cont)
out_c, rc, sc = alone(cont)
print(f"agent 1's turn after eviction: probe {probe}, reused {rc}/{len(cont)}, from archive {e.last_from_archive()}, "
      f"copied {e.last_copied()}", flush=True)
ok &= rc >= len(agents[1]) + len(a1_out) - 1 and probe == rc and e.last_from_archive()  # at least the live end
# 3. a new agent of the kind now attaches to the shared region (re-materialized) or copies the prefix
new = chat(SYS, filler(99, 1800) + "\nTask 9: explain the code above in two sentences.")
out_n, rn, sn = alone(new)
print(f"new agent: reused {rn} (pin {pin}), from archive {e.last_from_archive()}, copied {e.last_copied()}")
ok &= rn >= pin
e.reset()
ref_n, _, _ = alone(new)
dc = next((j for j in range(N) if out_c[j] != ref_c[j]), N)
dn = next((j for j in range(N) if out_n[j] != ref_n[j]), N)
print(f"restored turn {dc}/{N} identical to the live one, new agent {dn}/{N} identical to from scratch")
ok &= dc == N  # the same rows (archive) and the same new-row prefill: exact
if dn < N:
    pending.append(("new agent", new, ref_n, dn, agents[0]))
for label, ids, seq, d, creator in pending:
    if creator is None:  # the creator of the prefix: the same computation as alone, must be exact
        print(f"{label}: diverges at {d} without attaching")
        ok = False
        continue
    ok &= near_tie(label, ids, seq, d, creator)
print("PASS" if ok else "FAIL")
e.close()
