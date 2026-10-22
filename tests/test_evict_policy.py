"""Value-based eviction (branch cache-priority): a main agent (prio 2) with a long context waits
while parallel subagents fill the KV pool; finished subagents (prio 0) then fill the RAM
archive. With a pool holding ~4 contexts and an archive holding ~2, LRU evicts the main agent
from the GPU first and then pushes its archive out behind the finished subagents; the policy
must keep it in the archive, so that when it returns it resumes from there with (nearly)
everything cached. Also: the system-prompt prefix of the subagents is pinned, so a new subagent
of that kind starts from it even after every slot that held it was overwritten; probe_cached()
predicts the reuse of start(); greedy continuation after the restore equals a fresh computation.
usage: python3 tests/test_evict_policy.py [--lru]
  --lru: every conversation prio 1 (the old behaviour: the main agent is expected to be lost)"""
import os, sys, time, random
os.environ.setdefault("QE_POOL_TOKENS", "65536")
os.environ.setdefault("QE_MAX_SEQS", "6")
os.environ.setdefault("QE_POOL_HEADROOM", "256")
os.environ.setdefault("QE_POOL_GROW_FRAC", "0")
os.environ.setdefault("QE_KV_ARCHIVE_GB", "4")
os.environ.setdefault("QE_MTP_ADAPT", "0")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from engine import Engine
from tokenizers import Tokenizer

LRU = "--lru" in sys.argv
MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
text = open(os.path.join(ROOT, "tests", "data", "cache_text.txt")).read()
W = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()


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


SYS_MAIN = "You are the main coding agent. Project notes:\n" + text[:4000]
SYS_SUB = "You are a research subagent. Tool descriptions:\n" + text[4000:12000]
SYS_OTHER = "You are a reviewer agent. Guidelines:\n" + text[12000:20000]
main = chat(SYS_MAIN, filler(1, 13000) + "\nSummarize the notes in one sentence.")
subs = [chat(SYS_SUB, filler(10 + i, 6800) + f"\nTask {i}: report the first word.") for i in range(5)]
others = [chat(SYS_OTHER, filler(30 + i, 6800) + f"\nReview {i}: report the last word.") for i in range(3)]
e = Engine(MODEL, max_t=262144)
e.set_sampling(0)
e.set_concurrent(True)
N = 24
P = lambda p: 1 if LRU else p
print(f"pool {e.pool_tokens} tokens; main {len(main)} (system part {sys_end(main)}), subagents {[len(s) for s in subs]} "
      f"(system part {sys_end(subs[0])}), archive {os.environ['QE_KV_ARCHIVE_GB']} GB, {'LRU' if LRU else 'value-based'}",
      flush=True)


def start(ids, prio, affinity):
    t0 = time.time()
    t, reused, s = e.start(ids, 3, affinity, sys_end(ids))
    e.set_prio(s, P(prio))
    return [t], reused, s, time.time() - t0, e.last_from_archive(), e.last_copied()


def run(ids, prio, affinity, done_prio=None):
    out, reused, s, dt, arch, cp = start(ids, prio, affinity)
    while len(out) < N:
        out += e.step()
    if done_prio is not None:
        e.set_prio(s, P(done_prio))
    e.release(s)
    return out, reused, s, dt, arch, cp  # every generated token (the engine processed them all)


def parallel(prompts, prio, affinity, done_prio, keep=()):
    """start all (they decode together with the ones in `keep`), then they finish and are released"""
    live = list(keep)
    for ids in prompts:
        out, reused, s, dt, arch, cp = start(ids, prio, affinity)
        print(f"  agent {len(live)}: slot {s} reused {reused} (from archive {arch}, copied {cp}) {dt * 1e3:.0f} ms", flush=True)
        live.append((out, s))
    while min(len(o) for o, _ in live) < N:
        for (o, _), r in zip(live, e.step_batch([s for _, s in live])):
            o += r
    return live


def finish(live, done_prio):
    for _, s in live:
        e.set_prio(s, P(done_prio))
        e.release(s)


out_main, r, s, dt, arch, cp = run(main, 2, 0x1)
print(f"main: slot {s} reused {r} ({dt * 1e3:.0f} ms)")
print("3 subagents in parallel (they fit next to the idle main agent):")
live = parallel(subs[:3], 1, 0x2, 0)
print("2 more while those run (the main agent is the only idle context: evicted to the archive):")
live = parallel(subs[3:], 1, 0x2, 0, keep=live)
finish(live, 0)
print("3 reviewers (the finished subagents are evicted to the archive, which cannot hold them all):")
finish(parallel(others, 1, 0x3, 0), 0)
info = e.pool_info()
print(f"  archives in RAM: {info['archives']['ram']} ({info['archives']['ram_bytes'] / 1e9:.2f} GB), pinned {e.num_pinned()}")
cont = main + out_main + tok.encode("<|im_end|>\n<|im_start|>user\n<task-notification>done</task-notification>"
                                    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
probe = e.probe_cached(cont)
o1, r, s, dt, arch, cp = run(cont, 2, 0x1)
print(f"main returns: probe {probe}, reused {r}/{len(cont)} (from archive {arch}), start {dt * 1e3:.0f} ms, slot {s}")
ok_main = r >= len(main) + len(out_main) - 1 and probe == r
# a new subagent of the known kind: every slot that held its system part was overwritten, so it
# comes from the pinned prefix
new_sub = chat(SYS_SUB, filler(99, 9000) + "\nTask 9: report the last word.")
probe2 = e.probe_cached(new_sub)
o2, r2, s2, dt2, arch2, cp2 = run(new_sub, 1, 0x2)
pin = sys_end(new_sub)
print(f"new subagent: probe {probe2}, reused {r2} (pinned prefix at {pin}; from archive {arch2}, copied {cp2}), "
      f"start {dt2 * 1e3:.0f} ms")
ok_sub = r2 >= pin and probe2 == r2
# correctness: the restored continuations equal fresh computations
e.reset()
of, rf, _, _, _, _ = run(cont, 2, 0x1)
d = next((j for j in range(N) if o1[j] != of[j]), N)
of2, _, _, _, _, _ = run(new_sub, 1, 0x2)
d2 = next((j for j in range(N) if o2[j] != of2[j]), N)
print(f"restored vs fresh continuation: main {d}/{N}, new subagent {d2}/{N} tokens identical")
ok = ok_main and ok_sub and d == N and d2 == N
if LRU:
    print("(LRU mode: main expected to be lost)", "lost" if r < len(main) else "kept")
    ok = d == N and d2 == N
print("PASS" if ok else "FAIL")
e.close()
