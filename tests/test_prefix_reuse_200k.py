"""Prompt-cache reuse of a 200K+ prompt with other traffic alongside (GPU validation of
fix-prefix-reuse): cold 205K prompt, then (with other requests in between) a growing turn, an
identical repeat, another tail diverging ~1500 tokens before the end, and two Claude Code
compaction-style requests (conversation + summarize instruction, with / without the last reply). Prints cached tokens
and wall time per request. usage: python tests/test_prefix_reuse_200k.py [base_url] [model] [words]"""
import random, sys, time
import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qengine"
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 205000
rnd = random.Random(77)
WORDS = ("cache state token layer prefix kernel memory batch window queue model vector index chapter "
         "section engine latency stream buffer region archive snapshot request schema answer").split()
words = lambda n: " ".join(rnd.choice(WORDS) for _ in range(n))


def ask(msgs, max_tokens=8):
    t0 = time.time()
    r = requests.post(BASE + "/v1/chat/completions", timeout=3600, json={
        "model": MODEL, "temperature": 0.0, "max_tokens": max_tokens, "messages": msgs})
    r.raise_for_status()
    d = r.json()
    u = d["usage"]
    return u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), \
        d["choices"][0]["message"].get("content") or "", time.time() - t0


def other_traffic():
    for i in range(4):
        ask([{"role": "user", "content": f"Other {i}: " + words(6000)}], 4)
    doc = words(12000)
    for i in range(2):
        ask([{"role": "user", "content": doc + "\n\nChunk: " + words(300)}], 4)


ok = True
sysm = {"role": "system", "content": "You are a coding agent."}
doc = words(NW)
m1 = [sysm, {"role": "user", "content": doc + "\n\nQuestion one?"}]
p, c, out1, t = ask(m1, 16)
print(f"cold:            prompt {p} cached {c} {t:.1f}s", flush=True)
other_traffic()
m2 = m1 + [{"role": "assistant", "content": out1}, {"role": "user", "content": words(3000) + "\nNext?"}]
p2, c2, out2, t2 = ask(m2)
print(f"growing turn:    prompt {p2} cached {c2} {t2:.1f}s", flush=True)
ok &= c2 >= p - 1
other_traffic()
p3, c3, _, t3 = ask(m2)
print(f"identical:       prompt {p3} cached {c3} {t3:.1f}s", flush=True)
ok &= c3 >= p3 - 2048
other_traffic()
tail = m2[-1]["content"]
m4 = m2[:-1] + [{"role": "user", "content": tail[:-9000] + words(800)}]
p4, c4, _, t4 = ask(m4)
print(f"other tail:      prompt {p4} cached {c4} {t4:.1f}s", flush=True)
ok &= c4 >= p4 - 4096
# Claude Code compaction near the max context: the whole conversation (+ the last reply) and a
# summarize instruction; the conversation's slot was taken over by the other-tail request, so its
# state comes from the archive / snapshots
summ = {"role": "user", "content": "Your task is to create a detailed summary of the conversation so far."}
m5 = m2 + [{"role": "assistant", "content": out2}, summ]
p5, c5, _, t5 = ask(m5, 32)
print(f"compaction (+reply): prompt {p5} cached {c5} {t5:.1f}s", flush=True)
ok &= c5 >= p2 - 1
other_traffic()
m6 = m2 + [summ]  # without the last reply: diverges at the last assistant header
p6, c6, _, t6 = ask(m6, 32)
print(f"compaction (no reply): prompt {p6} cached {c6} {t6:.1f}s", flush=True)
ok &= c6 >= p2 - 2048 - 16
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
