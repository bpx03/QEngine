"""Prompt-cache reuse against a running server (GPU validation of the prefix-reuse fix). Synthetic text only (fixed seed).
 1. fill the snapshot store with 40 distinct prompts (their prompt-end states),
 2. main's repro: X ending "Question A?", X again, the same text with "Question B?", then "Question C?",
 3. the RAG ingest pattern: one warm-up request with a ~12.5K-token chapter prefix, then 8
    parallel requests sharing it (6 chunk batches + 2 section summaries).
Prints usage.prompt_tokens_details.cached_tokens of every request and checks the expected reuse.
usage: python tests/test_prefix_reuse_http.py [base_url] [model]"""
import random, sys, threading
import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qengine"
SYSTEM = "You help index a technical book for search. Answer briefly."
rnd = random.Random(1234)
WORDS = ("cache state token layer prefix kernel memory batch window queue model vector index chapter "
         "section engine latency stream buffer region archive snapshot request schema answer").split()


def words(n):
    return " ".join(rnd.choice(WORDS) for _ in range(n))


def ask(user, max_tokens=8):
    r = requests.post(BASE + "/v1/chat/completions", timeout=1800, json={
        "model": MODEL, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    d = r.json()
    u = d["usage"]
    return u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), \
        d["choices"][0]["message"].get("content") or ""


ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  ok    " if cond else "  FAIL  ") + msg)


print("1. filling the snapshot store (40 distinct prompts)")
for i in range(40):
    ask(f"Note {i}: " + words(300), 1)

print("2. repeated prompt")
doc = "<document>\n" + words(10000) + "\n</document>\n\n"
p1, c1, o1 = ask(doc + "Question A?", 16)
p2, c2, o2 = ask(doc + "Question A?", 16)
p3, c3, _ = ask(doc + "Question B?")
p4, c4, _ = ask(doc + "Question C?")
for k, (p, c) in enumerate([(p1, c1), (p2, c2), (p3, c3), (p4, c4)], 1):
    print(f"  request {k}: prompt {p} cached {c}")
check(c2 >= p2 - 1, f"identical prompt reuses all but <= 1 token ({c2}/{p2})")
check(o1 == o2, "identical prompt, greedy: the same answer" + ("" if o1 == o2 else f" ({o1!r} vs {o2!r})"))
check(c3 >= 8192, f"other question: at least the 8192 checkpoint ({c3})")
check(c4 >= p4 - 40, f"third question: the divergence checkpoint of the second ({c4}/{p4})")

print("3. RAG ingest pattern: warm-up + 8 parallel")
prefix = "<document title=\"Chapter X\">\n" + words(12000) + "\n</document>\n\n"
chunks = lambda: "Chunks from this document:\n\n" + "\n\n".join(
    f"<chunk n=\"{i}\">\n{words(120)}\n</chunk>" for i in range(1, 4)) + "\n\nFor EACH chunk return a context."
pw, cw, _ = ask(prefix + chunks())
print(f"  warm-up: prompt {pw} cached {cw}")
reqs = [prefix + chunks() for _ in range(6)] + [prefix + f"Summarize the section {s}." for s in "AB"]
res = [None] * len(reqs)


def one(i):
    res[i] = ask(reqs[i])


ts = [threading.Thread(target=one, args=(i,)) for i in range(len(reqs))]
for t in ts:
    t.start()
for t in ts:
    t.join()
for i, (p, c, _) in enumerate(res):
    print(f"  parallel {i + 1} ({'batch' if i < 6 else 'summary'}): prompt {p} cached {c}")
check(all(c >= 8192 for _, c, _ in res), "every parallel request reuses at least the 8192 checkpoint")
near = sum(c >= pw - 600 for _, c, _ in res[:6])
check(near >= 3, f"batches started after the first two reuse ~the whole prefix ({near}/6)")
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
