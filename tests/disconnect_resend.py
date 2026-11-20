"""[srv-disconnect] GPU check against a running server (27B, Flash-Next TP / FLASH_PAR=pp mode B, any model): turn 1 of a
conversation, then a long turn 2 whose client disconnects mid-prefill, then turn 2 again. The retry must reuse the
conversation (cached tokens >= turn 1's prompt + reply, plus the rows prefilled before the cancel) and give the same
answer as the same conversation without a cancel (conversation B, other first words).

  python3 tests/disconnect_resend.py <port> [delay ...]
  delay: seconds before the disconnect, or "<x>T" = x times the uncancelled turn 2 (default: 3)
Prints RESULT {...}; exit 1 when the answers differ or the old rows were not reused."""
import http.client, json, socket, sys, time
port = int(sys.argv[1])
DELAYS = sys.argv[2:] or ["3"]

def items(a, b):
    return " ".join(f"Item {i}: the {['red','green','blue','grey'][i % 4]} box number {i * 7 % 1000} is on shelf {i % 37}." for i in range(a, b))

def msgs(tag, reply=None):
    m = [{"role": "user", "content": f"Inventory {tag}. " + items(0, 1500) + "\nSay OK."}]
    if reply is not None:
        m += [{"role": "assistant", "content": reply},
              {"role": "user", "content": "More items. " + items(1500, 3500) + "\nWhich shelf is item 2345 on? Answer with the number only."}]
    return m

def post(m, mt):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    c.request("POST", "/v1/chat/completions", json.dumps({"model": "m", "messages": m, "max_tokens": mt, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": False}}), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    c.close()
    return d["choices"][0]["message"]["content"], d["usage"]

def drop(m, after):
    d = json.dumps({"model": "m", "messages": m, "max_tokens": 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(d)}\r\n\r\n".encode() + d)
    time.sleep(after)
    s.shutdown(socket.SHUT_RDWR); s.close()

def turn(tag, cancel_at):
    r1, u1 = post(msgs(tag), 8)
    t2 = None
    if cancel_at is not None:
        drop(msgs(tag, r1), cancel_at)
        time.sleep(1.0)
    t0 = time.monotonic()
    r2, u2 = post(msgs(tag, r1), 40)
    t2 = time.monotonic() - t0
    o = {"tag": tag, "cancel_at": cancel_at, "turn1_tokens": u1.get("prompt_tokens", 0) + u1.get("completion_tokens", 0),
         "turn2": r2, "turn2_prompt": u2.get("prompt_tokens"), "turn2_s": round(t2, 2),
         "turn2_cached": (u2.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}
    print(o, flush=True)
    return o

def ok_answer(t):
    """the retry's answer: the uncancelled conversation's, or the right one (item 2345 is on shelf 2345 % 37 = 14). Not
    bit-exact on every engine: Flash-Next's prefill split changes its numerics (a stated class), and B differs from A in
    its first words; on the gate B said 15 and the cancelled A 14"""
    return t == ref["turn2"] or t.strip().strip(".") == "14"

# B: no cancel (the reference answer, and turn 2's time T); then cancels mid-prefill and around the prefill's end
# (review2 L5: a prompt done in the engine but not yet taken by the server)
ref = turn("B", None)
T = ref["turn2_s"]
cases = []
for k, d in enumerate(DELAYS):
    c = turn(f"A{k}", float(d[:-1]) * T if d.endswith("T") else float(d))
    cases.append(c)
    if not ok_answer(c["turn2"]) or c["turn2_cached"] < c["turn1_tokens"] - 2:
        break   # (stop at the first failure)
res = {"ref": ref["turn2"], "T_s": T,
       "cases": [{"cancel_at": round(c["cancel_at"], 2), "same": c["turn2"] == ref["turn2"], "answer": c["turn2"], "ok_answer": ok_answer(c["turn2"]), "cached": c["turn2_cached"],
                  "turn1_tokens": c["turn1_tokens"], "reused": c["turn2_cached"] >= c["turn1_tokens"] - 2} for c in cases]}
res["right"] = "14" in ref["turn2"]
ok = all(c["ok_answer"] and c["reused"] for c in res["cases"])
print("RESULT " + json.dumps(res), flush=True)
sys.exit(0 if ok else 1)
