"""[srv-disconnect] CPU test: a client that disconnects while its request is queued, while its prompt is prefilled or
while it decodes is cancelled at the scheduler's next safe point, for streaming and non-streaming requests on both APIs
(OpenAI /v1/chat/completions, Anthropic /v1/messages); the slot is freed and the next request is served.

A real uvicorn server (the disconnect is a closed socket) over the DSV4 adapter (server/dsv4_engine.py) with a slow
fake DSV4 C library (tests/test_dsv4_shared.FakeCLib + prefill chunks that take time and read the cancel flag, as
dsv4_capi does): a prompt alone in service is one start call whose chunks check the flag inside the call; a prompt next
to a decoding request is fed in prompt chunks between scheduler passes (the path every chunked engine takes: 27B,
Flash-Next, gpt-oss, GLM windows). Needs the Maya-S GGUF tokenizer (skipped without it).

  python3 tests/test_disconnect.py
"""
import http.client
import json
import os
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_dsv4_shared as T  # noqa: E402  (stubs NVML, sets the env, puts server/ on the path)
import dsv4_engine  # noqa: E402

CHUNK, CHUNK_S = 10240, 0.15   # the fake engine's prefill chunk (DSV4's 10,240 rows) and its time
FAILS = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + str(info) if info else ''}", flush=True)
    if not ok:
        FAILS.append(name)


class SlowCLib(T.FakeCLib):
    """FakeCLib whose prefill takes CHUNK_S per CHUNK rows and stops at a chunk start once the request's cancel flag
    (CLib.abort, armed around start / prefill_part) is set: the engine keeps the prefix done and the call raises
    PrefillCancelled (dsv4_capi's -5)"""
    PASS_S = 0.004
    has_abort = True

    def __init__(self, lib, args):
        super().__init__(lib, args)
        self.abort, self.aborts, self.rows = None, 0, 0
        self.on_lost = None

    def _feed(self, ids, part):
        reuse = 0
        if not self.nocache:
            reuse, _ = self._reuse(ids)
            reuse = min(reuse, len(ids) - (0 if part else 1))
        self.nocache = False
        self._save_held()
        for s0 in range(0, max(0, len(ids) - reuse), CHUNK):
            if self.abort is not None and self.abort.value:
                self.aborts += 1
                self.toks, self.next, self.live = list(ids[:reuse + s0]), -1, 0
                if self.on_lost:
                    self.on_lost()
                raise dsv4_engine.PrefillCancelled("cancelled while its prompt was prefilled (the client left)")
            time.sleep(CHUNK_S)
            self.rows += min(CHUNK, len(ids) - reuse - s0)
        return reuse


def body(api, stream, text, max_tokens=64):
    b = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "stream": stream,
         "temperature": 0}
    if api == "anthropic":
        b["model"], b["thinking"] = "x", {"type": "disabled"}
    else:
        b["reasoning_effort"] = "none"
    return b


def path(api):
    return "/v1/messages" if api == "anthropic" else "/v1/chat/completions"


def send_and_drop(port, api, stream, text, after_s, max_tokens=64):
    """POST, then close the socket after after_s without reading the reply (a client that left)"""
    data = json.dumps(body(api, stream, text, max_tokens)).encode()
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(f"POST {path(api)} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    time.sleep(after_s)
    t = time.monotonic()
    s.shutdown(socket.SHUT_RDWR)
    s.close()
    return t


def request(port, api, stream, text, max_tokens=64, out=None):
    """a whole request; (status, reply text, seconds)"""
    t0 = time.monotonic()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", path(api), json.dumps(body(api, stream, text, max_tokens)), {"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read().decode()
    c.close()
    txt = ""
    if stream:
        for line in raw.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if api == "anthropic":
                if d.get("type") == "content_block_delta" and d["delta"].get("type") == "text_delta":
                    txt += d["delta"]["text"]
            elif d.get("choices"):
                txt += d["choices"][0]["delta"].get("content") or ""
    else:
        d = json.loads(raw)
        txt = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text") if api == "anthropic" \
            else (d["choices"][0]["message"]["content"] or "")
    res = (r.status, txt, time.monotonic() - t0)
    if out is not None:
        out.append(res)
    return res


def main():
    if not os.path.isdir(T.MAYA):
        print(f"SKIP: no DSV4 GGUF dir at {T.MAYA} (QE_DSV4_MODEL)")
        return 0
    T.FakeCLib = SlowCLib   # (make() installs T.FakeCLib as dsv4_engine.CLib)
    app, gen, _, f = T.make(["--max-ctx", "131072", "--prefill-chunk", "10240", "--max-active", "2"])
    T.TOK = gen.tok
    notes = []
    real_note = gen.cancel_note

    def note(j, where):
        c = j.cancelled
        if getattr(c, "why", "") == "client":
            notes.append((where, time.monotonic() - c.t, len(j.ids)))
        real_note(j, where)

    gen.cancel_note = note
    import uvicorn
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = uvicorn.Server(uvicorn.Config(app.build_app(gen, gen.args), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    long_text = "alpha " * 45000   # ~45K tokens: 5 fake chunks of 0.15 s alone
    n_long = len(T.TOK.encode(long_text, add_special_tokens=False).ids)
    print(f"long prompt {n_long} tokens: {(n_long + CHUNK - 1) // CHUNK} chunks of {CHUNK_S} s", flush=True)
    want_short = T.want_text("", False)

    def wait_note(k, timeout=10.0):
        t = time.monotonic()
        while len(notes) < k and time.monotonic() - t < timeout:
            time.sleep(0.01)
        return notes[k - 1] if len(notes) >= k else None

    def idle(timeout=3.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not gen.engine.seqs and not gen.queue and not gen.jobs_ref[0] and not gen.jobs_ref[1]:
                return True
            time.sleep(0.01)
        return False

    def next_ok(api, label):
        st, txt, dt = request(port, api, False, "What is the capital of Poland?")
        check(f"{label}: the next request is served", st == 200 and txt.strip() == want_short.strip(), (st, txt[:60], f"{dt:.2f}s"))

    for api in ("openai", "anthropic"):
        for stream in (False, True):
            tag = f"{api} {'stream' if stream else 'non-stream'}"
            # 1. prefill, alone in service: one start call; its chunks read the cancel flag
            k, a0 = len(notes), f.aborts
            send_and_drop(port, api, stream, f"{tag} alone. " + long_text, 0.3)   # (own text: nothing cached)
            n = wait_note(k + 1)
            check(f"{tag} prefill alone: cancelled mid-prefill inside the call",
                  n is not None and "prefilled" in n[0] and n[1] < CHUNK_S + 0.25 and f.aborts == a0 + 1, n)
            check(f"{tag} prefill alone: slot free", idle())
            next_ok(api, f"{tag} prefill alone")
            # 2. prefill next to a decoding request: prompt chunks between scheduler passes (27B / QFN / gpt-oss / GLM)
            k, a0 = len(notes), f.aborts
            res = []
            th = threading.Thread(target=request, args=(port, api, False, "Count to 400.", 2000, res))
            th.start()
            time.sleep(0.15)
            send_and_drop(port, api, stream, f"{tag} beside. " + long_text, 0.35)
            n = wait_note(k + 1)
            check(f"{tag} prefill beside a decode: cancelled at a chunk boundary",
                  n is not None and "prefilled" in n[0] and n[1] < CHUNK_S + 0.25, (n, f.aborts - a0))
            th.join()
            check(f"{tag} prefill beside a decode: the decoding request ends whole",
                  res and res[0][0] == 200 and res[0][1].strip().endswith("399"), res and (res[0][0], res[0][1][-30:]))
            check(f"{tag} prefill beside a decode: slot free", idle())
            # 3. queued: two requests decode (both slots), the third waits; its client leaves
            k = len(notes)
            res = []
            ths = [threading.Thread(target=request, args=(port, api, False, "Count to 400.", 300, res)) for _ in range(2)]
            for t in ths:
                t.start()
            time.sleep(0.15)
            send_and_drop(port, api, stream, "What is the capital of Poland?", 0.15)
            n = wait_note(k + 1)
            check(f"{tag} queued: cancelled in the queue", n is not None and n[0] == "queued" and n[1] < 0.1, n)
            for t in ths:
                t.join()
            check(f"{tag} queued: the running requests end whole", len(res) == 2 and all(r[0] == 200 for r in res),
                  [(r[0], r[1][-20:]) for r in res])
            check(f"{tag} queued: slot free", idle())
            next_ok(api, f"{tag} queued")
            # 4. decode: the client leaves while its reply is generated
            k = len(notes)
            send_and_drop(port, api, stream, "Count to 400.", 0.35, max_tokens=2000)
            n = wait_note(k + 1)
            check(f"{tag} decode: cancelled between steps", n is not None and "generated" in n[0] and n[1] < 0.1, n)
            check(f"{tag} decode: slot free", idle())
            next_ok(api, f"{tag} decode")
    # 5. no false cancel: a normal streamed reply, a normal non-streamed one and two requests on one keep-alive connection
    k, c0 = len(notes), gen.sched.get("cancelled", 0)
    for api in ("openai", "anthropic"):
        for stream in (False, True):
            st, txt, _ = request(port, api, stream, "What is the capital of Poland?")
            check(f"normal {api} {'stream' if stream else 'non-stream'} reply", st == 200 and txt.strip() == want_short.strip(), (st, txt[:40]))
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    for i in range(2):
        c.request("POST", "/v1/chat/completions", json.dumps(body("openai", True, "What is the capital of Poland?")),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        r.read()
        check(f"keep-alive request {i + 1}", r.status == 200)
    c.close()
    time.sleep(0.3)
    check("no cancel counted for normal requests", len(notes) == k and gen.sched.get("cancelled", 0) == c0,
          (notes[k:], gen.sched.get("cancelled", 0) - c0))
    # 6. a cancel is no engine error; a resend of the cancelled prompt reuses the rows done before the cancel
    e0 = gen.metrics.session["errors"]
    text = "resend. " + long_text
    send_and_drop(port, "openai", False, text, 0.5)
    n = wait_note(len(notes) + 1)
    check("cancel while prefilled (resend case)", n is not None and "prefilled" in n[0], n)
    check("a cancel is not counted as an engine error", gen.metrics.session["errors"] == e0, gen.metrics.session["errors"] - e0)
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", "/v1/chat/completions", json.dumps(body("openai", False, text)), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    c.close()
    cached = d.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    check("resend: the rows prefilled before the cancel are reused", cached >= CHUNK, cached)
    # 7. decode cancel with two replies taking turns (DSV4 time slices): the other one ends whole
    k = len(notes)
    res = []
    th = threading.Thread(target=request, args=(port, "openai", False, "Count to 400.", 2000, res))
    th.start()
    time.sleep(0.1)
    send_and_drop(port, "openai", True, "Count to 400 please.", 0.4, max_tokens=2000)
    n = wait_note(k + 1)
    check("two decoding, one leaves: cancelled between steps", n is not None and "generated" in n[0], n)
    th.join()
    check("two decoding, one leaves: the other ends whole", res and res[0][0] == 200 and res[0][1].strip().endswith("399"),
          res and res[0][1][-20:])
    check("two decoding, one leaves: slot free", idle())
    lat = {}
    for where, dt, _ in notes:
        key = "queued" if where == "queued" else "prefill" if "prefilled" in where else "decode"
        lat.setdefault(key, []).append(dt)
    print("cancel latency (disconnect -> slot free), max: " +
          ", ".join(f"{k} {max(v) * 1000:.0f} ms ({len(v)})" for k, v in sorted(lat.items())), flush=True)
    srv.should_exit = True
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
