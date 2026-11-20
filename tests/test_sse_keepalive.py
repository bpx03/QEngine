"""CPU test: streamed replies get an SSE keep-alive while nothing else is sent (a request queued, its prompt prefilled,
a reply parked / waiting for its turn): OpenAI streams the comment ": keep-alive", Anthropic a ping event; the reply's
events and text are unchanged, non-streamed replies and QE_SSE_KEEPALIVE=0 send none.

A real uvicorn server over the DSV4 adapter with the slow fake DSV4 library of test_disconnect (a ~45K-token prompt
takes 5 x 0.15 s of prefill), keep-alive every 0.1 s. Needs the Maya-S GGUF tokenizer (skipped without it).

  python3 tests/test_sse_keepalive.py
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
import test_disconnect as D  # noqa: E402  (test_dsv4_shared stubs + SlowCLib)

T = D.T
FAILS = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + str(info) if info else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def raw_request(port, api, stream, text, max_tokens=16):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", D.path(api), json.dumps(D.body(api, stream, text, max_tokens)), {"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read().decode()
    c.close()
    return r.status, raw


def stream_text(api, raw):
    txt = ""
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if api == "anthropic":
            if d.get("type") == "content_block_delta" and d["delta"].get("type") == "text_delta":
                txt += d["delta"]["text"]
        elif d.get("choices"):
            txt += d["choices"][0]["delta"].get("content") or ""
    return txt


def main():
    if not os.path.isdir(T.MAYA):
        print(f"SKIP: no DSV4 GGUF dir at {T.MAYA} (QE_DSV4_MODEL)")
        return 0
    T.FakeCLib = D.SlowCLib
    app, gen, _, f = T.make(["--max-ctx", "131072", "--prefill-chunk", "10240", "--max-active", "2"])
    T.TOK = gen.tok
    app.SSE_KEEPALIVE_S = 0.1
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
    long_text = "alpha " * 45000
    for api in ("openai", "anthropic"):
        ka = ": keep-alive" if api == "openai" else "event: ping"
        # 1. a long prompt (~0.75 s of prefill, nothing to send): keep-alives first, then the reply as before
        st, raw = raw_request(port, api, True, f"{api} ka. " + long_text)
        n = sum(1 for ln in raw.splitlines() if ln.startswith(ka))
        first_data = next((i for i, ln in enumerate(raw.splitlines()) if ln.startswith("data: ")), -1)
        first_ka = next((i for i, ln in enumerate(raw.splitlines()) if ln.startswith(ka)), -1)
        check(f"{api}: keep-alives during the prefill", st == 200 and n >= 3, (st, n))
        if api == "openai":
            check(f"{api}: keep-alives before the first data event", 0 <= first_ka < first_data, (first_ka, first_data))
            d0 = json.loads(raw.splitlines()[first_data][6:])
            check(f"{api}: the first data event is still the role chunk", d0["choices"][0]["delta"].get("role") == "assistant")
            check(f"{api}: [DONE] last", raw.rstrip().endswith("data: [DONE]"))
        else:
            check(f"{api}: ping events parse as JSON",
                  all(json.loads(b.split("data: ", 1)[1]) == {"type": "ping"} for b in raw.split("\n\n") if b.startswith(ka)))
            check(f"{api}: message_stop last", raw.rstrip().endswith('{"type": "message_stop"}'))
        # the same prompt with keep-alives off: the same reply text, no keep-alive
        app.SSE_KEEPALIVE_S = 0.0
        st2, raw2 = raw_request(port, api, True, f"{api} off. " + long_text)   # (own text: a full prefill, nothing cached)
        app.SSE_KEEPALIVE_S = 0.1
        check(f"{api}: off (0): none sent through a full prefill", st2 == 200 and ka not in raw2)
        check(f"{api}: same reply text with and without", stream_text(api, raw) == stream_text(api, raw2) != "",
              (stream_text(api, raw)[:40], stream_text(api, raw2)[:40]))
        # 2. non-streamed: plain JSON
        st3, raw3 = raw_request(port, api, False, f"{api} ns. " + long_text)
        try:
            json.loads(raw3)
            ok = True
        except Exception:
            ok = False
        check(f"{api}: non-streamed reply is plain JSON", st3 == 200 and ok, raw3[:80])
    # 3. review M1: a streamed tool call, held back by the parser for its whole decode (tokens flow, nothing is written)
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}}]
    app.SSE_KEEPALIVE_S = 0.03
    D.SlowCLib.PASS_S = 0.01
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", "/v1/chat/completions", json.dumps({
        "messages": [{"role": "user", "content": "What is the weather in Warsaw right now? Use the tool."}], "tools": tools,
        "max_tokens": 300, "temperature": 0, "stream": True, "reasoning_effort": "none"}), {"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read().decode()
    c.close()
    lines = raw.splitlines()
    calls = [d for ln in lines if ln.startswith("data: {") for d in [json.loads(ln[6:])]
             if (d.get("choices") or [{}])[0].get("delta", {}).get("tool_calls")]
    i_call = next((i for i, ln in enumerate(lines) if ln.startswith("data: {") and '"tool_calls"' in ln), -1)
    n_before = sum(1 for ln in lines[:max(i_call, 0)] if ln.startswith(": keep-alive"))
    check("openai: keep-alives while a tool call is held back", r.status == 200 and len(calls) == 1 and n_before >= 2,
          (r.status, len(calls), n_before))
    srv.should_exit = True
    print("ok" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
