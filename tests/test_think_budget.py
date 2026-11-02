"""CPU test of the hard thinking budget (server/think_budget.py) on the mock engine with the grammar tier
(tests/mock_engine.c -DMOCK_GRAMMAR): request fields and aliases, the forced close at the budget, the answer tail
when max_tokens would end inside <think>, streaming split, tool call after a forced close, /api/stats counters.
No GPU.

  python tests/test_think_budget.py           # Flash-Next model directory (tokenizer / template)
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
_nv = types.ModuleType("pynvml")


def _no_nvml():
    raise RuntimeError("NVML disabled in the mock test")


_nv.nvmlInit = _no_nvml
sys.modules["pynvml"] = _nv

MODEL = os.environ.get("QE_MOCK_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
TMP = tempfile.mkdtemp(prefix="qe_think_")
os.environ["QE_DIAG_LOG"] = os.path.join(TMP, "diag.log")
os.environ["QE_WARMUP"] = "0"

import app  # noqa: E402
import think_budget  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

TOOLS_OAI = [{"type": "function", "function": {"name": "Bash", "description": "Run a shell command",
              "parameters": {"type": "object", "required": ["command"],
                             "properties": {"command": {"type": "string"}}}}}]
THINK = "".join(f"Step {i}: check case {i * 7 % 13} against rule {i % 5}, then note result {i * 3}.\n"
                for i in range(1, 40))  # ~800 tokens, not periodic (loop guard)
ANSWER = "The answer is 42."
R_LONG = THINK + "\n</think>\n\n" + ANSWER + "<|im_end|>"
R_TOOL = THINK + "\n</think>\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n" \
                 "</function>\n</tool_call><|im_end|>"
LONG_ANSWER = " ".join(f"Point {i} is fine." for i in range(1, 120))
R_LONGANS = THINK + "\n</think>\n\n" + LONG_ANSWER + "<|im_end|>"
R_SHORT = "Easy.\n</think>\n\n" + ANSWER + "<|im_end|>"
R_THINK_EOS = THINK + "<|im_end|>"                 # end of turn inside <think>: reasoning only
R_CLOSED_EMPTY = THINK + "\n</think>\n\n<|im_end|>"  # closes the thinking, then ends the turn: no answer


def sse(text):
    out = []
    for block in text.split("\n\n"):
        data = [l[6:] for l in block.splitlines() if l.startswith("data: ")]
        if data and data[0] != "[DONE]":
            out.append(json.loads(data[0]))
    return out


def main():
    so = os.path.join(TMP, "mockg.so")
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-DMOCK_GRAMMAR", "-o", so,
                           os.path.join(HERE, "mock_engine.c")])
    log = os.path.join(TMP, "log.txt")
    script = os.path.join(TMP, "script.txt")
    # unit: request fields
    rb = think_budget.request_budget
    assert rb({}, {}) is None and rb({}, {}, 300) == 300
    assert rb({"thinking_budget": 100}, {}, 300) == 100
    assert rb({"thinking_token_budget": 101}, {}) == 101 and rb({"max_thinking_tokens": 102}, {}) == 102
    assert rb({"custom_params": {"thinking_budget": 103}}, {}) == 103
    assert rb({"reasoning": {"max_tokens": 104}}, {}) == 104
    assert rb({"thinking": {"type": "enabled", "budget_tokens": 105}}, {}) == 105
    assert rb({"thinking": {"type": "disabled", "budget_tokens": 105}}, {}) is None
    kw = {"thinking_budget": 106}
    assert rb({"thinking_budget": 1}, kw) == 106 and "thinking_budget" not in kw  # kwargs first, popped
    assert rb({"thinking_budget": -1}, {}, 300) is None  # explicit off over the server default
    kw = {}
    assert rb({"thinking_budget": 0}, kw, 300) is None and kw["enable_thinking"] is False
    for bad in ("x", True, [1]):
        try:
            rb({"thinking_budget": bad}, {})
            raise AssertionError(bad)
        except think_budget.BudgetError:
            pass
    print("  request fields / aliases ok")

    args = app.parse_args(["--model", MODEL, "--engine-lib", so, "--metrics-port", "0",
                           "--stats-file", os.path.join(TMP, "stats.json")])
    tok = args.profile.tokenizer()[0]
    order = [R_LONG,      # 1 OpenAI budget 64
             R_LONG,      # 2 no budget (reference)
             R_LONG,      # 3 Anthropic stream budget_tokens 100
             R_TOOL,      # 4 OpenAI stream tool call, budget 32
             R_LONG,      # 5 max_tokens 300, no budget: answer tail
             R_LONGANS,   # 6 max_tokens 300: answer cut -> length
             R_SHORT,     # 7 budget larger than the thinking: no forced close
             R_LONG,      # 8 budget 0 -> no thinking
             R_LONG,      # 9 max_tokens 5: bare close
             R_THINK_EOS,     # 10 empty reply: EOS inside <think>
             R_CLOSED_EMPTY,  # 11 empty reply (Anthropic stream): </think> then EOS
             R_LONG,          # 12 empty reply: max_tokens 50, answer tail off -> length
             R_LONG,          # 13 normal turn: no warning
             R_TOOL,          # 14 tool call only: no warning
             R_SHORT, R_SHORT, R_SHORT, R_SHORT,  # 15-18 history with reasoning-only turns (HF #50 log)
             R_LONG]          # 19 max_tokens 300 + ignore_eos: no answer tail
    with open(script, "w") as f:
        for r in order:
            f.write(",".join(map(str, tok.encode(r, add_special_tokens=False).ids)) + "\n")
    os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, log
    gen = app.Generator(args)
    assert gen.can_force, "mock with the grammar tier must enforce"
    c = TestClient(app.build_app(gen, args))
    name = args.served_name
    ti = gen.think_ids
    wrap_n = len(ti.wrap[""])
    msgs = [{"role": "user", "content": "Solve it."}]

    def klines():
        return [l.split() for l in open(log) if l[0] in "KC"]

    # 1. OpenAI budget 64: forced close, answer, reasoning tokens <= budget + one step (3) + wrap-up
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "thinking_budget": 64}).json()
    m = r["choices"][0]["message"]
    assert m["content"] == ANSWER and r["choices"][0]["finish_reason"] == "stop", r
    assert m["reasoning_content"].endswith(think_budget.WRAP), m["reasoning_content"][-200:]
    assert m["reasoning_content"].endswith("\n\n" + think_budget.WRAP), m["reasoning_content"][-200:]  # own paragraph
    rt = r["usage"]["completion_tokens_details"]["reasoning_tokens"]
    assert 64 <= rt <= 64 + 3 + wrap_n, (rt, wrap_n)
    k = klines()
    assert k[0] == ["K", "1"] and k[1] == ["K", "0"] and k[2][0] == "C", k
    print(f"  openai budget 64: {rt} reasoning tokens (wrap-up {wrap_n}), {k[2][1]} constrained steps, answer ok")

    # 2. no budget: the model's own thinking, untouched
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs}).json()
    m2 = r["choices"][0]["message"]
    assert m2["content"] == ANSWER and m2["reasoning_content"] == THINK.strip(), m2
    assert len(klines()) == 3, klines()
    print(f"  no budget: {r['usage']['completion_tokens_details']['reasoning_tokens']} reasoning tokens, no forcing")

    # 3. Anthropic streaming, thinking.budget_tokens 100: thinking block, then the text block
    t = c.post("/v1/messages", json={"model": name, "max_tokens": 4000, "stream": True, "messages": msgs,
                                     "thinking": {"type": "enabled", "budget_tokens": 100}}).text
    evs = [json.loads(l[6:]) for l in t.splitlines() if l.startswith("data: ")]
    starts = [e["content_block"]["type"] for e in evs if e["type"] == "content_block_start"]
    th = "".join(e["delta"].get("thinking", "") for e in evs if e["type"] == "content_block_delta")
    tx = "".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta")
    md = [e for e in evs if e["type"] == "message_delta"][0]
    assert starts == ["thinking", "text"] and tx == ANSWER and "</think>" not in th + tx, (starts, tx)
    assert th.rstrip().endswith(think_budget.WRAP) and md["delta"]["stop_reason"] == "end_turn", md
    print("  anthropic stream budget 100: thinking -> text split ok")

    # 4. OpenAI streaming tool call after a forced close (budget 32)
    t = c.post("/v1/chat/completions", json={"model": name, "stream": True, "tools": TOOLS_OAI, "messages": msgs,
                                             "chat_template_kwargs": {"thinking_budget": 32}}).text
    ev = sse(t)
    calls = [d for e in ev for ch in e["choices"] for d in [ch["delta"]] if "tool_calls" in d]
    rc = "".join(ch["delta"].get("reasoning_content", "") for e in ev for ch in e["choices"])
    assert len(calls) == 1 and json.loads(calls[0]["tool_calls"][0]["function"]["arguments"]) == {"command": "ls -la"}
    assert rc.rstrip().endswith(think_budget.WRAP), rc[-200:]
    assert [ch["finish_reason"] for e in ev for ch in e["choices"] if ch["finish_reason"]] == ["tool_calls"]
    print("  openai stream budget 32: tool call ok after the forced close")

    # 5. max_tokens 300, no budget: the thinking (~800 tokens) closes early, the answer fits
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "max_tokens": 300}).json()
    m = r["choices"][0]["message"]
    assert m["content"] == ANSWER and r["choices"][0]["finish_reason"] == "stop", r
    assert r["usage"]["completion_tokens"] <= 300, r["usage"]
    tail = think_budget.answer_tail(300)
    # the answer keeps the tail, less at most one step's overshoot (3 tokens per mock step)
    assert r["usage"]["completion_tokens_details"]["reasoning_tokens"] <= 300 - tail + 2, r["usage"]
    print(f"  max_tokens 300: thinking closed for the answer tail ({tail}), {r['usage']['completion_tokens']} tokens")

    # 6. max_tokens 300, the answer itself is longer than the tail: finish_reason length, the answer is there
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "max_tokens": 300}).json()
    m = r["choices"][0]["message"]
    assert r["choices"][0]["finish_reason"] == "length" and m["content"].startswith("Point 1 is fine."), r
    assert r["usage"]["completion_tokens"] == 300, r["usage"]
    print("  max_tokens 300, long answer: finish length, answer present")

    # 7. budget above the thinking: no forced close
    n0 = len(klines())
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "thinking_budget": 500}).json()
    assert r["choices"][0]["message"]["content"] == ANSWER and len(klines()) == n0
    # 8. budget 0: no thinking at all (the template's empty think block)
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "thinking_budget": 0}).json()
    p = [l for l in open(log) if l.startswith("P ")][-1].split()
    ptxt = tok.decode([int(x) for x in p[3:]], skip_special_tokens=False)
    assert ptxt.endswith("<think>\n\n</think>\n\n"), ptxt[-60:]
    # 9. max_tokens 6: the bare close (3 tokens) after the first token, then the answer (the mock repeats its "\n\n")
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "max_tokens": 6}).json()
    assert r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 6, r
    assert r["choices"][0]["message"]["content"], r
    # bad value: 400
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "thinking_budget": "lots"})
    assert r.status_code == 400 and "thinking_budget" in r.text, r.text
    r = c.post("/v1/messages", json={"model": name, "max_tokens": 10, "messages": msgs,
                                     "thinking": {"type": "enabled", "budget_tokens": "x"}})
    assert r.status_code == 400, r.text
    st = gen.metrics.snapshot()["thinking"]
    assert st["forced_budget"] == 3 and st["forced_max_tokens"] == 3 and st["forced_failed"] == 0, st
    assert st["requests_with_budget"] == 4, st
    assert st["empty_replies"] == 0, st  # nothing above was an empty turn (tool call, bare close included)
    print("  no-force above budget, budget 0, max_tokens 6, 400s, /api/stats", st)

    # empty replies: one WARNING line each, with the slot / conversation / finish / thinking / limits / tail
    def logged(fn):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = fn()
        return r, [l for l in buf.getvalue().splitlines() if "WARNING empty reply" in l]

    # (greedy: a sampled empty reply would be sampled again, --empty-retries; tests/test_server_mock.py)
    r, w = logged(lambda: c.post("/v1/chat/completions", json={"model": name, "messages": msgs,
                                                               "temperature": 0}).json())
    assert r["choices"][0]["message"]["content"] in ("", None) and r["choices"][0]["finish_reason"] == "stop", r
    assert len(w) == 1 and "finish stop;" in w[0] and "</think> no" in w[0] and "reasoning only" in w[0], w
    assert "slot 0, conversation " in w[0] and "max_tokens " in w[0] and "(context room)" in w[0], w
    assert "thinking budget none" in w[0] and w[0].endswith("thinking ends ' note result 117.\\n'"), w  # the last thinking tokens
    print("  empty (EOS inside <think>):", w[0])
    t, w = logged(lambda: c.post("/v1/messages", json={"model": name, "max_tokens": 4000, "stream": True,
                                                        "messages": msgs}).text)
    assert len(w) == 1 and "finish stop;" in w[0] and "</think> yes" in w[0] and "max_tokens 4000," in w[0], w
    print("  empty (</think> then EOS, anthropic stream):", w[0])
    think_budget.TAIL = 0  # QE_THINK_TAIL=0: max_tokens ends inside <think>
    try:
        r, w = logged(lambda: c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "stream": True,
                                                                   "max_tokens": 50}).text)
    finally:
        think_budget.TAIL = -1
    assert len(w) == 1 and "finish length;" in w[0] and "</think> no" in w[0] and "50 thinking tokens of 50" in w[0], w
    print("  empty (max_tokens 50, tail off, openai stream):", w[0])
    _, w = logged(lambda: c.post("/v1/chat/completions", json={"model": name, "messages": msgs}).json())
    assert not w, w
    _, w = logged(lambda: c.post("/v1/chat/completions", json={"model": name, "messages": msgs,
                                                               "tools": TOOLS_OAI}).json())
    assert not w, w
    st = gen.metrics.snapshot()["thinking"]
    assert st["empty_replies"] == 3 and st["empty_stop"] == 2 and st["empty_length"] == 1, st
    assert st["empty_after_forced_close"] == 0, st
    print("  normal turn / tool call: no warning; /api/stats", st)

    # HF #50: the client resends reasoning-only turns with every request: one log line per conversation, again
    # only when the count grows
    empty = {"role": "assistant", "content": "", "reasoning_content": "Thinking only."}
    hist = [{"role": "system", "content": "You are a helper."}, {"role": "user", "content": "Start."}, empty,
            {"role": "user", "content": "Go on."}, empty, {"role": "user", "content": "And?"}]

    def drop_lines(h):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = c.post("/v1/chat/completions", json={"model": name, "messages": h}).json()
        assert r["choices"][0]["message"]["content"] == ANSWER, r
        return [l for l in buf.getvalue().splitlines() if l.startswith("history:")]

    h1 = drop_lines(hist)
    assert len(h1) == 1 and "has 2 assistant turn(s) with reasoning only; left out" in h1[0], h1
    assert "--drop-empty-turns 0 keeps them" in h1[0] and "conversation " in h1[0], h1
    hist2 = hist + [{"role": "assistant", "content": ANSWER}, {"role": "user", "content": "More."}]
    assert drop_lines(hist2) == []  # the same conversation, the same two turns: no line
    hist3 = hist2 + [empty, {"role": "user", "content": "Last."}]
    h3 = drop_lines(hist3)
    assert len(h3) == 1 and "has 3 assistant turn(s)" in h3[0] and "(was 2)" in h3[0], h3
    assert drop_lines(hist3 + [{"role": "assistant", "content": ANSWER}, {"role": "user", "content": "x"}]) == []
    snap = gen.metrics.snapshot()
    hs = snap["history_drops"]
    assert hs["requests"] == 4 and hs["turns"] == 10 and hs["logged"] == 2, hs
    # the dashboards' throughput chart (qetop, dashboard.html) reads "history" as the per-second list
    assert isinstance(snap["history"], list), type(snap["history"])
    print("  HF #50 log:", h1[0], "|", h3[0], "| /api/stats", hs)

    # 19. ignore_eos (benchmarks): exactly max_tokens of raw continuation, no forced close for the answer tail
    nk = len(klines())
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "max_tokens": 300,
                                             "ignore_eos": True}).json()
    u = r["usage"]
    assert r["choices"][0]["finish_reason"] == "length" and u["completion_tokens"] == 300, r
    assert u["completion_tokens_details"]["reasoning_tokens"] == 300, u
    assert not any(k[0] == "C" for k in klines()[nk:]), klines()[nk:]
    print("  ignore_eos max_tokens 300: no answer tail, 300 reasoning tokens")
    print("ok")


if __name__ == "__main__":
    main()
