"""CPU test of the cross-turn loop guard in the server, on the mock engine with the grammar tier (tests/mock_engine.c
-DMOCK_GRAMMAR), --think 0: the guarded turn's prompt (think block, note), the back-off over a stretch, cycles,
the forced reasoning on top of the client's max_tokens, client stop sequences only after the reasoning, the Anthropic
path, conversations the guard must leave alone. No GPU.

  python tests/test_loop_guard_server.py
"""
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
TMP = tempfile.mkdtemp(prefix="qe_guard_")
os.environ["QE_DIAG_LOG"] = os.path.join(TMP, "diag.log")
os.environ["QE_WARMUP"] = "0"

import app  # noqa: E402
import think_budget  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "Bash", "description": "Run a shell command",
          "parameters": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}}}}]
TOOLS_A = [{"name": "Bash", "description": "Run a shell command",
            "input_schema": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}}}]
THINK = "".join(f"Step {i}: check case {i * 7 % 13} against rule {i % 5}, then note result {i * 3}.\n\n"
                for i in range(1, 40))  # ~800 tokens, not periodic; paragraph breaks and "Step" for the stop tests
ANSWER = "The answer is 42."
CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
R_TOOL = THINK + "\n</think>\n\n" + CALL + "<|im_end|>"
R_SHORT = "Easy.\n</think>\n\n" + CALL + "<|im_end|>"
R_ANSWER = THINK + "\n</think>\n\n" + ANSWER + "<|im_end|>"
R_LONGANS = THINK + "\n</think>\n\n" + " ".join(f"Point {i} is fine." for i in range(1, 120)) + "<|im_end|>"
R_PLAIN = CALL + "<|im_end|>"  # a turn generated without thinking
BUDGET = 64
OPEN, EMPTY = "<think>\n", app.EMPTY_THINK


def hist(k, res="Exit code 1"):
    """OpenAI history: k identical (call, result) turns."""
    m = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "find the start time"}]
    for _ in range(k):
        m += [{"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {
            "name": "Bash", "arguments": json.dumps({"command": "ps -p 4242"})}}]},
              {"role": "tool", "tool_call_id": "x", "content": res}]
    return m


def hist_a(k, res="Exit code 1"):
    """Anthropic history: k identical (tool_use, tool_result) turns."""
    m = [{"role": "user", "content": "find the start time"}]
    for i in range(k):
        m += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash",
                                                 "input": {"command": "ps -p 4242"}}]},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": res}]}]
    return m


def main():
    so = os.path.join(TMP, "mockg.so")
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-DMOCK_GRAMMAR", "-o", so,
                           os.path.join(HERE, "mock_engine.c")])
    log, script = os.path.join(TMP, "log.txt"), os.path.join(TMP, "script.txt")
    args = app.parse_args(["--model", MODEL, "--engine-lib", so, "--metrics-port", "0", "--think", "0",
                           "--loop-guard-think", str(BUDGET), "--stats-file", os.path.join(TMP, "stats.json")])
    tok = args.profile.tokenizer()[0]
    order = [R_PLAIN,    # below the threshold
             R_TOOL,     # the first guarded turn, max_tokens small
             R_PLAIN,    # 4 turns: quiet
             R_SHORT,    # 6 turns: guarded again, thinking included
             R_PLAIN,    # 7 turns: quiet
             R_SHORT,    # human just spoke: thinking, no note
             R_LONGANS,  # answer cut at the client's max_tokens
             R_ANSWER,   # stops: inside the reasoning they do not match, in the answer they do
             R_ANSWER,   # stop "\n\n": no match in the blank line after </think>
             R_ANSWER,   # stop in a forced wrap-up word
             R_PLAIN,    # no tool results: never guarded
             R_PLAIN,    # image results: unknown
             R_TOOL,     # Anthropic: guarded turn
             R_PLAIN,    # Anthropic: next turn, the reasoning stays in the history once
             R_ANSWER,   # context room: the thinking closes before the slot ends
             R_TOOL,     # context room with max_tokens far above it
             R_PLAIN,    # period 22: quiet the turn after a firing
             R_SHORT,    # period 2: fires at 4 turns
             R_ANSWER,   # stops without a client max_tokens
             R_SHORT,    # streamed chat: reasoning, tool call
             R_ANSWER,   # streamed chat: stops
             R_TOOL,     # streamed messages: thinking block, tool call
             R_ANSWER,   # streamed messages: stops
             R_ANSWER,   # non-streamed messages: stops
             R_PLAIN]    # --loop-guard-think 0 is checked at the prompt: a turn with no thinking opened
    with open(script, "w") as f:
        for r in order:
            f.write(",".join(map(str, tok.encode(r, add_special_tokens=False).ids)) + "\n")
    os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, log
    gen = app.Generator(args)
    assert gen.can_force and gen.think_ids.ok, "the mock with the grammar tier must enforce"
    c = TestClient(app.build_app(gen, args))
    name = args.served_name
    wrap_n = len(gen.think_ids.wrap[""])

    def prompt():  # the last prompt the engine got, as text
        ps = [ln.split() for ln in open(log) if ln.startswith("P ")]
        return tok.decode([int(x) for x in ps[-1][3:]], skip_special_tokens=False)

    def chat(messages, **kw):
        r = c.post("/v1/chat/completions", json={"model": name, "messages": messages, "tools": TOOLS, **kw}).json()
        return r, r["choices"][0]["message"], r["choices"][0]["finish_reason"]

    def fires():
        return gen.guards["triggers"]

    # below the threshold: an ordinary no-think turn
    r, m, fin = chat(hist(2), max_tokens=64)
    assert prompt().endswith(EMPTY) and "server note" not in prompt() and fires() == 0, prompt()[-200:]
    print("  2 turns: not guarded")

    # the first guarded turn: thinking opened, the note with the count of calls (review M1), and the reasoning does not
    # take the client's max_tokens: the tool call comes whole with max_tokens smaller than the reasoning (review M5)
    r, m, fin = chat(hist(3), max_tokens=40)
    p = prompt()
    assert p.endswith(OPEN) and not p.endswith(EMPTY), p[-100:]
    assert "your last 3 turns" in p and "(1 different call(s)" in p and fires() == 1, p[-600:]
    assert m["reasoning_content"].endswith(think_budget.WRAP), m["reasoning_content"][-200:]
    assert m["tool_calls"][0]["function"]["name"] == "Bash" and fin == "tool_calls", r
    n = r["usage"]["completion_tokens"]
    assert 40 < n <= BUDGET + 3 + wrap_n + 40, n
    print(f"  3 turns: thinking opened, note ok, tool call whole with max_tokens 40 ({n} tokens with reasoning)")

    # back-off: 4 quiet, 6 guarded again with thinking (a loop that survived the first thinking turn), 7 quiet
    r, m, fin = chat(hist(4), max_tokens=64)
    assert prompt().endswith(EMPTY) and "server note" not in prompt() and fires() == 1, prompt()[-200:]
    r, m, fin = chat(hist(6), max_tokens=64)
    p = prompt()
    assert p.endswith(OPEN) and "your last 6 turns" in p and fires() == 2, p[-200:]
    assert gen.guards["recent"][-1]["presence"] == 2.0 and m["reasoning_content"].startswith("Easy"), (gen.guards, m)
    r, m, fin = chat(hist(7), max_tokens=64)
    assert prompt().endswith(EMPTY) and "server note" not in prompt() and fires() == 2
    print("  back-off: 4 and 7 quiet, 6 guarded with thinking")

    # a human just spoke on a long stretch: only thinking, whatever the turn count
    r, m, fin = chat(hist(7) + [{"role": "user", "content": "you are looping, stop"}], max_tokens=64)
    p = prompt()
    assert p.endswith(OPEN) and "server note" not in p and fires() == 3, p[-300:]
    assert gen.guards["recent"][-1]["presence"] == 0.0 and m["reasoning_content"].startswith("Easy"), m
    print("  human after 7 turns: thinking, no note, no penalty")

    # the answer has the client's max_tokens, no more: the unused budget is no extra room
    r, m, fin = chat(hist(3), max_tokens=30)
    n, rt = r["usage"]["completion_tokens"], r["usage"]["completion_tokens_details"]["reasoning_tokens"]
    assert fin == "length" and m["content"].startswith("Point 1") and n - rt <= 30 + 3, (fin, n, rt)
    print(f"  answer cut at max_tokens 30: {n} tokens, {rt} reasoning")

    # client stops match only in the answer: "Step" is all over the reasoning, "answer" in the answer
    r, m, fin = chat(hist(3), max_tokens=60, stop=["Step", "answer"])
    assert fin == "stop" and m["content"] == "The" and m["reasoning_content"].count("Step") >= 1, r
    print("  stop: not inside the reasoning, cut in the answer")
    r, m, fin = chat(hist(3), max_tokens=60, stop=["\n\n"])
    assert m["content"] == ANSWER and fin == "stop", r
    print("  stop \\n\\n: no match in the blank line after </think>")
    r, m, fin = chat(hist(3), max_tokens=60, stop=["limited time"])  # (a word of the forced wrap-up)
    assert m["content"] == ANSWER and m["reasoning_content"].endswith(think_budget.WRAP), r
    print("  stop in the forced wrap-up: no match")

    # no tool results: same-shape JSON answers to different items are never guarded (review H1)
    before = fires()
    m = [{"role": "system", "content": "Classify."}]
    for i in range(6):
        m += [{"role": "user", "content": f"classify item {i}"},
              {"role": "assistant", "content": '{"label": "positive", "score": 0.9, "x": %d}' % i}]
    m.append({"role": "user", "content": "classify item 6"})
    r, _, _ = chat(m, max_tokens=64)
    assert prompt().endswith(EMPTY) and fires() == before, prompt()[-100:]
    print("  few-shot JSON without tool results: not guarded")

    # image results on a server without vision: unknown, not "the same placeholder text" (review L2)
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    h = [{"role": "user", "content": "watch the page"}]
    for i in range(3):
        h += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"s{i}", "name": "Bash", "input": {"command": "shot"}}]},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"s{i}",
                                            "content": [{"type": "text", "text": "Captured"}, img]}]}]
    a = c.post("/v1/messages", json={"model": name, "max_tokens": 64, "messages": h, "tools": TOOLS_A})
    assert a.status_code == 200 and prompt().endswith(EMPTY) and fires() == before, (a.text, prompt()[-100:])
    print("  image results: not guarded")

    # Anthropic path: a thinking block, the tool call; the next request (client echoes no thinking block) gets the
    # reasoning from the server's memory once, and no thinking on a turn that is not due
    h = hist_a(3)
    a = c.post("/v1/messages", json={"model": name, "max_tokens": 40, "messages": h, "tools": TOOLS_A}).json()
    kinds = [b["type"] for b in a["content"]]
    assert kinds == ["thinking", "tool_use"] and a["stop_reason"] == "tool_use" and prompt().endswith(OPEN), a
    used = next(b for b in a["content"] if b["type"] == "tool_use")
    h += [{"role": "assistant", "content": [used]},
          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": used["id"], "content": "new output"}]}]
    a = c.post("/v1/messages", json={"model": name, "max_tokens": 64, "messages": h, "tools": TOOLS_A}).json()
    p = prompt()
    assert p.endswith(EMPTY) and p.count(think_budget.WRAP) == 1 and p.count("Step 1:") == 1, p[-400:]
    print("  anthropic: thinking block once, in the history once")

    # no room in the context for budget + answer: the answer tail still closes the thinking before the slot ends
    h = hist(3)
    slots = list(gen.engine.slots)
    r = gen.prompt(app.openai_messages(h), app.canonical_tools(TOOLS), {"enable_thinking": False}, None, track=True)
    n = len(r[0])
    gen.engine.slots[:] = [n + 60] * len(slots)
    try:
        r, m, fin = chat(h, max_tokens=30)
    finally:
        gen.engine.slots[:] = slots
    assert m["content"] == ANSWER and m["reasoning_content"], r
    assert r["usage"]["completion_tokens"] <= 52, r["usage"]
    print(f"  context room 60: thinking closed in time, answer whole ({r['usage']['completion_tokens']} tokens)")

    # max_tokens above the room (agent clients send tens of thousands): the same, not reasoning cut at the slot end
    gen.engine.slots[:] = [n + 60] * len(slots)
    try:
        r, m, fin = chat(h, max_tokens=4096)
    finally:
        gen.engine.slots[:] = slots
    assert fin == "tool_calls" and m["tool_calls"][0]["function"]["name"] == "Bash" and m["reasoning_content"], r
    assert r["usage"]["completion_tokens"] <= 52, r["usage"]
    print(f"  context room 60, max_tokens 4096: tool call whole ({r['usage']['completion_tokens']} tokens)")

    # real cycles through the server: period 22 is quiet the turn after a firing, period 2 fires with thinking
    def cycle(p, turns):
        m = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "go"}]
        for t in range(turns):
            cmd = "cmd " + chr(97 + t % p)
            m += [{"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {
                "name": "Bash", "arguments": json.dumps({"command": cmd})}}]},
                  {"role": "tool", "tool_call_id": "x", "content": "result of " + cmd}]
        return m

    before = fires()
    chat(cycle(22, 25), max_tokens=64)
    assert prompt().endswith(EMPTY) and fires() == before, prompt()[-100:]
    chat(cycle(2, 4), max_tokens=64)
    assert prompt().endswith(OPEN) and "your last 3 turns" in prompt() and "(2 different call(s)" in prompt() and fires() == before + 1
    print("  cycles: period 22 quiet after its firing, period 2 guarded with thinking")

    # the rule holds without a client max_tokens: there is no budget on top, the stops still wait for the answer
    r, m, fin = chat(hist(3), stop=["Step", "answer"])
    assert prompt().endswith(OPEN) and fin == "stop" and m["content"] == "The" and "Step" in m["reasoning_content"], r
    print("  stop without max_tokens: not inside the reasoning, cut in the answer")

    # streams, both APIs, on the no-think request: reasoning in its own channel, the tool call runs, no tag in the text
    def sse(path, **body):
        with c.stream("POST", path, json={"model": name, "messages": body.pop("messages"), "stream": True, **body}) as s:
            assert s.status_code == 200
            return [json.loads(x[6:]) for x in s.iter_lines() if x.startswith("data: ") and x != "data: [DONE]"]

    def chat_stream(**body):
        ev = sse("/v1/chat/completions", messages=hist(3), tools=TOOLS, **body)
        ds = [e["choices"][0] for e in ev if e.get("choices")]
        d = [x["delta"] for x in ds]
        return ("".join(x.get("reasoning_content") or "" for x in d), "".join(x.get("content") or "" for x in d),
                [t for x in d for t in x.get("tool_calls") or []], ds[-1]["finish_reason"])

    def msg_stream(**body):
        ev = sse("/v1/messages", messages=hist_a(3), tools=TOOLS_A, max_tokens=60, **body)
        ds = [e["delta"] for e in ev if e["type"] == "content_block_delta"]
        return ("".join(x.get("thinking") or "" for x in ds), "".join(x.get("text") or "" for x in ds),
                [e["content_block"]["name"] for e in ev if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use"],
                next(e["delta"]["stop_reason"] for e in ev if e["type"] == "message_delta"))

    th, tx, calls, fin = chat_stream(max_tokens=40)
    assert prompt().endswith(OPEN) and th.startswith("Easy") and "think>" not in th + tx and not tx.strip(), (th, tx)
    assert fin == "tool_calls" and calls[0]["function"]["name"] == "Bash", (calls, fin)
    th, tx, calls, fin = chat_stream(max_tokens=60, stop=["Step", "answer"])
    assert "Step" in th and tx == "The " and fin == "stop" and not calls, (th[-80:], tx, fin)
    print("  chat stream: reasoning apart, tool call whole, stop only in the answer")
    th, tx, calls, fin = msg_stream()
    assert prompt().endswith(OPEN) and th.startswith("Step 1") and "think>" not in th + tx and not tx.strip(), (th[:80], tx)
    assert fin == "tool_use" and calls == ["Bash"], (calls, fin)
    th, tx, calls, fin = msg_stream(stop_sequences=["Step", "answer"])
    assert "Step" in th and tx == "The " and fin == "end_turn" and not calls, (th[-80:], tx, fin)
    a = c.post("/v1/messages", json={"model": name, "max_tokens": 60, "messages": hist_a(3), "tools": TOOLS_A,
                                     "stop_sequences": ["Step", "answer"]}).json()
    assert "Step" in a["content"][0]["thinking"] and a["content"][1]["text"] == "The", a
    print("  messages: stream and not, thinking block apart, tool call whole, stop only in the answer")

    # the off-switch: no thinking opened, the stretch only gets the note
    gen.args.loop_guard_think = 0
    try:
        r, m, fin = chat(hist(3), max_tokens=64)
    finally:
        gen.args.loop_guard_think = BUDGET
    assert prompt().endswith(EMPTY) and "server note" in prompt() and fin == "tool_calls" and not m.get("reasoning_content"), r
    print("  --loop-guard-think 0: the guarded turn stays without thinking")

    print(f"ok ({fires()} guarded turns)")


if __name__ == "__main__":
    main()
