"""CPU test of tool_choice (server/tool_choice.py) on both APIs, streamed and not, on the mock engine: with the grammar
tier (tests/mock_engine.c -DMOCK_GRAMMAR, mask-aware with QE_MOCK_MASK) `none` blocks a scripted call, `required` /
Anthropic `any` force one where the script answers in text, a named tool is forced, also after a forced close of
<think>; without the grammar tier (tier 0+1 mock) `none` ends the reply at the call and `required` decodes as auto.
Bad values are 400s; auto / absent are unchanged. No GPU.

  python tests/test_tool_choice.py            # Flash-Next model directory (tokenizer / template)
  QE_MOCK_MODEL=models/Qwen3.8-27B-INT4 python tests/test_tool_choice.py   # 27B
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
TMP = tempfile.mkdtemp(prefix="qe_toolchoice_")
os.environ["QE_DIAG_LOG"] = os.path.join(TMP, "diag.log")
os.environ["QE_WARMUP"] = "0"

import app  # noqa: E402
import think_budget  # noqa: E402
import tool_choice  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

PARAMS = {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}}
BASH = {"type": "function", "function": {"name": "Bash", "description": "Run a shell command", "parameters": PARAMS}}
READ = {"type": "function", "function": {"name": "Read", "description": "Read a file",
                                         "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
ANT_BASH = {"name": "Bash", "description": "Run a shell command", "input_schema": PARAMS}
ANT_READ = {"name": "Read", "description": "Read a file", "input_schema": READ["function"]["parameters"]}
CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
R_TOOL = "Run ls.\n</think>\n\n" + CALL + "<|im_end|>"
R_TEXT_CALL = "Run ls.\n</think>\n\nI will list the files.\n\n" + CALL + "<|im_end|>"
R_TEXT = "Easy.\n</think>\n\nThe answer is 4.<|im_end|>"
R_TOOL_OFF = CALL + "<|im_end|>"  # thinking off: the prompt ends with <think>\n\n</think>\n\n
R_TEXT_OFF = "The answer is 4.<|im_end|>"
THINK = "".join(f"Step {i}: check case {i * 7 % 13} against rule {i % 5}, then note result {i * 3}.\n" for i in range(1, 30))
R_LONG = THINK + "\n</think>\n\nThe answer is 4.<|im_end|>"
R_BLOCK_CALL = "plan " + CALL + "<|im_end|>"
READ_CALL = "<tool_call>\n<function=Read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
R_TWO = "Both.\n</think>\n\n" + CALL + "\n" + READ_CALL + "<|im_end|>"  # two calls: Bash, then Read
READFILE = {"type": "function", "function": {"name": "ReadFile", "description": "Read a whole file",
                                             "parameters": READ["function"]["parameters"]}}  # a call inside the open <think>, then the end of turn
MSGS = [{"role": "user", "content": "list the files"}]


def sse(text):
    out = []
    for block in text.split("\n\n"):
        data = [l[6:] for l in block.splitlines() if l.startswith("data: ")]
        if data and data[0] != "[DONE]":
            out.append(json.loads(data[0]))
    return out


def oai_stream(evs):
    """(content, reasoning, tool calls, finish reasons) of an OpenAI stream."""
    content = "".join(ch["delta"].get("content") or "" for e in evs for ch in e.get("choices", []))
    reasoning = "".join(ch["delta"].get("reasoning_content") or "" for e in evs for ch in e.get("choices", []))
    calls = [tc for e in evs for ch in e.get("choices", []) for tc in ch["delta"].get("tool_calls") or []]
    fins = [ch["finish_reason"] for e in evs for ch in e.get("choices", []) if ch.get("finish_reason")]
    return content, reasoning, calls, fins


def ant_stream(evs):
    """(text, tool_use blocks with their input, stop reason) of an Anthropic stream."""
    text = "".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta")
    uses = []
    for e in evs:
        if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use":
            uses.append({"name": e["content_block"]["name"], "json": ""})
        elif e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta":
            uses[-1]["json"] += e["delta"]["partial_json"]
    stop = [e["delta"]["stop_reason"] for e in evs if e["type"] == "message_delta"]
    return text, uses, stop


def units():
    p = tool_choice.parse
    tools = [BASH, READ]
    assert p(None, tools) is None and p("auto", tools) is None and p({"type": "auto"}, tools) is None
    assert p("none", tools) == {"mode": "none"} and p({"type": "none"}, tools) == {"mode": "none"}
    assert p("none", None) == {"mode": "none", "cut": True} and p("auto", None) is None  # nothing to mask: accepted
    assert p("required", tools) == {"mode": "required", "names": ["Bash", "Read"], "subset": False}
    assert p({"type": "any"}, tools) == p("required", tools)
    assert p({"type": "tool", "name": "Read"}, tools) == {"mode": "required", "names": ["Read"], "subset": True}
    fc = tool_choice.filter_calls
    calls = [{"name": "Bash", "arguments": {}}, {"name": "Read", "arguments": {}}]
    assert fc(None, calls) == calls and fc({"mode": "none"}, calls) == [] and fc(p("required", tools), calls) == calls
    assert fc(p({"type": "tool", "name": "Read"}, tools), calls) == calls[1:]
    assert p({"type": "function", "function": {"name": "Bash"}}, tools)["names"] == ["Bash"]
    assert p({"type": "tool", "name": "Bash", "disable_parallel_tool_use": True}, tools)["names"] == ["Bash"]
    at = {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [{"type": "function",
                                                                                   "function": {"name": "Read"}}]}}
    assert p(at, tools)["names"] == ["Read"]
    assert p(dict(at, allowed_tools=dict(at["allowed_tools"], mode="auto")), tools) is None
    assert p("required", [ANT_BASH])["names"] == ["Bash"]  # (a bare name form is read too)
    for bad, want in ((("required", None), "no tools"), (({"type": "any"}, []), "no tools"),
                      (({"type": "tool", "name": "Nope"}, tools), "not in tools"),
                      (({"type": "function", "function": {"name": "Nope"}}, tools), "not in tools"),
                      (({"type": "tool"}, tools), "needs a function name"), (("sometimes", tools), "unsupported"),
                      ((7, tools), "string or an object"), (({"type": "allowed_tools"}, tools), "allowed_tools")):
        try:
            p(*bad)
            raise AssertionError(bad)
        except tool_choice.ToolChoiceError as ex:
            assert want in str(ex), (bad, ex)
    print("  tool_choice forms ok")


def check_header(gen):
    """The forced header is the start of the template's own tokenization of a call (both leads)."""
    tier = gen.tool_tier
    for lead in ("", "\n\n"):
        for name in ("Bash", "Read", "mcp__srv__get_weather", "x"):
            h = tier.header(lead, name)
            full = tier.enc(lead + "<tool_call>\n<function=" + name + ">\n<parameter=a>\n1\n</parameter>\n</function>\n"
                                                                          "</tool_call>")
            assert full[:len(h)] == h and tier.call in h, (lead, name)
            assert gen.tok.decode(h, skip_special_tokens=False).startswith(lead + "<tool_call>\n<function="), (lead, name)
    print("  forced headers are prefixes of the template's call tokenization")


def build(flags, name):
    so = os.path.join(TMP, name)
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", *flags, "-o", so, os.path.join(HERE, "mock_engine.c")])
    return so


def server(so, order, tag):
    log = os.path.join(TMP, f"log_{tag}.txt")
    script = os.path.join(TMP, f"script_{tag}.txt")
    args = app.parse_args(["--model", MODEL, "--engine-lib", so, "--metrics-port", "0",
                           "--stats-file", os.path.join(TMP, f"stats_{tag}.json")])
    tok = args.profile.tokenizer()[0]
    with open(script, "w") as f:
        for r in order:
            f.write(",".join(map(str, tok.encode(r, add_special_tokens=False).ids)) + "\n")
    os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, log
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gen = app.Generator(args)
    return gen, TestClient(app.build_app(gen, args)), args.served_name, log, out.getvalue(), tok


def masked(so):
    order = [R_TEXT_CALL, R_TEXT_CALL, R_TEXT_CALL, R_TEXT_CALL,  # 1-4 none: OpenAI, OpenAI stream, Anthropic, stream
             R_TEXT, R_TOOL, R_TEXT, R_TOOL,                      # 5-8 required / any / named
             R_TOOL_OFF, R_TEXT_OFF,                              # 9-10 thinking off: named, any of two
             R_TOOL, R_LONG, R_TEXT_CALL, R_TOOL,                 # 11 auto, 12 budget + required, 13 logprobs, 14 none
             R_TOOL, R_TEXT_OFF, R_TWO]                           # 15 absent, 16 shared name prefix, 17 extra call
    gen, c, name, log, boot, tok = server(so, order, "mask")
    assert gen.tool_tier is not None and "tool_choice: not enforced" not in boot, boot
    check_header(gen)

    def klines():
        return [l.split() for l in open(log) if l[0] in "KC"]

    def tools_in_last_prompt():
        p = [l for l in open(log) if l.startswith("P ")][-1].split()
        return "<tools>" in tok.decode([int(x) for x in p[3:]], skip_special_tokens=False)

    def oai(tc, tools=(BASH,), stream=False, **kw):
        body = {"model": name, "messages": MSGS, "tools": list(tools), "temperature": 0, **kw}
        if tc is not None:
            body["tool_choice"] = tc
        return c.post("/v1/chat/completions", json=dict(body, stream=stream))

    def ant(tc, tools=(ANT_BASH,), stream=False, **kw):
        body = {"model": name, "messages": MSGS, "tools": list(tools), "max_tokens": 400, "temperature": 0, **kw}
        if tc is not None:
            body["tool_choice"] = tc
        return c.post("/v1/messages", json=dict(body, stream=stream))

    # 1. OpenAI none: the scripted call is masked, the model ends its text answer there; the tools stay in the prompt
    k0 = len(klines())
    r = oai("none").json()
    ch = r["choices"][0]
    assert ch["message"]["content"] == "I will list the files." and "tool_calls" not in ch["message"], r
    assert ch["finish_reason"] == "stop" and tools_in_last_prompt(), r
    assert klines()[k0] == ["K", "1"] and klines()[k0 + 1][0] == "C", klines()[k0:]  # constrained to the end
    # 2. OpenAI none, streamed
    content, _, calls, fins = oai_stream(sse(oai("none", stream=True).text))
    assert content.strip() == "I will list the files." and not calls and fins == ["stop"], (content, calls, fins)
    # 3. Anthropic none
    r = ant({"type": "none"}).json()
    assert [b["type"] for b in r["content"]] == ["thinking", "text"] and r["stop_reason"] == "end_turn", r
    assert r["content"][1]["text"] == "I will list the files.", r
    # 4. Anthropic none, streamed
    text, uses, stop = ant_stream(sse(ant({"type": "none"}, stream=True).text))
    assert text.strip() == "I will list the files." and not uses and stop == ["end_turn"], (text, uses, stop)
    print("  none: the scripted call is masked on both APIs, streamed and not; tools stay in the prompt")

    # 5. OpenAI required, the script answers in text: the call header is forced after </think>, then released
    k0 = len(klines())
    r = oai("required").json()
    ch = r["choices"][0]
    assert ch["finish_reason"] == "tool_calls" and ch["message"]["tool_calls"][0]["function"]["name"] == "Bash", r
    k = klines()[k0:]
    assert k[0] == ["K", "1"] and k[1] == ["K", "0"], k  # released after the header
    # 6. OpenAI required, streamed, the script calls itself: the forced header is the script's own, arguments intact
    content, reasoning, calls, fins = oai_stream(sse(oai("required", stream=True).text))
    assert len(calls) == 1 and calls[0]["function"]["name"] == "Bash", calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"} and fins == ["tool_calls"], calls
    assert reasoning.strip() == "Run ls." and not content.strip(), (reasoning, content)
    # 7. Anthropic any: text script -> a call
    r = ant({"type": "any"}).json()
    assert r["stop_reason"] == "tool_use" and [b["name"] for b in r["content"] if b["type"] == "tool_use"] == ["Bash"], r
    # 8. Anthropic named tool, streamed: the script calls Bash, Read is forced
    text, uses, stop = ant_stream(sse(ant({"type": "tool", "name": "Read"}, tools=(ANT_BASH, ANT_READ),
                                          stream=True).text))
    assert [u["name"] for u in uses] == ["Read"] and stop == ["tool_use"], (text, uses, stop)
    print("  required / any / named: a call is forced on both APIs, streamed and not, then the slot is released")

    # 9. thinking off, OpenAI named function (the script calls Bash): Read forced from the first token
    r = oai({"type": "function", "function": {"name": "Read"}}, tools=(BASH, READ),
            chat_template_kwargs={"enable_thinking": False}).json()
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "Read", r
    # 10. thinking off, any of two tools, text script: the header branches at the name (a trie), one of them is chosen
    r = oai("required", tools=(BASH, READ), chat_template_kwargs={"enable_thinking": False}).json()
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] in ("Bash", "Read"), r
    # 11. auto: unchanged (no constraint)
    k0 = len(klines())
    r = oai("auto").json()
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "Bash", r
    assert [x for x in klines()[k0:] if x[0] == "K"] == [], klines()[k0:]
    # 12. a thinking budget closes <think> (forced), then the required header follows: both constraints in turn
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r = oai("required", thinking_budget=24).json()
    m = r["choices"][0]["message"]
    assert m["reasoning_content"].endswith(think_budget.WRAP) and m["tool_calls"][0]["function"]["name"] == "Bash", r
    assert "forced close failed" not in out.getvalue() and "| tool_choice required" in out.getvalue(), out.getvalue()
    # 13. none with logprobs: constrained for both, the text answer comes with its entries
    if gen.engine.has("logprobs"):
        r = oai("none", logprobs=True).json()
        ch = r["choices"][0]
        assert ch["message"]["content"].strip() == "I will list the files." and "tool_calls" not in ch["message"], r
        assert "".join(e["token"] for e in ch["logprobs"]["content"]) == ch["message"]["content"], ch["logprobs"]
    else:
        oai("none")
    print("  thinking off named / any, auto unchanged, budget close + required, none + logprobs ok")

    # 14. errors: bad name, no tools, structured output with a forced call (no start for any of them)
    n0 = len([l for l in open(log) if l.startswith("P ")])
    for r in (oai({"type": "function", "function": {"name": "Nope"}}), oai("required", tools=()),
              oai("required", response_format={"type": "json_object"}), oai("sometimes")):
        e = r.json()["error"]
        assert r.status_code == 400 and e["type"] == "invalid_request_error" and e["param"] == "tool_choice", r.text
    for r in (ant({"type": "tool", "name": "Nope"}), ant({"type": "any"}, tools=())):
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error", r.text
        assert "tool_choice" in r.json()["error"]["message"], r.text
    assert len([l for l in open(log) if l.startswith("P ")]) == n0
    # none without tools: accepted, nothing to mask (no constraint), a call written anyway is not returned
    k0 = len(klines())
    r = oai("none", tools=())
    assert r.status_code == 200 and "tool_calls" not in r.json()["choices"][0]["message"], r.text
    assert [x for x in klines()[k0:] if x[0] == "K"] == [], klines()[k0:]
    # 15. no tool_choice field: unconstrained, the call is returned
    k0 = len(klines())
    r = oai(None).json()
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "Bash", r
    assert [x for x in klines()[k0:] if x[0] == "K"] == [], klines()[k0:]
    # 16. a named tool whose name extends another's (Read / ReadFile): exactly the named one
    for want in ("ReadFile",):
        r = oai({"type": "function", "function": {"name": want}}, tools=(READ, READFILE),
                chat_template_kwargs={"enable_thinking": False}).json()
        assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == want, r
    # 17. named Bash, the model adds a call to Read: only the calls the choice allows are returned
    r = oai({"type": "function", "function": {"name": "Bash"}}, tools=(BASH, READ)).json()
    assert [t["function"]["name"] for t in r["choices"][0]["message"]["tool_calls"]] == ["Bash"], r
    print("  400 for a bad name, a forced call without tools or with response_format, an unknown value; none "
          "without tools ok")


def fallback(so):
    order = [R_TEXT_CALL, R_TEXT_CALL, R_BLOCK_CALL, R_TOOL, R_TEXT_CALL]
    gen, c, name, log, boot, tok = server(so, order, "plain")
    assert gen.tool_tier is None and "tool_choice: not enforced through masks" in boot, boot
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        # none: the reply ends at the call (content before it), streamed and not
        r = c.post("/v1/chat/completions", json={"model": name, "messages": MSGS, "tools": [BASH], "temperature": 0,
                                                 "tool_choice": "none"}).json()
        ch = r["choices"][0]
        assert ch["message"]["content"] == "I will list the files." and "tool_calls" not in ch["message"], r
        assert ch["finish_reason"] == "stop", r
        text, uses, stop = ant_stream(sse(c.post("/v1/messages", json={
            "model": name, "messages": MSGS, "tools": [ANT_BASH], "max_tokens": 400, "temperature": 0, "stream": True,
            "tool_choice": {"type": "none"}}).text))
        assert text.strip() == "I will list the files." and not uses and stop == ["end_turn"], (text, uses, stop)
        # none: a call inside the open <think> stays reasoning (not run as a call)
        r = c.post("/v1/chat/completions", json={"model": name, "messages": MSGS, "tools": [BASH], "temperature": 0,
                                                 "tool_choice": "none"}).json()
        assert "tool_calls" not in r["choices"][0]["message"], r
        # required: not enforced, decodes as auto (one log line)
        for _ in range(2):
            r = c.post("/v1/chat/completions", json={"model": name, "messages": MSGS, "tools": [BASH],
                                                     "temperature": 0, "tool_choice": "required"}).json()
            assert r["choices"][0]["message"]["content"] is not None or r["choices"][0]["message"].get("tool_calls")
    log_out = out.getvalue()
    assert log_out.count("tool_choice none: the reply ends at the tool call") == 2, log_out
    assert log_out.count("tool_choice required: not enforced") == 1, log_out
    r = c.post("/v1/chat/completions", json={"model": name, "messages": MSGS, "tools": [BASH],
                                             "tool_choice": {"type": "function", "function": {"name": "Nope"}}})
    assert r.status_code == 400 and r.json()["error"]["param"] == "tool_choice", r.text
    print("  without the grammar tier: none ends at the call, a call in <think> is not run, required decodes as auto "
          "(logged once), bad name 400")


def main():
    units()
    os.environ["QE_MOCK_MASK"] = "0"  # (replaced by the end token id below, once the tokenizer is known)
    so_g = build(["-DMOCK_GRAMMAR"], "mock_grammar.so")
    so_p = build([], "mock_plain.so")
    tok = app.parse_args(["--model", MODEL, "--metrics-port", "0"]).profile.tokenizer()[0]
    os.environ["QE_MOCK_MASK"] = str(tok.token_to_id("<|im_end|>"))
    masked(so_g)
    fallback(so_p)
    print("ok")


if __name__ == "__main__":
    main()
