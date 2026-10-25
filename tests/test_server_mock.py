"""CPU-only end-to-end HTTP test of the server (OpenAI + Anthropic APIs) on a mock engine library
(tests/mock_engine.c, built here with gcc) that exports only the TIER 0 C API of server/qe_capi.h
(and once more with TIER 1). Flash-Next model directory for the tokenizer / template / configs,
replies scripted as Flash-Next token ids. No GPU (NVML stubbed out).

  python tests/test_server_mock.py            # Flash-Next, tier 0 and tier 0+1 mocks
  QE_MOCK_MODEL=/path/27B python tests/...    # another model directory
"""
import json
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
# monitoring reads GPU stats through NVML: not in this test
_nv = types.ModuleType("pynvml")


def _no_nvml():
    raise RuntimeError("NVML disabled in the mock test")


_nv.nvmlInit = _no_nvml
sys.modules["pynvml"] = _nv

FLASH = os.environ.get("QE_MOCK_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
TMP = tempfile.mkdtemp(prefix="qe_mock_")
os.environ["QE_DIAG_LOG"] = os.path.join(TMP, "diag.log")
os.environ["QE_WARMUP"] = "0"

import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

TOOLS_OAI = [{"type": "function", "function": {"name": "Bash", "description": "Run a shell command",
              "parameters": {"type": "object", "required": ["command"],
                             "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}}}}]
TOOLS_ANT = [{"name": "Bash", "description": "Run a shell command",
              "input_schema": TOOLS_OAI[0]["function"]["parameters"]}]
R_TEXT = "Ich denke darüber nach.\n</think>\n\nHallo! नमस्ते, ฉันสบายดี<|im_end|>"
R_TOOL = ("Run ls.\n</think>\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la ~/größe\n"
          "</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call><|im_end|>")
R_EMPTY = "nothing to add\n</think>\n\n<|im_end|>"
R_DONE = "ok\n</think>\n\nGotowe. Done here.<|im_end|>"
# replies in the order of the requests below (each start takes the next line)
SCRIPT = [R_TEXT, R_TOOL, R_EMPTY, R_DONE, R_TOOL, R_DONE, R_DONE, R_TEXT]


def build_mock(tier0):
    so = os.path.join(TMP, "mock0.so" if tier0 else "mock1.so")
    cmd = ["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "mock_engine.c")]
    subprocess.check_call(cmd + (["-DMOCK_TIER0"] if tier0 else []))
    return so


def sse_events(text):
    out = []
    for block in text.split("\n\n"):
        data = [l[6:] for l in block.splitlines() if l.startswith("data: ")]
        if data and data[0] != "[DONE]":
            out.append(json.loads(data[0]))
    return out


def run(tier0):
    so = build_mock(tier0)
    log = os.path.join(TMP, f"log{int(tier0)}.txt")
    script = os.path.join(TMP, f"script{int(tier0)}.txt")
    args = app.parse_args(["--model", FLASH, "--engine-lib", so, "--metrics-port", "0",
                           "--stats-file", os.path.join(TMP, "stats.json")])
    tok = args.profile.tokenizer()[0]
    with open(script, "w") as f:
        for r in SCRIPT:
            f.write(",".join(map(str, tok.encode(r, add_special_tokens=False).ids)) + "\n")
    os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, log
    if os.path.exists(log):
        os.remove(log)
    gen = app.Generator(args)
    c = TestClient(app.build_app(gen, args))
    name = "qwen3.8-flash-next-qengine" if args.profile.is_flash else "qwen3.8-27b-qengine"
    assert c.get("/v1/models").json()["data"][0]["id"] == name

    def prompts():
        out = []
        for line in open(log):
            if line.startswith("P "):
                p = line.split()
                out.append((int(p[1]), int(p[2]), tok.decode([int(x) for x in p[3:]], skip_special_tokens=False)))
        return out

    # 1. OpenAI, no sampling parameters: generation_config defaults, reasoning / content split
    r = c.post("/v1/chat/completions", json={"model": name, "messages": [
        {"role": "system", "content": "Du bist hilfreich."}, {"role": "user", "content": "Hallo"}]}).json()
    m = r["choices"][0]["message"]
    assert m["reasoning_content"] == "Ich denke darüber nach." and m["content"] == "Hallo! नमस्ते, ฉันสบายดี", m
    assert r["choices"][0]["finish_reason"] == "stop" and r["model"] == name
    s = [l.split() for l in open(log) if l.startswith("S ")][0]
    want_t = "1.000" if args.profile.is_flash else "0.600"
    assert s[1:] == [want_t, "0.950", "20"], s
    print("  openai: text + reasoning ok, default sampling", s[1:])

    # 2. OpenAI streaming tool call
    t = c.post("/v1/chat/completions", json={"model": name, "stream": True, "tools": TOOLS_OAI, "temperature": 0.7,
                                             "messages": [{"role": "user", "content": "zeige die Dateien"}]}).text
    ev = sse_events(t)
    calls = [d for e in ev for ch in e["choices"] for d in [ch["delta"]] if "tool_calls" in d]
    assert len(calls) == 1, ev
    tc = calls[0]["tool_calls"][0]["function"]
    assert tc["name"] == "Bash" and json.loads(tc["arguments"]) == {"command": "ls -la ~/größe", "timeout": 30}, tc
    assert [ch["finish_reason"] for e in ev for ch in e["choices"] if ch["finish_reason"]] == ["tool_calls"]
    assert ev[-1]["usage"]["completion_tokens"] > 0
    print("  openai stream: tool call ok")

    # 3. Anthropic: reasoning-only reply
    r = c.post("/v1/messages", json={"model": name, "max_tokens": 200,
                                     "messages": [{"role": "user", "content": "q1"}]}).json()
    assert [b["type"] for b in r["content"]] == ["thinking"] and r["stop_reason"] == "end_turn", r

    # 4. OpenAI: several leading system messages (HF #75/#76) + reasoning-only turn in history (HF #50)
    r = c.post("/v1/chat/completions", json={"model": name, "messages": [
        {"role": "system", "content": "Rule A."}, {"role": "system", "content": "Rule B."},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "", "reasoning_content": "SECRET-EMPTY-TURN"},
        {"role": "user", "content": "q2"}]})
    assert r.status_code == 200, r.text
    p = prompts()[-1][2]
    flash = args.profile.is_flash
    assert "Rule A.\nRule B.<|im_end|>" in p, p
    assert ("SECRET-EMPTY-TURN" not in p) == flash, p  # dropped by default for Flash only
    print(f"  merged system messages; reasoning-only turn {'dropped' if flash else 'kept (27B default)'}")

    # 5. Anthropic tool round trip: the second turn resumes from the cached first turn
    msgs = [{"role": "user", "content": "Liste der Dateien"}]
    r1 = c.post("/v1/messages", json={"model": name, "max_tokens": 500, "tools": TOOLS_ANT, "messages": msgs,
                                      "system": [{"type": "text", "text": "You are a coding agent."}]}).json()
    tu = [b for b in r1["content"] if b["type"] == "tool_use"]
    assert r1["stop_reason"] == "tool_use" and tu[0]["input"] == {"command": "ls -la ~/größe", "timeout": 30}, r1
    msgs += [{"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": ""},
                                               {"type": "tool_use", "id": tu[0]["id"], "name": "Bash", "input": tu[0]["input"]}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0]["id"], "content": "a.txt\nb.txt"}]}]
    t = c.post("/v1/messages", json={"model": name, "max_tokens": 500, "tools": TOOLS_ANT, "messages": msgs, "stream": True,
                                     "system": [{"type": "text", "text": "You are a coding agent."}]}).text
    evs = [json.loads(l[6:]) for l in t.splitlines() if l.startswith("data: ")]
    text = "".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta")
    assert text == "Gotowe. Done here.", text
    md = [e for e in evs if e["type"] == "message_delta"][0]
    n1 = prompts()[-2][0]
    assert md["usage"]["cache_read_input_tokens"] >= n1, (md, n1)  # previous prompt + reply reused
    print(f"  anthropic tool round trip ok, 2nd turn cached {md['usage']['cache_read_input_tokens']} "
          f"tokens (1st prompt {n1})")

    # 6. stop sequence cut (no stop-trim API in the mock)
    r = c.post("/v1/chat/completions", json={"model": name, "stop": ["Done"],
                                             "messages": [{"role": "user", "content": "x"}]}).json()
    assert r["choices"][0]["message"]["content"] == "Gotowe.", r

    # 7. count_tokens, structured output without the grammar API
    n = c.post("/v1/messages/count_tokens", json={"model": name, "messages": [{"role": "user", "content": "hej"}]}).json()
    assert n["input_tokens"] > 3
    r = c.post("/v1/chat/completions", json={"model": name, "messages": [{"role": "user", "content": "x"}],
                                             "response_format": {"type": "json_object"}})
    assert r.status_code == 400 and "not supported by this engine" in r.text, r.text

    # 8. max_tokens
    r = c.post("/v1/chat/completions", json={"model": name, "max_tokens": 4,
                                             "messages": [{"role": "user", "content": "y"}]}).json()
    assert r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 4, r
    print("  stop sequence, count_tokens, structured-output 400, max_tokens ok")
    print("  engine fallbacks:", ", ".join(gen.engine.missing_optional()))


def main():
    for tier0 in (True, False):
        print(f"mock engine {'TIER 0 only' if tier0 else 'TIER 0 + 1'} ({FLASH}):")
        run(tier0)
    print("ok")


if __name__ == "__main__":
    main()
