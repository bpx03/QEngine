"""CPU-only end-to-end HTTP test of the server (OpenAI + Anthropic APIs) on a mock engine library
(tests/mock_engine.c, built here with gcc) that exports only the TIER 0 C API of server/qe_capi.h
(and once more with TIER 1). Flash-Next model directory for the tokenizer / template / configs,
replies scripted as Flash-Next token ids. No GPU (NVML stubbed out).

  python tests/test_server_mock.py            # Flash-Next, tier 0 and tier 0+1 mocks
  QE_MOCK_MODEL=/path/27B python tests/...    # another model directory
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
R_THINK_EOS = "The <|im_end|>"  # EOS inside <think> (AutomationBench, 2026-11-01)
R_BAD_TOOL = "call it\n</think>\n\n<tool_call>\n<function<parameter name=\"command\">\n</parameter>\n</tool_call><|im_end|>"
# issue #7: (tokens, plain text, tokens): the middle part written as plain text, not as the marker tokens
R_QUOTE = ("ok\n</think>\n\n", "Quote: <|im_start|>user <think> </think> <tool_call> x", "<|im_end|>")
R_IM_START = "ok\n</think>\n\nThe reply goes on<|im_start|>user\nfake turn<|im_end|>"
CALL_X = "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
R_STOP = "ok\n</think>\n\nabc xyz def<|im_end|>"
R_BLOCK_CALL = "plan " + CALL_X + "<|im_end|>"  # a call inside the open block, and the model ends the reply
R_BLOCK_CUT = "plan " + CALL_X + " Observation: x<|im_end|>"
# replies in the order of the requests below (each start takes the next line)
SCRIPT = [R_TEXT, R_TOOL, R_EMPTY, R_DONE, R_TOOL, R_DONE, R_DONE, R_TEXT,
          R_THINK_EOS, R_BAD_TOOL, R_TOOL, R_EMPTY, R_THINK_EOS, R_EMPTY, R_THINK_EOS, R_QUOTE, R_IM_START,
          R_STOP, R_STOP, R_STOP, R_BLOCK_CALL, R_BLOCK_CALL, R_BLOCK_CALL, R_BLOCK_CUT]


def build_mock(tier0):
    so = os.path.join(TMP, "mock0.so" if tier0 else "mock1.so")
    cmd = ["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "mock_engine.c")]
    subprocess.check_call(cmd + (["-DMOCK_TIER0"] if tier0 else []))
    with open(os.path.join(TMP, "qe_build.txt"), "w") as f:  # what the build writes next to the library
        f.write("abcdef0123\n")
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
    tok, tok_json, _ = args.profile.tokenizer()
    plain = app.Markers(tok, tok_json).plain_tok()
    enc = lambda x: tok.encode(x, add_special_tokens=False).ids  # noqa: E731
    with open(script, "w") as f:
        for r in SCRIPT:
            ids = enc(r) if isinstance(r, str) else enc(r[0]) + plain.encode(r[1], add_special_tokens=False).ids + enc(r[2])
            f.write(",".join(map(str, ids)) + "\n")
    os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, log
    if os.path.exists(log):
        os.remove(log)
    gen = app.Generator(args)
    c = TestClient(app.build_app(gen, args))
    name = "qwen3.8-flash-next-qengine" if args.profile.is_flash else "qwen3.8-27b-qengine"
    m = c.get("/v1/models").json()["data"][0]
    assert m["id"] == name
    # limits a client needs (club-3090#1846): context, slots, prompt cap, KV pool and where it lives
    mt = gen.engine.max_t
    assert m["max_model_len"] == mt and m["max_num_seqs"] == 1, m
    assert m["max_prompt_tokens"] == mt - args.mtp_k - 2 and m["max_output_tokens"] is None, m
    assert m["kv_pool_tokens"] == mt, m  # tier 0: no qe_pool_tokens -> max_context; mock tier 1: max_t
    assert m["kv_cache"] == ("ram" if os.environ.get("QE_MOCK_KV_RAM") == "1" and gen.engine.has("qe_kv_in_ram")
                             else "vram"), m
    p = c.get("/props").json()
    assert p["total_slots"] == 1 and p["default_generation_settings"]["n_ctx"] == mt and p["max_num_seqs"] == 1, p

    def prompts():
        out = []
        for line in open(log):
            if line.startswith("P "):
                p = line.split()
                out.append((int(p[1]), int(p[2]), tok.decode([int(x) for x in p[3:]], skip_special_tokens=False)))
        return out

    # 1. OpenAI, no sampling parameters: the model card's thinking defaults, reasoning / content split
    r = c.post("/v1/chat/completions", json={"model": name, "messages": [
        {"role": "system", "content": "Du bist hilfreich."}, {"role": "user", "content": "Hallo"}]}).json()
    m = r["choices"][0]["message"]
    assert m["reasoning_content"] == "Ich denke darüber nach." and m["content"] == "Hallo! नमस्ते, ฉันสบายดี", m
    assert r["choices"][0]["finish_reason"] == "stop" and r["model"] == name
    s = [l.split() for l in open(log) if l.startswith("S ")][0]
    assert s[1:] == ["1.000", "0.950", "20"], s
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

    # 3. Anthropic: reasoning-only reply (greedy: passed through, no --empty-retry)
    r = c.post("/v1/messages", json={"model": name, "max_tokens": 200, "temperature": 0,
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
    # 8b. a prompt over the context: OpenAI's wording (verifiers / LiteLLM detect overlong prompts by it), no start
    n0 = len(prompts())
    r = c.post("/v1/chat/completions", json={"model": name, "messages": [{"role": "user", "content": "x " * (mt + 64)}]})
    e = r.json()["error"]
    assert r.status_code == 400 and e["code"] == "context_length_exceeded", r.text
    assert e["message"].startswith("This model's maximum context length is ") and "prompt is too long" in e["message"], e
    r = c.post("/v1/messages", json={"model": name, "max_tokens": 10, "messages": [{"role": "user", "content": "x " * (mt + 64)}]})
    assert r.status_code == 400 and r.json()["error"]["message"].startswith("prompt is too long"), r.text  # Claude Code
    assert len(prompts()) == n0
    print("  stop sequence, count_tokens, structured-output 400, max_tokens, overlong prompt 400s ok")

    # 9. --empty-retry (default 2): EOS inside <think>, then a tool call that does not parse, then a good one
    n0 = len(prompts())
    r = c.post("/v1/chat/completions", json={"model": name, "tools": TOOLS_OAI,
                                             "messages": [{"role": "user", "content": "pliki?"}]}).json()
    ch = r["choices"][0]
    assert ch["finish_reason"] == "tool_calls" and ch["message"]["tool_calls"][0]["function"]["name"] == "Bash", r
    assert len(prompts()) == n0 + 3, prompts()[n0:]
    assert prompts()[-1][1] > 0 and gen.think["empty_retried"] == 2 and gen.think["empty_retry_recovered"] == 1
    want = sum(len(tok.encode(x, add_special_tokens=False).ids) - 1 for x in (R_THINK_EOS, R_BAD_TOOL, R_TOOL))
    assert r["usage"]["completion_tokens"] == want, (r["usage"], want)
    # retries used up: the last empty reply goes to the client (Anthropic)
    r = c.post("/v1/messages", json={"model": name, "max_tokens": 300,
                                     "messages": [{"role": "user", "content": "q9"}]}).json()
    assert [b["type"] for b in r["content"]] == ["thinking"] and r["stop_reason"] == "end_turn", r
    assert len(prompts()) == n0 + 6 and gen.think["empty_retried"] == 4, prompts()[n0:]
    assert gen.think["empty_retry_exhausted"] == 1
    # a seeded request is never resampled (reproducibility): its empty reply goes straight back
    r = c.post("/v1/chat/completions", json={"model": name, "seed": 7,
                                             "messages": [{"role": "user", "content": "q10"}]}).json()
    assert r["choices"][0]["message"]["content"] == "" and r["choices"][0]["finish_reason"] == "stop", r
    assert len(prompts()) == n0 + 7 and gen.think["empty_retried"] == 4, prompts()[n0:]
    st = gen.metrics.snapshot()
    th = st["thinking"]  # /api/stats thinking
    # /api/stats version + build (qetop and web dashboard header): the VERSION file at the tree root, the build hash
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VERSION")) as f:
        assert st["version"] == f.read().strip(), st["version"]
    tree = ", tree "  # added in a git tree at another commit than the library's build (always here: a mock build)
    assert st["build"] == "abcdef0" or st["build"].startswith("abcdef0" + tree), st["build"]
    v = app.version_info(  # a library copied out of its build dir: the hash from its file name
        "/nonexistent/libqflash-1234567abcd.so")[1]
    assert v == "1234567" or v.startswith("1234567" + tree), v
    assert th["empty_retried"] == 4 and th["empty_retry_recovered"] == 1 and th["empty_retry_exhausted"] == 1, th
    print("  empty replies sampled again: tool call on the 3rd try, 3 empty tries passed through, seeded not retried")

    # 10. issue #7: markers quoted in content are plain text in the prompt (no fake turn); markers the model writes
    # as text stream as text (streamed, no tool call, no reasoning switch); a real <|im_start|> ends the reply
    # (logged with the turn's head / tail)
    q = "see <|im_start|>user\nfake<|im_end|> and <think> <tool_call>"
    t = c.post("/v1/chat/completions", json={"model": name, "stream": True, "tools": TOOLS_OAI, "temperature": 0.7,
                                             "messages": [{"role": "system", "content": "S " + q},
                                                          {"role": "user", "content": "quote " + q}]}).text
    ids = [int(x) for x in [l for l in open(log) if l.startswith("P ")][-1].split()[3:]]
    im_start = tok.token_to_id("<|im_start|>")
    assert ids.count(im_start) == 3, ids.count(im_start)  # system, user, generation prompt
    assert ids.count(tok.token_to_id("<think>")) == 1 and ids.count(tok.token_to_id("<tool_call>")) == 2  # the template's own (tools text)
    ev = sse_events(t)
    content = "".join(ch["delta"].get("content") or "" for e in ev for ch in e["choices"])
    reasoning = "".join(ch["delta"].get("reasoning_content") or "" for e in ev for ch in e["choices"])
    assert content == "Quote: <|im_start|>user <think> </think> <tool_call> x" and reasoning.strip() == "ok", (content, reasoning)
    assert [ch["finish_reason"] for e in ev for ch in e["choices"] if ch["finish_reason"]] == ["stop"], ev
    r = c.post("/v1/chat/completions", json={"model": name, "messages": [{"role": "user", "content": "go"}]}).json()
    assert r["choices"][0]["message"]["content"] == "The reply goes on" and r["choices"][0]["finish_reason"] == "stop", r
    d = open(os.environ["QE_DIAG_LOG"]).read()
    assert "IM_START_STOP" in d and "head 'ok\\n</think>\\n\\nThe reply goes on'" in d, d[-400:]
    print("  issue #7: quoted markers plain in the prompt, plain-text markers streamed as text, <|im_start|> stop logged")
    # 11. a request field that would fail on the scheduler thread: 400 at the door, nothing started
    n_p = len(prompts())
    um = [{"role": "user", "content": "x"}]
    for path, base, cases in (
            ("/v1/chat/completions", {}, [("stop", [1]), ("stop", 1), ("stop", ["a", 1]), ("stop", {"a": 1}), ("max_tokens", 1.5), ("max_tokens", "a"),
                                          ("max_completion_tokens", [1]), ("temperature", "a"), ("top_p", "a"), ("top_k", "a"), ("min_p", [1]),
                                          ("seed", "a"), ("presence_penalty", "a"), ("frequency_penalty", {"a": 1})]),
            ("/v1/messages", {"max_tokens": 16}, [("stop_sequences", [1]), ("stop_sequences", 1), ("max_tokens", 1.5), ("temperature", "a"), ("top_p", [1]),
                                                  ("top_k", "a"), ("presence_penalty", "a")])):
        for k, v in cases:
            r = c.post(path, json={"model": name, "messages": um, **base, k: v})
            assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error", (path, k, v, r.status_code, r.text[:200])
    assert len(prompts()) == n_p, "a refused request reached the engine"
    # 12. stop strings on streamed requests and as one string (R_STOP x 3): only the text before the stop, on both APIs
    t = c.post("/v1/chat/completions", json={"model": name, "messages": um, "stop": ["xyz"], "stream": True}).text
    evs = sse_events(t)
    assert "".join(e["choices"][0]["delta"].get("content") or "" for e in evs if e.get("choices")) == "abc ", t[:400]
    assert [e["choices"][0]["finish_reason"] for e in evs if e.get("choices")][-1] == "stop", t[-300:]
    t = c.post("/v1/messages", json={"model": name, "messages": um, "max_tokens": 100, "stop_sequences": ["xyz"], "stream": True}).text
    assert "".join(e["delta"].get("text", "") for e in sse_events(t) if e["type"] == "content_block_delta") .endswith("abc ") and "xyz" not in t, t[:400]
    r = c.post("/v1/messages", json={"model": name, "messages": um, "max_tokens": 100, "stop_sequences": "xyz"}).json()
    assert [b["text"] for b in r["content"] if b["type"] == "text"] == ["abc"], r  # (a non-streamed reply comes back stripped)
    # 13. a call inside the open block: runs when the model ended the reply, on all paths; stays reasoning after a stop cut
    r = c.post("/v1/chat/completions", json={"model": name, "messages": um, "tools": TOOLS_OAI}).json()
    m = r["choices"][0]
    assert m["finish_reason"] == "tool_calls" and m["message"]["tool_calls"][0]["function"]["name"] == "Bash" and m["message"]["reasoning_content"] == "plan", r
    assert r["usage"]["completion_tokens_details"]["reasoning_tokens"] == len(enc("plan ")), r["usage"]
    t = c.post("/v1/chat/completions", json={"model": name, "messages": um, "tools": TOOLS_OAI, "stream": True}).text
    assert '"tool_calls"' in t and [e["choices"][0]["finish_reason"] for e in sse_events(t) if e.get("choices")][-1] == "tool_calls", t[-300:]
    t = c.post("/v1/messages", json={"model": name, "messages": um, "max_tokens": 200, "tools": TOOLS_ANT, "stream": True}).text
    assert '"tool_use"' in t and '"stop_reason": "tool_use"' in t, t[-300:]
    r = c.post("/v1/chat/completions", json={"model": name, "messages": um, "tools": TOOLS_OAI, "stop": ["Observation:"]}).json()
    m = r["choices"][0]
    assert m["finish_reason"] == "stop" and not m["message"].get("tool_calls") and "<tool_call>" in m["message"]["reasoning_content"], r
    print("  bad fields 400, streamed stops, call in an open block ok")
    # non-thinking request: the card's non-thinking defaults
    c.post("/v1/chat/completions", json={"model": name, "messages": [{"role": "user", "content": "Cześć"}],
                                         "chat_template_kwargs": {"enable_thinking": False}})
    s = [l.split() for l in open(log) if l.startswith("S ")][-1]
    assert s[1:] == ["0.700", "0.800", "20"], s
    print("  openai: non-thinking default sampling", s[1:])
    print("  engine fallbacks:", ", ".join(gen.engine.missing_optional()))


def check_disk_cache_flag():
    """Issue #15: --disk-cache turns the disk tier on for both models; QE_DISK_CACHE alone only moves the start-up cache."""
    env = dict(os.environ)
    os.environ.pop("QE_DISK_CACHE", None)
    os.environ["FLASH_DISK_CACHE"] = "/nonexistent"  # (only the flag decides)
    d = os.path.join(TMP, "dc")
    argv = ["--engine-lib", "none", "--metrics-port", "0"]
    a = app.parse_args(["--model", FLASH] + argv + ["--disk-cache", d, "--disk-cache-gb", "7"])
    if a.profile.is_flash:
        assert os.environ.get("FLASH_DISK_CACHE") == d and float(os.environ["FLASH_DISK_CACHE_GB"]) == 7.0
        assert os.environ.get("QE_DISK_CACHE") == d
    os.environ.pop("QE_DISK_CACHE", None)
    os.environ["FLASH_DISK_CACHE"] = "/nonexistent"
    os.environ["QE_DISK_CACHE"] = d
    a = app.parse_args(["--model", FLASH] + argv)
    assert not a.disk_cache and "FLASH_DISK_CACHE" not in os.environ and os.environ.get("QE_DISK_CACHE") == d
    q27 = os.path.join(HERE, "..", "models", "Qwen3.8-27B-INT4")
    if not os.path.isdir(q27):
        q27 = os.path.expanduser("~/GIT/QEngine/models/Qwen3.8-27B-INT4")
    if os.path.isdir(q27):
        os.environ.pop("QE_DISK_CACHE", None)
        a = app.parse_args(["--model", q27] + argv + ["--disk-cache", d])
        assert a.disk_cache == d and os.environ.pop("QE_DISK_CACHE") == d and "FLASH_DISK_CACHE" not in os.environ
    os.environ.clear()
    os.environ.update(env)
    print("  issue #15: --disk-cache sets the Flash-Next tier, QE_DISK_CACHE alone does not, the 27B unchanged")


def check_effort_none():
    """club-3090 #1966: reasoning_effort "none" (chat_template_kwargs, top level, the Anthropic path, any case) is thinking
    off: the engine gets the very prompt of enable_thinking=false and the reply carries no reasoning; an explicit
    enable_thinking true wins over it; QE_REASONING_EFFORT=none = thinking off by default."""
    so = os.path.join(TMP, "mock_none.so")
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "mock_engine.c")])
    log, script = os.path.join(TMP, "log_none.txt"), os.path.join(TMP, "script_none.txt")

    def server(extra):
        args = app.parse_args(["--model", FLASH, "--engine-lib", so, "--metrics-port", "0",
                               "--stats-file", os.path.join(TMP, "stats_none.json")] + extra)
        tok, _, _ = args.profile.tokenizer()
        with open(script, "w") as f:  # no </think>: a reply rendered with thinking on would be reasoning only
            f.write(",".join(map(str, tok.encode("Hello there.<|im_end|>", add_special_tokens=False).ids)) + "\n")
        os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, log
        if os.path.exists(log):
            os.remove(log)
        return TestClient(app.build_app(app.Generator(args), args))

    def last_prompt():
        return [ln.split()[3:] for ln in open(log) if ln.startswith("P ")][-1]

    def chat(c, **kw):
        r = c.post("/v1/chat/completions", json=dict({"model": "x", "messages": msgs, "max_tokens": 50}, **kw)).json()
        m = r["choices"][0]["message"]
        return last_prompt(), m.get("reasoning_content"), m["content"], r["usage"].get("completion_tokens_details")

    def anth(c, **kw):
        r = c.post("/v1/messages", json=dict({"model": "x", "messages": msgs, "max_tokens": 50}, **kw)).json()
        return last_prompt(), [b["type"] for b in r["content"]], [b.get("text") for b in r["content"]]

    msgs = [{"role": "user", "content": "hi"}]
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    c = server([])
    p_off, rs, txt, det = chat(c, **off)
    assert rs is None and txt == "Hello there." and det is None, (rs, txt, det)
    p_on, rs, txt, _ = chat(c)
    assert p_on != p_off and rs == "Hello there." and not txt, (rs, txt)  # (the mock tells thinking on from off)
    for body in ({"chat_template_kwargs": {"reasoning_effort": "none"}}, {"reasoning_effort": "none"},
                 {"reasoning_effort": "None"}, {"reasoning_effort": " NONE "},
                 {"chat_template_kwargs": {"reasoning_effort": "none"}, "reasoning_effort": "low"},
                 {"reasoning_effort": "none", "thinking_budget": 100},
                 {"reasoning": {"effort": "none"}}, {"reasoning": {"effort": "NONE", "max_tokens": 100}}):
        p, rs, txt, det = chat(c, **body)
        assert p == p_off and rs is None and txt == "Hello there." and det is None, (body, rs, txt, det)
    # streamed: no reasoning delta either
    t = c.post("/v1/chat/completions", json={"model": "x", "messages": msgs, "max_tokens": 50, "stream": True,
                                             "reasoning_effort": "none"}).text
    ev = [e["choices"][0]["delta"] for e in sse_events(t) if e.get("choices")]
    assert last_prompt() == p_off and not any(d.get("reasoning_content") for d in ev), t[:400]
    assert "".join(d.get("content") or "" for d in ev) == "Hello there.", t[:400]
    # a request's none wins over its own enable_thinking true (any "off" wins)
    p, rs, _, _ = chat(c, chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "none"})
    assert p == p_off and rs is None, rs
    p_low, rs, _, _ = chat(c, reasoning_effort="low")
    assert p_low not in (p_on, p_off) and rs == "Hello there.", rs
    # OpenAI Responses / OpenRouter reasoning.effort: read below the top-level field and the kwargs
    p, rs, _, _ = chat(c, reasoning={"effort": "low"})
    assert p == p_low and rs == "Hello there.", rs
    p, rs, _, _ = chat(c, reasoning={"effort": "none"}, reasoning_effort="low")
    assert p == p_low and rs == "Hello there.", rs
    p, rs, _, _ = chat(c, reasoning={"effort": "low"}, chat_template_kwargs={"reasoning_effort": "none"})
    assert p == p_off and rs is None, rs
    t = c.post("/v1/chat/completions", json={"model": "x", "messages": msgs, "max_tokens": 50, "stream": True,
                                             "reasoning": {"effort": "none"}}).text
    ev = [e["choices"][0]["delta"] for e in sse_events(t) if e.get("choices")]
    assert last_prompt() == p_off and not any(d.get("reasoning_content") for d in ev), t[:400]
    # Anthropic: the same prompt as enable_thinking=false, a text block only
    pa_off, types_, _ = anth(c, **off)
    assert pa_off == p_off and types_ == ["text"], types_
    for body in ({"reasoning_effort": "none"}, {"output_config": {"effort": "none"}},
                 {"chat_template_kwargs": {"reasoning_effort": "NONE"}}, {"reasoning": {"effort": "none"}}):
        p, types_, texts = anth(c, **body)
        assert p == p_off and types_ == ["text"] and texts == ["Hello there."], (body, types_, texts)
    p, types_, _ = anth(c, output_config={"effort": "none"}, thinking={"type": "enabled", "budget_tokens": 1000})
    assert p == p_off and types_ == ["text"], types_  # the request's none wins over its thinking enabled
    r = c.post("/v1/messages/count_tokens", json={"model": "x", "messages": msgs, "reasoning_effort": "none"}).json()
    assert r["input_tokens"] == len(p_off), r
    # server default none: thinking off unless a request turns it on or names an effort
    old = os.environ.pop("QE_REASONING_EFFORT", None)
    os.environ["QE_REASONING_EFFORT"] = "None"
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            c = server([])
        assert "warning: --reasoning-effort" not in out.getvalue(), out.getvalue()
    finally:
        os.environ.pop("QE_REASONING_EFFORT", None)
        if old is not None:
            os.environ["QE_REASONING_EFFORT"] = old
    p, rs, _, _ = chat(c)
    assert p == p_off and rs is None, rs
    p, rs, _, _ = chat(c, chat_template_kwargs={"enable_thinking": True})
    assert p == p_on and rs == "Hello there.", rs  # (template default effort: none is not passed on)
    p, rs, _, _ = chat(c, reasoning_effort="low")
    assert p == p_low and rs == "Hello there.", rs
    p, types_, _ = anth(c, thinking={"type": "adaptive"})
    assert p == p_on and types_[0] == "thinking", types_
    p, rs, _, _ = chat(c, chat_template_kwargs={"enable_thinking": 0})  # (not a bool: not a switch)
    assert p == p_off and rs is None, rs
    c = server(["--think", "0"])
    p, rs, _, _ = chat(c, reasoning_effort="none")
    assert p == p_off and rs is None, rs
    print("  reasoning_effort none: prompt of enable_thinking=false, no reasoning (chat, stream, messages, default)")


def check_logprobs():
    """Issue #10: the fields, the 400s, the numbers (mock logits: raw pattern around the emitted token, log-softmax over the
    whole vocabulary), the streaming shape, reasoning / tool-call / stop tokens, a forced close, requests without the fields."""
    import numpy as np
    so = os.path.join(TMP, "mock_lp.so")
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-DMOCK_GRAMMAR", "-o", so, os.path.join(HERE, "mock_engine.c")])
    log = os.path.join(TMP, "log_lp.txt")
    script = os.path.join(TMP, "script_lp.txt")
    args = app.parse_args(["--model", FLASH, "--engine-lib", so, "--metrics-port", "0",
                           "--stats-file", os.path.join(TMP, "stats_lp.json")])
    tok, _, _ = args.profile.tokenizer()
    enc = lambda x: tok.encode(x, add_special_tokens=False).ids  # noqa: E731
    think = "".join(f"Step {i}: check case {i * 7 % 13}.\n" for i in range(1, 40))
    plain, multi = "The capital of France is Paris.", "Cześć! नमस्ते, ฉันสบายดี"
    replies = [plain, plain, plain, plain, multi, "Reason here.\n</think>\n\n" + plain, "Reason here.\n</think>\n\n" + plain,
               "abc xyz def", "abc xyz def", "Run it.\n</think>\n\n" + CALL_X, think + "</think>\n\nDone now.",
               "\n\n" + plain + "\n", plain, plain, plain]
    with open(script, "w") as f:
        for r in replies:
            f.write(",".join(map(str, enc(r + "<|im_end|>"))) + "\n")
    os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, log
    gen = app.Generator(args)
    c = TestClient(app.build_app(gen, args))
    msgs = [{"role": "user", "content": "hi"}]
    off = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0}
    on = {"chat_template_kwargs": {"enable_thinking": True}, "temperature": 0}

    def post(**kw):
        return c.post("/v1/chat/completions", json=dict({"model": "x", "messages": msgs, "max_tokens": 400}, **kw))

    for bad, field in (({"logprobs": True, "top_logprobs": 21}, "top_logprobs"), ({"logprobs": True, "top_logprobs": -1}, "top_logprobs"),
                       ({"top_logprobs": 5}, "top_logprobs"), ({"logprobs": False, "top_logprobs": 5}, "top_logprobs"),
                       ({"logprobs": True, "top_logprobs": 1.5}, "top_logprobs"), ({"logprobs": True, "top_logprobs": True}, "top_logprobs"),
                       ({"logprobs": "yes"}, "logprobs")):
        r = post(**bad)
        assert r.status_code == 400 and r.json()["error"]["param"] == field, (bad, r.text)
    x = np.full(248320, -50.0)
    x[:30] = 3.0 - 0.5 * np.arange(30)
    want = x - np.log(np.exp(x).sum())

    def check_entries(ents, text, top):
        assert "".join(e["token"] for e in ents) == text, ents
        assert b"".join(bytes(e["bytes"]) for e in ents) == text.encode(), ents
        for e in ents:
            assert abs(e["logprob"] - want[0]) < 1e-6, e
            assert len(e["top_logprobs"]) == top
            assert [abs(a["logprob"] - want[i]) < 1e-6 for i, a in enumerate(e["top_logprobs"])] == [True] * top, e
            assert not top or e["top_logprobs"][0]["bytes"] == e["bytes"], e

    r = post(logprobs=False, top_logprobs=0, **off)
    assert r.status_code == 200 and "logprobs" not in r.json()["choices"][0], r.text  # (ignored before: still accepted)
    # no fields: the key stays away, the constraint and the capture too
    r = post(**off).json()
    assert r["choices"][0]["message"]["content"] == plain and "logprobs" not in r["choices"][0], r
    assert not [ln for ln in open(log) if ln[0] in "KL"]
    r = post(logprobs=True, top_logprobs=5, **off).json()
    assert r["choices"][0]["message"]["content"] == plain
    check_entries(r["choices"][0]["logprobs"]["content"], plain, 5)
    assert [ln.split()[1] for ln in open(log) if ln[0] in "L"] == ["1"], "capture on once"
    # streamed, multi-byte: the same entries on the chunks; top_logprobs default 0
    ev = sse_events(post(logprobs=True, stream=True, **off).text)
    assert all("logprobs" in e["choices"][0] for e in ev if e["choices"])
    ents = [x for e in ev if e["choices"] and e["choices"][0]["logprobs"] for x in e["choices"][0]["logprobs"]["content"]]
    check_entries(ents, plain, 0)
    assert sum(1 for e in ev if e["choices"] and e["choices"][0]["logprobs"]) > 1, "more than one chunk carries entries"
    r = post(logprobs=True, top_logprobs=20, **off).json()
    assert r["choices"][0]["message"]["content"] == multi
    check_entries(r["choices"][0]["logprobs"]["content"], multi, 20)
    # reasoning: none of its tokens, nor </think> and the newlines after it
    r = post(logprobs=True, top_logprobs=1, **on).json()
    assert r["choices"][0]["message"]["reasoning_content"] and r["choices"][0]["message"]["content"] == plain, r
    check_entries(r["choices"][0]["logprobs"]["content"], plain, 1)
    ev = sse_events(post(logprobs=True, stream=True, **on).text)
    assert not [e for e in ev if e["choices"] and "reasoning_content" in e["choices"][0]["delta"] and e["choices"][0]["logprobs"]]
    check_entries([x for e in ev if e["choices"] and e["choices"][0]["logprobs"] for x in e["choices"][0]["logprobs"]["content"]], plain, 0)
    # a stop string: nothing past the cut, streamed or not
    r = post(logprobs=True, stop=["xyz"], **off).json()
    assert r["choices"][0]["message"]["content"].strip() == "abc" and r["choices"][0]["finish_reason"] == "stop", r
    assert "".join(e["token"] for e in r["choices"][0]["logprobs"]["content"]).strip() == "abc", r
    ev = sse_events(post(logprobs=True, stop=["xyz"], stream=True, **off).text)
    assert "".join(x["token"] for e in ev if e["choices"] and e["choices"][0]["logprobs"] for x in e["choices"][0]["logprobs"]["content"]).strip() == "abc", ev
    # a tool call: its tokens are not content
    r = post(logprobs=True, tools=TOOLS_OAI, **on).json()
    assert r["choices"][0]["message"]["tool_calls"] and r["choices"][0]["logprobs"]["content"] == [], r
    # a forced close of <think>: the slot stays constrained (one token per step), the answer gets its entries
    r = post(logprobs=True, thinking_budget=16, **dict(on, max_tokens=400)).json()
    assert r["choices"][0]["message"]["content"] == "Done now.", r
    check_entries(r["choices"][0]["logprobs"]["content"], "Done now.", 0)
    # whitespace around the reply: with logprobs the non-streamed content stays unstripped, the entries join to it
    r = post(logprobs=True, **off).json()
    text = "\n\n" + plain + "\n"
    assert r["choices"][0]["message"]["content"] == text, r
    check_entries(r["choices"][0]["logprobs"]["content"], text, 0)
    # non-finite logits: valid JSON, no 500 (one NaN row, then a row with no finite value); a missing row: no entry
    for mode, n in (("nan", 7), ("allnan", 7), ("norow", 0)):
        os.environ["QE_MOCK_LP"] = mode
        r = post(logprobs=True, top_logprobs=2, **off)
        os.environ.pop("QE_MOCK_LP")
        assert r.status_code == 200, r.text
        json.loads(r.text, parse_constant=lambda c: 1 / 0)
        ents = r.json()["choices"][0]["logprobs"]["content"]
        assert len(ents) >= n and all(e["logprob"] <= 0 for e in ents), ents
        assert mode != "norow" or ents == [], ents
        assert mode != "allnan" or ents[0]["logprob"] == -9999.0, ents
    print("  logprobs: fields, 400s, values, streaming, reasoning, stop, tool call, forced close, whitespace, NaN, no row")
    # an engine library without the read-back: a clear 400, not a reply without numbers
    so0 = build_mock(False)
    args0 = app.parse_args(["--model", FLASH, "--engine-lib", so0, "--metrics-port", "0",
                            "--stats-file", os.path.join(TMP, "stats_lp0.json")])
    c0 = TestClient(app.build_app(app.Generator(args0), args0))
    r = c0.post("/v1/chat/completions", json={"model": "x", "messages": msgs, "logprobs": True})
    assert r.status_code == 400 and "no logits" in r.json()["error"]["message"], r.text
    print("  logprobs: no read-back in the engine library: 400")


def check_window_pool():
    """club-3090 #1966: a library that reports a context larger than its KV pool. The server starts, serves the pool's
    length (/v1/models, /props, the request length check) and says so in one start-up line; the normal case (pool >=
    context) keeps the requested context and prints no "lowered"."""
    import contextlib
    import io
    from types import SimpleNamespace as NS
    so = os.path.join(TMP, "mock_pool.so")
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "mock_engine.c")])
    log, script = os.path.join(TMP, "log_pool.txt"), os.path.join(TMP, "script_pool.txt")
    args = app.parse_args(["--model", FLASH, "--engine-lib", so, "--metrics-port", "0", "--max-ctx", "8K",
                           "--stats-file", os.path.join(TMP, "stats_pool.json")])
    tok, _, _ = args.profile.tokenizer()
    with open(script, "w") as f:
        f.write(",".join(map(str, tok.encode("Fine.<|im_end|>", add_special_tokens=False).ids)) + "\n")
    os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, log
    os.environ["QE_MOCK_POOL_NOW"] = "4096"  # (the mock's pool at start: smaller than the 8192 context it reports)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            gen = app.Generator(args)
    finally:
        del os.environ["QE_MOCK_POOL_NOW"]
    line = [ln for ln in out.getvalue().splitlines() if ln.startswith("context: ")]
    assert line == ["context: requested 8192 tokens, KV pool 4096 tokens, served 4096 tokens per conversation (lowered: "
                    "the engine reported more than its KV pool; longer requests get 400 context_length_exceeded)"], line
    c = TestClient(app.build_app(gen, args))
    m = c.get("/v1/models").json()["data"][0]
    assert m["max_model_len"] == 4096 and m["kv_pool_tokens"] == 4096, m
    assert m["max_prompt_tokens"] == 4096 - args.mtp_k - 2, m
    assert c.get("/props").json()["default_generation_settings"]["n_ctx"] == 4096
    msgs = [{"role": "user", "content": "x " * 5000}]  # over the pool, under the requested context
    r = c.post("/v1/chat/completions", json={"model": "x", "messages": msgs, "max_tokens": 8})
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded", r.text[:300]
    r = c.post("/v1/chat/completions", json={"model": "x", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}],
                                             "chat_template_kwargs": {"enable_thinking": False}})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Fine.", r.text[:300]
    # the line's other cases (no engine needed)
    cl = app.context_line
    assert cl(262144, NS(max_t=262144, max_t_engine=262144, pool_tokens=661504)) == \
        "context: requested 262144 tokens, KV pool 661504 tokens, served 262144 tokens per conversation"
    assert "lowered: by the engine (KV pool, GPU memory" in cl(262144, NS(max_t=99328, max_t_engine=99328, pool_tokens=100352))
    assert "lowered: by the engine" in cl(1 << 20, NS(max_t=262144, max_t_engine=262144, pool_tokens=262144))
    assert "lowered" not in cl(1000000, NS(max_t=1000000, max_t_engine=1000000, pool_tokens=1000448))
    print("  context larger than the KV pool: served the pool, one start-up line, 400 past it, starts anyway")


def main():
    check_window_pool()
    check_disk_cache_flag()
    check_effort_none()
    check_logprobs()
    for tier0 in (True, False):
        print(f"mock engine {'TIER 0 only' if tier0 else 'TIER 0 + 1'} ({FLASH}):")
        os.environ["QE_MOCK_KV_RAM"] = "0" if tier0 else "1"  # tier 1: /v1/models reports kv_cache "ram"
        run(tier0)
    print("ok")


if __name__ == "__main__":
    main()
