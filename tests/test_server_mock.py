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
    print("  engine fallbacks:", ", ".join(gen.engine.missing_optional()))


def check_disk_cache_flag():
    """Issue #15: --disk-cache on Flash-Next stops the start with a clear error; the 27B keeps the flag."""
    env = dict(os.environ)
    os.environ.pop("QE_DISK_CACHE", None)
    d = os.path.join(TMP, "dc")
    argv = ["--engine-lib", "none", "--metrics-port", "0", "--disk-cache", d]
    a = app.parse_args(["--model", FLASH] + argv)
    if a.profile.is_flash:
        try:
            app.refuse_disk_cache_on_flash(a)
            assert False, "--disk-cache on Flash-Next did not stop the start"
        except SystemExit as e:
            assert "not supported on Flash-Next" in str(e.code), e.code
    q27 = os.path.join(HERE, "..", "models", "Qwen3.8-27B-INT4")
    if not os.path.isdir(q27):
        q27 = os.path.expanduser("~/GIT/QEngine/models/Qwen3.8-27B-INT4")
    if os.path.isdir(q27):
        a = app.parse_args(["--model", q27] + argv)
        app.refuse_disk_cache_on_flash(a)
        assert a.disk_cache == d and os.environ.pop("QE_DISK_CACHE") == d
    os.environ.clear()
    os.environ.update(env)
    print("  issue #15: --disk-cache stops the start on Flash-Next, unchanged on the 27B")


def main():
    check_disk_cache_flag()
    for tier0 in (True, False):
        print(f"mock engine {'TIER 0 only' if tier0 else 'TIER 0 + 1'} ({FLASH}):")
        os.environ["QE_MOCK_KV_RAM"] = "0" if tier0 else "1"  # tier 1: /v1/models reports kv_cache "ram"
        run(tier0)
    print("ok")


if __name__ == "__main__":
    main()
