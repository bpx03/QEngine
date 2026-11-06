"""CPU test of the hard thinking budget (server/think_budget.py) on the mock engine with the grammar tier
(tests/mock_engine.c -DMOCK_GRAMMAR): request fields and aliases, the forced close at the budget, the answer tail
when max_tokens would end inside <think>, streaming split, tool call after a forced close, /api/stats counters;
the default budget per reasoning effort (issue #4): map parsing, the effort rules of every chat template we serve
(qwen38_syssafe.jinja, the Qwen3.8-27B / -Flash-Next own templates, GLM-5.3-Flash's when present), precedence, the
Anthropic output_config.effort, the budget source in /api/stats and the request log line. No GPU.

  python tests/test_think_budget.py           # Flash-Next model directory (tokenizer / template)
  QE_MOCK_MODEL=models/Qwen3.8-27B-INT4 python tests/test_think_budget.py   # 27B
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


# synthetic templates (machine-independent): GLM-5.3-Flash's effort lines as in its chat_template.jinja, and one that
# renders no effort at all (not recognised)
GLM_LIKE = ("{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and "
            "reasoning_effort in ['low', 'high'] else 'max' -%}"
            "<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}"
            "{% for m in messages %}<|user|>{{ m.content }}{% endfor %}<|assistant|><think>")
NO_EFFORT = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
             "<|im_start|>assistant\n<think>\n")


def _synthetic(name, text):
    path = os.path.join(TMP, name)
    with open(path, "w") as f:
        f.write(text)
    return path


TEMPLATES = [  # (label, path, rules): the chat templates the server may render with
    ("qwen38_syssafe (server default, both Qwen models)",
     os.path.join(HERE, "..", "server", "templates", "qwen38_syssafe.jinja"), "qwen"),
    ("model dir own (QE_MOCK_MODEL)", os.path.join(MODEL, "chat_template.jinja"), "qwen"),
    ("GLM-5.3-Flash effort lines (synthetic)", _synthetic("glm_like.jinja", GLM_LIKE), "glm"),
    ("no effort rendered (synthetic)", _synthetic("no_effort.jinja", NO_EFFORT), "raw"),
] + [(f"{label} (skipped if missing)", path, rules) for label, path, rules in (  # real files, where present
    ("Qwen3.8-27B own", os.path.join(os.environ.get("QE_27B_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-27B-INT4")),
                                     "chat_template.jinja"), "qwen"),
    ("GLM-5.3-Flash own", os.environ.get("QE_GLM_TEMPLATE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "GLM-5.3-Flash", "chat_template.jinja")),
     "glm"))]


def rendered_effort(text):
    """The effort a rendered prompt carries (Qwen3.8: the xhigh / low instruction, medium = none; GLM: the
    'Reasoning Effort: X' line)."""
    if "Reasoning Effort: " in text:
        return text.split("Reasoning Effort: ", 1)[1].split("<", 1)[0].strip().lower()
    for lv in ("xhigh", "low"):
        if f"Reasoning effort is set to {lv}." in text:
            return lv
    return "medium"


def effort_units():
    tb = think_budget
    BM = tb.BudgetMap
    # map parsing: valid
    assert BM.parse(None).config() == 0 and not BM.parse("") and not BM.parse("0") and BM.parse("8192").config() == 8192
    assert BM.parse(" 300 ").plain and BM.parse(300).default == 300 and not BM.parse("-5")  # plain: as 1.3.4
    m = BM.parse("low:4096,medium:16384,xhigh:32768")
    assert m.config() == {"low": 4096, "medium": 16384, "xhigh": 32768} and not m.plain, m.config()
    assert BM.parse(" Low : 5 , *:9").config() == {"low": 5, "default": 9}
    assert BM.parse("default:7").config() == {"default": 7} and BM.parse("max:3,high:2").by_effort == {"max": 3, "high": 2}
    # malformed: a clear error naming the entry
    assert BM.parse("low:-1,default:9").by_effort == {"low": 0}  # negative entry = no budget, as 0 / plain -5
    for bad in ("low", "abc", "low:x", "foo:1", "low:1,low:2", "default:1,*:2", "low:1,", ":5",
                "low:1.5", "minimal:3"):
        try:
            BM.parse(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError as ex:
            assert "thinking budget" in str(ex), ex
    # the start-up check: a malformed value stops the server with the message (argparse exit 2)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            app.parse_args(["--model", MODEL, "--metrics-port", "0", "--thinking-budget", "low:4k"])
            raise AssertionError("malformed --thinking-budget accepted")
        except SystemExit as ex:
            assert ex.code == 2
    assert "QE_THINKING_BUDGET" in err.getvalue() and "'4k' is not a token count" in err.getvalue(), err.getvalue()
    os.environ["QE_THINKING_BUDGET"] = "low:40,xhigh:60"
    try:
        a = app.parse_args(["--model", MODEL, "--metrics-port", "0"])
        assert a.thinking_budget.config() == {"low": 40, "xhigh": 60}
        a = app.parse_args(["--model", MODEL, "--metrics-port", "0", "--thinking-budget", "77"])
        assert a.thinking_budget.plain and a.thinking_budget.default == 77  # the flag wins over the env
    finally:
        del os.environ["QE_THINKING_BUDGET"]
    assert not app.parse_args(["--model", MODEL, "--metrics-port", "0"]).thinking_budget  # unset: no budget

    # effort rules: exactly what the templates do (exact match, no lower-casing)
    Q, G = tb.QWEN_RULES, tb.GLM_RULES
    for e, want in ((None, "xhigh"), ("low", "low"), ("medium", "medium"), ("xhigh", "xhigh"), ("high", "xhigh"),
                    ("minimal", "medium"), ("max", "medium"), ("none", "medium"), ("LOW", "medium"), ("", "medium"),
                    (3, "medium")):
        assert Q.level(e) == want, (e, Q.level(e))
    for e, want in ((None, "max"), ("low", "low"), ("high", "high"), ("xhigh", "max"), ("medium", "max"),
                    ("max", "max"), ("Low", "max")):
        assert G.level(e) == want, (e, G.level(e))
    # client aliases, rewritten before rendering: Qwen max = xhigh, minimal = low; GLM keeps max, minimal = low
    ce = lambda r, e: r.client_effort({"reasoning_effort": e})["reasoning_effort"]  # noqa: E731
    assert ce(Q, "max") == "xhigh" and ce(Q, "minimal") == "low" and ce(Q, "none") == "none" and ce(Q, "MAX") == "MAX"
    assert ce(G, "max") == "max" and ce(G, "minimal") == "low" and ce(G, "xhigh") == "xhigh"
    assert Q.client_effort({}) == {} and ce(Q, 3) == 3
    # probe errors: logged once (template errors only when nothing fits, other errors always)
    logs = []
    assert tb.probe_rules(lambda e: (_ for _ in ()).throw(RuntimeError("boom")), logs.append) is tb.RAW_RULES
    assert len(logs) == 1 and "9 of 9 renders failed" in logs[0] and "RuntimeError: boom" in logs[0], logs
    logs = []
    tb.probe_rules(lambda e: "x" if e is None else 1 / 0, logs.append)
    assert len(logs) == 1 and "ZeroDivisionError" in logs[0], logs
    assert tb.RAW_RULES.level(None) == "xhigh" and tb.RAW_RULES.level("low") == "low"  # unset: the xhigh entry

    # precedence: request > the effort's entry > default > none
    bs = tb.budget_source
    m = BM.parse("low:40,medium:50,xhigh:60")
    for e, want in (("low", (40, "effort:low")), ("medium", (50, "effort:medium")), ("xhigh", (60, "effort:xhigh")),
                    ("high", (60, "effort:xhigh")), ("minimal", (50, "effort:medium")), ("bogus", (50, "effort:medium")),
                    (None, (60, "effort:xhigh"))):
        kw = {} if e is None else {"reasoning_effort": e}
        assert bs({}, kw, m, Q) == want, (e, bs({}, kw, m, Q))
    md = BM.parse("low:40,default:70")
    assert bs({}, {"reasoning_effort": "medium"}, md, Q) == (70, "default")
    assert bs({}, {"reasoning_effort": "low"}, md, Q) == (40, "effort:low")
    assert bs({}, {"reasoning_effort": "low"}, BM.parse("low:0,default:70"), Q) == (None, "effort:low")  # 0: none
    assert bs({}, {"reasoning_effort": "medium"}, BM.parse("low:40"), Q) == (None, "none")
    for body in ({"thinking_budget": 30}, {"chat_template_kwargs": {}, "thinking_token_budget": 30},
                 {"max_thinking_tokens": 30}, {"custom_params": {"thinking_budget": 30}}, {"reasoning": {"max_tokens": 30}},
                 {"thinking": {"type": "enabled", "budget_tokens": 30}}):
        assert bs(body, {"reasoning_effort": "low"}, m, Q) == (30, "request"), body
    assert bs({}, {"reasoning_effort": "low", "thinking_budget": 31}, m, Q) == (31, "request")
    assert bs({"thinking_budget": -1}, {"reasoning_effort": "low"}, m, Q) == (None, "off")  # off over the map
    kw = {"reasoning_effort": "low"}
    assert bs({"thinking_budget": 0}, kw, m, Q) == (None, "request") and kw["enable_thinking"] is False
    # plain integer: one budget for every effort, as in 1.3.4 (int, str and BudgetMap)
    for d in (300, "300", BM.parse("300")):
        for e in (None, "low", "xhigh", "bogus"):
            assert bs({}, {} if e is None else {"reasoning_effort": e}, d, Q) == (300, "default")
    assert tb.request_budget({}, {}, 300) == 300 and tb.request_budget({}, {}) is None
    assert bs({}, {}, 0, Q) == (None, "none")
    # GLM rules: low / high / max (no medium / xhigh level); the default entry covers the rest
    mg = BM.parse("low:40,high:50,max:60")
    assert bs({}, {}, mg, G) == (60, "effort:max") and bs({}, {"reasoning_effort": "xhigh"}, mg, G) == (60, "effort:max")
    assert bs({}, {"reasoning_effort": "high"}, mg, G) == (50, "effort:high")
    mq = BM.parse("low:40,medium:50,xhigh:60,default:70")  # a Qwen-style map on GLM: medium / xhigh never apply
    assert bs({}, {"reasoning_effort": "medium"}, mq, G) == (70, "default") and mq.unreachable(G) == ["medium", "xhigh"]
    assert bs({}, {"reasoning_effort": "low"}, mq, G) == (40, "effort:low")
    assert any("never applies" in l for l in tb.budget_banner(mq, G))
    assert tb.budget_banner(BM.parse("300"), Q) == ["thinking budget: 300 tokens by default (a request's budget wins)"]
    assert tb.budget_banner(BM.parse(""), Q) == []  # nothing set: no line, as before
    assert "low 40, medium 50, xhigh 60, default none" in tb.budget_banner(m, Q)[0]

    # Anthropic output_config.effort (/v1/messages only); the top-level reasoning_effort and the kwargs win
    tk = app.template_kwargs
    assert tk({"output_config": {"effort": "low"}}, True, None, anthropic=True)["reasoning_effort"] == "low"
    assert "reasoning_effort" not in tk({"output_config": {"effort": "low"}}, True)  # OpenAI path: not read
    assert tk({"output_config": {"effort": "low"}, "reasoning_effort": "xhigh"}, True, None,
              anthropic=True)["reasoning_effort"] == "xhigh"
    assert tk({"output_config": {"effort": "low"}, "chat_template_kwargs": {"reasoning_effort": "medium"}}, True, None,
              anthropic=True)["reasoning_effort"] == "medium"
    assert tk({"output_config": {"effort": "low"}}, True, "xhigh", anthropic=True)["reasoning_effort"] == "low"
    assert tk({"output_config": {"format": {"type": "json_schema"}}}, True, "medium",
              anthropic=True)["reasoning_effort"] == "medium"  # no effort in output_config: the server default
    print("  effort map: parsing (valid / malformed / start-up error), rules, precedence, output_config.effort ok")

    # per template: the rules probed from the template, and for every effort a request can send, the budget's
    # level is the effort the rendered prompt carries
    msgs = [{"role": "user", "content": "probe"}]
    for label, path, kind in TEMPLATES:
        if not os.path.exists(path):
            print(f"  template {label}: {path} missing, skipped")
            continue
        t = app.Template(path)
        rules = tb.probe_rules(lambda e: t.render([dict(x) for x in msgs], None, enable_thinking=True,
                                                  reasoning_effort=e))
        glm = kind == "glm"
        assert rules is {"qwen": Q, "glm": G, "raw": tb.RAW_RULES}[kind], (label, rules.describe())
        if kind == "raw":
            assert tb.budget_source({}, {}, BM.parse("low:40,xhigh:60"), rules) == (60, "effort:xhigh")
            # review L3: thinking off gets no server budget (the request's own budget still wins)
            assert tb.budget_source({}, {"enable_thinking": False}, BM.parse("low:40,xhigh:60"), rules) == (None, "none")
            assert tb.budget_source({"thinking_budget": 30}, {"enable_thinking": False}, BM.parse("xhigh:60"), rules) == (30, "request")
            print(f"  template {label}: {rules.describe()}")
            continue
        seen = []
        for e in tb.PROBE_EFFORTS + ("LOW", ""):
            for body, dflt in (({"reasoning_effort": e}, None), ({}, e), ({"output_config": {"effort": e}}, None)):
                kw = rules.client_effort(tk(body, True, dflt, anthropic=True))
                try:
                    text = t.render([dict(x) for x in msgs], None, **kw)
                except Exception:  # noqa: BLE001 (stock Qwen template: unknown efforts fail the request)
                    assert not glm and "own" in label and e not in ("low", "medium", "xhigh", "max", "minimal", None)
                    continue
                lv = rules.level(kw.get("reasoning_effort"))
                assert rendered_effort(text) == lv, (label, e, lv, rendered_effort(text))
                seen.append(lv)
        print(f"  template {label}: {rules.describe()}; rendered levels {sorted(set(seen))} match")


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
    effort_units()

    # (--drop-empty-turns 1: the Flash-Next default, explicit so the HF #50 part runs on a 27B directory too)
    args = app.parse_args(["--model", MODEL, "--engine-lib", so, "--metrics-port", "0", "--drop-empty-turns", "1",
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
             R_LONG,          # 19 max_tokens 300 + ignore_eos: no answer tail
             R_LONG, R_LONG, R_LONG, R_LONG,  # 20-23 effort map: low, high (= xhigh), minimal (= medium), unset
             R_LONG, R_LONG,  # 24-25 Anthropic output_config.effort low; request budget over the map
             R_LONG, R_LONG, R_LONG, R_LONG,  # 26-29 max (chat / Anthropic) = xhigh, none = medium, -1 = off
             R_LONG, R_LONG]  # 30-31 thinking off with the map; plain integer default
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

    # 20-27. default budget per reasoning effort (issue #4) on the served template
    assert gen.effort_rules is think_budget.QWEN_RULES, gen.effort_rules.describe()
    gen.budget_map = think_budget.BudgetMap.parse("low:40,medium:50,xhigh:60")
    st0 = gen.think_snapshot()["budget_from"]

    def eff_run(path, body, want):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if path == "chat":
                r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, **body}).json()
                m = r["choices"][0]["message"]
                rt = r["usage"]["completion_tokens_details"]["reasoning_tokens"]
                assert m["content"] == ANSWER, r
            else:
                r = c.post("/v1/messages", json={"model": name, "max_tokens": 4000, "messages": msgs, **body}).json()
                th = "".join(b["thinking"] for b in r["content"] if b["type"] == "thinking")
                rt = len(tok.encode(th, add_special_tokens=False).ids)
                assert any(b["type"] == "text" and b["text"] == ANSWER for b in r["content"]), r
        line = [l for l in buf.getvalue().splitlines() if l.startswith(f"[{path}]")][-1]
        b, src = want
        if b is None:
            assert " think " not in line or "/" not in line.split(" think ")[1], line
        else:
            assert b <= rt <= b + 3 + wrap_n + 2, (body, rt, b)
            assert f"/{b} ({src})" in line, line
        return rt, line

    cases = [("chat", {"reasoning_effort": "low"}, (40, "effort:low")),
             ("chat", {"reasoning_effort": "high"}, (60, "effort:xhigh")),
             ("chat", {"chat_template_kwargs": {"reasoning_effort": "minimal"}}, (40, "effort:low")),
             ("chat", {}, (60, "effort:xhigh")),                                       # template default xhigh
             ("messages", {"output_config": {"effort": "low"}}, (40, "effort:low")),
             ("chat", {"reasoning_effort": "low", "thinking_budget": 30}, (30, "request")),
             ("chat", {"reasoning_effort": "max"}, (60, "effort:xhigh")),              # client alias max = xhigh
             ("messages", {"output_config": {"effort": "max"}}, (60, "effort:xhigh")),
             ("chat", {"reasoning_effort": "none"}, (50, "effort:medium")),            # the template's fallback
             ("chat", {"reasoning_effort": "low", "thinking_budget": -1}, (None, "off"))]
    for path, body, want in cases:
        rt, line = eff_run(path, body, want)
        print(f"  effort map {path} {body}: {rt} reasoning tokens, log '{line.split(' | ')[-1]}'")
    # 26. thinking off: the prompt does not open <think>, no budget applies (no thinking state at all)
    nk = len(klines())
    r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "reasoning_effort": "low",
                                             "chat_template_kwargs": {"enable_thinking": False}}).json()
    assert not any(k[0] == "C" for k in klines()[nk:]), klines()[nk:]
    p = [l for l in open(log) if l.startswith("P ")][-1].split()
    assert tok.decode([int(x) for x in p[3:]], skip_special_tokens=False).endswith("<think>\n\n</think>\n\n")
    # 27. plain integer default: unchanged meaning (one budget for every effort, source "default")
    gen.budget_map = think_budget.BudgetMap.parse("45")
    eff_run("chat", {"reasoning_effort": "low"}, (45, "default"))
    th = gen.metrics.snapshot()["thinking"]
    got = {k: v - st0.get(k, 0) for k, v in th["budget_from"].items() if v - st0.get(k, 0)}
    assert got == {"effort:low": 3, "effort:xhigh": 4, "effort:medium": 1, "request": 1, "off": 1, "default": 1}, got
    assert th["budget_default"] == 45 and th["effort_rules"].startswith("qwen3.8"), th
    print("  thinking off: no forced close; plain 45: default; /api/stats budget_from", got)
    # 26b. enable_thinking null / 0: the template opens <think> (its test is `is false`), so the reply starts in reasoning on both APIs
    for v in (None, 0):
        kwa = {"chat_template_kwargs": {"enable_thinking": v}}
        r = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "max_tokens": 200, **kwa}).json()
        p = [l for l in open(log) if l.startswith("P ")][-1].split()
        assert tok.decode([int(x) for x in p[3:]], skip_special_tokens=False).endswith("<think>\n"), v
        m = r["choices"][0]["message"]
        assert m.get("reasoning_content") and "</think>" not in (m["content"] or ""), (v, m)
        r = c.post("/v1/messages", json={"model": name, "messages": msgs, "max_tokens": 200, **kwa}).json()
        kinds = [b["type"] for b in r["content"]]
        assert "thinking" in kinds and not any("</think>" in b.get("text", "") for b in r["content"]), (v, r["content"])
        t = c.post("/v1/chat/completions", json={"model": name, "messages": msgs, "max_tokens": 200, "stream": True, **kwa}).text
        assert "reasoning_content" in t and "</think>" not in t.replace("\\u003c", "<"), (v, t[:300])
        t = c.post("/v1/messages", json={"model": name, "messages": msgs, "max_tokens": 200, "stream": True, **kwa}).text
        assert "thinking_delta" in t and "</think>" not in t, (v, t[:300])
    print("  enable_thinking null / 0: reasoning on chat and messages, stream and not")
    # the prompt really carries the aliased effort: max renders the xhigh instruction on /v1/messages
    r = c.post("/v1/messages/count_tokens", json={"model": name, "messages": msgs, "output_config": {"effort": "max"}})
    r2 = c.post("/v1/messages/count_tokens", json={"model": name, "messages": msgs, "reasoning_effort": "xhigh"})
    r3 = c.post("/v1/messages/count_tokens", json={"model": name, "messages": msgs, "reasoning_effort": "none"})
    assert r.json() == r2.json() != r3.json(), (r.json(), r2.json(), r3.json())

    # start-up with a map: the banner (and the warning for a template whose efforts are not recognised)
    for tmpl, want, warn in ((None, "thinking budget by effort: low 40, xhigh 60, default none (template efforts "
                                    "qwen3.8:", False),
                             (os.path.join(TMP, "no_effort.jinja"), "thinking budget by effort: low 40, xhigh 60", True)):
        a2 = app.parse_args(["--model", MODEL, "--engine-lib", so, "--metrics-port", "0", "--thinking-budget",
                             "low:40,xhigh:60", "--stats-file", os.path.join(TMP, "stats2.json")] +
                            (["--template", tmpl] if tmpl else []))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            g2 = app.Generator(a2)
        out = buf.getvalue()
        assert want in out, out
        assert ("were not recognised" in out) == warn, out
        assert (g2.effort_rules is think_budget.RAW_RULES) == warn
        assert think_budget.budget_source({}, {}, g2.budget_map, g2.effort_rules) == (60, "effort:xhigh")
        print("  start-up:", " | ".join(l for l in out.splitlines() if "thinking budget" in l))
    print("ok")


if __name__ == "__main__":
    main()
