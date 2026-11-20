"""[srv-gptoss] CPU test of gpt-oss in server/app.py: model profile "gptoss", the model's own harmony template, the
harmony reply parser (server/harmony.py), tool calls, stops, OpenAI + Anthropic, on the tier-0 mock engine
(tests/mock_engine.c -DMOCK_TIER0, the tier gpt-oss's C API implements today) with the real tokenizer
(QE_GPTOSS_MODEL, default <tree>/models/gpt-oss-120b: a link to the checkpoint; skipped without it). No GPU.

  python3 tests/test_gptoss_server.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
_nv = types.ModuleType("pynvml")


def _no_nvml():
    raise RuntimeError("NVML disabled in the test")


_nv.nvmlInit = _no_nvml
sys.modules["pynvml"] = _nv
MODEL = os.environ.get("QE_GPTOSS_MODEL", os.path.join(HERE, "..", "models", "gpt-oss-120b"))
TMP = tempfile.mkdtemp(prefix="qe_gptoss_")
os.environ["QE_DIAG_LOG"] = os.path.join(TMP, "diag.log")
os.environ["QE_WARMUP"] = "0"
FAILS = []

# scripted replies, used in turn by the mock (one per request below, in this order); (text, plain) parts: plain text is
# tokenized without the special tokens (the model writing a tag as text)
R_ANS = "<|channel|>analysis<|message|>User asks the capital.<|end|><|start|>assistant<|channel|>final<|message|>Warsaw.<|return|>"
R_TOOL = ('<|channel|>analysis<|message|>Need the tool.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary json'
          '<|message|>{"city": "Warsaw"}<|call|>')
R_TOOL2 = ('<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "Paris"}<|call|>')
R_COUNT = "<|channel|>analysis<|message|>Count 1 2 3 4.<|end|><|start|>assistant<|channel|>final<|message|>1\n2\n3\n4\n5<|return|>"
R_QUOTE = ("<|channel|>final<|message|>The tag ", "<|channel|>", " is a control token.<|return|>")
SCRIPT = [R_ANS, R_ANS, R_TOOL, R_TOOL2, R_TOOL, R_COUNT, R_QUOTE, R_ANS]


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + str(info)[:400] if info else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def think_budget_checks(gen, enc):
    """[srv-gptoss-think] ThinkControl over harmony token ids (no engine: a fake one; a forced token takes the place of
    the model's next token, as the grammar mask makes it): the budget closes the analysis with the wrap-up and the
    final header, the answer follows; the model's own close needs nothing; the answer tail closes before max_tokens."""
    import numpy as np
    from think_budget import ThinkControl, empty_reply
    ti = gen.think_ids
    check("harmony think ids", getattr(ti, "harmony", False) and ti.ok and ti.bare[""] == enc(
        "<|end|><|start|>assistant<|channel|>final<|message|>"), ti.bare[""] if ti.ok else None)

    class Eng:
        def __init__(self):
            self.m = np.zeros((8, 6300), dtype=np.uint32)
            self.io = [0] * 32
            self.con, self.limit = False, None

        def set_limit(self, slot, n):
            self.limit = n

        def constrain(self, slot, on):
            self.con = on

        def mask_buffer(self, slot):
            return self.m

        def spec_io(self, slot):
            return self.io

    def run(model_text, budget, max_new=400):
        prompt = enc("<|start|>user<|message|>Hi<|end|><|start|>assistant")
        model = enc(model_text)
        e = Eng()
        det = types.SimpleNamespace(tok=gen.tok, ids=[])
        job = types.SimpleNamespace(ids=prompt, gen=types.SimpleNamespace(engine=e, im_start=None, args=types.SimpleNamespace(mtp_k=1)),
                                    max_new=max_new, n=0, slot=0, det=det, grammar=None, lp=None, p={})
        tc = ThinkControl(job, budget, ti, True)
        job.tc = tc
        job.n_out = lambda: job.n
        out, k = [], 0
        while job.n < max_new and k < len(model) + 200:
            t = tc.fz[tc.fi] if tc.active() else (model[k] if k < len(model) else ti.ret)
            if not tc.active():
                k += 1
            out.append(t)
            det.ids.append(t)
            job.n += 1
            tc.track([t])
            if t in (ti.ret, ti.call):
                break
            tc.after([t])
        return tc, gen.tok.decode(out, skip_special_tokens=False), job

    long_think = "<|channel|>analysis<|message|>" + " ".join(f"step{i}" for i in range(300)) + \
                 "<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    tc, text, job = run(long_think, 40)
    close = "<|end|><|start|>assistant<|channel|>final<|message|>"
    at = text.find(close)
    check("budget 40: analysis closed by the server (wrap-up + final header), then the answer", tc.forced == "budget" and
          at > 0 and 40 <= tc.n <= 40 + ti.wrap_max and tc.n_forced in [len(v) for v in ti.wrap.values()],
          (tc.forced, tc.n, tc.n_forced, text[max(0, at - 150):at + len(close) + 20]))
    check("budget 40: the tracker is in the final body after the close", tc.hs.state == "body" and tc.hs.channel == "final" and
          not tc.open, (tc.hs.state, tc.hs.channel))
    tc, text, job = run(R_ANS, 40)
    check("model closes the analysis itself: no forced close", tc.forced is None and tc.n_forced == 0 and tc.n > 0 and
          text.endswith("Warsaw.<|return|>"), (tc.forced, tc.n, text))
    job.finish, job.text_mk, job.text_all = "stop", "", "x"
    check("harmony reply with an answer is not empty", empty_reply(job) is None)
    tc, text, job = run(long_think, None, max_new=200)
    check("answer tail: the close comes before max_tokens", tc.forced == "max_tokens" and "<|channel|>final<|message|>" in text,
          (tc.forced, tc.n, len(text)))


def main():
    if not os.path.isdir(MODEL):
        print(f"SKIP: no gpt-oss model dir at {MODEL}")
        return 0
    t0 = time.time()
    import app
    from fastapi.testclient import TestClient
    so = os.path.join(TMP, "mock0.so")
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "mock_engine.c"), "-DMOCK_TIER0"])
    args = app.parse_args(["--model", MODEL, "--engine-lib", so, "--metrics-port", "0", "--max-ctx", "16384",
                           "--stats-file", os.path.join(TMP, "stats.json"), "--log-file", "0"])
    tok, tok_json, _ = args.profile.tokenizer()
    plain = app.Markers(tok, tok_json).plain_tok()
    enc = lambda x: tok.encode(x, add_special_tokens=False).ids  # noqa: E731
    script = os.path.join(TMP, "script.txt")
    with open(script, "w") as f:
        for r in SCRIPT:
            ids = enc(r) if isinstance(r, str) else enc(r[0]) + plain.encode(r[1], add_special_tokens=False).ids + enc(r[2])
            f.write(",".join(map(str, ids)) + "\n")
    os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, os.path.join(TMP, "log.txt")
    gen = app.Generator(args)
    c = http = TestClient(app.build_app(gen, args))
    check("profile", gen.profile.family == "gptoss" and app.TOOL_FORMAT == "harmony" and gen.seg_mark == "<|start|>", gen.profile.family)
    check("sampling defaults", gen.sampling_defaults[True]["temperature"] == 1.0 and gen.sampling_defaults[True]["top_p"] == 1.0)
    check("eos ids", {200002, 199999, 200012} <= gen.eos, sorted(gen.eos))
    dec = lambda ids: tok.decode(list(ids), skip_special_tokens=False)  # noqa: E731
    q = [{"role": "user", "content": "What is the capital of Poland?"}]
    # renderer: effort levels, the generation prompt, developer message, segment tokenization = whole-text tokenization
    for eff, want in ((None, "Reasoning: medium"), ("high", "Reasoning: high"), ("low", "Reasoning: low")):
        kw = {"enable_thinking": True} if eff is None else {"enable_thinking": True, "reasoning_effort": eff}
        ids, _ = gen.prompt([dict(m) for m in q], None, kw)
        t = dec(ids)
        check(f"render effort {eff}", want in t and t.endswith("<|start|>assistant"), t[-160:])
    ids, _ = gen.prompt([dict(m) for m in q], None, app.template_kwargs({"reasoning_effort": "none"}, True))
    check("render effort none = low", "Reasoning: low" in dec(ids), dec(ids)[:300])
    sysq = [{"role": "system", "content": "Be brief."}] + q
    ids, _ = gen.prompt([dict(m) for m in sysq], None, {"enable_thinking": True})
    t = dec(ids)
    check("render developer message", "<|start|>developer<|message|># Instructions\n\nBe brief." in t, t[:600])
    check("segment tokenization exact", list(ids) == enc(t), (len(ids), len(enc(t))))
    # [srv-gptoss-cache] the system-part hint: the first user message starts there; another user turn shares it
    h0 = ids.hint
    ids2, _ = gen.prompt([dict(m) for m in sysq[:1]] + [{"role": "user", "content": "Something else entirely."}], None,
                         {"enable_thinking": True})
    check("system-part hint at the first user message", h0 > 0 and dec(list(ids)[h0:]).startswith("<|start|>user") and
          ids2.hint == h0 and list(ids2)[:h0] == list(ids)[:h0], (h0, ids2.hint))
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}}]
    hist = q + [{"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
                 "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Warsaw"}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 4}'}]
    ids, _ = gen.prompt([dict(m) for m in hist], tools, {"enable_thinking": True})
    t = dec(ids)
    check("render tool history", 'to=functions.get_weather<|channel|>commentary json<|message|>{"city": "Warsaw"}<|call|>' in t
          and "<|channel|>analysis<|message|>Need the tool.<|end|>" in t and "namespace functions" in t, t[-700:])
    # [gptoss-rel1] blocker 8: several tool calls in one turn -> one call + its result per call (results matched by id)
    tools2 = tools + [{"type": "function", "function": {"name": "get_time", "description": "Local time",
                                                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    hist2 = q + [{"role": "assistant", "content": None, "reasoning_content": "Two tools.", "tool_calls": [
                     {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Warsaw"}'}},
                     {"id": "b", "type": "function", "function": {"name": "get_time", "arguments": '{"city": "Warsaw"}'}}]},
                 {"role": "tool", "tool_call_id": "b", "content": "12:00"},
                 {"role": "tool", "tool_call_id": "a", "content": "4 C"}]
    ids, _ = gen.prompt([dict(m) for m in hist2], tools2, {"enable_thinking": True})
    t = dec(ids)
    w1 = t.find("to=functions.get_weather<|channel|>commentary json<|message|>")
    r1 = t.find('<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>"4 C"')
    w2 = t.find("to=functions.get_time<|channel|>commentary json<|message|>")
    r2 = t.find('<|start|>functions.get_time to=assistant<|channel|>commentary<|message|>"12:00"')
    check("render two calls in one turn: call, result, call, result", 0 <= w1 < r1 < w2 < r2 and t.count("Two tools.") == 1,
          (w1, r1, w2, r2, t[-560:]))
    # blocker 9: a second leading system / developer message is merged; a later one becomes a user message
    ids, _ = gen.prompt([{"role": "system", "content": "Be brief."}, {"role": "developer", "content": "Use metric units."}] + q
                        + [{"role": "assistant", "content": "Warsaw."}, {"role": "system", "content": "Now answer in French."},
                           {"role": "user", "content": "Again?"}], None, {"enable_thinking": True})
    t = dec(ids)
    check("render two leading system messages merged", "Be brief.\n\nUse metric units." in t, t[:700])
    check("render a later system message kept", "<|start|>user<|message|>[system message] Now answer in French.<|end|>" in t, t[-400:])
    # blocker 6: content parts (text + an image without vision) render as text
    ids, _ = gen.prompt([{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image", "_src": "x"}]}],
                        None, {"enable_thinking": True})
    t = dec(ids)
    check("render content parts as text", "<|start|>user<|message|>What is this?\n\n[image omitted" in t, t[-300:])
    import harmony as _h
    check("unknown content part kept as JSON (review rel1 L15)", _h._text([{"type": "json", "json": {"a": 1}}]) == '{"type": "json", "json": {"a": 1}}')
    # blocker 7: other APIs' effort names -> the harmony levels
    check("effort rules: harmony", gen.effort_rules.name.startswith("harmony"), gen.effort_rules.describe())
    for eff, want in (("max", "high"), ("xhigh", "high"), ("minimal", "low"), ("none", "low"), ("bogus", "medium"), ("high", "high"), (" High", "high")):
        kw = gen.effort_rules.client_effort(app.template_kwargs({"reasoning_effort": eff}, True))
        ids, _ = gen.prompt([dict(m) for m in q], None, kw)
        t = dec(ids)
        check(f"render effort {eff} = {want}", f"Reasoning: {want}\n" in t, t[:400])
    # 1. chat, non-streamed: reasoning / content
    r = c.post("/v1/chat/completions", json={"messages": q, "max_tokens": 200})
    m = r.json()["choices"][0]
    check("chat", r.status_code == 200 and m["message"].get("content") == "Warsaw." and
          m["message"].get("reasoning_content") == "User asks the capital." and m["finish_reason"] == "stop", m)
    # 2. streamed
    r = c.post("/v1/chat/completions", json={"messages": q, "max_tokens": 200, "stream": True})
    rs = cs = ""
    for line in r.text.splitlines():
        if line.startswith("data: {"):
            de = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta", {})
            rs += de.get("reasoning_content") or ""
            cs += de.get("content") or ""
    check("stream", rs == "User asks the capital." and cs == "Warsaw.", (rs, cs))
    # 3. tool call (recipient before the channel), 4. (recipient after the channel, <|constrain|>)
    for tag, city in (("tool call", "Warsaw"), ("tool call (recipient after channel)", "Paris")):
        r = c.post("/v1/chat/completions", json={"messages": q, "tools": tools, "max_tokens": 200})
        m = r.json()["choices"][0]
        tc = m["message"].get("tool_calls") or []
        check(tag, m["finish_reason"] == "tool_calls" and len(tc) == 1 and tc[0]["function"]["name"] == "get_weather"
              and json.loads(tc[0]["function"]["arguments"]) == {"city": city} and not (m["message"].get("content") or ""), m)
    # 5. Anthropic tool_use
    r = c.post("/v1/messages", json={"model": "x", "max_tokens": 200, "messages": q,
                                     "tools": [{"name": "get_weather", "description": "w", "input_schema": tools[0]["function"]["parameters"]}]})
    a = r.json()
    tu = [b for b in a.get("content", []) if b.get("type") == "tool_use"]
    th = [b for b in a.get("content", []) if b.get("type") == "thinking"]
    check("anthropic tool_use", r.status_code == 200 and tu and tu[0]["input"] == {"city": "Warsaw"} and a.get("stop_reason") == "tool_use"
          and th and th[0]["thinking"] == "Need the tool.", a)
    # 6. stop strings: the final channel only (the analysis has "3" too)
    r = c.post("/v1/chat/completions", json={"messages": q, "max_tokens": 200, "stop": ["3"]})
    m = r.json()["choices"][0]
    check("stop in the final channel only", m["message"].get("reasoning_content") == "Count 1 2 3 4." and
          m["message"].get("content", "").rstrip() == "1\n2" and m["finish_reason"] == "stop", m)
    # 7. a control tag written as text stays text
    r = c.post("/v1/chat/completions", json={"messages": q, "max_tokens": 200})
    m = r.json()["choices"][0]
    check("quoted tag is text", m["message"].get("content") == "The tag <|channel|> is a control token.", m)
    # 8. Anthropic text
    r = c.post("/v1/messages", json={"model": "x", "max_tokens": 200, "messages": q})
    a = r.json()
    txt = "".join(b.get("text", "") for b in a.get("content", []) if b.get("type") == "text")
    check("anthropic text", txt == "Warsaw." and a.get("stop_reason") == "end_turn", a)
    # 9. grammar tier automata (server/harmony.py; the engine side is the C API's): tool_choice none / required / named
    # and structured output's forced final header, walked over real token ids with the masks checked per token
    import numpy as np
    import harmony
    from grammar import allowed
    tier = harmony.HarmonyTier(gen.tok, None, sorted(gen.eos), None)
    W = (201088 + 31) // 32

    def walk(m, text):
        """tokens of text accepted one by one (each inside the mask before it); returns (accepted count, total)"""
        ids = gen.tok.encode(text, add_special_tokens=False).ids
        buf = np.zeros(W, np.uint32)
        for k, t in enumerate(ids):
            m.unsafe_compute_mask_ptr(buf.ctypes.data, buf.nbytes)
            if not allowed(buf, t) or not m.consume_token(t):
                return k, len(ids)
        return len(ids), len(ids)

    req = {"mode": "required", "names": ["get_weather"]}
    k, n = walk(tier.matcher(req, False), R_TOOL2)
    check("tool_choice required: call header accepted", k == n, (k, n))
    m = tier.matcher(req, False)
    walk(m, "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>")
    check("tool_choice required: released after the header", m.released())
    k, n = walk(tier.matcher(req, False), R_ANS)
    ids = gen.tok.encode(R_ANS, add_special_tokens=False).ids
    check("tool_choice required: no final answer", k < n and "final" in gen.tok.decode(ids[:k + 4], skip_special_tokens=False)[-40:],
          gen.tok.decode(ids[:k + 1], skip_special_tokens=False))
    m = tier.matcher(req, False)
    ff = m.compute_ff_tokens()
    check("tool_choice required: the header's first token forced", ff[:1] == gen.tok.encode("<|channel|>", add_special_tokens=False).ids, ff)
    k, n = walk(tier.matcher({"mode": "required", "names": ["get_weather"]}, False),
                "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>")
    check("tool_choice required: analysis, then the call", k == n, (k, n))
    k, n = walk(tier.matcher({"mode": "none"}, False), R_ANS)
    check("tool_choice none: answer accepted", k == n, (k, n))
    k, n = walk(tier.matcher({"mode": "none"}, False), R_TOOL2)
    check("tool_choice none: no call", k < n, (k, n))
    k, n = walk(tier.matcher({"mode": "required", "names": ["a_tool", "get_weather"]}, False), R_TOOL2)
    check("tool_choice named of two: accepted", k == n, (k, n))
    if gen.grammars is None:   # (the tier-0 mock has no grammar C API: the server's Grammars over the real tokenizer)
        from grammar import Grammars
        gen.grammars = Grammars(open(os.path.join(MODEL, "tokenizer.json")).read(), 201088, sorted(gen.eos))
    if gen.grammars.available:
        c = app.structured(gen, {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}}, {})
        k, n = walk(c.m, '<|channel|>final<|message|>{"city": "Paris"}')
        check("structured output: final header + schema body", k == n and c.m.is_accepting(), (k, n))
        c = app.structured(gen, {"response_format": {"type": "json_object"}}, {})
        k, n = walk(c.m, '<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>{"a": 1}')
        check("structured output: analysis, then final + body", k == n and c.m.is_accepting(), (k, n))
        k2, _ = walk(app.structured(gen, {"response_format": {"type": "json_object"}}, {}).m, "<|channel|>commentary<|message|>x")
        check("structured output: no commentary", k2 == 1, k2)
    think_budget_checks(gen, enc)
    # review L9: a header without <|start|> and a reply cut inside a header give no header text as content
    import harmony
    from content_markers import mark
    hp = harmony.HarmonyParser(True)
    o = hp.feed(mark("<|channel|>") + "analysis" + mark("<|message|>") + "think" + mark("<|end|>") + mark("<|channel|>") + "final" +
                mark("<|message|>") + "Answer." + mark("<|return|>"), final=True)
    txt = "".join(t for k, t in o if k == "content")
    check("L9 header without <|start|>: content is the answer only", txt == "Answer.", o)
    hp = harmony.HarmonyParser(True)
    o = hp.feed(mark("<|channel|>") + "analysis" + mark("<|message|>") + "t" + mark("<|end|>") + mark("<|start|>") + "assistant" +
                mark("<|channel|>") + "fin", final=True)
    check("L9 reply cut inside a header: no header text as content", not any(k == "content" for k, _ in o), o)
    # [gptoss-rel1] blocker 6 over HTTP: an image part (no vision) is answered, not HTTP 500 (the script cycles: R_ANS)
    img = [{"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    r = http.post("/v1/chat/completions", json={"messages": img, "max_tokens": 200})
    check("image part (chat): 200", r.status_code == 200, (r.status_code, r.text[:300]))
    r = http.post("/v1/messages", json={"model": "x", "max_tokens": 200, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is this?"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]})
    check("image part (anthropic): 200", r.status_code == 200, (r.status_code, r.text[:300]))
    # srv L8: a rollback longer than the kept history is refused with a clear error, a long one inside it works
    m = harmony.HarmonyTier(gen.tok, None, sorted(gen.eos), None).matcher({"mode": "none"}, False)
    body = gen.tok.encode("<|channel|>final<|message|>" + "word " * 300, add_special_tokens=False).ids
    for t_ in body:
        m.consume_token(t_)
    m.rollback(200)
    try:
        m.rollback(10_000)
        check("L8 rollback past the history refused", False)
    except ValueError:
        check("L8 rollback past the history refused", True)
    # blocker 11: a missing gpt-oss library stops the start with the reason (no fallback to the 27B library)
    lib = os.path.join(HERE, "..", "build_gptoss", "libqgptoss.so")
    if not os.path.exists(lib) and not os.environ.get("QE_LIB"):
        try:
            app.parse_args(["--model", MODEL, "--metrics-port", "0"])
            check("missing gpt-oss library stops the start", False)
        except SystemExit as ex:
            check("missing gpt-oss library stops the start", "src/gptoss/build.sh" in str(ex.code), ex.code)
    print(f"{len(FAILS)} failed {FAILS} in {time.time() - t0:.1f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
