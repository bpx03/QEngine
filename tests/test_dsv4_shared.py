"""[shared-srv] CPU test of DSV4 behind server/app.py (model profile "dsv4", server/dsv4_engine.py) with a fake DSV4 C
library (no GPU, no weights; the real tokenizer from the Maya-S GGUF metadata, skipped without it).

  python3 tests/test_dsv4_shared.py

The fake (FakeCLib) keeps dsv4_capi's bookkeeping: one held sequence (tokens + pending token), an archive of parked
conversations, prefill_part / start / resume / trim / step(cap) / step_logits / set_next, live ids. Its "model" replies
from a script chosen by the prompt (an answer, a DSML tool call, a JSON object, a long count) at the reply position, 2
tokens per pass, so reuse, switches and caps change nothing in the text. Checks: chat (thinking off / on, streaming),
multi-turn reuse (cached = prompt_1 + completion_1 - 1), DSML tool calls (OpenAI + Anthropic), stop strings, cache_prompt
false, time-sliced concurrency (4 requests at once, every reply exact, switches > 0), chunked prefill beside a decode,
the shared system-part hint, response_format JSON, tool_choice none / required / named, the thinking budget,
max_tokens caps, /api/stats engine.dsv4 + qetop's DSV4 panel, one-at-a-time with --tier-reset (and the tier reset per
request).
"""
import json
import os
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
_nv = types.ModuleType("pynvml")


def _no_nvml():
    raise RuntimeError("NVML disabled in the test")


_nv.nvmlInit = _no_nvml
sys.modules["pynvml"] = _nv
MAYA = os.environ.get("QE_DSV4_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "DeepSeek-V4.1-Flash-1017-GGUF", "Maya-S"))
TMP = tempfile.mkdtemp(prefix="qe_dsv4_shared_")
os.environ["QE_DIAG_LOG"] = os.path.join(TMP, "diag.log")
os.environ["QE_WARMUP"] = "0"

import numpy as np  # noqa: E402

import dsv4_chat as C  # noqa: E402
import dsv4_engine  # noqa: E402

V = 129280
EOS, ASST, THINK, ETHINK = 1, 128804, 128821, 128822
TOK = None   # set in main()
FAILS = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + str(info) if info else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def script(prompt_text, thinking):
    """the fake model's whole reply (text) for a prompt"""
    pre = "I think it over.</think>" if thinking else ""
    if "weather" in prompt_text:
        body = ('\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n<｜DSML｜ parameter name="city" string="true">'
                'Warsaw</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    elif "JSON" in prompt_text:
        body = '{"answer": 42}'
    elif "Count" in prompt_text:
        body = " ".join(str(i) for i in range(1, 400))
    else:
        body = "The answer is 42. Numbers: 1, 2, 3, 4, 5, 6."
    return pre + body + C.EOS


class FakeL:
    def __init__(self, f):
        self.f = f

    def dsv4_pos(self, h):
        return len(self.f.toks)

    def dsv4_set_nocache(self, h):
        self.f.nocache = True

    def dsv4_reset(self, h):
        self.f.toks, self.f.next = [], -1

    def dsv4_vocab(self):
        return V

    def dsv4_eos(self):
        return EOS


class FakeCLib:
    """dsv4_engine.CLib's interface; the C API's held-sequence / archive bookkeeping in Python"""
    PASS_S = 0.002
    PER_PASS = 2     # tokens per pass (optgram review L9: long passes up to cap)
    TRIM_FAIL = 0    # the next n trims fail and drop the hold (optgram review L9)
    inst = None

    def __init__(self, lib, args):
        FakeCLib.inst = self
        self.args = args
        self.max_ctx = int(args.split("--ctx ")[1].split()[0])
        self.L, self.h = FakeL(self), 1
        self.has_sched, self.has_pfx, self.has_kvhost, self.has_logits = True, True, True, True
        self.toks, self.next, self.live, self.nocache, self.hint = [], -1, 0, False, 0
        self.arch = []   # [toks, next, live]
        self.calls = {"start": 0, "prefill_part": 0, "resume": 0, "resume_miss": 0, "step": 0, "step_logits": 0,
                      "end_request": 0, "trim": 0, "set_live": 0}
        self.hints, self.parts, self.samp = [], [], []
        self.follows = []
        self.lock = threading.Lock()
        self.replies = {}

    # the "model": the scripted reply token at the reply position of a sequence
    def tok_at(self, seq):
        k = len(seq) - 1 - seq[::-1].index(ASST)
        p = k + 1
        thinking = p < len(seq) and seq[p] == THINK
        if p < len(seq) and seq[p] in (THINK, ETHINK):
            p += 1
        key = tuple(seq[:p])
        r = self.replies.get(key)
        if r is None:
            r = self.replies[key] = TOK.encode(script(TOK.decode(seq[:p], skip_special_tokens=False), thinking),
                                               add_special_tokens=False).ids
        i = len(seq) - p
        return r[i] if i < len(r) else EOS

    def _save_held(self):
        if self.toks:
            self.arch = [e for e in self.arch if not (e[0] == self.toks and e[1] == self.next)]
            self.arch.append([list(self.toks), self.next, self.live])
        self.toks, self.next, self.live = [], -1, 0

    def _reuse(self, ids):
        """tokens of ids already held / archived (whole-sequence prefixes, not live)"""
        best, src = 0, None
        cands = ([(self.toks + ([self.next] if self.next >= 0 else []), None)] if not self.live else []) + \
                [(e[0] + ([e[1]] if e[1] >= 0 else []), e) for e in self.arch if not e[2]]
        for seq, e in cands:
            n = 0
            while n < min(len(seq), len(ids)) and seq[n] == ids[n]:
                n += 1
            if n == len(seq) or n >= 1:
                if n > best:
                    best, src = n, e
        return best, src

    def _feed(self, ids, part):
        reuse = 0
        if not self.nocache:
            reuse, src = self._reuse(ids)
            reuse = min(reuse, len(ids) - (0 if part else 1))
        self.nocache = False
        self._save_held()
        time.sleep(0.001 * max(0, len(ids) - reuse) / 1000)
        return reuse

    def set_sampling(self, t, p, k, m, seed):
        self.samp.append((t, p, k, m, seed))

    def start(self, ids, cache=True):
        with self.lock:
            self.calls["start"] += 1
            self.hints.append(self.hint)
            r = self._feed(ids, False)
            self.toks, self.next = list(ids), self.tok_at(list(ids))
            return self.next, r

    def prefill_part(self, ids, follow=0):
        with self.lock:
            self.calls["prefill_part"] += 1
            self.parts.append(len(ids))
            self.follows.append(follow)   # [dsv4-xcall]
            self.hints.append(self.hint)
            r = self._feed(ids, True)
            self.toks, self.next = list(ids), -1
            return r

    def resume(self, ids):
        with self.lock:
            self.calls["resume"] += 1
            if self.toks + [self.next] == list(ids) and self.next >= 0:
                return True
            for e in self.arch:
                if e[0] + [e[1]] == list(ids) and e[1] >= 0:
                    self.arch.remove(e)
                    self._save_held()
                    self.toks, self.next, self.live = e[0], e[1], e[2]
                    return True
            self.calls["resume_miss"] += 1
            return False

    def reusable(self, ids):
        return self._reuse(ids)[0]

    def prefix_hint(self, n):
        self.hint = n

    def prefix_info(self):
        return (len([h for h in self.hints if h]), 0)

    def set_live(self, jid):
        self.calls["set_live"] += 1
        self.live = jid

    def end_live(self, jid):
        if self.live == jid:
            self.live = 0
        for e in self.arch:
            if e[2] == jid:
                e[2] = 0

    def step(self, cap=32):
        with self.lock:
            self.calls["step"] += 1
            time.sleep(self.PASS_S)
            if self.next < 0 or len(self.toks) + 16 > self.max_ctx:
                return []
            out = []
            seq = self.toks + [self.next]
            for _ in range(min(self.PER_PASS, max(1, cap))):
                t = self.tok_at(seq)
                out.append(t)
                seq.append(t)
            self.toks = seq[:-1]
            self.next = out[-1]
            return out

    def step_logits(self):
        with self.lock:
            self.calls["step_logits"] += 1
            if self.next < 0 or len(self.toks) + 16 > self.max_ctx:
                return None
            seq = self.toks + [self.next]
            x = np.zeros(V, np.float32)
            x[self.tok_at(seq)] = 10.0
            self.toks, self.next = seq, -1
            return x

    def set_next(self, t):
        if self.next >= 0:
            return 0
        self.next = int(t)
        return 1

    def trim(self, keep):
        with self.lock:
            self.calls["trim"] += 1
            seq = self.toks + ([self.next] if self.next >= 0 else [])
            if keep >= len(seq):
                return True
            if self.TRIM_FAIL > 0:   # (as dsv4_trim when it cannot move back exactly: the hold is dropped)
                FakeCLib.TRIM_FAIL -= 1
                self.toks, self.next = [], -1
                return False
            self.toks, self.next = seq[:keep - 1], seq[keep - 1]
            return True

    def end_request(self):
        self.calls["end_request"] += 1
        return 0.0

    def archive_info(self):
        return len(self.arch), 1e6 * len(self.arch), "archive", [0, 0, 0, 0]

    def kvpool_info(self):
        return 0, 0.0, 0.0, [0, 0, 0, 0, 0]

    def kvhost_info(self):
        return 0.0, 0.0, [0, 0, 0]

    def archive_pressure(self):
        pass

    def ctx_idle(self):   # [dsv4-rel8-giveback]
        self.calls["ctx_idle"] = self.calls.get("ctx_idle", 0) + 1
        return 1

    def stats(self):
        return None


def make(extra=()):
    import app
    dsv4_engine.CLib = FakeCLib
    args = app.parse_args(["--model", MAYA, "--metrics-port", "0", "--max-ctx", "16384", "--engine-lib", "/nonexistent.so",
                           "--stats-file", os.path.join(TMP, "stats.json"), "--log-file", "0", "--prefill-chunk", "1024"] + list(extra))
    gen = app.Generator(args)
    from fastapi.testclient import TestClient
    return app, gen, TestClient(app.build_app(gen, args)), FakeCLib.inst


def chat(c, msgs, **kw):
    r = c.post("/v1/chat/completions", json=dict(messages=msgs, **kw))
    return r.status_code, r.json()


def want_text(prompt, thinking):
    t = script(prompt, thinking).split(C.EOS)[0]
    return t.split(C.ETHINK)[-1] if thinking else t


def main():
    global TOK
    if not os.path.isdir(MAYA):
        print(f"SKIP: no DSV4 GGUF dir at {MAYA} (QE_DSV4_MODEL)")
        return 0
    t0 = time.time()
    app, gen, c, f = make()
    import dsv4_gguf
    TOK = gen.tok
    check("profile", gen.profile.family == "dsv4" and isinstance(gen.engine, dsv4_engine.Dsv4Engine), gen.profile.family)
    check("sampling defaults from the GGUF", gen.sampling_defaults[True]["temperature"] == 1.0 and gen.sampling_defaults[True]["top_k"] == 0)
    q = [{"role": "user", "content": "What is the capital of Poland? One word."}]
    # 1. chat, thinking off: same prompt ids as dsv4_app (dsv4_chat.encode + tokenizer)
    ids, _ = gen.prompt([dict(m) for m in q], None, {"enable_thinking": False})
    ref = TOK.encode(C.encode(q, None, thinking=False), add_special_tokens=False).ids
    check("prompt ids = dsv4_app's", list(ids) == ref, (len(ids), len(ref)))
    st, d = chat(c, q, reasoning_effort="none", max_tokens=64, temperature=0)
    m = d["choices"][0]["message"]
    check("chat thinking off", st == 200 and m["content"] == want_text("", False) and d["choices"][0]["finish_reason"] == "stop", m)
    u1 = d["usage"]
    tm = d.get("timings") or {}  # [dsv4-rel8-timings] llama.cpp-style
    check("timings (llama.cpp style)", tm.get("prompt_n") == u1["prompt_tokens"] - u1["prompt_tokens_details"]["cached_tokens"]
          and tm.get("predicted_n") == u1["completion_tokens"] and tm.get("prompt_ms", -1) >= 0 and tm.get("predicted_ms", 0) > 0
          and tm.get("predicted_per_second", 0) > 0 and "prompt_per_second" in tm and tm.get("ttft_s", -1) >= 0
          and tm.get("tok_s") == tm.get("predicted_per_second"), tm)
    # 2. multi-turn reuse: turn 2 continues the held sequence (cached = prompt_1 + completion_1 - 1)
    q2 = q + [{"role": "assistant", "content": m["content"]}, {"role": "user", "content": "And of France?"}]
    st, d2 = chat(c, q2, reasoning_effort="none", max_tokens=64, temperature=0)
    c2 = d2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check("REUSE multi-turn", c2 >= u1["prompt_tokens"] + u1["completion_tokens"] - 1, (c2, u1))
    # 3. thinking on, streaming: reasoning / content split
    r = c.post("/v1/chat/completions", json=dict(messages=[{"role": "user", "content": "Is 91 prime?"}], stream=True,
                                                 max_tokens=200, temperature=0))
    rs = cs = ""
    fin = None
    for line in r.text.splitlines():
        if line.startswith("data: {"):
            x = json.loads(line[6:])
            ch = (x.get("choices") or [{}])[0]
            de = ch.get("delta", {})
            rs += de.get("reasoning_content") or ""
            cs += de.get("content") or ""
            fin = ch.get("finish_reason") or fin
    check("stream thinking", rs == "I think it over." and cs == want_text("", False), (rs, cs, fin))
    tl = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: {")]
    check("timings in the stream's usage chunk", tl and tl[-1].get("usage") and (tl[-1].get("timings") or {}).get("predicted_n")
          == tl[-1]["usage"]["completion_tokens"], tl[-1] if tl else None)
    # 4. DSML tool call
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}}]
    wq = [{"role": "user", "content": "What is the weather in Warsaw right now? Use the tool."}]
    seen = []  # [dsv4-rel8-dsml] the agent tracker's view of each reply (call or final answer)
    fin0 = gen.agents.finished
    gen.agents.finished = lambda info, txt, call, split=True: (seen.append((call, txt)), fin0(info, txt, call, split))
    st, d = chat(c, wq, tools=tools, max_tokens=300, temperature=0)
    m = d["choices"][0]
    tc = m["message"].get("tool_calls") or []
    check("tool call (DSML)", m["finish_reason"] == "tool_calls" and len(tc) == 1 and tc[0]["function"]["name"] == "get_weather"
          and json.loads(tc[0]["function"]["arguments"]) == {"city": "Warsaw"} and not (m["message"].get("content") or ""), m)
    st, _ = chat(c, q, max_tokens=300, temperature=0)  # a text answer (thinking on)
    gen.agents.finished = fin0
    check("agent tracker sees the DSML call (L6)", len(seen) == 2 and seen[0][0] is True and "DSML" not in seen[0][1] and not seen[0][1].strip()
          and seen[1] == (False, want_text("", False)), seen)
    # review rel8 L9: a DSML block that does not parse is no call for the tracker (the HTTP side returns it as text)
    ns = types.SimpleNamespace(gen=gen, ids=list(ids), p={}, text_mk="\n\n<｜DSML｜ calls>\nbroken</｜DSML｜ calls>")
    ok_call = types.SimpleNamespace(gen=gen, ids=list(ids), p={}, text_mk=want_text("weather", False))
    check("agent tracker: a malformed DSML block is no call (review L9)", app.Job.answer_call(ns)[1] is False
          and app.Job.answer_call(ok_call)[1] is True, (app.Job.answer_call(ns), app.Job.answer_call(ok_call)))
    # rel re-review L3 [kb-server]: a call the request's tool_choice filters out (none / another tool) is no call either
    import tool_choice as _tcm
    no_call = types.SimpleNamespace(gen=gen, ids=list(ids), text_mk=want_text("weather", False),
                                    p={"tool_choice": {"mode": _tcm.NONE}})
    other = types.SimpleNamespace(gen=gen, ids=list(ids), text_mk=want_text("weather", False),
                                  p={"tool_choice": {"mode": _tcm.REQUIRED, "subset": True, "names": {"get_time"}}})
    check("agent tracker: a call tool_choice filters out is no call", app.Job.answer_call(no_call)[1] is False
          and app.Job.answer_call(other)[1] is False, (app.Job.answer_call(no_call), app.Job.answer_call(other)))
    # streamed tool call
    r = c.post("/v1/chat/completions", json=dict(messages=wq, tools=tools, max_tokens=300, temperature=0, stream=True))
    calls = [x for line in r.text.splitlines() if line.startswith("data: {")
             for x in [json.loads(line[6:])] if (x.get("choices") or [{}])[0].get("delta", {}).get("tool_calls")]
    streamed_c = "".join((json.loads(line[6:]).get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                         for line in r.text.splitlines() if line.startswith("data: {"))
    check("tool call streamed", len(calls) >= 1 and "DSML" not in streamed_c, streamed_c[:80])
    # Anthropic endpoint
    r = c.post("/v1/messages", json={"model": "x", "max_tokens": 300, "messages": wq,
                                     "tools": [{"name": "get_weather", "description": "w", "input_schema": tools[0]["function"]["parameters"]}]})
    a = r.json()
    tu = [b for b in a.get("content", []) if b.get("type") == "tool_use"]
    check("anthropic tool_use", r.status_code == 200 and tu and tu[0]["input"] == {"city": "Warsaw"} and a.get("stop_reason") == "tool_use", a)
    r = c.post("/v1/messages", json={"model": "x", "max_tokens": 64, "messages": q, "thinking": {"type": "disabled"}})
    a = r.json()
    txt = "".join(b.get("text", "") for b in a.get("content", []) if b.get("type") == "text")
    check("anthropic text", r.status_code == 200 and txt == want_text("", False), a)
    check("timings on /v1/messages (review L10)", (a.get("timings") or {}).get("predicted_n") == a["usage"]["output_tokens"], a.get("timings"))
    # 5. stop string: content cut, finish stop, the held sequence trimmed
    n0 = f.calls["trim"]
    st, d = chat(c, q, reasoning_effort="none", max_tokens=64, temperature=0, stop=["3,"])
    cont = d["choices"][0]["message"]["content"]
    check("STOP cut", cont.rstrip() == want_text("", False).split("3,")[0].rstrip() and d["choices"][0]["finish_reason"] == "stop", cont)
    check("STOP trim", f.calls["trim"] > n0, f.calls["trim"])
    st, d = chat(c, [{"role": "user", "content": "Is 91 prime?"}], max_tokens=400, temperature=0, stop=["think", "3,"])
    m = d["choices"][0]["message"]
    check("STOP never in the reasoning", m.get("reasoning_content") == "I think it over." and
          m["content"].rstrip() == want_text("", False).split("3,")[0].rstrip(), m)
    # [dsv4-rel8-think] L7: stops never match inside a DSML block (nor in the blank line before it), thinking on and off
    for eff in (None, "none"):
        kw = {"reasoning_effort": eff} if eff else {}
        st, d = chat(c, wq, tools=tools, max_tokens=300, temperature=0, stop=["\n\n", "Warsaw", "invoke", "\n"], **kw)
        m = d["choices"][0]
        tc = m["message"].get("tool_calls") or []
        check(f"STOP not inside a DSML block ({eff or 'thinking'})", st == 200 and m["finish_reason"] == "tool_calls" and tc
              and json.loads(tc[0]["function"]["arguments"]) == {"city": "Warsaw"}, m)
    # 6. cache_prompt false: nothing reused
    st, d = chat(c, q, reasoning_effort="none", max_tokens=8, temperature=0, cache_prompt=False)
    check("cache_prompt false", d["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, d["usage"])
    # 7. max_tokens: the engine is capped (no pass past what the client keeps)
    st, d = chat(c, [{"role": "user", "content": "Count to many."}], reasoning_effort="none", max_tokens=7, temperature=0)
    check("max_tokens cap", d["usage"]["completion_tokens"] == 7 and d["choices"][0]["finish_reason"] == "length", d["usage"])
    held = f.toks + [f.next]
    check("held sequence = what the client got", len(held) == d["usage"]["prompt_tokens"] + 7, (len(held), d["usage"]))
    # 8. four at once: time-sliced, each reply exact
    res = {}
    sw0 = gen.engine.counters["switches"]
    beats = []  # [dsv4-rel8-hang] the hang deadline per engine call
    beat0 = gen.beat
    gen.beat = lambda what, limit=None: (beats.append((what, limit)), beat0(what, limit))

    def one(k):
        _, x = chat(c, [{"role": "user", "content": f"Count for request {k}."}], reasoning_effort="none", max_tokens=120,
                    temperature=0)
        res[k] = x
    ths = [threading.Thread(target=one, args=(k,)) for k in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    _, ser = chat(c, [{"role": "user", "content": "Count for request 0."}], reasoning_effort="none", max_tokens=120, temperature=0)
    ok = all(res[k]["choices"][0]["message"]["content"] == ser["choices"][0]["message"]["content"] for k in res)
    check("CONC 4 at once exact", ok and len(res) == 4, [(res[k]["usage"]["completion_tokens"], res[k]["choices"][0]["message"]["content"][-30:]) for k in res])
    check("CONC time-sliced (switches)", gen.engine.counters["switches"] > sw0, gen.engine.counters)
    gen.beat = beat0
    steps = [l for w, l in beats if w in ("step", "batch step")]
    pre = [l for w, l in beats if w == "prefill"]
    check("hang limit: decode passes QE_HANG_S, switches / prefills scaled (L4)", steps and pre
          and sum(l is None for l in steps) > len(steps) // 2 and any(l is not None for l in steps)
          and all(l is not None and app.HANG_S < l < app.HANG_S + 2 for l in pre)
          and all(l is None or app.HANG_S < l < app.HANG_S + 2 for l in steps),
          (len(steps), sum(l is None for l in steps), pre[:8], sorted(set(round(l, 2) for l in steps if l))[:6]))
    # [dsv4-rel8-refill] re-prefills (the K/V left the archive: every resume misses) count as prefill
    res.clear()
    rs0 = dict(gen.engine.counters)
    pt0 = gen.metrics.session["prefill_tokens"]
    res_orig = f.resume
    f.resume = lambda ids: (f.arch.clear(), f.calls.__setitem__("resume_miss", f.calls["resume_miss"] + 1), False)[2]
    ths = [threading.Thread(target=one, args=(k,)) for k in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    f.resume = res_orig
    nref = gen.engine.counters["reprefills"] - rs0["reprefills"]
    rtok = gen.engine.counters["refill_tokens"] - rs0["refill_tokens"]
    comp = sum(x["usage"]["prompt_tokens"] - x["usage"]["prompt_tokens_details"]["cached_tokens"] for x in res.values())
    dpt = gen.metrics.session["prefill_tokens"] - pt0
    rf = gen.engine.pool_info()["dsv4"]["refills"]
    import io
    import qetop
    panel = qetop.dsv4_view(gen.engine.pool_info()["dsv4"])
    from rich.console import Console
    buf = io.StringIO()
    Console(file=buf, width=160).print(panel)
    ok = all(res[k]["choices"][0]["message"]["content"] == ser["choices"][0]["message"]["content"] for k in res)
    check("re-prefills counted as prefill (qetop L2)", ok and nref >= 1 and rtok > 0 and dpt == comp + rtok
          and rf["count"] == gen.engine.counters["reprefills"] and gen.engine.counters["refill_s"] > 0 and "prefill" in buf.getvalue(),
          (nref, rtok, comp, dpt, rf, buf.getvalue().splitlines()[-2:]))
    # review rel8 L3: a redo inside a prompt's prefill adds its tokens, not its seconds (prefill_done has them)
    ps0, pt1 = gen.metrics.session["prefill_s"], gen.metrics.session["prefill_tokens"]
    gen.engine._in_pf = True
    gen.engine._refilled(0, 10, time.time() - 0.5)
    gen.engine._in_pf = False
    gen.count_refills()
    check("refill inside a prefill: tokens only (review L3)", gen.metrics.session["prefill_tokens"] == pt1 + 10
          and gen.metrics.session["prefill_s"] == ps0, (gen.metrics.session["prefill_tokens"] - pt1, gen.metrics.session["prefill_s"] - ps0))
    # 9. chunked prefill beside a decode + a shared system part (hint >= 1024 tokens)
    sysm = {"role": "system", "content": "You are a careful assistant. " + "Rule: answer briefly. " * 400}
    nparts = f.calls["prefill_part"]
    res.clear()
    def long1():
        _, x = chat(c, [{"role": "user", "content": "Count for the long one."}], reasoning_effort="none", max_tokens=700,
                    temperature=0)
        res[9] = x
    th = threading.Thread(target=long1)
    th.start()
    time.sleep(0.05)
    st, d = chat(c, [sysm, {"role": "user", "content": "Hello?"}], reasoning_effort="none", max_tokens=64, temperature=0)
    th.join()
    check("CHUNK prefill parts beside a decode", f.calls["prefill_part"] > nparts, (f.calls["prefill_part"], f.parts[-3:]))
    check("CHUNK follow counts (xcall)", f.follows and f.follows[-1] > 0, f.follows[-4:])
    check("XPFX system-part hint", any(h >= 1024 for h in f.hints[-4:]), f.hints[-4:])
    check("CHUNK reply exact", d["choices"][0]["message"]["content"] == want_text("", False), d["choices"][0]["message"])
    # 10. response_format JSON (constrained: host-side masks over step_logits)
    nl = f.calls["step_logits"]
    st, d = chat(c, [{"role": "user", "content": "Reply in JSON."}], reasoning_effort="none", max_tokens=64, temperature=0,
                 response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": {
                     "type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}})
    ok = st == 200
    try:
        ok = ok and json.loads(d["choices"][0]["message"]["content"]) == {"answer": 42}
    except Exception:
        ok = False
    check("response_format json_schema", ok and f.calls["step_logits"] > nl, d)
    st, d = chat(c, [{"role": "user", "content": "What is 6 x 7? Reply in JSON."}], reasoning_effort="none", max_tokens=64,
                 temperature=0.7, response_format={"type": "json_object"})
    try:
        ok = isinstance(json.loads(d["choices"][0]["message"]["content"]), dict)
    except Exception:
        ok = False
    check("response_format json_object sampled", ok, d.get("choices", d))
    # 11. tool_choice none: no call (the DSML token masked)
    st, d = chat(c, wq, tools=tools, tool_choice="none", max_tokens=200, temperature=0)
    m = d["choices"][0]
    check("tool_choice none", st == 200 and not m["message"].get("tool_calls") and m["finish_reason"] != "tool_calls", m)
    check("tool_choice none: DSpark passes, the banned token redone", gen.engine.counters.get("ban_redo", 0) >= 1
          and "DSML" not in (m["message"].get("content") or ""), gen.engine.counters)
    # tool_choice required / named on a prompt whose script answers in text: the header is forced
    for tcv in ("required", {"type": "function", "function": {"name": "get_weather"}}):
        st, d = chat(c, [{"role": "user", "content": "Hello there."}], tools=tools, tool_choice=tcv, max_tokens=200, temperature=0)
        m = d["choices"][0]
        tc = m["message"].get("tool_calls") or []
        check(f"tool_choice {tcv if isinstance(tcv, str) else 'named'}", st == 200 and tc and tc[0]["function"]["name"] == "get_weather", m)
    # review rel8 L8: required + client stops: the forced call stays whole (stops never inside the block)
    for eff in (None, "none"):
        kw = {"reasoning_effort": eff} if eff else {}
        st, d = chat(c, [{"role": "user", "content": "Hello there."}], tools=tools, tool_choice="required", max_tokens=200,
                     temperature=0, stop=["\n\n", "\n", "city"], **kw)
        m = d["choices"][0]
        tc = m["message"].get("tool_calls") or []
        check(f"tool_choice required + stops ({eff or 'thinking'})", st == 200 and tc and tc[0]["function"]["name"] == "get_weather", m)
    # thinking + required: reasoning first, then the forced header
    st, d = chat(c, [{"role": "user", "content": "Hello there."}], tools=tools, tool_choice="required", max_tokens=200, temperature=0)
    m = d["choices"][0]
    check("tool_choice required (thinking)", (m["message"].get("tool_calls") or [{}])[0].get("function", {}).get("name") == "get_weather"
          and m["message"].get("reasoning_content") == "I think it over.", m)
    # [dsv4-rel8-optgram] constrained replies on DSpark passes = one masked step per token (greedy), with redone tokens
    sch = {"type": "object", "properties": {"city": {"type": "string"}, "n": {"type": "integer"}}, "required": ["city", "n"]}
    cases = [("json_schema legal", [{"role": "user", "content": "Reply in JSON."}],
              dict(response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": {
                  "type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}})),
             ("json_schema other keys", [{"role": "user", "content": "Reply in JSON."}],
              dict(response_format={"type": "json_schema", "json_schema": {"name": "b", "schema": sch}})),
             ("required", [{"role": "user", "content": "Hello there."}], dict(tools=tools, tool_choice="required")),
             ("named", [{"role": "user", "content": "Hello there."}],
              dict(tools=tools, tool_choice={"type": "function", "function": {"name": "get_weather"}}))]
    for name, msgs, kw in cases:
        outs = []
        for on in (False, True):
            dsv4_engine.GRAM_PASS = on
            g0 = dict(gen.engine.counters)
            st, d = chat(c, msgs, reasoning_effort="none", max_tokens=120, temperature=0, **kw)
            m = d["choices"][0]["message"]
            outs.append((st, m.get("content"), [(x["function"]["name"], x["function"]["arguments"]) for x in m.get("tool_calls") or []],
                         {k: gen.engine.counters[k] - g0.get(k, 0) for k in ("gram_passes", "gram_viol", "constrained_steps")}))
        dsv4_engine.GRAM_PASS = True
        same = outs[0][:3] == outs[1][:3] and outs[0][0] == 200
        ok = same and outs[1][3]["gram_passes"] > 0 and outs[1][3]["constrained_steps"] < outs[0][3]["constrained_steps"]
        if name == "json_schema other keys":   # (the fake's script is illegal there: every pass redone)
            ok = ok and outs[1][3]["gram_viol"] > 0
        check(f"GRAM_PASS {name}: passes = masked steps", ok, (outs[0][3], outs[1][3], outs[1][1:3] if not same else ""))
    # optgram review L9: long passes (up to 32 tokens: ToolMatcher history, the release point mid-pass, an end token
    # mid-pass), a failed trim (re-prefill), sampled replies (masked steps), a forced think close under tool_choice (L5)
    ref = {}
    for name, msgs, kw in cases:
        for on in (False, True):
            dsv4_engine.GRAM_PASS = on
            FakeCLib.PER_PASS = 32 if on else 2
            g0 = dict(gen.engine.counters)
            st, d = chat(c, msgs, reasoning_effort="none", max_tokens=120, temperature=0, **kw)
            m = d["choices"][0]["message"]
            r = (st, m.get("content"), [(x["function"]["name"], x["function"]["arguments"]) for x in m.get("tool_calls") or []])
            if not on:
                ref[name] = r
        FakeCLib.PER_PASS, dsv4_engine.GRAM_PASS = 2, True
        gp = gen.engine.counters["gram_passes"] - g0.get("gram_passes", 0)
        check(f"GRAM_PASS {name}: 32-token passes = masked steps", r == ref[name] and r[0] == 200 and gp > 0, (r, ref[name], gp))
    for name, msgs, kw in cases[1:2]:   # (every pass redone: the trims run)
        FakeCLib.TRIM_FAIL = 1
        g0 = dict(gen.engine.counters)
        st, d = chat(c, msgs, reasoning_effort="none", max_tokens=120, temperature=0, **kw)
        rp = gen.engine.counters["reprefills"] - g0.get("reprefills", 0)
        r = (st, d["choices"][0]["message"].get("content"), [])
        check("GRAM_PASS a failed trim re-prefills, same reply", FakeCLib.TRIM_FAIL == 0 and rp >= 1 and r == ref[name], (r, ref[name], rp))
        FakeCLib.TRIM_FAIL = 0
    # (rel9 review L4) the end-of-reply trim inside a long pass fails: the hold is dropped, the reply is the same and the
    # next request is served
    FakeCLib.PER_PASS, FakeCLib.TRIM_FAIL = 32, 1
    st, d = chat(c, cases[0][1], reasoning_effort="none", max_tokens=120, temperature=0, **cases[0][2])
    r = (st, d["choices"][0]["message"].get("content"), [])
    left = FakeCLib.TRIM_FAIL
    FakeCLib.PER_PASS, FakeCLib.TRIM_FAIL = 2, 0
    st2, d2 = chat(c, cases[0][1], reasoning_effort="none", max_tokens=120, temperature=0, **cases[0][2])
    check("GRAM_PASS a failed end-of-reply trim: hold dropped, same reply, next request ok",
          left == 0 and r == ref[cases[0][0]] and st2 == 200 and d2["choices"][0]["message"].get("content") == r[1], (r, left, st2))
    g0 = dict(gen.engine.counters)
    st, d = chat(c, cases[0][1], reasoning_effort="none", max_tokens=120, temperature=0.6, seed=7, **cases[0][2])
    gp = gen.engine.counters["gram_passes"] - g0.get("gram_passes", 0)
    cs = gen.engine.counters["constrained_steps"] - g0.get("constrained_steps", 0)
    check("GRAM_PASS sampled: masked steps only (review M1)", st == 200 and gp == 0 and cs > 0, (st, gp, cs))
    fz, sc0 = [], gen.engine.set_check
    for on in (False, True):
        dsv4_engine.GRAM_PASS = on
        calls = []
        gen.engine.set_check = lambda slot, fn, _c=calls: (_c.append(fn is not None), sc0(slot, fn))
        g0 = dict(gen.engine.counters)
        st, d = chat(c, [{"role": "user", "content": "Hello there."}], tools=tools, tool_choice="required", max_tokens=200,
                     temperature=0, thinking_budget=2)
        gen.engine.set_check = sc0
        fz.append((st, d["choices"][0]["message"], gen.engine.counters["gram_passes"] - g0.get("gram_passes", 0), calls))
    dsv4_engine.GRAM_PASS = True
    # (rel9 review L3: the checker is set again after the close: two registrations, the start and the close's end)
    check("GRAM_PASS after a forced think close under tool_choice required (review L5)",
          fz[0][0] == 200 and fz[0][:2] == fz[1][:2] and fz[0][2] == 0 and fz[1][2] > 0 and fz[1][3] == [True, True], fz)
    # 12. thinking budget: the reasoning is closed by force at the budget
    st, d = chat(c, [{"role": "user", "content": "Count to many."}], max_tokens=300, temperature=0, thinking_budget=2)
    m = d["choices"][0]["message"]
    check("thinking budget", st == 200 and m.get("content") is not None and gen.think.get("forced_budget", 0) >= 1, (gen.think, m))
    # [dsv4-rel8-think] L8: the forced close is DSV4's rendered form ("</think>" + content): turn 2 (tools given, so the
    # reasoning stays in the history) renders turn 1's held tokens as its prefix
    ttools = [{"type": "function", "function": {"name": "get_time", "description": "Current time",
                                                "parameters": {"type": "object", "properties": {}}}}]
    t1 = [{"role": "user", "content": "Count to 97."}]
    st, d = chat(c, t1, tools=ttools, max_tokens=80, temperature=0, thinking_budget=2)
    m = d["choices"][0]["message"]
    held = list(f.toks)  # (the fake's pending token after the end token is not part of the turn)
    t2 = t1 + [{"role": "assistant", "content": m["content"], "reasoning_content": m["reasoning_content"]},
               {"role": "user", "content": "And 98?"}]
    ids2, _ = gen.prompt([dict(x) for x in t2], ttools, {})
    ok = list(ids2[:len(held)]) == list(held)
    k = next((i for i, (a, b) in enumerate(zip(ids2, held)) if a != b), min(len(ids2), len(held)))
    check("forced close: turn 2 extends turn 1's held tokens (L8)", st == 200 and ok and held[-1] == EOS
          and "Considering" in m["reasoning_content"],
          (len(held), len(ids2), k, TOK.decode(held[max(0, k - 3):k + 3], skip_special_tokens=False),
           TOK.decode(list(ids2[max(0, k - 3):k + 3]), skip_special_tokens=False)))
    st, d2 = chat(c, t2, tools=ttools, max_tokens=64, temperature=0)
    c2 = d2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check("forced close: turn 2 reuses all of turn 1", c2 >= len(held), (c2, len(held)))
    # 13. /api/stats engine.dsv4 and qetop's panel
    info = gen.engine.pool_info()
    try:
        panel = qetop.dsv4_view(info["dsv4"])
        ok = panel is not None
    except Exception as ex:
        ok, panel = False, ex
    check("qetop DSV4 panel", ok and info["pool_tokens"] == 0, panel)
    check("tier reset call after each idle end", f.calls["end_request"] >= 1, f.calls)
    print(f"engine calls: {f.calls}; adapter: {gen.engine.counters}")
    # [dsv4-rel8-giveback] rel7 L5: the lazy context's give-back runs on the idle tick (mem_guard), never with a request in service
    k0 = f.calls.get("ctx_idle", 0)
    gen.engine.mem_guard()  # (review rel8 L2: the memory check runs also while requests wait: no give-back there)
    km = f.calls.get("ctx_idle", 0)
    gen.engine.idle_tick()
    k1 = f.calls.get("ctx_idle", 0)
    gen.engine.seqs[99] = object()  # (a request in service)
    gen.engine.idle_tick()
    del gen.engine.seqs[99]
    deadline = time.time() + 3  # the scheduler's idle wait calls it by itself (about once a second)
    while f.calls.get("ctx_idle", 0) == k1 and time.time() < deadline:
        time.sleep(0.05)
    check("give-back on the idle tick only (rel7 L5, review L2)", km == k0 and k1 == k0 + 1
          and f.calls.get("ctx_idle", 0) > k1, (k0, km, k1, f.calls.get("ctx_idle")))
    # [dsv4-rel8-ctxdef] --max-ctx default: 262144 asked, 262144 with the static reservation too (user 11-19) (the engine decides at load);
    # an explicit --max-ctx (also abbreviated) wins and asks for no static default
    base = ["--model", MAYA, "--metrics-port", "0", "--engine-lib", "/nonexistent.so", "--log-file", "0"]
    ad = app.parse_args(base)
    ae = app.parse_args(base + ["--max-c", "65536"])
    af = app.parse_args(base + ["--max-ctx=262144"])
    ok = (ad.max_ctx == 262144 and ad.ctx_static == 262144 and "--ctx 262144 --dspark 1 --ctx-static 262144" in
          dsv4_engine.engine_args(ad) and ae.max_ctx == 65536 and not getattr(ae, "ctx_static", 0) and "--ctx-static" not in
          dsv4_engine.engine_args(ae) and af.max_ctx == 262144 and not getattr(af, "ctx_static", 0))
    check("--max-ctx default auto (262144 lazy and static), explicit wins", ok,
          (ad.max_ctx, getattr(ad, "ctx_static", None), ae.max_ctx, getattr(ae, "ctx_static", None), dsv4_engine.engine_args(ad)[:120]))
    # [dsv4-rel8-ctxcap] rel7 L6: a load that could map less than --max-ctx serves what it mapped (the engine caps its
    # max context instead of aborting; the adapter takes dsv4_max_ctx and says so)
    init0 = FakeCLib.__init__

    def init8(self, lib, args):
        init0(self, lib, args)
        self.max_ctx = 8192
    FakeCLib.__init__ = init8
    try:
        e8 = dsv4_engine.Dsv4Engine(app.parse_args(["--model", MAYA, "--metrics-port", "0", "--max-ctx", "16384",
                                                    "--engine-lib", "/x.so", "--log-file", "0"]))
    finally:
        FakeCLib.__init__ = init0
    check("ctx cap: the served context is what the load mapped (rel7 L6)", e8.max_t == 8192 and max(e8.slots) == 8192 - 9,
          (e8.max_t, e8.slots[:1]))
    # 13b. [kb-dsv4-ctxgrow] the context K/V cannot grow in a decode pass of one reply: that reply fails with an error,
    # the other one in service finishes as it would alone, the server goes on
    _, solo = chat(c, [{"role": "user", "content": "Count for request 1."}], reasoning_effort="none", max_tokens=120, temperature=0)
    f = gen.engine.c   # (FakeCLib.inst is the 8K-cap engine of section 13 by now)
    step0, hit = f.step, {"n": 0}

    def step_room(cap=32):
        if "request 0." in TOK.decode(f.toks[-400:], skip_special_tokens=False):
            hit["n"] += 1
            if hit["n"] > 3:
                raise dsv4_engine.CtxNoRoom("no VRAM for this request's context K/V (test)")
        return step0(cap)
    f.step = step_room
    res.clear()
    sc = {}

    def one_sc(k):
        st, x = chat(c, [{"role": "user", "content": f"Count for request {k}."}], reasoning_effort="none", max_tokens=120,
                     temperature=0)
        sc[k], res[k] = st, x
    ths = [threading.Thread(target=one_sc, args=(k,)) for k in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    f.step = step0
    check("ctx no room: that reply fails, the other one finishes exact",
          sc.get(0) is not None and sc[0] >= 400 and sc.get(1) == 200
          and res[1]["choices"][0]["message"]["content"] == solo["choices"][0]["message"]["content"],
          (sc, str(res.get(0))[:160], hit))
    _, after = chat(c, [{"role": "user", "content": "Count for request 1."}], reasoning_effort="none", max_tokens=120, temperature=0)
    check("ctx no room: the server serves on", after["choices"][0]["message"]["content"] == solo["choices"][0]["message"]["content"])
    # review kb M1: the prefill path (a new prompt whose context cannot grow): an error, also as a stream's error event
    start0 = f.start

    def start_room(ids, cache=True):
        if "request 7." in TOK.decode(list(ids)[-40:], skip_special_tokens=False):
            raise dsv4_engine.CtxNoRoom("no VRAM for this request's context K/V (test)")
        return start0(ids, cache)
    f.start = start_room
    st7, x7 = chat(c, [{"role": "user", "content": "Count for request 7."}], reasoning_effort="none", max_tokens=50, temperature=0)
    r = c.post("/v1/chat/completions", json=dict(messages=[{"role": "user", "content": "Count for request 7."}],
                                                 reasoning_effort="none", max_tokens=50, temperature=0, stream=True))
    f.start = start0
    check("ctx no room at prefill: an error reply, the stream ends with an error event and [DONE]",
          st7 >= 400 and "context K/V" in json.dumps(x7) and "context K/V" in r.text and "[DONE]" in r.text,
          (st7, str(x7)[:120], r.text[-200:]))
    # 14. --tier-reset: one at a time, end_request after every request
    gen.stop_requests()
    app2, gen2, c2, f2 = make(["--tier-reset"])
    res.clear()
    c = c2
    ths = [threading.Thread(target=one, args=(k,)) for k in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    check("tier-reset: one at a time", gen2.engine.counters["switches"] == 0 and len(gen2.engine.slots) == 1 and f2.calls["end_request"] == 2,
          (gen2.engine.counters, f2.calls["end_request"]))
    check("tier-reset: no shared prefix hint", not any(f2.hints), f2.hints)
    print(f"{len(FAILS)} failed {FAILS} in {time.time() - t0:.1f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
