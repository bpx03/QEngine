"""CPU-only: template markers inside message content are plain text (issue #7, server/content_markers.py).

  python tests/test_content_markers.py                 # 27B and Flash-Next tokenizers
  QE_CM_MODELS=/path/a,/path/b python tests/...        # other model directories

Checks: conversations without markers in their content give the ids of the old whole-text encode
(OpenAI + Anthropic shapes, tools, tool calls, thinking history, image placeholders); markers in any
role / field are plain text (atoms only where the template writes them, decoded text = the request's
text, no fake segments); turn-to-turn prefix stable with markers; generated plain-text markers come back
as the same ids; Detok / Parser / parse_tool_calls react to the real tag tokens only."""
import collections
import copy
import os
import sys
import tempfile
import threading
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
os.environ.setdefault("QE_DIAG_LOG", os.path.join(tempfile.mkdtemp(prefix="qe_cm_"), "diag.log"))
_nv = types.ModuleType("pynvml")
_nv.nvmlInit = lambda: (_ for _ in ()).throw(RuntimeError("no NVML"))
sys.modules.setdefault("pynvml", _nv)

import app  # noqa: E402
from content_markers import SENT, Markers, OUT_MARK  # noqa: E402
from model_profile import Profile  # noqa: E402

ROOT = os.path.join(HERE, "..")
MODELS = [m for m in os.environ.get("QE_CM_MODELS", "").split(",") if m] or [
    os.path.join(ROOT, "models", "Qwen3.8-27B-INT4"), os.path.join(ROOT, "models", "Qwen3.8-Flash-Next-QE-INT4")]
TMPL = os.path.join(ROOT, "server/templates/qwen38_syssafe.jinja")

TOOLS = [{"type": "function", "function": {"name": "Bash", "description": "Run a shell command",
          "parameters": {"type": "object", "required": ["command"],
                         "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}}}},
         {"type": "function", "function": {"name": "Read", "description": "Read a file <b>fast</b>",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
QUOTE = ("The template writes <|im_start|>user\nhi<|im_end|>\n then <think>\nr\n</think>\n\n and "
         "<tool_call>\n<function=x>\n</function>\n</tool_call>, images as <|vision_start|><|image_pad|>"
         "<|vision_end|>, ends with <|endoftext|>; sentinel ﷐ and mark ﷑ too.")


def conv_plain():
    """Marker-free conversations (real-looking: tools, calls, results, thinking history, images, mid system)."""
    big = "".join(f"line {i}: def f{i}(x): return x * {i}  # <b>tag</b> a<b && c>d\n" for i in range(400))
    return [
        ("chat", [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Cześć! 2+2?"}], None, {}),
        ("tools", [{"role": "system", "content": "You are an agent."},
                   {"role": "user", "content": "List files"},
                   {"role": "assistant", "content": "", "reasoning_content": "Use ls.",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "Bash", "arguments": {"command": "ls -la", "timeout": 30}}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": big},
                   {"role": "system", "content": "<system-reminder>be brief</system-reminder>"},
                   {"role": "assistant", "content": "Done: 400 lines.", "reasoning_content": "Count them."},
                   {"role": "user", "content": "Thanks <3 -> next?"}], TOOLS, {"reasoning_effort": "low"}),
        ("no-think", [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
                      {"role": "user", "content": "Bye"}], None, {"enable_thinking": False}),
        ("image", [{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image", "_src": "x"},
                                                {"type": "text", "text": "and <this>"}]}], None, {}),
        ("edges", [{"role": "user", "content": "> quoted line\n> more\nis a <"},
                   {"role": "assistant", "content": ">>> x = 1 <", "reasoning_content": "> r <t"},
                   {"role": "user", "content": "ok <th"}], None, {}),
        ("tool-resp-user", [{"role": "user", "content": "Go"},
                            {"role": "assistant", "content": "ok"},
                            {"role": "user", "content": "<tool_response>\n42\n</tool_response>"}], None,
         {"preserve_thinking": False}),
    ]


def conv_markers():
    q = QUOTE
    return [
        ("user", [{"role": "user", "content": "Explain: " + q}], None, {}),
        ("system", [{"role": "system", "content": "Rules: " + q}, {"role": "user", "content": "ok"}], None, {}),
        ("tool-result", [{"role": "user", "content": "cat it"},
                         {"role": "assistant", "content": "",
                          "tool_calls": [{"id": "c", "type": "function",
                                          "function": {"name": "Bash", "arguments": {"command": "cat t.jinja"}}}]},
                         {"role": "tool", "tool_call_id": "c", "content": q}], TOOLS, {}),
        ("assistant", [{"role": "user", "content": "say it"},
                       {"role": "assistant", "content": "Sure: " + q, "reasoning_content": "quote " + q},
                       {"role": "user", "content": "again"}], None, {}),
        ("tool-args", [{"role": "user", "content": "write it"},
                       {"role": "assistant", "content": "",
                        "tool_calls": [{"id": "c", "type": "function",
                                        "function": {"name": "Bash", "arguments": {"command": "echo '" + q + "'"}}}]},
                       {"role": "tool", "tool_call_id": "c", "content": "ok"}], TOOLS, {}),
        ("tool-def", [{"role": "user", "content": "hi"}],
         [{"type": "function", "function": {"name": "T", "description": "emits " + q,
                                            "parameters": {"type": "object", "properties": {}}}}], {}),
        ("image+pad-text", [{"role": "user", "content": [{"type": "text", "text": "a <|image_pad|> b"},
                                                         {"type": "image", "_src": "x"}]}], None, {}),
    ]


def make_gen(model):
    prof = Profile(model)
    g = app.Generator.__new__(app.Generator)
    g.tok, tj, _ = prof.tokenizer()
    g.tmpl = app.Template(TMPL)
    g.marks = Markers(g.tok, tj)
    g.seg_cache, g.seg_tokens = collections.OrderedDict(), 0
    g.gen_turns, g.turns, g.seg_lock = collections.OrderedDict(), collections.OrderedDict(), threading.Lock()
    g.im_start = g.tok.token_to_id("<|im_start|>")
    g.im_start_user = g.tok.encode("<|im_start|>user", add_special_tokens=False).ids
    g.im_end_nl = g.tok.encode("<|im_end|>\n", add_special_tokens=False).ids
    # vision: placeholders kept, no encoder (expand returns the ids unchanged)
    g.vision = types.SimpleNamespace(expand=lambda ids, raw: (ids, types.SimpleNamespace(shape=(len(raw),)), None, 0))
    g.vision_lock = threading.Lock()
    fake = types.ModuleType("vision")
    fake.load_image_bytes = lambda s: s
    sys.modules["vision"] = fake
    g.args = types.SimpleNamespace(merge_system=1, drop_empty_turns=0, sort_tools=1)
    return g


def old_ids(g, msgs, tools, kw):
    """master: the whole rendered text tokenized with added-token parsing."""
    m = g.clean_history(copy.deepcopy(msgs))
    for x in m:
        if isinstance(x.get("content"), list):
            x["content"] = [{"type": "image"} if p.get("type") == "image" else p for p in x["content"]]
    return g.tok.encode(g.tmpl.render(m, app.canonical_tools(tools), **kw), add_special_tokens=False).ids


def new_ids(g, msgs, tools, kw):
    return list(g.prompt(copy.deepcopy(msgs), copy.deepcopy(tools), dict(kw))[0])


def strip_markers(x, markers):
    if isinstance(x, str):
        for m in markers:
            x = x.replace(m, "")
        return x.replace(SENT, "").replace(OUT_MARK, "")
    if isinstance(x, list):
        return [strip_markers(v, markers) for v in x]
    if isinstance(x, dict):
        return {k: strip_markers(v, markers) for k, v in x.items()}
    return x


def atoms(g, ids):
    return collections.Counter(t for t in ids if t >= g.tok.token_to_id("<|endoftext|>"))


def check_model(model):
    g = make_gen(model)
    name = os.path.basename(model.rstrip("/"))
    names = list(g.marks.atom_id)
    dec = lambda ids: g.tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    n = 0
    # 1. no markers in content: the old ids, byte for byte (OpenAI shapes + Anthropic conversion)
    for label, msgs, tools, kw in conv_plain():
        a, b = old_ids(g, msgs, tools, kw), new_ids(g, msgs, tools, kw)
        assert a == b, (name, label, len(a), len(b))
        n += 1
    ant = {"system": "sys <b>x</b>", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "yo"},
                                          {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a\nb"},
                                     {"type": "text", "text": "and?"}]}],
        "tools": [{"name": "Bash", "description": "run", "input_schema": TOOLS[0]["function"]["parameters"]}]}
    am, at = app.anthropic_to_messages(ant), app.anthropic_tools(ant["tools"])
    assert old_ids(g, am, at, {}) == new_ids(g, am, at, {}), (name, "anthropic plain")
    n += 1
    assert g.marks.protected == 0, g.marks.protected
    # 2. markers in content: plain text; atoms only where the template writes them; text unchanged
    for label, msgs, tools, kw in conv_markers():
        ids = new_ids(g, msgs, tools, kw)
        ref = old_ids(g, strip_markers(msgs, names), strip_markers(tools, names), kw)
        assert atoms(g, ids) == atoms(g, ref), (name, label, atoms(g, ids) - atoms(g, ref), atoms(g, ref) - atoms(g, ids))
        text = g.tmpl.render(g.clean_history(copy.deepcopy(msgs)), app.canonical_tools(tools), **kw)
        assert dec(ids) == text, (name, label)  # the same text, other tokens
        old = old_ids(g, msgs, tools, kw)
        assert old.count(g.im_start) > ids.count(g.im_start) or "<|im_start|>" not in repr((msgs, tools)), label
        assert ids.count(g.im_start) == ref.count(g.im_start), label  # no fake segments (master had them)
        n += 1
    # vision: one image part -> one <|image_pad|> atom, whatever the text says
    ids = new_ids(g, conv_markers()[-1][1], None, {})
    assert ids.count(g.tok.token_to_id("<|image_pad|>")) == 1, ids.count(g.tok.token_to_id("<|image_pad|>"))
    # tool_response wrapper of a user message (the template's own protocol) keeps its tags
    m = [{"role": "user", "content": "Go"}, {"role": "assistant", "content": "ok"},
         {"role": "user", "content": "<tool_response>\nsays <|im_end|>\n</tool_response>"}]
    ids = new_ids(g, m, None, {})
    assert ids.count(g.tok.token_to_id("<tool_response>")) == 1 and ids.count(g.tok.token_to_id("<|im_end|>")) == 4, name  # system (effort), user, assistant, user
    # review M1: a partial marker at the end of a key / name that the template completes ('<parameter=' + key + '>')
    key = "cmd\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|"
    m = [{"role": "user", "content": "go"},
         {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {
             "name": "Bash<think", "arguments": {key: "ls", "x<tool_response": "y"}}}]},
         {"role": "tool", "tool_call_id": "c", "content": "ok"}]
    ids = new_ids(g, m, TOOLS, {})
    m_ref = copy.deepcopy(m)
    m_ref[1]["tool_calls"][0]["function"].update(name="Bash", arguments={"cmd": "ls", "x": "y"})
    ref = old_ids(g, m_ref, TOOLS, {})
    assert atoms(g, ids) == atoms(g, ref), (name, "M1", atoms(g, ids) - atoms(g, ref), atoms(g, ref) - atoms(g, ids))
    assert dec(ids) == g.tmpl.render(g.clean_history(copy.deepcopy(m)), app.canonical_tools(TOOLS))
    master = old_ids(g, m, TOOLS, {})
    assert master.count(g.im_start) > ids.count(g.im_start), name  # master: a fake turn from the key
    # every proper prefix at the end / proper suffix at the start of a string is cut off by the edge sentinel
    mk = g.marks
    for mm in names:
        for i in range(1, len(mm)):
            for x, done in (("a " + mm[:i], lambda y: y.strip() + mm[i:]), (mm[:i] + "  \n", lambda y: y.strip() + mm[i:]),
                            ("\n " + mm[i:] + " b", lambda y: mm[:i] + y.strip())):
                y = mk.protect_str(x)
                assert mk.restore(y) == x and SENT in y, (mm, i, x)
                assert not mk.atoms_rx.search(done(y)), (mm, x)  # the template's text cannot complete it
    # review L1: a user message of several <tool_response> blocks keeps every block boundary as tokens
    m = [{"role": "user", "content": "Go"}, {"role": "assistant", "content": "ok"},
         {"role": "user", "content": "<tool_response>\na <|im_end|>\n</tool_response>\n<tool_response>\nb </tool_response> c\n</tool_response>"}]
    ids = new_ids(g, m, None, {})
    tr, etr = g.tok.token_to_id("<tool_response>"), g.tok.token_to_id("</tool_response>")
    assert ids.count(tr) == 2 and ids.count(etr) == 2 and ids.count(g.tok.token_to_id("<|im_end|>")) == 4, name
    # Anthropic path with markers in text / tool_result / thinking / tool_use input / system
    ant2 = {"system": "S " + QUOTE, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "U " + QUOTE}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "T " + QUOTE}, {"type": "text", "text": "A " + QUOTE},
                                          {"type": "tool_use", "id": "t9", "name": "Bash", "input": {"command": QUOTE}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t9", "content": [{"type": "text", "text": QUOTE}]}]}],
        "tools": ant["tools"]}
    am, at = app.anthropic_to_messages(ant2), app.anthropic_tools(ant2["tools"])
    ids = new_ids(g, am, at, {})
    ref = old_ids(g, strip_markers(am, names), at, {})
    assert atoms(g, ids) == atoms(g, ref), (name, "anthropic markers")
    n += 1
    # 3. reuse: turn 2 extends turn 1 token for token (markers in content), and a generated reply with
    # plain-text markers comes back as the generated ids
    t1 = [{"role": "system", "content": "sys " + QUOTE}, {"role": "user", "content": "quote: " + QUOTE}]
    p1 = new_ids(g, t1, TOOLS, {})
    gen_prompt = g.tok.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False).ids
    assert p1[-len(gen_prompt):] == gen_prompt
    reply_text = "ok\n</think>\n\nHere: " + QUOTE
    # what the model writes after the fix: the reasoning close as its token, the quote as plain text
    out = (g.tok.encode("ok\n", add_special_tokens=False).ids + [g.tok.token_to_id("</think>")] +
           g.marks.plain_tok().encode("\n\nHere: " + QUOTE, add_special_tokens=False).ids + [g.tok.token_to_id("<|im_end|>")])
    assert dec(out[:-1]) == reply_text
    g.remember_turn(p1, out[:-1])
    t2 = t1 + [{"role": "assistant", "content": "Here: " + QUOTE, "reasoning_content": "ok"},
               {"role": "user", "content": "again " + QUOTE}]
    p2 = new_ids(g, t2, TOOLS, {})
    assert p2[:len(p1) - len(gen_prompt)] == p1[:-len(gen_prompt)]  # the earlier turns keep their ids
    g2 = make_gen(model)  # a fresh server (no generated-turn cache, no segment cache): the same ids
    assert new_ids(g2, t2, TOOLS, {})[:len(p1) - len(gen_prompt)] == p1[:-len(gen_prompt)]
    k = len(p1) + len(out)
    assert p2[:k] == p1 + out[:-1] + g.im_end_nl[:1], (name, app.lcp_len(p2, p1 + out))
    p2b = new_ids(g, t2, TOOLS, {})
    assert p2b == p2  # stable between requests (segment cache)
    n += 1
    # review L2: a real tag token the Parser passes through as text (a real <tool_call> inside the reasoning) comes
    # back as plain text; the turn keeps its generated ids while its KV can be reused: 1100 later turns (past the
    # old 1024-turn bound) do not evict it; only the token bound (QE_GEN_TURNS_TOKENS) does
    pl0 = lambda s: g.marks.plain_tok().encode(s, add_special_tokens=False).ids  # noqa: E731
    eth0, tc0, ime = (g.tok.token_to_id(t) for t in ("</think>", "<tool_call>", "<|im_end|>"))
    out2 = pl0("plan: ") + [tc0] + pl0(" later") + [eth0] + pl0("\n\nDone.") + [ime]
    d0, pr0, got = app.Detok(g.tok, g.marks.out), app.Parser(True), {"reasoning": "", "content": ""}
    for t in out2[:-1]:
        for kind, x in pr0.feed(d0.push([t])):
            got[kind] += x
    for kind, x in pr0.feed("", final=True):
        got[kind] += x
    assert got == {"reasoning": "plan: <tool_call> later", "content": "Done."}, got
    assert not pr0.tool_text
    t3 = t1 + [{"role": "assistant", "content": got["content"], "reasoning_content": got["reasoning"]},
               {"role": "user", "content": "next"}]

    def turn_ids(gen):
        gen.seg_cache.clear()
        gen.seg_tokens = 0
        return new_ids(gen, t3, TOOLS, {})
    g3 = make_gen(model)
    g3.remember_turn(p1, out2[:-1])
    for i in range(1100):
        g3.remember_turn(p1, pl0(f"filler {i}"))
    k3 = len(p1) + len(out2) - 1
    assert turn_ids(g3)[:k3] == p1 + out2[:-1], "L2: generated ids of the turn not reused"
    plain3 = turn_ids(make_gen(model))  # no generated-turn cache: the plain tokenization (the fallback)
    assert plain3[:k3] != p1 + out2[:-1]
    keep = app.GEN_TURNS_TOKENS
    try:  # the bound is by tokens: a small bound evicts the turn, the history then tokenizes as plain text
        app.GEN_TURNS_TOKENS = 64
        g4 = make_gen(model)
        g4.remember_turn(p1, out2[:-1])
        for i in range(8):
            g4.remember_turn(p1, pl0(f"filler {i}"))
        assert sum(len(x) for x in g4.gen_turns.values()) == g4.gen_turn_tokens <= app.GEN_TURNS_TOKENS
        assert turn_ids(g4) == plain3
    finally:
        app.GEN_TURNS_TOKENS = keep
    n += 1
    # 4. output: only the real tag tokens switch the parser; plain-text tags stream as text
    det = app.Detok(g.tok, g.marks.out)
    th, eth, tc, etc = (g.tok.token_to_id(t) for t in ("<think>", "</think>", "<tool_call>", "</tool_call>"))
    pl = lambda s: g.marks.plain_tok().encode(s, add_special_tokens=False).ids  # noqa: E731
    seq = (pl("reason about </think> and <tool_call> here") + [eth] + pl("\n\nAnswer: <|im_start|> <think> </think> "
           "<tool_call> x") + [tc] + pl("\n<function=Bash>\n<parameter=command>\necho '</tool_call>'\n</parameter>\n"
                                       "</function>\n") + [etc])
    p = app.Parser(True)
    parts = {"reasoning": "", "content": ""}
    stream = ""
    for t in seq:  # one token at a time, as the decode loop does
        s = det.push([t])
        stream += s
        for kind, x in p.feed(s):
            parts[kind] += x
    for kind, x in p.feed("", final=True):
        parts[kind] += x
    assert app.unmark(stream) == g.tok.decode(seq, skip_special_tokens=False)
    assert parts["reasoning"] == "reason about </think> and <tool_call> here", parts
    assert parts["content"] == "Answer: <|im_start|> <think> </think> <tool_call> x", parts
    calls = app.parse_tool_calls(p.tool_text, TOOLS)
    assert calls == [{"name": "Bash", "arguments": {"command": "echo '</tool_call>'"}}], calls
    # the old plain-text interface (tests, tools) still parses
    assert app.parse_tool_calls("<tool_call>\n<function=Read>\n<parameter=path>\n/a\n</parameter>\n</function>\n</tool_call>",
                                TOOLS) == [{"name": "Read", "arguments": {"path": "/a"}}]
    # review L6: a value that quotes </parameter> / </function> is not cut there (marked and plain interface)
    v = "{% if x %}</parameter>\n</function>{% endif %}\nend </parameter> ok"
    body = "\n<function=Bash>\n<parameter=command>\n" + v + "\n</parameter>\n</function>\n"
    assert app.parse_tool_calls("<tool_call>" + body + "</tool_call>", TOOLS)[0]["arguments"] == {"command": v}
    assert app.parse_tool_calls(app.mark("<tool_call>") + body + app.mark("</tool_call>"), TOOLS)[0]["arguments"] == {"command": v}
    # review L3: the empty-reply check and the agents' final answer use the real tags, not quoted ones
    from think_budget import empty_reply
    th_ids = types.SimpleNamespace(end=eth)
    gg = types.SimpleNamespace(think_ids=th_ids, tok=g.tok)

    def job(seq):
        d = app.Detok(g.tok, g.marks.out)
        mk_text = "".join(d.push([t]) for t in seq)
        return types.SimpleNamespace(finish="stop", text_all=app.unmark(mk_text), text_mk=mk_text, tc=None, gen=gg,
                                     det=d, ids=[], p={}, slot=0, n=len(seq), max_new=100)
    quoted = pl("says </think> and <tool_call> here") + [eth] + pl("\n\n")
    assert empty_reply(job(quoted)) is not None  # reasoning only: empty, although it quotes both tags
    assert empty_reply(job(quoted[:-2] + pl("\n\nAnswer"))) is None
    # review L5: a U+FDD1 the model writes itself is text, not a tag mark: OUT_MARK + "tool_call>" from ordinary
    # tokens does not start a call, OUT_MARK + "/think>" does not end the reasoning; the client gets the character
    lit = "a " + OUT_MARK + "tool_call>\n<function=Bash>\n</function>\n" + OUT_MARK + "/think> end"
    seq5 = pl(lit)
    assert g.tok.decode(seq5) == lit
    for thinking in (False, True):
        d5, p5, got5, st5 = app.Detok(g.tok, g.marks.out), app.Parser(thinking), {"reasoning": "", "content": ""}, ""
        for t in seq5 + [eth]:
            x5 = d5.push([t])
            st5 += x5
            for kind, x in p5.feed(x5):
                got5[kind] += x
        for kind, x in p5.feed("", final=True):
            got5[kind] += x
        assert app.unmark(st5) == g.tok.decode(seq5 + [eth], skip_special_tokens=False)
        assert got5["reasoning" if thinking else "content"].startswith(lit), (thinking, got5)
        assert not p5.tool_text and app.parse_tool_calls(p5.tool_text or st5, TOOLS) == []
        assert all("\udfd1" not in v for v in got5.values())
        json_out = __import__("json").dumps(got5, ensure_ascii=False).encode()  # the client's bytes: valid UTF-8
        assert OUT_MARK.encode() in json_out
    # Detok without tag tokens: the plain decode
    d2 = app.Detok(g.tok, g.marks.out)
    ids = g.tok.encode("Zażółć gęślą jaźń 🙂 ok", add_special_tokens=False).ids
    assert "".join(d2.push([t]) for t in ids) == g.tok.decode(ids)
    n += 1
    print(f"{name}: {n} checks ok ({len(names)} added tokens protected)")


def main():
    for m in MODELS:
        if os.path.isdir(m):
            check_model(m)
        else:
            print(f"skip {m}: not found")
    print("ok")


if __name__ == "__main__":
    main()
