"""CPU-only: the reply text must not depend on where decode steps end (under 10 s).

A decode step emits 1..k+1 tokens and k is chosen from measured step times, so the same tokens reach the server in different
chunks from run to run. Every chunking of a reply has to give what one single chunk gives: same reasoning, content and tool text,
same order of reasoning and content runs (the Anthropic blocks).

  python tests/test_parser_chunks.py     # 1: Parser on marked text, 3: Job.feed with a byte tokenizer, always;
                                         # 2 and 4 (Detok + Parser on token ids, ThinkControl) need the 27B tokenizer: skipped is said in the last line
  QE_SERVER_DIR=<other server/> python tests/test_parser_chunks.py     # another version of the server"""
import itertools
import os
import random
import sys
import types

sys.path.insert(0, os.environ.get("QE_SERVER_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import app  # noqa: E402
import think_budget  # noqa: E402

TH, ETH, TC, ETC = (app.mark(t) for t in ("<think>", "</think>", "<tool_call>", "</tool_call>"))
CALL = TC + "\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n" + ETC


def outcome(p, feed_out):
    """Reasoning and content runs, adjacent runs of a kind merged; tool text."""
    runs = []
    for kind, t in feed_out:
        if runs and runs[-1][0] == kind:
            runs[-1][1] += t
        else:
            runs.append([kind, t])
    return tuple((k, t) for k, t in runs), p.tool_text


def parse(chunks, thinking, promote=True):
    p, out = app.Parser(thinking), []
    for c in chunks:
        out += p.feed(c)
    out += p.feed("", final=True, promote=promote)
    return outcome(p, out)


def splits(n, rng, tries=200):
    """Cut lists of a sequence of n items: all of them if few, else all with 1 and 2 cuts, one-per-item, and random ones."""
    pts = range(1, n)
    if n <= 13:
        for r in range(n):
            yield from itertools.combinations(pts, r)
        return
    yield ()
    yield from itertools.combinations(pts, 1)
    yield from itertools.combinations(pts, 2)
    yield tuple(pts)
    for _ in range(tries):
        yield tuple(sorted(rng.sample(pts, rng.randint(3, min(12, n - 1)))))


def cut(seq, cuts):
    e = (0,) + tuple(cuts) + (len(seq),)
    return [seq[a:b] for a, b in zip(e, e[1:])]


# (name, thinking, stream, expected single-chunk outcome: reasoning, content, tool text present)
TEXTS = [
    ("plain answer", False, "Hello there.\nBye\n", ("", "Hello there.\nBye\n", False)),
    ("leading newline, no block", False, "\nHello", ("", "\nHello", False)),
    ("think first, thinking off", False, TH + "plan" + ETH + "\n\nanswer", ("plan", "answer", False)),
    ("whitespace, then think", False, "\n " + TH + "plan" + ETH + "\nanswer", ("plan", "answer", False)),
    ("real think after content stays text", False, "abc\n" + TH + "\n\n" + ETH + "\n\nRule 1", ("", "abc\n<think>\n\n</think>\n\nRule 1", False)),
    ("the 69f543ce tail", False, "<|im_end|>\n<|im_start|>assistant\n" + TH + "\n\n" + ETH + "\n\nRule 1: answer",
     ("", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nRule 1: answer", False)),
    ("thinking on, normal", True, "plan\n" + ETH + "\n\nanswer", ("plan\n", "answer", False)),
    ("thinking on, second block right after the first", True, "one" + ETH + "\n\n" + TH + "two" + ETH + "\n\nanswer", ("onetwo", "answer", False)),
    ("thinking on, think after the answer began", True, "one" + ETH + "\n\nanswer " + TH + "x" + ETH + " end", ("one", "answer <think>x</think> end", False)),
    ("think in reasoning stays text", True, "a" + TH + "b" + ETH + "c", ("a<think>b", "c", False)),
    ("stray end tag in content stays text", False, "a" + ETH + "b", ("", "a</think>b", False)),
    ("reply ends inside a block", False, TH + "never closed", ("never closed", "", False)),
    ("bare think at the very end", False, "abc" + TH, ("", "abc<think>", False)),
    ("bare think as the whole reply", False, TH, ("", "", False)),
    ("tool call after text", False, "I will run it.\n" + CALL, ("", "I will run it.\n", True)),
    ("tool call first", False, CALL, ("", "", True)),
    ("thinking, then a tool call", True, "plan" + ETH + "\n\n" + CALL, ("plan", "", True)),
    ("think, then a tool call, thinking off", False, TH + "plan" + ETH + CALL, ("plan", "", True)),
    ("real think after content does not hide a tool call", False, "abc " + TH + "x " + CALL, ("", "abc <think>x ", True)),
    ("tool call in an open block at the end: run", True, "plan " + CALL, ("plan ", "", True)),
    ("tool call in a re-opened block at the end: run", False, TH + "plan " + CALL + "\n", ("plan ", "", True)),
    ("two blocks, tool call in the second: run", True, "one" + ETH + TH + "two " + CALL, ("onetwo ", "", True)),
    ("tool call quoted in a block that closes: reasoning", True, "see " + CALL + " so" + ETH + "\n\nanswer",
     ("see <tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call> so", "answer", False)),
    ("quoted in the block, then the real call: one call", True, "see " + CALL + ETH + "\n\n" + CALL,
     ("see <tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>", "", True)),
    ("complete call, then more reasoning, then the end: reasoning", True, "see " + CALL + " more thoughts",
     ("see <tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call> more thoughts", "", False)),
    ("two calls end the open block: both run", True, "plan " + CALL + "\n" + CALL + "\n", ("plan ", "", True)),
    ("quoted call, more reasoning, then a call at the end: the last runs", True, "see " + CALL + " so " + CALL,
     ("see <tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call> so ", "", True)),
    ("complete call, then a cut-off one, in an open block: reasoning", True, "plan " + CALL + TC + "\n<function=Bash>",
     ("plan <tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call><tool_call>\n<function=Bash>", "", False)),
    ("a tag pair in reasoning is text", True, "the " + TC + " and " + ETC, ("the <tool_call> and </tool_call>", "", False)),
    ("a tag with other text, then a real call", True, "the " + TC + " tag " + CALL, ("the <tool_call> tag ", "", True)),
    ("call cut off in an open block stays reasoning", True, "plan " + TC + "\n<function=Bash>", ("plan <tool_call>\n<function=Bash>", "", False)),
    ("think after a tool call is tool text", False, "x" + CALL + TH + "y" + ETH, ("", "x", True)),
    ("literal tags, thinking off", False, "<think> quoted </think> and <tool_call> too", ("", "<think> quoted </think> and <tool_call> too", False)),
    ("literal think first stays content", False, "<think>quoted", ("", "<think>quoted", False)),
    ("literal tags, thinking on", True, "about </think> and <think>" + ETH + "\n\nsee <think> and <tool_call> x",
     ("about </think> and <think>", "see <think> and <tool_call> x", False)),
    ("text that looks like a partial tag", False, "a < b, <thin", ("", "a < b, <thin", False)),
]


def check_texts():
    rng = random.Random(7)
    bad = n = 0
    for name, thinking, s, (want_r, want_c, want_tool) in TEXTS:
        one = parse([s], thinking)
        r = "".join(t for k, t in one[0] if k == "reasoning")
        c = "".join(t for k, t in one[0] if k == "content")
        if (r, c, bool(one[1])) != (want_r, want_c, want_tool):
            print(f"FAIL {name}: single chunk gives reasoning {r!r} content {c!r} tool {one[1]!r}")
            bad += 1
        seen = {}
        for cuts in splits(len(s), rng):
            seen.setdefault(parse(cut(s, cuts), thinking), cuts)
            n += 1
        if len(seen) > 1:
            print(f"FAIL {name}: {len(seen)} outcomes over the chunkings, e.g.")
            for res, cuts in list(seen.items())[:2]:
                print(f"     cuts {cuts}: {res}")
            bad += 1
    # without promote (any end but the model's own): no call from inside a block, the text stays reasoning
    q = "plan " + CALL
    for chunks in ([q], list(q)):
        got = parse(chunks, True, promote=False)
        if got != ((("reasoning", app.unmark(q)),), ""):
            print(f"FAIL no promote: {got}")
            bad += 1
    # a tag that is no call is not held
    p = app.Parser(True)
    a0 = p.feed("The " + TC)
    a1 = p.feed(" tag")
    if a0 != [("reasoning", "The ")] or "".join(t for _, t in a1) != "<tool_call> tag":
        print(f"FAIL tag held: {a0} {a1}")
        bad += 1
    for _ in range(30000):
        if not p.feed(" word"):
            print("FAIL reasoning held after a tag")
            bad += 1
            break
    if len(p.buf) > 16:
        print(f"FAIL held {len(p.buf)} chars")
        bad += 1
    print(f"part 1: {len(TEXTS)} texts, {n} chunkings, {bad} failed")
    return bad


def check_tokens():
    root = os.environ.get("QE_MODEL", os.path.expanduser("~/GIT/QEngine/models/Qwen3.8-27B-INT4"))
    path = os.path.join(root, "tokenizer.json")
    if not os.path.exists(path):
        print(f"part 2 skipped: no {path}")
        return 0
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    th, eth, tc, etc = (tok.token_to_id(t) for t in ("<think>", "</think>", "<tool_call>", "</tool_call>"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    body = "\n<function=Read>\n<parameter=path>\n/a\n</parameter>\n</function>\n"
    # (name, thinking, ids, expected reasoning, content, text of the tool part with the marks restored)
    seqs = [
        ("answer", False, enc("Rule 1: answer record 7919, ąę क्ष."), "", "Rule 1: answer record 7919, ąę क्ष.", ""),
        ("tail with a real block", False, enc("Rule 0.") + enc("assistant\n") + [th] + enc("\n\n") + [eth] + enc("\n\nRule 1"),
         "", "Rule 0.assistant\n<think>\n\n</think>\n\nRule 1", ""),
        ("block first", False, [th] + enc("plan") + [eth] + enc("\n\nanswer"), "plan", "answer", ""),
        ("thinking on, two blocks", True, enc("one") + [eth] + enc("\n\n") + [th] + enc("two") + [eth] + enc("\n\nanswer"), "onetwo", "answer", ""),
        ("thinking on, tool call", True, enc("plan") + [eth] + enc("\n\ndo it\n") + [tc] + enc(body) + [etc], "plan", "do it\n", "<tool_call>" + body + "</tool_call>"),
        ("tool call in the open block", True, enc("plan ") + [tc] + enc(body) + [etc], "plan ", "", "<tool_call>" + body + "</tool_call>"),
        ("reply cut inside a character", False, enc("abc def ") + enc("\U0001F680")[:2], "", "abc def ", ""),
        ("multi-byte across tokens", False, enc("a \U0001F600\U0001F680 b"), "", "a \U0001F600\U0001F680 b", ""),
    ]
    rng = random.Random(8)
    bad = n = 0
    for name, thinking, ids, want_r, want_c, want_t in seqs:
        seen = {}
        for cuts in splits(len(ids), rng):
            det, p, out = app.Detok(tok), app.Parser(thinking), []
            for c in cut(ids, cuts):
                out += p.feed(det.push(c))
            out += p.feed(det.flush())
            out += p.feed("", final=True, promote=True)
            seen.setdefault(outcome(p, out), cuts)
            n += 1
        res = next(iter(seen))
        got = ("".join(t for k, t in res[0] if k == "reasoning"), "".join(t for k, t in res[0] if k == "content"), app.unmark(res[1]))
        if got != (want_r, want_c, want_t):
            print(f"FAIL {name}: got {got}")
            bad += 1
        if len(seen) > 1:
            print(f"FAIL {name}: {len(seen)} outcomes over the chunkings, e.g.")
            for res, cuts in list(seen.items())[:2]:
                print(f"     cuts {cuts}: {res}")
            bad += 1
    print(f"part 2: {len(seqs)} token sequences, {n} chunkings, {bad} failed")
    return bad


class ByteTok:
    """One token per byte, plus the tags: enough for Detok and Job.feed without a model directory."""
    SPECIAL = {300: "<think>", 301: "</think>", 302: "<tool_call>", 303: "</tool_call>", 304: "<eos>"}

    def decode(self, ids, skip_special_tokens=False):
        out, run = [], bytearray()
        for t in ids:
            if t in self.SPECIAL:
                out.append(run.decode("utf-8", "replace")); run = bytearray()
                out.append(self.SPECIAL[t])
            else:
                run.append(t)
        out.append(run.decode("utf-8", "replace"))
        return "".join(out)


LAST = {}


class Tc:
    """Counts thinking tokens as ThinkControl does for a block that stays open."""
    open, n = True, 0

    def track(self, toks):
        self.n += len(toks)

    def after(self, toks):
        pass


class Grammar:
    """Done after `left` tokens."""
    error = None

    def __init__(self, left):
        self.left = left

    def consume(self, t):
        self.left -= 1
        return True

    def done(self):
        return self.left <= 0

    def accepting(self):
        return self.done()

    def plan(self, *a):
        pass


def run_job(chunks, max_new=10000, stops=(), guard=0, eos=(), sink=None, room=10 ** 6, grammar=None, tc=None, forced=None):
    """The text Job.feed hands to the client, step by step: the real method on a stand-in for its state."""
    tok = ByteTok()
    tok.token_to_id = {v: k for k, v in tok.SPECIAL.items()}.get
    got = [] if sink is None else sink
    LAST.clear()
    g = types.SimpleNamespace(eos=set(eos), im_start=305, tok=tok, args=types.SimpleNamespace(loop_guard_tokens=guard, mtp_k=3),
                              engine=types.SimpleNamespace(slots={0: room}, spec_io=lambda s: None, mask_buffer=lambda s: None), marks=None,
                              think_ids=types.SimpleNamespace(ok=True, open_at_end=lambda ids, im: True))
    j = app.Job.__new__(app.Job)
    j.__dict__.update(gen=g, finished=False, max_new=max_new, n=0, grammar=grammar, p={}, tc=tc, slot=0, ids=[0],
                      det=app.Detok(tok, {i: app.mark(t) for i, t in tok.SPECIAL.items() if t != "<eos>"}), glk=None, gsteps=0, grammar_s=0.0,
                      text_all="", text_mk="", stops=list(stops), keep=None, cut=False, finish="length", loop_checked=0, on_text=got.append)
    if forced is not None:  # a guard-opened thinking turn, as finish_start sets it up: forced = the answer's max_tokens or 0
        j.answer_at, j.answer_max = None, forced or None
    LAST["job"] = j
    for c in chunks:
        j.feed(list(c))
    return "".join(got), j.finished, j.finish


def check_job():
    """Every end of a reply: the last step's text reaches the client, wherever steps end. A step ending inside a character holds its whole text."""
    rng = random.Random(9)
    rocket = list("\U0001F680".encode())
    txt = lambda s: list(s.encode())  # noqa: E731
    loop = txt("again and again ") + rocket
    cases = [
        ("max_tokens inside a character", txt("abc def ") + rocket[:2], dict(max_new=10), "abc def ", "length"),
        ("EOS after a partial character", txt("abc def ") + rocket[:2] + [304], dict(eos=[304]), "abc def ", "stop"),
        ("whole character before max_tokens", txt("abc ") + rocket, dict(max_new=8), "abc \U0001F680", "length"),
    ]
    bad = n = 0
    for name, ids, kw, want, want_finish in cases:
        seen = set()
        for cuts in splits(len(ids), rng):
            seen.add(run_job(cut(ids, cuts), **kw))
            n += 1
        if seen != {(want, True, want_finish)}:
            print(f"FAIL job, {name}: {sorted(seen)[:3]}")
            bad += 1
    # stop sequences: every split gives the single-chunk text, leftmost stop of the reply, and the kept tokens for the rollback
    E = [304]
    stops = [
        ("before a partial character", "abc xyz def ", rocket[:2] + E, dict(stops=["xyz"]), "abc ", "stop"),
        ("never completes", "abc xy def", E, dict(stops=["xyz"]), "abc xy def", "stop"),
        ("first part at the end, EOS", "abc xy", E, dict(stops=["xyz"]), "abc xy", "stop"),
        ("first part at max_tokens", "abc xy", [], dict(stops=["xyz"], max_new=6), "abc xy", "length"),
        ("first part, cut by max_tokens", "abc xyz", [], dict(stops=["xyz"], max_new=6), "abc xy", "length"),
        ("stop ends exactly at max_tokens", "abc xyz", [], dict(stops=["xyz"], max_new=7), "abc ", "stop"),
        ("overlapping start", "aaab", E, dict(stops=["aab"]), "a", "stop"),
        ("two occurrences", "a xyz b xyz c", E, dict(stops=["xyz"]), "a ", "stop"),
        ("second listed stop comes first", "a yy b zz c", E, dict(stops=["zz", "yy"]), "a ", "stop"),
        ("earlier-starting stop completes", "abcd", E, dict(stops=["bc", "abcd"]), "", "stop"),
        ("earlier-starting stop fails", "abce", E, dict(stops=["bc", "abcd"]), "a", "stop"),
        ("earlier-starting stop cut by EOS", "abc", E, dict(stops=["bc", "abcd"]), "a", "stop"),
        ("one is the start of the other", "x abc", E, dict(stops=["ab", "abc"]), "x ", "stop"),
        ("multi-byte stop", "cost: 5 \u20ac and", E, dict(stops=["\u20ac"]), "cost: 5 ", "stop"),
        ("longer than the old window", "go " + "ab" * 50 + "!", E, dict(stops=["ab" * 50 + "!"]), "go ", "stop"),
    ]
    for name, text, tail, kw, want, want_finish in stops:
        ids = txt(text) + tail
        seen = set()
        for cuts in splits(len(ids), rng):
            seen.add(run_job(cut(ids, cuts), eos=E if E[0] in ids else (), **kw))
            n += 1
        if seen != {(want, True, want_finish)}:
            print(f"FAIL stop, {name}: {sorted(seen)[:3]}, want {want!r} {want_finish}")
            bad += 1
    # the rollback keeps the tokens in front of the stop
    run_job([txt("abc "), txt("xy"), txt("z tail")], stops=["xyz"])
    if LAST["job"].keep != 3:  # (kept_tokens: trailing whitespace dropped)
        print(f"FAIL stop rollback: keep {LAST['job'].keep}")
        bad += 1
    # streaming: only what could still become a stop waits, and a request without stops sends every step's text at once
    sent = []
    run_job([txt("ab x"), txt("y"), txt("q r"), txt(" end")], stops=["xyz"], sink=sent)
    if sent != ["ab ", "xyq r", " end"]:
        print(f"FAIL stop hold: {sent}")
        bad += 1
    sent = []
    run_job([txt("ab x"), txt("y"), txt("\n"), txt(" end")], sink=sent)
    if sent != ["ab x", "y", "\n", " end"]:
        print(f"FAIL no stops, text delayed or merged: {sent}")
        bad += 1
    # text waiting for a stop goes out at the other ends too: grammar done, context limit, loop guard
    ids = txt("ab xy")
    for cuts in splits(len(ids), rng):
        for kw in (dict(grammar=Grammar(len(ids))), dict(room=1 + len(ids) + 8)):
            r = run_job(cut(ids, cuts), stops=["xyz"], **kw)
            n += 1
            if r[0] != "ab xy" or not r[1]:
                print(f"FAIL stop + other end {sorted(kw)}: {r}")
                bad += 1
    r = run_job([loop * 2, loop * 10 + txt("again ")], stops=["again x"], guard=32)
    if r[1:] != (True, "stop") or r[0] != "again and again \U0001F680" * 12 + "again ":
        print(f"FAIL stop + loop guard: {r[0][-30:]!r} {r[1:]}")
        bad += 1
    # a call in an open block runs only when the model ended the reply itself; its tokens are not thinking tokens
    call = [302] + txt("\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n") + [303]
    q = txt("plan ") + call
    ends = [
        ("EOS", q + E, dict(eos=E), True),
        ("max_tokens", q + txt(" tail"), dict(max_new=len(q)), False),
        ("context limit", q + txt(" tail"), dict(room=1 + len(q) + 8), False),
        ("stop string", q + txt(" Observation:") + E, dict(stops=["Observation:"], eos=E), False),
        ("stop string and EOS in one go", q + txt("Obs") + E, dict(stops=["Obs"], eos=E), False),
        ("no call form", txt("plan ") + [302] + txt(" and ") + [303] + E, dict(eos=E), False),
        ("no function name", txt("plan ") + [302] + txt("\n<function=\n") + [303] + E, dict(eos=E), False),
    ]
    for name, ids, kw, want in ends:
        res = set()
        for cuts in splits(len(ids), rng, tries=30):
            tc = Tc()
            run_job(cut(ids, cuts), tc=tc, **kw)
            j = LAST["job"]
            res.add((j.call_in_block(), tc.n))
            n += 1
        want_n = len(txt("plan ")) if want else None
        if {x[0] for x in res} != {want} or (want and {x[1] for x in res} != {want_n}):
            print(f"FAIL call in block, {name}: {sorted(res)[:3]}, want {want}")
            bad += 1
    # the same through the guard: a repeating call in an open block stays reasoning
    tc = Tc()
    run_job([call * 3, call * 40], guard=32, tc=tc)
    if LAST["job"].finish != "stop" or LAST["job"].call_in_block():
        print(f"FAIL call in block, loop guard: {LAST['job'].finish}")
        bad += 1
    # the loop guard ends the reply on a step that ends inside a character: that step's text must arrive
    ids = loop * 12
    text, done, fin = run_job([ids[:len(loop) * 2], ids[len(loop) * 2:-2]], guard=32)
    n += 1
    want = "again and again \U0001F680" * 11 + "again and again "
    if not done or fin != "stop" or text != want:
        print(f"FAIL job, loop guard end: done {done} finish {fin}, {len(text)} chars, want {len(want)}")
        bad += 1
    print(f"part 3: Job.feed, {len(cases) + 3 + len(stops) + len(ends)} ends of a reply, {n} chunkings, {bad} failed")
    return bad


def forced_tc():
    """The real ThinkControl over a reply that opens inside <think> (no forced close here: the mock server tests that)."""
    ti = types.SimpleNamespace(start=300, end=301, wrap_max=30, open_at_end=lambda ids, im: True,
                               is_space=lambda x: x < 256 and not bytes([x]).decode("latin-1").strip())
    job = types.SimpleNamespace(ids=[0], max_new=10000, p={}, gen=types.SimpleNamespace(im_start=305, engine=None))
    return think_budget.ThinkControl(job, None, ti, False)


def check_forced_stops():
    """A guard-opened thinking turn with client stops: they match only from the answer's first character, on every split,
    and what the client's parser sees is the same reasoning, answer and tool call."""
    rng = random.Random(11)
    txt = lambda s: list(s.encode())  # noqa: E731
    O, C, E = [300], [301], [304]
    rocket = list("\U0001F680".encode())
    call = [302] + txt("\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n") + [303]
    # name, tokens, stops, reasoning, answer, a call, finish, cut, extra
    cases = [
        ("in the reasoning and the answer", txt("plan Obs more") + C + txt("\n\nThe Obs rest") + E, ["Obs"], "plan Obs more", "The ", False, "stop", True, {}),
        ("in the reasoning only", txt("plan Obs more") + C + txt("\n\nDone") + E, ["Obs"], "plan Obs more", "Done", False, "stop", False, {}),
        ("second stop first in the answer", txt("a Obs b") + C + txt("\n\nx yy Obs") + E, ["Obs", "yy"], "a Obs b", "x ", False, "stop", True, {}),
        ("blank line, cut later", txt("a\n\nb") + C + txt("\n\nx\n\ny") + E, ["\n\n"], "a\n\nb", "x", False, "stop", True, {}),
        ("blank line after the close only", txt("a") + C + txt("\n\nx") + E, ["\n\n"], "a", "x", False, "stop", False, {}),
        ("spanning the close", txt("a b") + C + txt("\n\nx") + E, ["b</think>"], "a b", "x", False, "stop", False, {}),
        ("starting before the answer", txt("a") + C + txt("\n\nx y") + E, ["\n\nx"], "a", "x y", False, "stop", False, {}),
        ("at the answer's first character", txt("a") + C + txt("\n\nx y") + E, ["x"], "a", "", False, "stop", True, {}),
        ("part of a stop before the close", txt("a Ob") + C + txt("\n\ns Obs") + E, ["Obs"], "a Ob", "s ", False, "stop", True, {}),
        ("the close's string in the reasoning", txt("q </think> Obs z") + C + txt("\n\nA Obs") + E, ["Obs"], "q </think> Obs z", "A ", False, "stop", True, {}),
        ("the close's string in the answer", txt("q") + C + txt("\n\nObs </think> z") + E, ["Obs"], "q", "", False, "stop", True, {}),
        ("a block reopened", txt("q Obs") + C + txt("\n\n") + O + txt("r Obs") + C + txt("\n\nA Obs b") + E, ["Obs"], "q Obsr Obs", "A ", False, "stop", True, {}),
        ("never closes, EOS", txt("q Obs r") + E, ["Obs"], "q Obs r", "", False, "stop", False, {}),
        ("never closes, max_tokens", txt("q Obs r") + C, ["Obs"], "q Obs r", "", False, "length", False, dict(max_new=7)),
        ("partial character in the reasoning", txt("q Obs ") + rocket[:2] + E, ["Obs"], "q Obs ", "", False, "stop", False, {}),
        ("first part at the end, EOS", txt("q") + C + txt("\n\nab Ob") + E, ["Obs"], "q", "ab Ob", False, "stop", False, {}),
        ("tool call after the close", txt("q Obs") + C + txt("\n\n") + call + E, ["Obs"], "q Obs", "", True, "stop", False, {}),
        ("tool call in the open block", txt("q Obs ") + call + E, ["Obs"], "q Obs ", "", True, "stop", False, {}),
        ("no stops", txt("q Obs") + C + txt("\n\nA Obs") + E, [], "q Obs", "A Obs", False, "stop", False, {}),
    ]
    bad = n = 0
    for name, ids, stops, want_r, want_c, want_call, want_fin, want_cut, kw in cases:
        seen = set()
        for cuts in splits(len(ids), rng):
            sent = []
            _, done, fin = run_job(cut(ids, cuts), stops=stops, eos=E, sink=sent, tc=forced_tc(), forced=0, **kw)
            j = LAST["job"]
            pr, r, c = app.Parser(True), "", ""
            for x in sent + [None]:
                for kind, u in pr.feed("" if x is None else x, final=x is None, promote=x is None and j.call_in_block()):
                    r, c = (r + u, c) if kind == "reasoning" else (r, c + u)
            seen.add((r, c, bool(pr.tool_text), fin, j.cut, done))
            n += 1
        want = (want_r, want_c, want_call, want_fin, want_cut, True)
        if seen != {want}:
            print(f"FAIL forced stop, {name}: {sorted(seen)[:3]}, want {want}")
            bad += 1
    # the rollback keeps the tokens in front of the stop, reasoning included
    q = txt("q Obs") + C + txt("\n\nA ")
    run_job([q, txt("Ob"), txt("s z")], stops=["Obs"], tc=forced_tc(), forced=0)
    if LAST["job"].keep != len(q) - 1:  # (kept_tokens: trailing whitespace dropped)
        print(f"FAIL forced stop, rollback: keep {LAST['job'].keep}, want {len(q) - 1}")
        bad += 1
    # in the reasoning nothing waits for a stop; from the answer on, only what can still become one
    sent = []
    run_job([txt("q O"), txt("b") + C, txt("\n\n"), txt("a O"), txt("x")], stops=["Ob"], sink=sent, tc=forced_tc(), forced=0)
    if [app.unmark(x) for x in sent] != ["q O", "b</think>", "\n\n", "a ", "Ox"]:
        print(f"FAIL forced stop, what waits: {sent}")
        bad += 1
    # a loop guard end flushes the held text, in the reasoning (a step ending inside a character) and in the answer (part of a stop)
    loop = txt("again and again ") + rocket
    ids = loop * 12
    text = run_job([ids[:len(loop) * 2], ids[len(loop) * 2:-2]], guard=32, stops=["xyz"], tc=forced_tc(), forced=0)
    n += 1
    want = "again and again \U0001F680" * 11 + "again and again "
    if text != (want, True, "stop"):
        print(f"FAIL forced stop, loop guard end in the reasoning: {text[0][-30:]!r} {text[1:]}")
        bad += 1
    text = run_job([txt("q") + C + txt("\n\n"), loop * 2, loop * 10 + txt("again ")], guard=32, stops=["again x"], tc=forced_tc(), forced=0)
    n += 1
    want = "q</think>\n\n" + "again and again \U0001F680" * 12 + "again "
    if (app.unmark(text[0]),) + text[1:] != (want, True, "stop"):
        print(f"FAIL forced stop, loop guard end in the answer: {text[0][-30:]!r} {text[1:]}")
        bad += 1
    # the answer's max_tokens ends the reply: the part of a stop that waited goes out
    q, a = txt("q Obs") + C, txt("\n\nab Ob")
    for cuts in splits(len(a), rng, tries=40):
        text = run_job([q] + cut(a, cuts) + [txt(" more")], stops=["Obs"], tc=forced_tc(), forced=len(a))
        n += 1
        if (app.unmark(text[0]),) + text[1:] != ("q Obs</think>\n\nab Ob", True, "length"):
            print(f"FAIL forced stop, answer max_tokens: {text}")
            bad += 1
    print(f"part 5: guard-opened thinking with stops, {len(cases) + 5} checks, {n} runs, {bad} failed")
    return bad


def check_think_control():
    root = os.environ.get("QE_MODEL", os.path.expanduser("~/GIT/QEngine/models/Qwen3.8-27B-INT4"))
    path = os.path.join(root, "tokenizer.json")
    if not os.path.exists(path):
        print(f"part 4 skipped: no {path}")
        return 0
    from tokenizers import Tokenizer
    import think_budget as tb
    tok = Tokenizer.from_file(path)
    ti = tb.ThinkIds(tok)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    im = tok.token_to_id("<|im_start|>")
    open_p, shut_p = [im] + enc("assistant\n") + [ti.start] + enc("\n"), [im] + enc("assistant\n") + [ti.start] + enc("\n\n") + [ti.end] + enc("\n\n")
    plan, word = enc("plan"), enc("word")
    cases = [
        ("mid-answer <think> is text", open_p, plan + [ti.end] + enc("\n\nThe answer ") + [ti.start] + word, False, len(plan)),
        ("second block right after the first counts", open_p, plan + [ti.end] + enc("\n\n") + [ti.start] + word, True, len(plan) + len(word)),
        ("thinking off, block first", shut_p, [ti.start] + word, True, len(word)),
        ("thinking off, whitespace then block", shut_p, enc("\n") + [ti.start] + word, True, len(word)),
        ("thinking off, text then <think>", shut_p, enc("Hi ") + [ti.start] + word, False, 0),
        ("mid-answer block is never reopened by a close", open_p, plan + [ti.end] + enc("x") + [ti.start] + word + [ti.end] + [ti.start] + word, False, len(plan)),
    ]
    bad = 0
    for name, prompt, reply, want_open, want_n in cases:
        job = types.SimpleNamespace(ids=prompt, gen=types.SimpleNamespace(im_start=im), max_new=1000, p={})
        res = set()
        for cuts in itertools.chain([()], [(i,) for i in range(1, len(reply))]):
            tc = tb.ThinkControl(job, None, ti, False)
            for c in cut(reply, cuts):
                tc.track(c)
            res.add((tc.open, tc.n))
        if res != {(want_open, want_n)}:
            print(f"FAIL think control, {name}: {res}, want {(want_open, want_n)}")
            bad += 1
    # the parser's start state follows the rendered prompt; True and False are what they were before reply_thinking
    import jinja2
    msgs = [{"role": "user", "content": "hi"}]
    server = os.path.dirname(app.__file__)
    g = types.SimpleNamespace(think_ids=ti, im_start=im)
    for tp in (os.path.join(server, "templates", "qwen38_syssafe.jinja"), os.path.join(root, "chat_template.jinja")):
        if not os.path.exists(tp):
            continue
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        t = env.from_string(open(tp).read())
        for kw, want in (({"enable_thinking": True}, True), ({"enable_thinking": False}, False), ({"enable_thinking": None}, True),
                         ({"enable_thinking": 0}, True), ({}, True)):
            ids = tok.encode(t.render(messages=msgs, add_generation_prompt=True, **kw), add_special_tokens=False).ids
            if app.reply_thinking(g, ids, kw) is not want:
                print(f"FAIL start state {os.path.basename(tp)} {kw}: want {want}")
                bad += 1
    # streaming: reasoning before a quoted call comes out at once, only the call waits
    p = app.Parser(True)
    first = p.feed("thought ")
    second = p.feed(TC + "\n<function=Bash>")
    third = p.feed("\n" + app.mark("</tool_call>") + " more")
    if first != [("reasoning", "thought ")] or second != [] or "".join(x for _, x in third).find(" more") < 0:
        print(f"FAIL hold of a call in reasoning: {first} {second} {third}")
        bad += 1
    print(f"part 4: ThinkControl, {len(cases)} replies, {bad} failed")
    return bad


def main():
    bad = check_texts() + check_tokens() + check_job() + check_think_control() + check_forced_stops()
    have = os.path.exists(os.path.join(os.environ.get("QE_MODEL", os.path.expanduser("~/GIT/QEngine/models/Qwen3.8-27B-INT4")), "tokenizer.json"))
    print((f"{bad} FAILED") if bad else ("ok" if have else "ok, parts 2 and 4 SKIPPED (no tokenizer)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
