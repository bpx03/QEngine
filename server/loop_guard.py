"""Cross-turn loop guard: a conversation whose last assistant turns repeat earlier turns that got the same results.

Agents at long contexts sometimes repeat one turn forever: the identical tool call after the tool keeps returning
the same error, or near-copy text replies, or a short cycle of a few calls (A A A B A A A B ..., Claude Code session
92997599). Sampling within a reply does not help there: the loop is the model copying
its previous turns from the context. Tool calls whose arguments change each turn (probe69, probe70, ...) are not
caught since 1.3.4: they look the same as a legitimate batch over ids.

Detection works on the request's own history (so it follows the conversation, whichever slot or session serves it).
Each assistant turn is a (reply, tool result that came back) pair. A turn is *stale* when an earlier turn within the
last `window` turns has the same reply and the same result. The result must be a non-empty text from a tool message
and equal in full (compared by length and hash, whitespace collapsed; digits count, so a build log's progress counter
or a `date` line is new output; a result with an image part is unknown). The reply's tool calls must equal exactly
(name + canonical arguments without the cosmetic `description` argument Claude Code's Bash carries, no other
normalisation: a batch of calls that differ only in an id, e.g. 29 memory removes of "Legacy note N", is work, not a
loop; the text around a call is not compared). A reply without a call must match in text: exactly after normalising
(whitespace collapsed, digit runs -> 0) or nearly (token multiset overlap >= `near`, both at least `min_chars` long).
The trailing run of stale turns plus one is the repeated stretch (`turns`): 3 identical turns -> 3, a cycle
A A A B A A A B -> 5 (the period does not matter: each turn repeats an earlier one), while one repeated check between
new actions (A X A Y A) stays at 2.
Conservative on purpose:
  - a reply counts only with a tool call or at least `min_chars` characters of text: "ok", "Done." never trigger;
  - the results must repeat too: a repeated command that gets new output (polling a job) is no loop, an
    edit -> test -> edit -> test chain is no loop (each edit is new), and a conversation without tool results (batch
    extraction, few-shot examples) has nothing to repeat;
  - a human message (not a tool result) does not reset a stretch of tool calls (the model looped straight past "stop
    looping" in session 92997599), but when it is the newest message the guarded turn only thinks first: no note, no
    penalty (the human may well have asked to run the same thing again). A text reply a human answered is not stale;
  - unchanged results can be legitimate waiting (a long compile, CI pending, a 503), so a stretch is guarded with
    back-off, not on every turn (due()), for every period: at N, 2N, 4N turns, then every `window` turns of the
    conversation; and when a human just spoke.

The server then generates the next turn with reasoning on (also on a --think 0 server: the empty think block of the
generation prompt is opened, so the rendered history and its cache stay the same; those tokens come on top of the
client's max_tokens and its stop sequences start after them), a raised presence penalty whose counts start with the
repeated reply's tokens, without lookup drafts, and with NOTE appended to the latest tool result."""
import collections
import json
import re

WS = re.compile(r"\s+")
DIGITS = re.compile(r"\d+")
TOKENS = re.compile(r"\w+|[^\w\s]")  # words and single punctuation marks: compact JSON has no whitespace to split on
WINDOW = 24  # turns of history the detection looks at
# tool-call arguments that only describe the call (Claude Code Bash "description"): the model rewords them while it
# repeats the same command ("Decode branch targets with corrected imm*4" / "... with correct imm*4 scaling")
COSMETIC_ARGS = frozenset({"description"})


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(_text(p.get("text", p.get("content", ""))) if isinstance(p, dict) else str(p) for p in c)
    if isinstance(c, dict):
        return _text(c.get("text", c.get("content", "")))
    return "" if c is None else str(c)


def _norm(s, cap=4000):
    return DIGITS.sub("0", " ".join(s[:4 * cap].split()))[:cap]


def _has_non_text(c):
    return isinstance(c, list) and any(isinstance(p, dict) and p.get("type", "text") != "text" for p in c)


def _reply_sig(m):
    """(normalised visible text, exact tool calls): the text without reasoning, digit runs -> 0; the calls as
    name:canonical arguments without COSMETIC_ARGS, NOT normalised ("" without a call)."""
    txt = _text(m.get("content"))
    if "</think>" in txt:
        txt = txt.split("</think>", 1)[1]
    calls = []
    for c in m.get("tool_calls") or []:
        f = c.get("function", c)
        a = f.get("arguments")
        if isinstance(a, str):
            try:
                a = json.loads(a or "{}")
            except Exception:
                pass
        if isinstance(a, dict):
            a = {k: v for k, v in a.items() if k not in COSMETIC_ARGS}
        calls.append(f"{f.get('name')}:{json.dumps(a, sort_keys=True, ensure_ascii=False, default=str)}")
    return _norm(txt), " | ".join(calls)


Turn = collections.namedtuple("Turn", "text calls human result tokens")


def _similar(x, y, near, min_chars):
    """The text of two turns is equal, or a near copy: the share of tokens they have in common, as multisets. Linear in
    the text (at most 4000 characters, _norm): this runs on the event loop, where a character-level alignment cost
    seconds per request."""
    if x.text == y.text:
        return True
    if near >= 1.0 or len(x.text) < min_chars or len(y.text) < min_chars:
        return False
    a, b = (x.tokens, y.tokens) if len(x.tokens) <= len(y.tokens) else (y.tokens, x.tokens)
    na, nb = sum(a.values()), sum(b.values())
    if 2 * min(na, nb) < near * (na + nb):
        return False
    return 2 * sum(min(c, b.get(w, 0)) for w, c in a.items()) >= near * (na + nb)


def turns_of(messages, window=None):
    """The last `window` assistant turns of the history (all: None) as Turns, oldest first. text: normalised reply
    text; calls: exact tool calls; human: a human message (not a tool result) follows the turn; result: (length, hash)
    of the tool results, whitespace collapsed, None without one or with a non-text part (unknown). Walks back first:
    only the window's turns are normalised."""
    i, n = 0, len(messages)
    if window:
        seen = 0
        for k in range(n - 1, -1, -1):
            if messages[k].get("role") == "assistant":
                seen += 1
                if seen == window:
                    i = k
                    break
    out = []
    while i < n:
        m = messages[i]
        if m.get("role") != "assistant":
            i += 1
            continue
        sig, call = _reply_sig(m)
        j, res, human, unknown = i + 1, [], False, False
        while j < n and messages[j].get("role") != "assistant":
            r = messages[j].get("role")
            if r == "tool":
                c = messages[j].get("content")
                res.append(_text(c))
                unknown = unknown or _has_non_text(c)
            elif r == "user":
                t = _text(messages[j].get("content")).strip()
                if t and not t.startswith("<system-reminder>"):
                    human = True  # the human's words are not part of the result: the call's output is
            j += 1
        raw = " ".join(" | ".join(res).split())
        key = (len(raw), hash(raw)) if raw and not unknown else None
        out.append(Turn(sig, call, human, key, collections.Counter(TOKENS.findall(sig)) if key and not call else None))
        i = j
    return out


def _same_turn(x, y, near, min_chars):
    """The same reply that got the same result. The result: not empty (none, or an image: unknown) and equal in full.
    A tool call: the same exact call. A text reply: normalised / near copy text, and no human answered it (a repeated
    answer to new human messages is no loop)."""
    if x.calls != y.calls or x.result is None or x.result != y.result:
        return False
    if x.calls:  # tool calls repeat only exactly (same ids, paths, numbers)
        return True
    return not x.human and not y.human and _similar(x, y, near, min_chars)


def state(messages, near=0.92, min_chars=24, window=WINDOW):
    """{"turns": length of the repeated stretch at the end of the history (0: none, at most window), "distinct":
    different replies in it, "human": a human message follows the last assistant turn, "total": assistant turns in the
    history}. See the module doc."""
    t = turns_of(messages, 2 * window)  # a turn looks `window` turns back, so that any period the window sees gets to `window`
    out = {"turns": 0, "distinct": 0, "human": bool(t and t[-1].human),
           "total": sum(m.get("role") == "assistant" for m in messages)}
    if not t:
        return out

    def counts(x):
        return bool(x.calls) or len(x.text) >= min_chars

    run = 0  # trailing stale turns
    for i in range(len(t) - 1, 0, -1):
        if not counts(t[i]) or not any(_same_turn(t[j], t[i], near, min_chars) for j in range(i - 1, max(i - window, 0) - 1, -1)):
            break
        run += 1
        if run + 1 >= window:
            break
    if run == 0:
        out["turns"] = int(counts(t[-1]))  # a reply that repeats nothing (yet): 1, as a streak of one
        out["distinct"] = out["turns"]
        return out
    stretch = t[len(t) - run:]
    distinct = []
    for x in stretch:
        if not any(_same_turn(x, y, near, min_chars) for y in distinct):
            distinct.append(x)
    out["turns"] = run + 1  # + the turn the run starts from (A A A: 3; A A A B A A A B: 5)
    out["distinct"] = len(distinct)
    return out


def due(st, n, window=WINDOW):
    """A stretch of state() `st` is guarded on this turn (threshold n): at n, 2n, 4n ... turns; from `window` turns on,
    where `turns` stops growing whatever the period, every `window` turns of the conversation. Stateless like the
    detection. Unchanged polls and retries are often legitimate, so they must not get the guard on every turn."""
    k = st["turns"]
    if n <= 0 or k < n:
        return False
    if k >= window:
        return st["total"] % window == 0
    q = k // n
    return k % n == 0 and q & (q - 1) == 0


def streak(messages, near=0.92, min_chars=24, window=WINDOW):
    """Length of the repeated stretch at the end of the history (0: none). See state()."""
    return state(messages, near, min_chars, window)["turns"]


def last_reply_tokens(ids, im_start, assistant_ids, im_end, special_min, skip=()):
    """Token ids of the last assistant turn in a rendered prompt (after '<|im_start|>assistant\\n', up to
    '<|im_end|>'), without special tokens (id >= special_min) and the ids in skip (whitespace-only pieces)."""
    n = len(ids)
    L = len(assistant_ids)
    for i in range(n - L - 1, -1, -1):
        if ids[i] == im_start and list(ids[i + 1:i + 1 + L]) == list(assistant_ids):
            j = i + 1 + L
            out = []
            while j < n and ids[j] != im_end:
                t = ids[j]
                if t < special_min and t not in skip:
                    out.append(t)
                j += 1
            if j < n:  # a finished turn (the open generation header at the prompt end has no <|im_end|>)
                return out
    return []


NOTE = ("[server note: your last {n} turns repeated earlier turns of this conversation ({k} different call(s) in a "
        "cycle) and every one got the same result as before. Repeating them will not change anything. Stop and "
        "think: re-read the outputs and check your own command or script for a bug, then change the approach (a "
        "different command, tool or arguments), or tell the user what is blocking you.]")


def add_note(messages, n, k=1):
    """The messages with NOTE appended to the latest tool result (after the last assistant turn); the history is not
    changed in place. Only the turn being generated carries it: earlier renders stay as the client sent them."""
    last = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    at = max((i for i in range(last + 1, len(messages)) if messages[i].get("role") in ("tool", "user")), default=-1)
    if last < 0 or at < 0:
        return messages, False
    note = NOTE.format(n=n, k=k)
    m = dict(messages[at])
    c = m.get("content")
    if isinstance(c, list):
        c = list(c)
        if c and isinstance(c[-1], dict) and c[-1].get("type", "text") == "text" and "text" in c[-1]:
            c[-1] = {**c[-1], "text": c[-1]["text"] + "\n\n" + note}
        else:
            c.append({"type": "text", "text": note})
        m["content"] = c
    else:
        m["content"] = (_text(c) + "\n\n" if c else "") + note
    out = list(messages)
    out[at] = m
    return out, True
