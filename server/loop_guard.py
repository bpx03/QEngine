"""Cross-turn loop guard: a conversation whose last assistant turns are the same reply getting the same result.

Agents at long contexts sometimes repeat one turn forever: the identical tool call after the tool keeps returning
the same error, or a run of near-copies (probe69, probe70, ... one path segment more each time). Sampling within a
reply does not help there: the loop is the model copying its previous turn from the context.

Detection works on the request's own history (so it follows the conversation, whichever slot or session serves it):
the trailing assistant turns are compared pairwise, each as (reply, result that came back). A turn joins the streak
when both its reply and its result match the next turn's, either exactly after normalising (whitespace collapsed,
digit runs -> 0) or nearly (difflib ratio >= `near`, both at least `min_chars` long). A request is guarded when the
streak reaches `turns`. Conservative on purpose:
  - a reply counts only with a tool call or at least `min_chars` characters of text: "ok", "Done." never trigger;
  - the results must repeat too: a repeated command that gets new output (polling a job) is no loop;
  - a new user message (not a tool result) after the last assistant turn ends the streak: the human stepped in.

The server then generates the next turn with a raised presence penalty whose counts start with the repeated reply's
tokens, and without lookup drafts (they would propose the copy)."""
import difflib
import json
import re

WS = re.compile(r"\s+")
DIGITS = re.compile(r"\d+")


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(_text(p.get("text", p.get("content", ""))) if isinstance(p, dict) else str(p) for p in c)
    if isinstance(c, dict):
        return _text(c.get("text", c.get("content", "")))
    return "" if c is None else str(c)


def _norm(s, cap=4000):
    return DIGITS.sub("0", WS.sub(" ", s).strip())[:cap]


def _reply_sig(m):
    """(normalised reply, has a tool call): visible text + tool calls (name, canonical arguments); no reasoning."""
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
        calls.append(f"{f.get('name')}:{json.dumps(a, sort_keys=True, ensure_ascii=False, default=str)}")
    return _norm(txt + " | " + " | ".join(calls)), bool(calls)


def _similar(a, b, near, min_chars):
    if a == b:
        return True
    if near >= 1.0 or len(a) < min_chars or len(b) < min_chars:
        return False
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sm.real_quick_ratio() >= near and sm.quick_ratio() >= near and sm.ratio() >= near


def turns_of(messages):
    """Assistant turns at the end of the history as [(reply sig, has call, result sig)], oldest first, and whether a
    human message (not a tool result) follows the last one."""
    out, i, n = [], 0, len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") != "assistant":
            i += 1
            continue
        sig, call = _reply_sig(m)
        j, res, human = i + 1, [], False
        while j < n and messages[j].get("role") != "assistant":
            r = messages[j].get("role")
            if r == "tool":
                res.append(_text(messages[j].get("content")))
            elif r == "user":
                t = _text(messages[j].get("content")).strip()
                if t and not t.startswith("<system-reminder>"):
                    human = True
                res.append(t)
            j += 1
        out.append((sig, call, _norm(" | ".join(res)), human))
        i = j
    return out


def streak(messages, near=0.92, min_chars=24):
    """Consecutive repeated (reply, result) turns at the end of the history (0: none, or the human stepped in)."""
    t = turns_of(messages)
    if not t:
        return 0
    if t[-1][3]:  # a human message after the last assistant turn
        return 0
    sig, call, res, _ = t[-1]
    if not call and len(sig) < min_chars:
        return 0
    k = 1
    for j in range(len(t) - 2, -1, -1):
        s2, c2, r2, h2 = t[j]
        if h2:  # the human spoke between these turns: the streak starts after it
            break
        if c2 != call or not _similar(s2, sig, near, min_chars) or not _similar(r2, res, near, min_chars):
            break
        k += 1
        sig, res = s2, r2
    return k


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


NOTE = ("[server note: you have made this same call {n} times in a row and it returned the same result every time. "
        "Repeating it will not change the result. Do not make this call again: change your approach (a different "
        "command, tool or arguments), or tell the user what is blocking you.]")


def add_note(messages, n):
    """The messages with NOTE appended to the latest tool result (after the last assistant turn); the history is not
    changed in place. Only the turn being generated carries it: earlier renders stay as the client sent them."""
    last = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    k = max((i for i in range(last + 1, len(messages)) if messages[i].get("role") in ("tool", "user")), default=-1)
    if last < 0 or k < 0:
        return messages, False
    note = NOTE.format(n=n)
    m = dict(messages[k])
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
    out[k] = m
    return out, True
