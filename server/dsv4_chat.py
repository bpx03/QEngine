"""DeepSeek-V4.1 prompt format for the DSV4 server: OpenAI chat messages -> prompt text, completion text -> message.

Own implementation of the format DeepSeek documents for V4.1 (the model repository's encoding/README.md): turns
<｜User｜> ... <｜Assistant｜> (reasoning</think>)answer<｜end▁of▁sentence｜>; thinking mode ends the prompt with
<think> and starts with a numeric reasoning-effort line; chat mode ends it with </think>; tool results are
<tool_result> blocks of a user turn; tool calls are "<｜DSML｜ calls>" blocks; the tool schemas sit in the system turn;
earlier turns lose their reasoning unless tools are given. Text only (no image blocks, no internal task tokens).
Checked against the model repository's golden cases (tools/dsv4/chat_golden.py).
"""
import json
import re

BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
SYS, USER, ASST = "<｜System｜>", "<｜User｜>", "<｜Assistant｜>"
THINK, ETHINK = "<think>", "</think>"
DSML = "｜DSML｜"
EFFORTS = {"low": 50, "high": 75, "max": 100, "medium": 75, "xhigh": 100}
DEFAULT_EFFORT = 75

TOOLS_TEXT = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _json(v):
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(v, ensure_ascii=True)


def effort_budget(effort):
    if effort is None:
        return DEFAULT_EFFORT
    if isinstance(effort, bool):
        raise ValueError(f"reasoning effort {effort!r}")
    if isinstance(effort, int):
        if not 1 <= effort <= 100:
            raise ValueError(f"reasoning effort {effort} (1-100)")
        return effort
    if isinstance(effort, str) and effort in EFFORTS:
        return EFFORTS[effort]
    raise ValueError(f"reasoning effort {effort!r} (an integer 1-100 or low / high / max)")


def _qualified(name, namespace=None):
    if isinstance(namespace, dict):
        namespace = namespace.get("name")
    pre, sep, bare = name.partition("::")
    if sep:
        if namespace not in (None, pre):
            raise ValueError(f"conflicting tool namespaces {namespace} / {pre}")
        namespace, name = pre, bare
    return name if namespace is None else f"{namespace}::{name}"


def _tool_schemas(tools):
    out = []
    for t in tools:
        f = dict(t.get("function", t))
        ns = t.get("namespace", f.get("namespace"))
        f.pop("namespace", None)
        f["name"] = _qualified(f["name"], ns)
        if isinstance(ns, dict) and ns.get("description"):
            f["description"] = ns["description"] + "\n" + (f.get("description") or "")
        out.append(_json(f))
    return "\n".join(out)


def _text(content):
    """Text of a message content (a string, or a list of text blocks joined by blank lines)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text") or "")
        elif isinstance(b, dict):
            parts.append(f"[Unsupported {b.get('type')}]")
    return "\n\n".join(parts)


def _args_dsml(arguments):
    a = arguments
    for _ in range(2):   # JSON strings, also double-encoded ones
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except ValueError:
                break
    if not isinstance(a, dict):
        a = {"arguments": arguments}
    return "\n".join(f'<{DSML} parameter name="{k}" string="{"true" if isinstance(v, str) else "false"}">'
                     f'{v if isinstance(v, str) else _json(v)}</{DSML} parameter>' for k, v in a.items())


def _calls_dsml(tool_calls):
    items = []
    for c in tool_calls:
        f = c.get("function", c)
        name = _qualified(f["name"], c.get("namespace") or f.get("namespace"))
        items.append(f'<{DSML} invoke name="{name}">\n{_args_dsml(f.get("arguments") or {})}\n</{DSML} invoke>')
    return f"\n\n<{DSML} calls>\n" + "\n".join(items) + f"\n</{DSML} calls>"


def _merge(messages):
    """Tool results into user turns (consecutive user / tool messages share one turn), results in call order."""
    out = []
    order = {}
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            order = {tc.get("id") or tc.get("function", {}).get("id", ""): i for i, tc in enumerate(m["tool_calls"])}
            order.pop("", None)
        if role == "tool":
            blk = {"type": "tool_result", "id": m.get("tool_call_id", ""), "content": m.get("content", "")}
            if out and out[-1]["role"] == "user" and "blocks" in out[-1]:
                out[-1]["blocks"].append(blk)
            else:
                out.append({"role": "user", "blocks": [blk], "_order": order})
        elif role == "user":
            c = m.get("content", "")
            blks = c if isinstance(c, list) else [{"type": "text", "text": c}]
            blks = [dict(b) for b in blks]
            if out and out[-1]["role"] == "user" and "blocks" in out[-1]:
                out[-1]["blocks"].extend(blks)
            else:
                out.append(dict(m, blocks=blks, _order=order))
        else:
            out.append(dict(m))
    for m in out:   # tool results in the order of the calls before them
        if m["role"] != "user" or not m.get("_order"):
            continue
        res = [b for b in m["blocks"] if b.get("type") == "tool_result"]
        if len(res) > 1:
            res.sort(key=lambda b: m["_order"].get(b.get("id", ""), 0))
            it = iter(res)
            m["blocks"] = [next(it) if b.get("type") == "tool_result" else b for b in m["blocks"]]
    return out


def _is_turn_start(m, idx):
    """A user turn with text (not only tool results), or a mid-conversation system message."""
    r = m.get("role")
    if r == "user" or (r == "system" and idx > 0):
        blks = m.get("blocks")
        return not blks or any(b.get("type") in ("text", "image", "image_url") for b in blks)
    return False


def _last_user(ms):
    for i in range(len(ms) - 1, -1, -1):
        if _is_turn_start(ms[i], i):
            return i
    return -1


def encode(messages, tools=None, thinking=True, effort=None, drop_thinking=True, bos=True):
    """The prompt text for OpenAI-style messages (tools: OpenAI tool list, attached to the first message)."""
    ms = [dict(m) for m in messages]
    if tools:   # the schemas go into the system turn (an empty one when the request has none)
        if not ms or ms[0].get("role") != "system":
            ms.insert(0, {"role": "system", "content": ""})
        ms[0]["tools"] = tools
    budget = effort_budget(effort)
    ms = _merge(ms)
    drop = drop_thinking and not any(m.get("tools") for m in ms)
    if thinking and drop:   # earlier assistant turns lose their reasoning
        lu = _last_user(ms)
        keep = ("user", "system", "tool", "latest_reminder")
        ms = [(dict(m, reasoning_content=None) if (m["role"] == "assistant" and i < lu) else m) for i, m in enumerate(ms)
              if m["role"] in keep or m["role"] == "assistant" or i >= lu]
    lu = _last_user(ms)
    p = BOS if bos else ""
    for i, m in enumerate(ms):
        role = m["role"]
        eff = f"Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n" \
            if i == 0 and thinking else ""
        s = SYS if i == 0 and (eff or role == "system") else ""
        s += eff
        if role == "system":
            if i > 0:
                s += SYS
            s += _text(m.get("content"))
            if m.get("tools"):
                s += "\n\n" + TOOLS_TEXT.format(schemas=_tool_schemas(m["tools"]))
            if m.get("response_format"):
                s += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" + \
                     _json(m["response_format"])
        elif role == "user":
            s += USER
            parts = []
            for b in m.get("blocks") or []:
                t = b.get("type")
                if t == "text":
                    parts.append(b.get("text") or "")
                elif t == "tool_result":
                    parts.append("<tool_result>" + _text(b.get("content")) + "</tool_result>")
                else:
                    parts.append(f"[Unsupported {t}]")
            s += "\n\n".join(parts)
        elif role == "assistant":
            think = ""
            if thinking and (not drop or i > lu):
                think = (m.get("reasoning_content") or "") + ETHINK
            s += think + (_text(m.get("content")) if m.get("content") else "") + \
                (_calls_dsml(m["tool_calls"]) if m.get("tool_calls") else "")
            if not m.get("wo_eos"):
                s += EOS
        elif role == "latest_reminder":
            s += "<｜latest_reminder｜>" + _text(m.get("content"))
        else:
            raise ValueError(f"unknown role {role!r}")
        nxt = ms[i + 1]["role"] if i + 1 < len(ms) else None
        if nxt in (None, "assistant", "latest_reminder"):
            if role == "user" or (role == "system" and i > 0):
                s += ASST
                if thinking and (not drop or i >= lu):
                    s += THINK
                else:
                    s += ETHINK
        p += s
    return p


_PARAM = re.compile(r'^ name="(.*?)" string="(true|false)">(.*?)<$', re.DOTALL)


def parse_completion(text, thinking=True):
    """Completion text (generated after the prompt, up to and including the end token if any) -> dict with content,
    reasoning_content, tool_calls (OpenAI format, arguments as JSON strings). Tolerant: a missing end token or </think>
    leaves what came as content / reasoning."""
    reasoning, rest = "", text
    if thinking:
        k = rest.find(ETHINK)
        if k < 0:
            return {"content": "", "reasoning_content": rest.replace(EOS, ""), "tool_calls": []}
        reasoning, rest = rest[:k], rest[k + len(ETHINK):]
    start = f"\n\n<{DSML} calls>"
    k = rest.find(start)
    calls = []
    if k >= 0:
        content, body = rest[:k], rest[k + len(start):]
        for inv in re.finditer(rf'<{DSML} invoke name="(.*?)">\n(.*?)</{DSML} invoke>', body, re.DOTALL):
            name, args_s = inv.group(1), inv.group(2)
            args = []
            for pm in re.finditer(rf'<{DSML} parameter(.*?)/{DSML} parameter>', args_s, re.DOTALL):
                kv = _PARAM.match(pm.group(1))
                if kv:
                    pn, is_s, pv = kv.groups()
                    args.append(f"{_json(pn)}: {_json(pv) if is_s == 'true' else pv}")
            ns, _, bare = name.rpartition("::")
            call = {"type": "function", "function": {"name": bare if ns else name, "arguments": "{" + ", ".join(args) + "}"}}
            if ns:
                call["namespace"] = ns
            calls.append(call)
    else:
        content = rest
    content = content.split(EOS)[0]
    return {"content": content, "reasoning_content": reasoning, "tool_calls": calls}


def parse_calls(text):
    """[shared-srv] The DSML calls of a completion's tool part (from "\\n\\n<｜DSML｜ calls>" on): [{"name": the tool's
    name (namespace::name as written), "params": [(name, raw value, is_string)]}] (app.py parses each value on its own)."""
    start = f"\n\n<{DSML} calls>"
    k = text.find(start)
    if k < 0:
        return []
    out = []
    for inv in re.finditer(rf'<{DSML} invoke name="(.*?)">\n(.*?)</{DSML} invoke>', text[k + len(start):], re.DOTALL):
        name, args_s = inv.group(1), inv.group(2)
        ps = []
        for pm in re.finditer(rf'<{DSML} parameter(.*?)/{DSML} parameter>', args_s, re.DOTALL):
            kv = _PARAM.match(pm.group(1))
            if kv:
                pn, is_s, pv = kv.groups()
                ps.append((pn, pv, is_s == "true"))
        ns, _, bare = name.rpartition("::")
        out.append({"name": bare if ns else name, "params": ps})
    return out
