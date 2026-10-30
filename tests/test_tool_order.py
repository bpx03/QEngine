"""CPU-only, < 10 s: the tools render canonically (server/app.py canonical_tools), so a client that reorders the
same tools between turns (omp with MCP tools re-injected) keeps the prompt cache.
- the same tools in two orders (and with their JSON keys in another order) render a byte-identical prompt;
- a mock two-turn conversation with shuffled tools: the second prompt extends the first one's prefix up to the
  generation prompt (without canonical_tools it diverges inside the tools block);
- tool-call parsing gives the same calls for either order.
  python tests/test_tool_order.py
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
from app import Template, anthropic_tools, canonical_tools, parse_tool_calls  # noqa: E402

SAFE = os.path.join(HERE, "..", "server", "templates", "qwen38_syssafe.jinja")
MODEL_TMPL = [p for p in (os.path.join(HERE, "..", "models", "Qwen3.8-Flash-Next-QE-INT4", "chat_template.jinja"),
                          os.path.join(HERE, "..", "models", "Qwen3.8-27B-INT4", "chat_template.jinja"))
              if os.path.exists(p)]


def tool(name, props, req=None, desc=""):
    return {"type": "function", "function": {"name": name, "description": desc or f"The {name} tool",
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": req or list(props)[:1]}}}


TOOLS = [tool("Read", {"file_path": {"type": "string"}, "limit": {"type": "integer"}}),
         tool("Bash", {"command": {"type": "string"}, "timeout": {"type": "number"}}),
         tool("mcp__search__web", {"query": {"type": "string"}, "n": {"type": "integer", "default": 5}}),
         tool("Edit", {"file_path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}),
         tool("mcp__git__log", {"path": {"type": "string"}})]


def reorder_keys(x, rnd):
    if isinstance(x, dict):
        ks = list(x)
        rnd.shuffle(ks)
        return {k: reorder_keys(x[k], rnd) for k in ks}
    if isinstance(x, list):
        return [reorder_keys(v, rnd) for v in x]
    return x


def common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def check_template(path):
    t = Template(path)
    rnd = random.Random(7)
    msgs1 = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "List files."}]
    base = t.render(msgs1, canonical_tools(TOOLS))
    for _ in range(20):
        sh = [reorder_keys(x, rnd) for x in TOOLS]
        rnd.shuffle(sh)
        assert t.render(msgs1, canonical_tools(sh)) == base, "same tool set, different prompt"
    # Anthropic API: tools converted by anthropic_tools, then the same canonical form
    an = [{"name": x["function"]["name"], "description": x["function"]["description"],
           "input_schema": x["function"]["parameters"]} for x in TOOLS]
    an_sh = an[::-1]
    assert t.render(msgs1, canonical_tools(anthropic_tools(an))) == t.render(msgs1, canonical_tools(anthropic_tools(an_sh)))
    # two turns, the client shuffles the tools (and adds nothing): the second prompt extends the first one
    turn1_tools = list(TOOLS)
    turn2_tools = TOOLS[2:] + TOOLS[:2]
    p1 = t.render(msgs1, canonical_tools(turn1_tools))
    msgs2 = msgs1 + [{"role": "assistant", "content": "Here they are."}, {"role": "user", "content": "Thanks, now read a.txt"}]
    p2 = t.render(msgs2, canonical_tools(turn2_tools))
    gen = p1.rindex("<|im_start|>assistant")
    assert p2.startswith(p1[:gen]), f"turn 2 diverges at {common(p1, p2)} of {gen}"
    # without the canonical form the client's order breaks the prefix inside the tools block (the bug)
    q1, q2 = t.render(msgs1, turn1_tools), t.render(msgs2, turn2_tools)
    tools_end = q1.index("</tools>")
    assert common(q1, q2) < tools_end, "expected the raw order to diverge in the tools block"
    # the stock head keys keep their order: a client sending the usual layout sees the usual text
    assert '{"type": "function", "function": {"name": "Bash", "description": "The Bash tool", "parameters": ' in base
    return gen


def check_parse():
    text = ("<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n<parameter=timeout>\n30\n"
            "</parameter>\n</function>\n</tool_call>")
    a = parse_tool_calls(text, TOOLS)
    b = parse_tool_calls(text, canonical_tools(TOOLS[::-1]))
    assert a == b == [{"name": "Bash", "arguments": {"command": "ls -la", "timeout": 30}}], (a, b)


def check_canonical():
    # stable on equal names, non-list / empty pass through
    dup = [tool("A", {"x": {"type": "string"}}, desc="first"), tool("A", {"y": {"type": "string"}}, desc="second")]
    assert [x["function"]["description"] for x in canonical_tools(dup)] == ["first", "second"]
    assert canonical_tools(None) is None and canonical_tools([]) == []
    assert [x["function"]["name"] for x in canonical_tools(TOOLS)] == sorted(x["function"]["name"] for x in TOOLS)


if __name__ == "__main__":
    check_canonical()
    check_parse()
    for p in [SAFE] + MODEL_TMPL:
        n = check_template(p)
        print(f"{os.path.basename(os.path.dirname(p)) or 'server'}/{os.path.basename(p)}: "
              f"canonical tools, turn 2 shares the first {n} chars")
    print("test_tool_order: ok")
