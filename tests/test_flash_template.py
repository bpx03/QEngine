"""CPU-only: chat template for Qwen3.8-Flash-Next.
- the model's chat_template.jinja is the 27B one (byte-identical), and qwen38_syssafe.jinja renders
  every conversation the stock template accepts identically (with the Flash tokenizer: same ids);
- several leading system messages (HF #75/#76): stock template raises, syssafe merges; the server's
  merge_leading_system gives the same text with either template;
- HF #50 cleanup (drop_empty_turns) drops only earlier reasoning-only assistant turns.
  python tests/test_flash_template.py
"""
import hashlib
import os
import sys

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
from app import Template  # noqa: E402
from model_profile import Profile, drop_empty_turns, merge_leading_system  # noqa: E402

FLASH = os.environ.get("QE_FLASH_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
B27 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-27B-INT4")
SAFE = os.path.join(HERE, "..", "server", "templates", "qwen38_syssafe.jinja")

TOOLS = [{"type": "function", "function": {"name": "Bash", "description": "Run a shell command",
                                           "parameters": {"type": "object", "required": ["command"],
                                                          "properties": {"command": {"type": "string"},
                                                                         "timeout": {"type": "number"}}}}}]
CONVS = [
    ([{"role": "system", "content": "Du bist ein hilfreicher Assistent."}, {"role": "user", "content": "Hallo, wie geht's? Grüße aus Köln."}], None, {}),
    ([{"role": "user", "content": "hi"}], None, {}),
    ([{"role": "user", "content": "hi"}], None, {"enable_thinking": False}),
    ([{"role": "user", "content": "hi"}], None, {"reasoning_effort": "low"}),
    ([{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], TOOLS, {"reasoning_effort": "medium"}),
    ([{"role": "system", "content": "You are Claude Code."},
      {"role": "user", "content": "list files"},
      {"role": "assistant", "content": "", "reasoning_content": "I should run ls.",
       "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "Bash", "arguments": {"command": "ls -la", "timeout": 10}}}]},
      {"role": "tool", "content": "a.txt\nb.txt\nüberblick.md", "tool_call_id": "t1"},
      {"role": "assistant", "content": "Es gibt drei Dateien.", "reasoning_content": "Three files."},
      {"role": "user", "content": [{"type": "text", "text": "und jetzt "}, {"type": "text", "text": "lösche b.txt"}]},
      {"role": "assistant", "content": "Usuwam.", "tool_calls": [
          {"id": "t2", "type": "function", "function": {"name": "Bash", "arguments": {"command": "rm b.txt"}}},
          {"id": "t3", "type": "function", "function": {"name": "Bash", "arguments": {"command": "ls"}}}]},
      {"role": "tool", "content": "", "tool_call_id": "t2"},
      {"role": "tool", "content": "a.txt", "tool_call_id": "t3"}], TOOLS, {}),
    ([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
      {"role": "user", "content": "q2"}], None, {"preserve_thinking": False}),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def render(t, msgs, tools=None, **kw):
    return t.render([dict(m) for m in msgs], tools, **kw)


def main():
    stock_p = os.path.join(FLASH, "chat_template.jinja")
    assert sha(stock_p) == sha(os.path.join(B27, "chat_template.jinja")), "Flash template differs from 27B"
    hf = os.path.expanduser("~/.cache/qengine/hf/Qwen3.8-Flash-Next/chat_template.jinja")
    if os.path.exists(hf):
        assert sha(hf) == sha(stock_p), "HF official template differs from the model directory's"
    stock, safe = Template(stock_p), Template(SAFE)
    tok, _, _ = Profile(FLASH).tokenizer()
    for i, (msgs, tools, kw) in enumerate(CONVS):
        a, b = render(stock, msgs, tools, **kw), render(safe, msgs, tools, **kw)
        assert a == b, (i, a[-300:], b[-300:])
        assert tok.encode(a, add_special_tokens=False).ids == tok.encode(b, add_special_tokens=False).ids
    print(f"stock == syssafe on {len(CONVS)} conversations (text and Flash token ids)")

    # HF #75/#76: several leading system messages
    multi = [{"role": "system", "content": "You are Claude Code. "}, {"role": "system", "content": "  Extra rules."},
             {"role": "developer", "content": "Dev note"}, {"role": "user", "content": "go"}]
    try:
        render(stock, multi, TOOLS)
        raise AssertionError("stock template accepted several system messages")
    except jinja2.TemplateError:
        pass
    want = render(safe, multi, TOOLS)
    assert "You are Claude Code.\nExtra rules.\nDev note<|im_end|>" in want, want[-200:]
    merged = merge_leading_system(multi)
    assert len(merged) == 2 and merged[0]["content"] == "You are Claude Code.\nExtra rules.\nDev note"
    assert render(safe, merged, TOOLS) == want
    assert render(stock, [m for m in merged], TOOLS) == want
    assert merge_leading_system(CONVS[0][0]) == CONVS[0][0]  # one system message: unchanged
    print("several leading system messages: stock raises, syssafe merges, server merge identical with both")

    # mid-conversation system message (Claude Code <system-reminder>): syssafe renders it in place
    mid = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"},
           {"role": "assistant", "content": "a"}, {"role": "system", "content": "<system-reminder>x</system-reminder>"},
           {"role": "user", "content": "q2"}]
    out = render(safe, merge_leading_system(mid))
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\na<|im_end|>\n<|im_start|>system\n<system-reminder>x" in out
    print("mid-conversation system message rendered in place")

    # HF #50: reasoning-only assistant turns dropped from the history (not tool calls, not the last)
    hist = [{"role": "user", "content": "q1"},
            {"role": "assistant", "content": "", "reasoning_content": "thinking only"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "  \n", "reasoning_content": "x", "tool_calls": [
                {"id": "t", "type": "function", "function": {"name": "Bash", "arguments": {"command": "ls"}}}]},
            {"role": "tool", "content": "out", "tool_call_id": "t"},
            {"role": "assistant", "content": [{"type": "text", "text": " "}], "reasoning_content": "again"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "", "reasoning_content": "last (prefill)"}]
    cleaned, n = drop_empty_turns(hist)
    assert n == 2 and [m["role"] for m in cleaned] == ["user", "user", "assistant", "tool", "user", "assistant"], cleaned
    assert "thinking only" not in render(safe, cleaned[:-1], TOOLS)
    assert drop_empty_turns(CONVS[5][0]) == (CONVS[5][0], 0)
    print("HF #50 cleanup: 2 reasoning-only turns dropped, tool-call / final turns kept")
    print("ok")


if __name__ == "__main__":
    main()
