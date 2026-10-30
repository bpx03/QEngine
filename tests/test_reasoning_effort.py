"""CPU-only, < 10 s: --reasoning-effort / QE_REASONING_EFFORT (the default effort of requests that send none).
- the rendered prompt without the flag (template default xhigh), with the flag, with a per-request override
  (OpenAI reasoning_effort, chat_template_kwargs, the same fields on an Anthropic body), with thinking off;
- unknown efforts ('minimal', 'max', typos) render as medium with qwen38_syssafe.jinja (the server default
  template for both models); the models' own chat_template.jinja raises on them instead;
- the flag / env parsing: empty env = unset, the flag wins over the env, a value is lower-cased.
  python tests/test_reasoning_effort.py
"""
import io
import os
import sys
from contextlib import redirect_stdout

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
import app  # noqa: E402
from app import Template, template_kwargs  # noqa: E402

FLASH = os.environ.get("QE_FLASH_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
B27 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-27B-INT4")
SAFE = os.path.join(HERE, "..", "server", "templates", "qwen38_syssafe.jinja")

MSGS = [{"role": "user", "content": "hi"}]
XHIGH = "Reasoning effort is set to xhigh."
LOW = "Reasoning effort is set to low."


def effort_of(text):
    """Effort the prompt carries: xhigh / low by their instruction, medium = no instruction."""
    if XHIGH in text:
        return "xhigh"
    if LOW in text:
        return "low"
    return "medium"


def render(t, body, default_effort, think=True):
    return t.render([dict(m) for m in MSGS], None, **template_kwargs(body, think, default_effort))


def test_template_kwargs():
    assert "reasoning_effort" not in template_kwargs({}, True)
    assert template_kwargs({}, True, "low")["reasoning_effort"] == "low"
    assert template_kwargs({"reasoning_effort": "medium"}, True, "low")["reasoning_effort"] == "medium"
    assert template_kwargs({"reasoning_effort": None}, True, "low")["reasoning_effort"] == "low"
    kw = template_kwargs({"chat_template_kwargs": {"reasoning_effort": "xhigh"}, "reasoning_effort": "medium"},
                         True, "low")
    assert kw["reasoning_effort"] == "xhigh"  # chat_template_kwargs wins over the top-level field
    kw = template_kwargs({"chat_template_kwargs": {"reasoning_effort": None}, "reasoning_effort": "medium"}, True)
    assert kw["reasoning_effort"] == "medium"
    kw = template_kwargs({"thinking": {"type": "disabled"}}, True, "low")
    assert kw["enable_thinking"] is False


def test_render(path, safe):
    t = Template(path)
    cases = [  # (body, --reasoning-effort, expected effort in the prompt)
        ({}, None, "xhigh"),                                                  # unset: template default
        ({}, "low", "low"),                                                   # the flag
        ({}, "medium", "medium"),
        ({}, "high", "xhigh"),
        ({"reasoning_effort": "xhigh"}, "low", "xhigh"),                      # OpenAI per-request override
        ({"reasoning_effort": "low"}, None, "low"),
        ({"chat_template_kwargs": {"reasoning_effort": "medium"}}, "low", "medium"),
        ({"thinking": {"type": "enabled", "budget_tokens": 4096},            # Anthropic body + override
          "reasoning_effort": "xhigh"}, "low", "xhigh"),
        ({"thinking": {"type": "enabled", "budget_tokens": 4096}}, "low", "low"),
    ]
    if not safe:  # the stock template has no 'high' alias
        cases = [c for c in cases if c[1] != "high"]
    for body, flag, want in cases:
        got = effort_of(render(t, body, flag))
        assert got == want, (path, body, flag, got, want)
    # thinking off: no effort instruction at all, whatever the flag says
    for body in ({"thinking": {"type": "disabled"}}, {"chat_template_kwargs": {"enable_thinking": False}}):
        assert XHIGH not in render(t, body, "xhigh") and LOW not in render(t, body, "low")
    assert XHIGH not in render(t, {}, None, think=False)
    # efforts the template does not know
    for eff in ("minimal", "max", "none", "LOW", "bogus"):
        for body, flag in (({"reasoning_effort": eff}, None), ({}, eff)):
            if safe:
                assert effort_of(render(t, body, flag)) == "medium", (eff, body, flag)
            else:
                try:
                    render(t, body, flag)
                except jinja2.TemplateError:
                    pass
                else:
                    raise AssertionError(f"stock template accepted {eff!r}")


def parse(argv, env=None):
    old = os.environ.get("QE_REASONING_EFFORT")
    if env is None:
        os.environ.pop("QE_REASONING_EFFORT", None)
    else:
        os.environ["QE_REASONING_EFFORT"] = env
    try:
        out = io.StringIO()
        with redirect_stdout(out):
            a = app.parse_args(["--model", FLASH, "--metrics-port", "0"] + argv)
        return a.reasoning_effort, out.getvalue()
    finally:
        if old is None:
            os.environ.pop("QE_REASONING_EFFORT", None)
        else:
            os.environ["QE_REASONING_EFFORT"] = old


def test_args():
    assert parse([])[0] is None
    for empty in ("", "  "):  # compose "${QE_REASONING_EFFORT:-}" pass-through
        assert parse([], empty)[0] is None
        assert parse(["--reasoning-effort", empty])[0] is None
    assert parse([], "low")[0] == "low"
    assert parse(["--reasoning-effort", "Medium"], "low")[0] == "medium"  # the flag wins, lower-cased
    eff, log = parse(["--reasoning-effort", "minimal"])
    assert eff == "minimal" and "renders it as medium" in log
    assert "warning" not in parse(["--reasoning-effort", "low"])[1]


def main():
    test_template_kwargs()
    test_render(SAFE, True)
    n = 1
    for d in (FLASH, B27):
        p = os.path.join(d, "chat_template.jinja")
        if os.path.exists(p):
            test_render(p, False)
            n += 1
        else:
            print(f"skip (missing): {p}")
    if os.path.isdir(FLASH):
        test_args()
    else:
        print(f"skip args (missing model): {FLASH}")
    print(f"ok: reasoning effort ({n} templates)")


if __name__ == "__main__":
    main()
