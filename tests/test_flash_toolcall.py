"""CPU-only: tool calls / reasoning generated as Qwen3.8-Flash-Next token ids, streamed one token
at a time (and in 3-token steps) through the server's Detok + Parser, then parse_tool_calls.
  python tests/test_flash_toolcall.py
"""
import json
import os
import sys
import tempfile

os.environ["QE_DIAG_LOG"] = os.path.join(tempfile.mkdtemp(), "diag.log")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from app import Detok, Parser, parse_tool_calls  # noqa: E402
from model_profile import Profile  # noqa: E402

FLASH = os.environ.get("QE_FLASH_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
TOOLS = [{"type": "function", "function": {"name": "Write", "parameters": {"type": "object", "required": ["file_path", "content"],
          "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}}},
         {"type": "function", "function": {"name": "Edit", "parameters": {"type": "object", "required": ["file_path"],
          "properties": {"file_path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
                         "replace_all": {"type": "boolean"}, "lines": {"type": "array"}}}}}]
CASES = [
    ("Ich schreibe die Datei.\n</think>\n\n<tool_call>\n<function=Write>\n<parameter=file_path>\n/tmp/übersicht.md\n</parameter>\n"
     "<parameter=content>\n# नमस्ते\nสวัสดี <b>&lt;tag&gt;</b>\n\tdef f(): return '</think>'\n</parameter>\n</function>\n</tool_call>",
     "Ich schreibe die Datei.", "", [{"name": "Write", "arguments": {"file_path": "/tmp/übersicht.md",
                         "content": "# नमस्ते\nสวัสดี <b>&lt;tag&gt;</b>\n\tdef f(): return '</think>'"}}]),
    ("r\n</think>\n\nNajpierw edycja.\n\n<tool_call>\n<function=Edit>\n<parameter=file_path>\na.py\n</parameter>\n"
     "<parameter=replace_all>\ntrue\n</parameter>\n<parameter=lines>\n[1, 2]\n</parameter>\n</function>\n</tool_call>\n"
     "<tool_call>\n<function=Write>\n<parameter=file_path>\nb\n</parameter>\n<parameter=content>\nx\n</parameter>\n</function>\n</tool_call>",
     "r", "Najpierw edycja.", [{"name": "Edit", "arguments": {"file_path": "a.py", "replace_all": True, "lines": [1, 2]}},
                               {"name": "Write", "arguments": {"file_path": "b", "content": "x"}}]),
    ("\n</think>\n\nAntwort ohne Werkzeuge: क्षत्रिय, äöü, `<tool` im Text.", "", "Antwort ohne Werkzeuge: क्षत्रिय, äöü, `<tool` im Text.", []),
]


def main():
    tok, _, _ = Profile(FLASH).tokenizer()
    for step in (1, 3):
        for text, want_r, want_c, want_calls in CASES:
            ids = tok.encode(text, add_special_tokens=False).ids
            assert tok.decode(ids, skip_special_tokens=False) == text
            det, par = Detok(tok), Parser(True)
            got = {"reasoning": "", "content": ""}
            for i in range(0, len(ids), step):
                for k, t in par.feed(det.push(ids[i:i + step])):
                    got[k] += t
            for k, t in par.feed("", final=True):
                got[k] += t
            calls = parse_tool_calls(par.tool_text, TOOLS) if par.tool_text else []
            assert got["reasoning"].strip() == want_r and got["content"].strip() == want_c, (got, text[:40])
            assert calls == want_calls, json.dumps(calls, ensure_ascii=False)
    print(f"{len(CASES)} cases x 2 step sizes: reasoning / content / tool calls ok")
    print("ok")


if __name__ == "__main__":
    main()
