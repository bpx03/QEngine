"""CPU-only: the server's tokenizer gives the same ids as the reference HF tokenizer for both
models (27B: the model's tokenizer.json; Flash-Next: the official Qwen/Qwen3.8-Flash-Next
tokenizer.json, whose pre-tokenizer keeps combining marks with their letters), and the server's
segment-wise encode (Generator.encode) equals encoding the whole prompt.

  python tests/test_flash_tokenizer.py
Reference Flash tokenizer: $QE_FLASH_HF_TOKENIZER, else ~/.cache/qengine/hf/Qwen3.8-Flash-Next/
tokenizer.json (downloaded from huggingface.co if missing). With transformers importable (e.g.
a venv from requirements-ref.txt) AutoTokenizer is checked too.
"""
import collections
import glob
import os
import random
import sys
import threading
import unicodedata
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
from tokenizers import Tokenizer  # noqa: E402

from model_profile import Profile  # noqa: E402

FLASH = os.environ.get("QE_FLASH_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
B27 = os.environ.get("QE_27B_MODEL", os.path.join(HERE, "..", "models", "Qwen3.8-27B-INT4"))
if not os.path.exists(B27):
    B27 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-27B-INT4")
REF = os.environ.get("QE_FLASH_HF_TOKENIZER",
                     os.path.expanduser("~/.cache/qengine/hf/Qwen3.8-Flash-Next/tokenizer.json"))


def ref_flash():
    if not os.path.exists(REF):
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        urllib.request.urlretrieve("https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/tokenizer.json", REF)
    return REF


def corpus():
    c = [
        "Příliš žluťoučký kůň úpěl ďábelské ódy. Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich.",
        "Bitte schreibe eine Funktion, die Wörter zählt; „Anführungszeichen“ — Gedankenstrich… ÄÖÜŘŠČŽŮ",
        unicodedata.normalize("NFD", "Příliš žluťoučký kůň — NFD-Zerlegung"),
        "def f(x):\n    return {'a': [1, 2, 3], \"b\": x ** 2}  # ok\n\tif x != 0: pass\r\n",
        "const re = /^\\p{L}+$/u; let s = `${a}äř`; // Kommentar\nfn main() -> Result<(), E> { Ok(()) }",
        "नमस्ते दुनिया, हिन्दी में लिखा गया पाठ। क्षत्रिय",
        "สวัสดีครับ ภาษาไทย ไม่มีช่องว่างระหว่างคำ",
        "தமிழ் மொழி ಕನ್ನಡ ಭಾಷೆ తెలుగు বাংলা ગુજરાતી ਪੰਜਾਬੀ മലയാളം සිංහල",
        "שָׁלוֹם עֲלֵיכֶם; مَرْحَبًا بِكُمْ; Tiếng Việt có dấu; Ελληνικά ἄνθρωπος",
        "é́́ z̶a̷l̸g̵o 1⃣ ✋🏽 👩‍👩‍👧 x̨́",
        "한국어 한글 (jamo), 日本語のテキスト、中文文本。",
        "  leading spaces\n\n\n  trailing \t \n",
        "<|im_start|>user\nHallo<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la ~/äř\n</parameter>\n</function>\n</tool_call>",
    ]
    for p in sorted(glob.glob(os.path.join(HERE, "..", "server", "*.py")))[:6]:
        c.append(open(p, encoding="utf-8").read()[:20000])
    rnd = random.Random(1)
    marks = [chr(x) for x in range(0x300, 0x370)] + [chr(x) for x in range(0x93E, 0x94D)] + \
        [chr(x) for x in range(0xE31, 0xE3A)] + ["̨", "̇", "⃝", "️"]
    base = list("aäeëoöuüßřšžčůxyz AÄ.,;:!?()[]{}<>'\"-_/\\0123456789\n\t") + ["क", "ส", "ש", "ع", "й", "語"]
    for _ in range(300):
        c.append("".join(rnd.choice(marks) if rnd.random() < 0.3 else rnd.choice(base)
                         for _ in range(rnd.randint(1, 80))))
    return c


def check(name, tok, ref, texts):
    bad = 0
    for t in texts:
        a = tok.encode(t, add_special_tokens=False).ids
        b = ref.encode(t, add_special_tokens=False).ids
        if a != b:
            bad += 1
            if bad <= 3:
                print(f"  {name} MISMATCH {t[:40]!r}: {a[:12]} vs {b[:12]}")
        assert tok.decode(a, skip_special_tokens=False) == ref.decode(b, skip_special_tokens=False)
    print(f"{name}: {len(texts) - bad}/{len(texts)} texts identical ids")
    return bad


def segmentwise(tok, text):
    """Generator.encode on a bare object (no engine)."""
    import app
    g = app.Generator.__new__(app.Generator)
    g.tok, g.seg_cache, g.seg_tokens = tok, collections.OrderedDict(), 0
    g.gen_turns, g.seg_lock = collections.OrderedDict(), threading.Lock()
    g.im_end_nl = tok.encode("<|im_end|>\n", add_special_tokens=False).ids
    return list(g.encode(text))


def main():
    texts = corpus()
    fails = 0
    # Flash-Next: server tokenizer (model dir + patch) == official HF tokenizer.json
    prof = Profile(FLASH)
    assert prof.is_flash and prof.eos_ids == [248044, 248046], (prof.family, prof.eos_ids)
    tok, _, note = prof.tokenizer()
    print("flash tokenizer:", note)
    ref = Tokenizer.from_file(ref_flash())
    fails += check("flash server vs HF official", tok, ref, texts)
    # a directory with the 27B tokenizer.json (Intel AutoRound): patched to the same ids
    ar = os.environ.get("QE_FLASH_AUTOROUND", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-W4A16-AutoRound"))
    if os.path.exists(os.path.join(ar, "tokenizer.json")):
        raw = Tokenizer.from_file(os.path.join(ar, "tokenizer.json"))
        diff = sum(raw.encode(t, add_special_tokens=False).ids != ref.encode(t, add_special_tokens=False).ids
                   for t in texts)
        tar, _, note_ar = Profile(ar).tokenizer()
        print(f"  {ar}: its own tokenizer.json differs on {diff}/{len(texts)} texts; server: {note_ar}")
        fails += check("flash (AutoRound dir) server vs HF official", tar, ref, texts)
    # 27B: unchanged (the model's own file, no patch)
    p27 = Profile(B27)
    assert not p27.is_flash
    t27, _, note27 = p27.tokenizer()
    print("27B tokenizer:", note27)
    fails += check("27B server vs model file", t27, Tokenizer.from_file(os.path.join(B27, "tokenizer.json")), texts)
    # segment-wise encode of a chat prompt == whole-text encode
    chat = "".join(f"<|im_start|>{r}\n{t}<|im_end|>\n" for r, t in
                   zip(["system", "user", "assistant", "user"] * 8, texts[:32]))
    for name, t in (("flash", tok), ("27B", t27)):
        assert segmentwise(t, chat) == t.encode(chat, add_special_tokens=False).ids, name
        print(f"{name}: segment-wise encode == whole encode ({len(t.encode(chat).ids)} tokens)")
    try:
        from transformers import AutoTokenizer
    except Exception:
        print("transformers not importable: AutoTokenizer check skipped")
    else:
        import transformers
        at = AutoTokenizer.from_pretrained(os.path.dirname(ref_flash()))
        bad = sum(at.encode(t, add_special_tokens=False) != tok.encode(t, add_special_tokens=False).ids for t in texts)
        print(f"transformers {transformers.__version__} AutoTokenizer (official Flash files): "
              f"{len(texts) - bad}/{len(texts)} identical")
        pre = getattr(getattr(at, "backend_tokenizer", None), "pre_tokenizer", None)
        if bad and pre is not None and "\\p{M}" not in pre.__getstate__().decode():
            # transformers 5 builds Qwen2Tokenizer's pre-tokenizer from its own (27B) regex and
            # ignores the one in tokenizer.json: a transformers issue, not ours (4.x uses the file)
            print("  WARNING: this transformers ignores the tokenizer.json pre-tokenizer (27B regex in "
                  "Qwen2Tokenizer); not counted")
        else:
            fails += bad
    assert fails == 0, fails
    print("ok")


if __name__ == "__main__":
    main()
