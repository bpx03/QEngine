"""DSV4 (DeepSeek-V4.1-Flash GGUF, e.g. Maya-S) for the server: metadata reader and tokenizer.

The model directory holds a GGUF split set; the first file's header carries the config, the tokenizer (gpt2 BPE:
tokens, merges, token types) and a chat template. Own reader of the documented GGUF container (header and key/values
only; tensors stay with the engine). The tokenizer is built once from the GGUF metadata as a tokenizers.Tokenizer
(byte-level BPE with DeepSeek's pre-tokenizer) and cached in the engine's cache dir.
"""
import glob
import hashlib
import json
import os
import struct

# gguf value types
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    if n > (1 << 26):
        raise ValueError("GGUF: string too long")
    return f.read(n).decode("utf-8", errors="replace")


def _read_val(f, t):
    if t in _SCALAR:
        fmt = _SCALAR[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8:
        return _read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if et == 8:
            return [_read_str(f) for _ in range(n)]
        if et not in _SCALAR:
            raise ValueError(f"GGUF: array of type {et}")
        fmt = _SCALAR[et]
        sz = struct.calcsize(fmt)
        raw = f.read(sz * n)
        return list(struct.unpack("<" + fmt[1:] * n, raw)) if n else []
    raise ValueError(f"GGUF: value type {t}")


def read_kv(path):
    """The key/values of one GGUF file's header."""
    kv = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        (ver,) = struct.unpack("<I", f.read(4))
        if ver not in (2, 3):
            raise ValueError(f"{path}: GGUF version {ver}")
        _nt, nkv = struct.unpack("<QQ", f.read(16))
        for _ in range(nkv):
            k = _read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            kv[k] = _read_val(f, t)
    return kv


def first_file(model_dir):
    """The split set's first file (split.no 0) of a model directory, or None."""
    files = sorted(glob.glob(os.path.join(model_dir, "*.gguf")))
    for p in files:
        if "-00001-of-" in os.path.basename(p) or len(files) == 1:
            return p
    return files[0] if files else None


def is_dsv4_dir(model_dir):
    p = first_file(model_dir) if os.path.isdir(model_dir or "") else None
    if not p:
        return False
    try:
        return read_kv(p).get("general.architecture") == "deepseek41"
    except (OSError, ValueError):
        return False


# DeepSeek-V3 / V4 pre-tokenizer (the model's tokenizer.json): digits in runs of up to 3, CJK runs, punctuation, words
_PRE = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": False},
        {"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated", "invert": False},
        {"type": "Split",
         "pattern": {"Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"},
         "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    ],
}


def tokenizer_json(kv):
    """A tokenizers JSON (byte-level BPE) from the GGUF tokenizer metadata."""
    toks = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(toks)
    merges = kv["tokenizer.ggml.merges"]
    vocab = {t: i for i, t in enumerate(toks)}
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False,
              "special": types[i] == 3} for i, t in enumerate(toks) if types[i] in (3, 4)]
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
        "pre_tokenizer": _PRE,
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    }


def load_tokenizer(model_dir, cache_dir=None):
    """(tokenizers.Tokenizer, tokenizer json text, metadata kv) for a DSV4 GGUF directory; the JSON is cached."""
    from tokenizers import Tokenizer
    p = first_file(model_dir)
    kv = read_kv(p)
    st = os.stat(p)
    key = hashlib.sha1(f"{os.path.basename(p)}:{st.st_size}".encode()).hexdigest()[:16]
    cache_dir = cache_dir or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                                          "qengine", "dsv4")
    cp = os.path.join(cache_dir, f"tokenizer-{key}.json")
    if os.path.exists(cp):
        txt = open(cp, encoding="utf-8").read()
    else:
        txt = json.dumps(tokenizer_json(kv), ensure_ascii=False)
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(txt)
        os.replace(tmp, cp)
    return Tokenizer.from_str(txt), txt, kv
