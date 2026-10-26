"""Model family detection and per-model server defaults (Qwen3.8-27B vs Qwen3.8-Flash-Next).

Everything the server needs to know about the model that is not the engine's business: tokenizer
(with the right pre-tokenizer), end-of-turn ids, recommended sampling, served name, MTP drafts,
the engine library and the chat-history cleanups that are on by default for the model.
"""
import json
import os

# Qwen/Qwen3.8-Flash-Next tokenizer.json (HF, 2026-10): the same vocab / merges / added tokens /
# normalizer as Qwen3.8-27B, but combining marks (\p{M}) join the letters before them. The
# Intel AutoRound checkpoint (and so our converted QE-INT4 directory) ships the 27B tokenizer.json
# instead (sha256 06b95093...), which splits e.g. Devanagari / Thai words at every vowel sign:
# different ids than the model was trained on. Patched here when loading a Flash-Next model.
FLASH_PRETOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split",
         "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}|"
                              " ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
         "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
    ],
}
FLASH_DECODER = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}

FAMILIES = {
    # 27B defaults are the ones the server always had (unchanged)
    "qwen38-27b": dict(
        served_name="qwen3.8-27b-qengine", temperature=0.6, mtp_k=7, split="",
        lib="build/libqengine.so", vision=True, prefill_chunk_idle=0,
        drop_empty_turns=False,   # HF #50 cleanup: off (would change 27B prompts)
        merge_system=False,       # server-side merge of leading system messages (syssafe does it)
    ),
    "qwen38-flash-next": dict(
        served_name="qwen3.8-flash-next-qengine",
        temperature=None,         # None: generation_config.json (1.0)
        mtp_k=3,                  # best k 2..4 at c=1 (measured)
        split="176,160,176",      # expert groups per card (the middle card, often the desktop one, less)
        lib="build/libqflash.so", vision=False,
        prefill_chunk_idle=4096,  # a prompt alone on the server: the engine's big batch (prefill-r3)
        drop_empty_turns=True, merge_system=True,
    ),
}


class Profile:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        cfg = _load(os.path.join(model_dir, "config.json"))
        text = cfg.get("text_config") or cfg
        self.model_type = cfg.get("model_type") or text.get("model_type") or ""
        self.family = "qwen38-flash-next" if self.model_type.startswith("qwen4_exp") else "qwen38-27b"
        self.vocab_size = text.get("vocab_size")
        self.max_position = text.get("max_position_embeddings")
        self.mtp_layers = text.get("mtp_num_hidden_layers") or 0
        self.gen = _load(os.path.join(model_dir, "generation_config.json"))
        eos = self.gen.get("eos_token_id", text.get("eos_token_id"))
        self.eos_ids = sorted(set(eos if isinstance(eos, list) else [eos] if eos is not None else []))
        self.defaults = dict(FAMILIES[self.family])

    @property
    def is_flash(self):
        return self.family == "qwen38-flash-next"

    def sampling(self, temperature=None, top_p=None, top_k=None, min_p=None):
        """Server defaults for requests without them: command line > family > generation_config."""
        g = self.gen
        t = temperature if temperature is not None else self.defaults["temperature"]
        if t is None:
            t = g.get("temperature", 1.0)
        return {"temperature": float(t),
                "top_p": float(top_p or g.get("top_p") or 0.95),
                "top_k": int(top_k or g.get("top_k") or 20),
                "min_p": float(min_p if min_p is not None else g.get("min_p", 0.0))}

    def tokenizer_json(self, override=None):
        """(tokenizer.json text, note): the model's file (or `override`), with the Flash-Next
        pre-tokenizer when the file has the 27B one."""
        path = override or os.path.join(self.model_dir, "tokenizer.json")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if not self.is_flash or override:
            return text, path
        tj = json.loads(text)
        if tj.get("pre_tokenizer") == FLASH_PRETOKENIZER and tj.get("decoder") == FLASH_DECODER:
            return text, path
        tj["pre_tokenizer"] = FLASH_PRETOKENIZER
        tj["decoder"] = FLASH_DECODER
        return json.dumps(tj, ensure_ascii=False), path + " (pre-tokenizer patched to Qwen3.8-Flash-Next)"

    def tokenizer(self, override=None):
        from tokenizers import Tokenizer
        text, note = self.tokenizer_json(override)
        return Tokenizer.from_str(text), text, note


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------------------- history cleanup


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return "" if c is None else str(c)


def drop_empty_turns(messages):
    """HF Qwen3.8-Flash-Next #50: an earlier assistant turn with nothing but reasoning (no text,
    no tool call) in the history makes the model end later turns empty ("No response requested."
    after tool results; 3/40 -> 0/40 with this). Drops such turns except the last message.
    Returns (messages, dropped count)."""
    out, n = [], 0
    last = len(messages) - 1
    for i, m in enumerate(messages):
        if (i < last and m.get("role") == "assistant" and not m.get("tool_calls")
                and not _text(m.get("content")).strip()):
            n += 1
            continue
        out.append(m)
    return out, n


def merge_leading_system(messages):
    """HF #75/#76: the stock template rejects more than one system message at the start (some
    clients send several). Merges them into one, joined like qwen38_syssafe.jinja does (trimmed,
    '\\n'), so the rendering with that template is byte-identical. Developer role counts too."""
    k = 0
    while k < len(messages) and messages[k].get("role") in ("system", "developer"):
        k += 1
    if k <= 1:
        return messages
    parts = [_text(m.get("content")).strip() for m in messages[:k]]
    return [{"role": "system", "content": "\n".join(p for p in parts if p)}] + list(messages[k:])
