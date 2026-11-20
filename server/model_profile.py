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

# Sampling recommended by the Qwen3.8-27B and Qwen3.8-Flash-Next model cards (the same values on both), per mode
CARD_SAMPLING = {
    True: dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),   # thinking
    False: dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5),  # instruct (non-thinking)
}

FAMILIES = {
    # 27B defaults are the ones the server always had (unchanged)
    "qwen38-27b": dict(
        served_name="qwen3.8-27b-qengine", mtp_k=7, split="",
        lib="build/libqengine.so", vision=True, prefill_chunk_idle=0,
        drop_empty_turns=False,   # HF #50 cleanup: off (would change 27B prompts)
        merge_system=False,       # server-side merge of leading system messages (syssafe does it)
    ),
    "glm53-flash": dict(
        served_name="glm-5.3-flash-qengine",
        temperature=None,         # generation_config.json (the engine samples: T / top_p / top_k / min_p / seed per slot)
        mtp_k=3,                  # MTP chain k=3 since round 13 (tau 0.10 + EFF: Maya-like k=2 104.8 -> k=3 104.3, listing 114.5 -> 122.6, std +1..15 %)
        split="", lib="build/libqglm.so", vision=False, prefill_chunk_idle=0,
        drop_empty_turns=False, merge_system=False,
        tool_format="glm",        # <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        # default checkpoint of --model glm: the QEngine INT4 quant (hf_repo) downloaded to models/; QE_GLM_MODEL
        # overrides
        default_dir="models/GLM-5.3-Flash-INT4-QEngine",
        hf_repo="bpx03/GLM-5.3-Flash-INT4-QEngine",  # the upload package (not uploaded yet: the user's OK)
        own_template=True,        # the model's chat_template.jinja (reasoning_effort low / high / max)
    ),
    # [shared-srv] DeepSeek-V4.1-Flash from a GGUF split set (e.g. Maya-S) on the DSV4 engine: a single-sequence engine
    # behind the slot interface through its adapter (server/dsv4_engine.py); prompt format dsv4_chat.py, DSML tool calls.
    # Extension point: another engine family (gpt-oss, harmony) names its adapter / renderer / tool_format the same way
    "dsv4": dict(
        served_name="deepseek-v4.1-flash-qengine",
        mtp_k=5,                  # DSpark drafts per pass (at most 5; the engine plans them itself)
        split="", lib="build_dsv4/libqdsv4.so", vision=False, prefill_chunk_idle=0,
        drop_empty_turns=False, merge_system=False,
        tool_format="dsml",       # "\n\n<｜DSML｜ calls>" blocks (dsv4_chat.parse_completion)
        think_close="",           # the forced think close writes "</think>" alone: dsv4_chat renders reasoning + "</think>" + content
        adapter="dsv4_engine:Dsv4Engine",
        renderer="dsv4",          # dsv4_chat.encode instead of a jinja template
        prefill_chunk=10240,      # the engine's prefill pass (2,048-token chunks cost -20 / -10 % aggregate at c = 4 / 8)
        # [dsv4-rel8-ctxdef] without --max-ctx: 262144 with both context reservations (user 11-19). The lazy one (peer
        # access between every pair) holds memory for the base context only, the rest is lent by the expert pool on
        # demand; the static one maps every position at load (fewer experts in VRAM). An explicit --max-ctx always wins
        max_ctx="auto", max_ctx_lazy=262144, max_ctx_static=262144,
        prefill_turns="slices",   # a prompt chunk, then one 16-pass slice per decoding request (dsv4_app's turns)
        stats_file="~/.cache/qengine/stats-dsv4.json",
    ),
    # [srv-gptoss] gpt-oss-120B (src/gptoss, harmony format): the engine implements server/qe_capi.h itself (no adapter);
    # the model's own chat template renders, server/harmony.py parses the reply (channels, tool calls)
    "gptoss": dict(
        served_name="gpt-oss-120b-qengine",
        mtp_k=1,                  # (the C API takes k and ignores it: greedy replies draft by prompt lookup, GPTOSS_LOOKUP)
        split="", lib="build_gptoss/libqgptoss.so", vision=False,
        # [gptoss-pf90-idle] a prompt alone on the server: one engine chunk (10,240 rows, 3 staggered lanes) per call
        # instead of --prefill-chunk 2048 (2 lanes of 1,024: fixed costs per lane, attention grid of 1.2 waves at depth);
        # server (club bench.sh) 10K 9.7K -> 13.1K tok/s, 90K 3.7K -> 4.6K tok/s (TTFT ~24 -> 19 s). Busy or a request
        # waiting: 2048 as before; idle, a cut-in / cancel waits for the call in flight (<= ~3.5 s at 90K depth)
        prefill_chunk_idle=10240,
        drop_empty_turns=False, merge_system=False,
        tool_format="harmony",
        own_template=True,        # chat_template.jinja (Reasoning: low / medium / high, the functions namespace)
        seg_mark="<|start|>",     # encode() splits the prompt at this atomic token
        no_effort="low",          # gpt-oss always reasons: effort none = the lowest level (open decision, design 9.2 (5))
    ),
    "qwen38-flash-next": dict(
        served_name="qwen3.8-flash-next-qengine",
        mtp_k=3,                  # best k 2..4 at c=1 (measured)
        split="176,160,176",      # expert groups per card (the middle card, often the desktop one, less)
        lib="build/libqflash.so", vision=True,
        prefill_chunk_idle=4096,  # a prompt alone on the server: the engine's big batch (prefill-r3)
        preempt_max=8192,         # tool results up to 8K join the shared prefill batches at once
        drop_empty_turns=True, merge_system=True,
    ),
}


class Profile:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.gguf = None
        try:   # [shared-srv] a DSV4 GGUF split set (the DSV4 modules are not in the public export: then never)
            import dsv4_gguf
            if dsv4_gguf.is_dsv4_dir(model_dir):
                self.gguf = dsv4_gguf
        except ImportError:
            pass
        if self.gguf is not None:
            kv = self.gguf.read_kv(self.gguf.first_file(model_dir))
            self.model_type, self.family = "deepseek_v4_gguf", "dsv4"
            self.vocab_size = len(kv.get("tokenizer.ggml.tokens") or []) or None
            self.max_position, self.mtp_layers = None, 0
            self.gen = {"temperature": kv.get("general.sampling.temp"), "top_p": kv.get("general.sampling.top_p")}
            eos = kv.get("tokenizer.ggml.eos_token_id")
            self.eos_ids = [int(eos)] if eos is not None else [1]
            self.defaults = dict(FAMILIES[self.family])
            return
        cfg = _load(os.path.join(model_dir, "config.json"))
        text = cfg.get("text_config") or cfg
        self.model_type = cfg.get("model_type") or text.get("model_type") or ""
        self.family = ("qwen38-flash-next" if self.model_type.startswith("qwen4_exp") else
                       "glm53-flash" if self.model_type.startswith("glm5_next") else
                       "gptoss" if self.model_type == "gpt_oss" else "qwen38-27b")
        self.vocab_size = text.get("vocab_size")
        self.max_position = text.get("max_position_embeddings")
        self.mtp_layers = text.get("mtp_num_hidden_layers") or 0
        self.gen = _load(os.path.join(model_dir, "generation_config.json"))
        eos = self.gen.get("eos_token_id", text.get("eos_token_id"))
        self.eos_ids = sorted(set(eos if isinstance(eos, list) else [eos] if eos is not None else []))
        self.defaults = dict(FAMILIES[self.family])

    @property
    def is_glm(self):
        return self.family == "glm53-flash"

    @property
    def is_gptoss(self):
        return self.family == "gptoss"

    @property
    def is_dsv4(self):
        return self.family == "dsv4"

    @property
    def is_flash(self):
        return self.family == "qwen38-flash-next"

    def sampling(self, thinking, temperature=None, top_p=None, top_k=None, min_p=None, presence_penalty=None):
        """Server defaults of one mode for requests without them: command line > the model card's for this mode."""
        c = CARD_SAMPLING[bool(thinking)]
        if self.family == "gptoss":   # OpenAI's recommendation for gpt-oss (generation_config.json has no values)
            t0, p0 = self.gen.get("temperature"), self.gen.get("top_p")
            return {"temperature": float(temperature if temperature is not None else (t0 if t0 is not None else 1.0)),
                    "top_p": float(top_p or (p0 if p0 is not None else 1.0)),
                    "top_k": int(top_k or 0), "min_p": float(min_p or 0.0),
                    "presence_penalty": float(presence_penalty or 0.0)}
        if self.family == "dsv4":   # the GGUF's general.sampling.* (dsv4_app's defaults), else 1.0; the engine sampler: top_k 0 = 64
            t0, p0 = self.gen.get("temperature"), self.gen.get("top_p")
            return {"temperature": float(temperature if temperature is not None else (t0 if t0 is not None else 1.0)),
                    "top_p": float(top_p or (p0 if p0 is not None else 1.0)),
                    "top_k": int(top_k or 0), "min_p": float(min_p or 0.0),
                    "presence_penalty": float(presence_penalty or 0.0)}
        return {"temperature": float(temperature if temperature is not None else c["temperature"]),
                "top_p": float(top_p or c["top_p"]),
                "top_k": int(top_k or c["top_k"]),
                "min_p": float(min_p if min_p is not None else c["min_p"]),
                "presence_penalty": float(presence_penalty if presence_penalty is not None else c["presence_penalty"])}

    def tokenizer_json(self, override=None):
        """(tokenizer.json text, note): the model's file (or `override`) without truncation / padding, with the
        Flash-Next pre-tokenizer when the file has the 27B one. (Some finetunes ship `"truncation": {"max_length":
        4096}` (Swift-1.5 27B): every longer prompt would be cut silently.)"""
        path = override or os.path.join(self.model_dir, "tokenizer.json")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        tj = json.loads(text)
        notes = []
        if tj.get("truncation") or tj.get("padding"):
            tj["truncation"] = tj["padding"] = None
            notes.append("truncation / padding off")
        if self.is_flash and not override and (tj.get("pre_tokenizer") != FLASH_PRETOKENIZER
                                                or tj.get("decoder") != FLASH_DECODER):
            tj["pre_tokenizer"] = FLASH_PRETOKENIZER
            tj["decoder"] = FLASH_DECODER
            notes.append("pre-tokenizer patched to Qwen3.8-Flash-Next")
        if not notes:
            return text, path
        return json.dumps(tj, ensure_ascii=False), path + " (" + ", ".join(notes) + ")"

    def tokenizer(self, override=None):
        if self.gguf is not None and not override:   # [shared-srv] built from the GGUF metadata (cached JSON)
            tok, text, _kv = self.gguf.load_tokenizer(self.model_dir, os.environ.get("QE_DSV4_CACHE_DIR") or None)
            return tok, text, "from the GGUF metadata"
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


class DroppedTurnLog:
    """The HF #50 cleanup's log line, once per conversation: a client resends the same reasoning-only turns with
    every request of a long conversation, and a line per request reads like an error. Logs a conversation again
    only when its count grows; counts for /api/stats. key: the conversation's identity (the agent tracker's
    conversation id, else a hash of its root)."""

    def __init__(self, cap=1024):
        import collections
        import threading
        self.seen, self.cap, self.lock = collections.OrderedDict(), cap, threading.Lock()
        self.counts = {"requests": 0, "turns": 0, "logged": 0}

    def note(self, key, n):
        """n reasoning-only turns left out of this request's prompt: the count seen before when it grew (log
        now), else None."""
        with self.lock:
            self.counts["requests"] += 1
            self.counts["turns"] += n
            prev = self.seen.get(key)
            self.seen[key] = max(n, prev or 0)
            self.seen.move_to_end(key)
            while len(self.seen) > self.cap:
                self.seen.popitem(last=False)
            if prev is not None and n <= prev:
                return None
            self.counts["logged"] += 1
            return prev or 0

    def stats(self):
        with self.lock:
            return dict(self.counts, conversations=len(self.seen))


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


# ----------------------------------------------------------------------------- what each model supports
# [feature-matrix] One table for the settings that work on some models only. The server logs at start one line per
# flag / environment variable that is set but has no effect for the loaded model, and at request time one line per
# unsupported request parameter (once per process). A new flag or variable needs one row here.
#   on:   the families where it has an effect (None: every family, then only `cap` decides)
#   cap:  a capability the loaded engine must have (Generator.caps: checked after the engine loads)
#   note: what happens instead (shown in parentheses)
#   when: f(value, args) -> True when the value matters at all (default: always)
# The README feature matrix (public README "Feature matrix") and list the same facts.

Q27, QFN, DSV4, GLM, GPTOSS = "qwen38-27b", "qwen38-flash-next", "dsv4", "glm53-flash", "gptoss"
MODEL_NAMES = {Q27: "Qwen3.8-27B", QFN: "Qwen3.8-Flash-Next", DSV4: "DeepSeek-V4.1-Flash", GLM: "GLM-5.3-Flash",
               GPTOSS: "gpt-oss-120B"}
ALL = frozenset(MODEL_NAMES)
QWEN = frozenset({Q27, QFN})


_OFF = ("", "0", "off", "false", "no", "none")  # (lsa_map.OFF)


def _nonzero(v):
    try:
        return float(str(v).strip()) != 0.0
    except ValueError:
        return True


def _card_ids(v):
    return [x.strip() for x in str(v or "").split(",") if x.strip()]


class Knob:
    __slots__ = ("on", "cap", "note", "when")

    def __init__(self, on=None, cap=None, note="", when=None):
        self.on = None if on is None else frozenset(on)
        self.cap, self.note, self.when = cap, note, when


# server flags (argparse dest -> row); only flags given on the command line are checked
FLAGS = {
    "lsa": Knob({Q27}, when=lambda v, a: str(v).strip().lower() not in _OFF, note="Latent Sparse Attention is for Qwen3.8-27B only; the model keeps its own attention"),
    "vision": Knob(QWEN, note="no image input; images in requests are replaced by a text note",
                   when=lambda v, a: bool(v)),
    "vision_device": Knob(QWEN, note="no image input"),
    "max_image_pixels": Knob(QWEN, note="no image input"),
    "mtp_k": Knob(ALL - {DSV4, GPTOSS}, note="the engine ignores the draft count: DSV4's DSpark plans its own, gpt-oss drafts by prompt lookup (GPTOSS_LOOKUP)"),
    "mtp_file": Knob({Q27}, note="the engine loads its own MTP head (Flash-Next: mtp*.safetensors in the model directory)"),
    "split": Knob(QWEN | {GLM}, note="the engine places its weights itself"),  # (GLM: routed expert columns per card)
    # (server/run.sh always passes --devices 0..n-1 of the visible cards: only another order or subset counts; GLM takes
    # the first 3 of more, unless GLM_NGPU is set)
    # (feature_matrix_review L2: DSV4, the only engine without the flag, always runs on the first 3 visible cards, so a
    # leading subset or a longer list is ignored too)
    "devices": Knob(ALL - {DSV4}, note="the engine uses the first 3 visible cards; choose them with CUDA_VISIBLE_DEVICES",
                    when=lambda v, a: _card_ids(v) != ["0", "1", "2"]),
    "max_ctx": Knob(QWEN | {DSV4}, when=lambda v, a: v is not None and v > 262144,
                    note="the engine serves its own maximum: GLM-5.3-Flash 262,144 (GLM_MAX_CTX sets it), gpt-oss-120B "
                         "131,072 (start-up 'context:' line)"),
    "template": Knob(ALL - {DSV4}, note="DeepSeek-V4.1-Flash renders its own prompt format"),
    "park": Knob(cap="park", when=lambda v, a: bool(v), note="the engine cannot park a reply in host memory; requests beyond the seats wait"),
    "park_reserve": Knob(cap="park", note="no parking on this engine"),
    "long_share": Knob(cap="multi", note="no shared prefill batches on this engine"),
    "fill_passes": Knob(cap="multi", when=lambda v, a: bool(v), note="no shared prefill batches on this engine"),
    "bg_archive": Knob(cap="archive", note="the engine has no background archive call"),
    "pause_wait": Knob(cap="pause", note="the engine cannot pause decoding sequences; a request that does not fit waits"),
    "disk_cache": Knob(QWEN, note="no disk cache: conversations live in VRAM and host RAM only",
                       when=lambda v, a: bool(v)),
    "disk_cache_gb": Knob(QWEN, note="no disk cache"),
    "persist_idle": Knob(QWEN, note="no disk cache"),
    "presence_penalty": Knob(cap="penalty", note="the sampler has no presence / frequency penalty",
                             when=lambda v, a: bool(v)),
    "frequency_penalty": Knob(cap="penalty", note="the sampler has no presence / frequency penalty",
                              when=lambda v, a: bool(v)),
    "loop_guard_penalty": Knob(cap="penalty", note="the loop guard runs without its penalty"),
    "thinking_budget": Knob(cap="force", note="no forced close of the reasoning: budgets are ignored",
                            when=lambda v, a: v not in (None, "", "0")),
    "think": Knob(ALL - {GPTOSS}, when=lambda v, a: not v,
                  note="gpt-oss always reasons: thinking off renders at reasoning effort low"),
    "reasoning_effort": Knob(ALL - {GPTOSS}, when=lambda v, a: str(v).strip().lower() == "none",
                             note="gpt-oss always reasons: none renders as low"),
    # the DSV4 engine's own flags (dsv4_flags); --tier-reset / --no-prefix-share also turn off gpt-oss's shared
    # system-part snapshot
    "engine_args": Knob({DSV4}), "cache_dir": Knob({DSV4}), "no_dspark": Knob({DSV4}), "archive_gib": Knob({DSV4}),
    "kv_pool_mib": Knob({DSV4}), "route_margin": Knob({DSV4}), "drop_w": Knob({DSV4}), "serial": Knob({DSV4}),
    "max_active": Knob({DSV4}), "tier_reset": Knob({DSV4, GPTOSS}), "no_prefix_share": Knob({DSV4, GPTOSS}),
}

# environment variables of an engine's own namespace: set for another model, they have no effect
ENV_OWNERS = (("QE_DSV4_", DSV4), ("FLASH_", QFN), ("GLM_", GLM), ("GPTOSS_", GPTOSS), ("DSV4_", DSV4))
# read by the server itself or by more than one engine (never warned by the prefix rule)
ENV_SERVER = frozenset({"FLASH_PAR_FALLBACK", "FLASH_DISK_CACHE", "FLASH_DISK_CACHE_GB", "QE_DSV4_LEGACY"})

_SEATS = "seats: Qwen3.8-27B QE_MAX_SEQS, Flash-Next FLASH_MAX_SEQS, gpt-oss GPTOSS_SLOTS, GLM GLM_CONVS, DSV4 --max-active"
_KVF = "K/V format: Qwen3.8-27B QE_KV_BITS / QE_KV_FP8, Flash-Next FLASH_KV_BITS, gpt-oss GPTOSS_KV"
_ARCH = ("host-RAM archive: Qwen3.8-27B / gpt-oss QE_KV_ARCHIVE_GB, Flash-Next FLASH_KV_ARCHIVE_GB, GLM GLM_ARC_GB "
         "(else QE_KV_ARCHIVE_GB), DSV4 --archive-gib")
ENV = {
    # rows with a note of their own (the prefix rule covers the rest of each namespace)
    "FLASH_KV": Knob({QFN}, note="KV stays in VRAM; KV in host RAM is Flash-Next only"),
    "FLASH_YARN": Knob({QFN}, note="Qwen3.8-27B: QE_YARN; the other models keep their own RoPE"),
    "FLASH_MAX_SEQS": Knob({QFN}, note=_SEATS),
    "FLASH_KV_BITS": Knob({QFN}, note=_KVF),
    "FLASH_KV_ARCHIVE_GB": Knob({QFN}, note=_ARCH),
    "FLASH_COMM": Knob({QFN}, note="Qwen3.8-27B: QE_COMM"),
    "FLASH_PIN_WAIT_S": Knob(QWEN),  # (the 27B's fallback for QE_GPU_LOCK_WAIT_S)
    "GPTOSS_DFLASH": Knob(frozenset(), note="the server ignores it in 1.5 and does not load the drafter (engine tool only)"),
    # the 27B engine's variables (Flash-Next reads its own FLASH_* names)
    "QE_MAX_SEQS": Knob({Q27}, note=_SEATS),
    "QE_KV_BITS": Knob({Q27}, note=_KVF), "QE_KV_FP8": Knob({Q27}, note=_KVF), "QE_KV_SCALES": Knob({Q27}),
    "QE_KV_SPLIT": Knob({Q27}), "QE_POOL_TOKENS": Knob({Q27}), "QE_SPLIT": Knob({Q27}),
    "QE_YARN": Knob({Q27}, note="Flash-Next: FLASH_YARN; the other models keep their own RoPE"),
    "QE_LSA": Knob({Q27}, note="Latent Sparse Attention is for Qwen3.8-27B only"),
    "QE_LSA_MODE": Knob({Q27}), "QE_LSA_PF_MIN": Knob({Q27}), "QE_LSA_DEC_MIN": Knob({Q27}),
    "QE_DFLASH": Knob({Q27}, note="the DFlash2 drafter is Qwen3.8-27B's"), "QE_DRAFTER": Knob({Q27}),
    "QE_DF_SIB": Knob({Q27}), "QE_DRAFT_VOCAB": Knob({Q27}), "QE_MTP_VOCAB": Knob({Q27}), "QE_LOOKUP": Knob({Q27}),
    "QE_MTP_FILE": Knob({Q27}, note="the engine loads its own MTP head"), "QE_INT8_MTP_FT": Knob({Q27}),
    "QE_ARCH_SUBS": Knob({Q27}), "QE_ARCH_DEMOTE": Knob({Q27}), "QE_ARCH_REPLY": Knob({Q27}),
    "QE_SNAPSHOTS": Knob({Q27}), "QE_PREFIX_GB": Knob({Q27}), "QE_ADMIT_HEADROOM": Knob({Q27}),
    "QE_ADMIT_BY_LIMIT": Knob({Q27}), "QE_RESERVE_GB": Knob({Q27}), "QE_VRAM_ADAPT": Knob({Q27}),
    # (GLM: only when GLM_ARC_GB is unset and GLM_ARC is on; a `when` cannot see the other variables)
    "QE_KV_ARCHIVE_GB": Knob({Q27, GPTOSS, GLM}, note=_ARCH),
    "QE_COMM": Knob({Q27}, note="Flash-Next: FLASH_COMM"),
    "QE_P2P": Knob({Q27, QFN, GLM}),
    "QE_STEER": Knob(QWEN, note="activation steering is for the Qwen3.8 models"),
    "QE_STEER_DIR": Knob(QWEN), "QE_STEER_DEFAULT": Knob(QWEN),
    "QE_DISK_CACHE": Knob(QWEN, note="no disk cache on this engine"),
    "QE_GLM_MODEL": Knob({GLM}),
    # server defaults that need an engine capability
    "QE_VRAM_LOW_MB": Knob({Q27}), "QE_VRAM_TARGET_MB": Knob({Q27}), "QE_SNAP_ASYNC": Knob({Q27}), "QE_SNAP_DEMOTE": Knob({Q27}), "QE_ARCHIVE_WAIT_S": Knob({Q27}),
    "QE_DF_BATCH": Knob({Q27}), "QE_SHARE_PREFIX": Knob({Q27}),
    "QE_GPU_LOCK_WAIT_S": Knob(QWEN), "QE_ALLOW_SHARED_GPU": Knob(QWEN),
    "QE_REASONING_EFFORT": Knob(ALL - {GPTOSS}, when=lambda v, a: str(v).strip().lower() == "none",
                                note="gpt-oss always reasons: none renders as low"),
    "QE_PRESENCE_PENALTY": Knob(cap="penalty", when=lambda v, a: _nonzero(v), note="the sampler has no presence / frequency penalty"),
    "QE_FREQUENCY_PENALTY": Knob(cap="penalty", when=lambda v, a: _nonzero(v), note="the sampler has no presence / frequency penalty"),
    "QE_LOOP_GUARD_PENALTY": Knob(cap="penalty", note="the loop guard runs without its penalty"),
    "QE_THINKING_BUDGET": Knob(cap="force", when=lambda v, a: str(v).strip() not in ("", "0"), note="no forced close of the reasoning: budgets are ignored"),
}


def engine_caps(engine, family, grammar, force, vision, tool_tier=None):
    """{capability: bool} of a loaded engine (engine.Engine or an adapter) for the FLAGS / ENV / PARAMS rows. grammar /
    force / vision: the server's own view (grammar tier built, thinking close enforceable, vision encoder loaded)."""
    fns = getattr(engine, "fns", set())
    return {"penalty": "qe_set_penalty" in fns, "park": bool(getattr(engine, "can_park", lambda: False)()),
            "multi": bool(getattr(engine, "can_prefill_multi", lambda: False)()),
            "archive": engine.has("archive") and family != DSV4,  # (DSV4: archive_more is a no-op)
            "pause": "qe_pause_for" in fns, "grammar": bool(grammar), "force": bool(force),
            "logprobs": engine.has("logprobs"), "vision": bool(vision), "steering": "qe_set_steering" in fns,
            "tool_tier": bool(grammar if tool_tier is None else tool_tier)}  # (the tool-call mask tier loaded)


def _env_row(name):
    """(row, owner family or None) of an environment variable; (None, None): not a model-specific one."""
    row = ENV.get(name)
    if row is not None or name in ENV_SERVER:
        return row, None
    for prefix, fam in ENV_OWNERS:
        if name.startswith(prefix):
            return Knob({fam}, note=f"a {MODEL_NAMES[fam]} engine setting"), fam
    return None, None


def _ignored(row, family, caps):
    """True when the row has no effect: phase 1 (caps None) checks the family, phase 2 the engine's capabilities
    (rows whose family check passed or that have none)."""
    if caps is None:
        return row.on is not None and family not in row.on
    if row.on is not None and family not in row.on:
        return False  # (logged in phase 1)
    return row.cap is not None and not caps.get(row.cap, False)


def _why(row):
    """ (note), or the models the setting is for"""
    if row.note:
        return f" ({row.note})"
    if row.on:
        return " (a setting of " + " / ".join(MODEL_NAMES[f] for f in sorted(row.on, key=list(MODEL_NAMES).index)) + ")"
    return ""


def ignored_settings(family, given=None, env=None, caps=None, args=None):
    """Log lines for the settings that are set but have no effect for `family`. given: {argparse dest: value} of
    the flags on the command line; env: the environment at start (empty values = unset); caps: None = the start-up
    check by family, else {capability: bool} of the loaded engine (the second check)."""
    name = MODEL_NAMES.get(family, family)
    out = []
    for dest, v in sorted((given or {}).items()):
        row = FLAGS.get(dest)
        if row is None or (row.when is not None and not row.when(v, args)) or not _ignored(row, family, caps):
            continue
        flag = "--" + dest.replace("_", "-")
        shown = flag if isinstance(v, bool) else f"{flag} {v}"
        out.append(f"[settings] {shown} is set but has no effect for {name}" + _why(row))
    for k in sorted(env or {}):
        v = (env or {})[k]
        if v is None or not str(v).strip():
            continue
        row, _ = _env_row(k)
        if row is None or (row.when is not None and not row.when(v, args)) or not _ignored(row, family, caps):
            continue
        out.append(f"[settings] {k}={v} is set but has no effect for {name}" + _why(row))
    return out


# request parameters (OpenAI / Anthropic) that some engines do not support: (name, row, present(body), behaviour).
# Logged once per parameter per process when a request uses one; the behaviour is what the server does (documented in
# the README "Feature matrix").
def _has(body, *keys):
    return any(body.get(k) not in (None, 0, 0.0, False, {}, []) for k in keys)


def _tool_choice(body):
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        return "none" if tc.get("type") == "none" else "auto" if tc.get("type") == "auto" else "required"
    return tc if tc in ("none", "auto") else ("required" if tc in ("required", "any") else None)


def _budget(body):
    if _has(body, "thinking_budget", "thinking_token_budget", "max_thinking_tokens"):
        return True
    ck, cp, rs, th = (body.get(k) for k in ("chat_template_kwargs", "custom_params", "reasoning", "thinking"))
    return ((isinstance(ck, dict) and _has(ck, "thinking_budget")) or (isinstance(cp, dict) and _has(cp, "thinking_budget"))
            or (isinstance(rs, dict) and _has(rs, "max_tokens"))
            or (isinstance(th, dict) and th.get("type") == "enabled" and _has(th, "budget_tokens")))


def _images(body):
    def has(c):
        return isinstance(c, list) and any(isinstance(p, dict) and (p.get("type") in ("image", "image_url", "input_image")
                                                                    or (p.get("type") == "tool_result" and has(p.get("content"))))
                                           for p in c)
    return any(isinstance(m, dict) and has(m.get("content")) for m in body.get("messages") or ())


def _think_off(body):
    ck = body.get("chat_template_kwargs")
    rs = body.get("reasoning")
    th = body.get("thinking")
    def none(v):  # (effort none in any case, blanks ignored: app.template_kwargs)
        return isinstance(v, str) and v.strip().lower() == "none"
    return ((isinstance(ck, dict) and (ck.get("enable_thinking") is False or none(ck.get("reasoning_effort"))))
            # (issue #31: the top-level form, when chat_template_kwargs has none, as app.template_kwargs reads it)
            or (body.get("enable_thinking") is False
                and (ck if isinstance(ck, dict) else {}).get("enable_thinking") is None)
            or none(body.get("reasoning_effort")) or (isinstance(rs, dict) and (none(rs.get("effort"))
                                                                                  or rs.get("enabled") is False))
            or (isinstance(th, dict) and th.get("type") == "disabled"))


PARAMS = (
    ("presence_penalty", Knob(cap="penalty"), lambda b: _has(b, "presence_penalty"),
     "ignored (the sampler has no presence / frequency penalty)"),
    ("frequency_penalty", Knob(cap="penalty"), lambda b: _has(b, "frequency_penalty"),
     "ignored (the sampler has no presence / frequency penalty)"),
    ("logprobs", Knob(cap="logprobs"), lambda b: b.get("logprobs") is True, "HTTP 400 (no logits read-back)"),
    ("response_format", Knob(cap="grammar"),
     lambda b: any(isinstance(b.get(k), dict) and b[k].get("type") not in (None, "text")
                   for k in ("response_format", "output_format")) or isinstance((b.get("output_config") or {}).get("format"), dict),
     "HTTP 400 (no grammar masks: structured output is not available)"),
    ("tool_choice", Knob(cap="tool_tier"), lambda b: _tool_choice(b) == "required",
     "not enforced: the model decides as with auto"),
    ("tool_choice none", Knob(cap="tool_tier"), lambda b: _tool_choice(b) == "none",
     "the reply ends at its first tool call (no token masks)"),
    ("thinking budget", Knob(cap="force"), _budget, "ignored (no forced close of the reasoning)"),
    ("steering", Knob(cap="steering"), lambda b: b.get("steering") not in (None, False),
     "HTTP 400 for a named vector, ignored otherwise (no activation steering)"),
    ("images", Knob(cap="vision"), _images, "replaced by a text note (no image input on this server)"),
    ("thinking off", Knob(ALL - {GPTOSS}), _think_off, "gpt-oss always reasons: rendered at reasoning effort low"),
    ("n", Knob(frozenset()), lambda b: b.get("n") not in (None, 1), "ignored: one choice per request"),
    ("logit_bias", Knob(frozenset()), lambda b: _has(b, "logit_bias"), "ignored"),
    ("parallel_tool_calls", Knob(frozenset()), lambda b: b.get("parallel_tool_calls") is False,
     "ignored: the model may still call several tools in one reply"),
)


class ParamNotes:
    """Request parameters the loaded model does not support: one log line per parameter per process."""

    def __init__(self, family, caps, log=print):
        import threading
        self.name = MODEL_NAMES.get(family, family)
        self.rows = [(p, pred, how) for p, row, pred, how in PARAMS
                     if (row.on is not None and family not in row.on) or (row.cap is not None and not caps.get(row.cap, False))]
        self.seen, self.lock, self.log = set(), threading.Lock(), log

    def check(self, body):
        """Logs the unsupported parameters of this request not logged before; returns their names (all of them)."""
        hit = []
        for p, pred, how in self.rows:
            try:
                if not pred(body):
                    continue
            except Exception:  # (a malformed field: the request's own checks answer it)
                continue
            hit.append(p)
            with self.lock:
                if p in self.seen:
                    continue
                self.seen.add(p)
            self.log(f"[api] request parameter '{p}' is not supported for {self.name}: {how} (logged once)", flush=True)
        return hit
