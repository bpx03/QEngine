"""ctypes wrapper around an engine library (one engine per process, used from a single thread).

The same wrapper serves libqengine (Qwen3.8-27B, every function below) and engines that export
a subset of the C API (Qwen3.8-Flash-Next while it is being ported): REQUIRED must exist, each
OPTIONAL group is used when all its functions exist and falls back to single-sequence behaviour
otherwise (contract: server/qe_capi.h)."""
import ctypes
import os

from steering import SteeringMixin

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

c_void_p, c_int, c_char_p, c_float = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_float
c_u64, c_i64, c_double, P_int = ctypes.c_ulonglong, ctypes.c_longlong, ctypes.c_double, ctypes.POINTER(ctypes.c_int)
P_i64, P_double = ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_double)

# name: (restype, argtypes)
SIGS = {
    # ---- required
    "qe_create": (c_void_p, [c_char_p, c_char_p, c_int, c_char_p]),
    "qe_destroy": (None, [c_void_p]),
    "qe_max_context": (c_int, [c_void_p]),
    "qe_set_sampling": (None, [c_void_p, c_float, c_float, c_int, c_float, c_u64]),
    "qe_start3": (c_int, [c_void_p, P_int, c_int, c_int, c_u64, c_int]),
    "qe_spec_step": (c_int, [c_void_p, P_int]),
    "qe_last_reused": (c_int, [c_void_p]),
    "qe_last_slot": (c_int, [c_void_p]),
    "qe_release": (None, [c_void_p, c_int]),
    # ---- optional
    "qe_vocab": (c_int, [c_void_p]),
    "qe_num_slots": (c_int, [c_void_p]),
    "qe_pool_tokens": (c_i64, [c_void_p]),
    "qe_can_admit": (c_int, [c_void_p, c_int]),
    "qe_set_limit": (None, [c_void_p, c_int, c_int]),
    "qe_warmup": (c_double, [c_void_p, c_int]),
    "qe_reset": (None, [c_void_p]),
    "qe_start_begin": (c_int, [c_void_p, P_int, c_int, c_int, c_u64, c_int, c_void_p, c_int, P_int, c_int]),
    "qe_prefill_more": (c_int, [c_void_p, c_int]),
    "qe_start_finish": (c_int, [c_void_p]),
    "qe_prefill_more_slot": (c_int, [c_void_p, c_int, c_int]),
    "qe_start_finish_slot": (c_int, [c_void_p, c_int]),
    "qe_num_prefills": (c_int, [c_void_p]),
    "qe_prefill_pos": (c_int, [c_void_p, c_int]),
    "qe_set_lookahead": (None, [c_void_p, c_int]),
    "qe_set_concurrent": (None, [c_void_p, c_int]),
    "qe_resume": (None, [c_void_p, c_int]),
    "qe_step_batch": (None, [c_void_p, P_int, c_int, P_int, P_int]),
    "qe_probe_cached": (c_int, [c_void_p, P_int, c_int]),
    "qe_set_prio": (None, [c_void_p, c_int, c_int]),
    "qe_num_pinned": (c_int, [c_void_p]),
    "qe_archive_more": (c_int, [c_void_p, c_i64]),
    "qe_persist": (None, [c_void_p]),
    "qe_num_archives": (c_int, [c_void_p]),
    "qe_num_disk_archives": (c_int, [c_void_p]),
    "qe_last_copied": (c_int, [c_void_p]),
    "qe_last_from_archive": (c_int, [c_void_p]),
    "qe_slot_info": (None, [c_void_p, c_int, P_i64]),
    "qe_archive_info": (None, [c_void_p, P_double]),
    "qe_pool_counters": (None, [c_void_p, P_i64]),
    "qe_watch_stops": (None, [c_void_p, c_int, c_int]),
    "qe_set_no_ahead": (None, [c_void_p, c_int, c_int]),
    "qe_drafts": (c_int, [c_void_p, c_int, P_int]),
    "qe_trim": (c_int, [c_void_p, c_int, c_int]),
    "qe_mask_buffer": (ctypes.POINTER(ctypes.c_uint32), [c_void_p, c_int]),
    "qe_spec_io": (ctypes.POINTER(ctypes.c_int32), [c_void_p, c_int]),
    "qe_constrain": (None, [c_void_p, c_int, c_int]),
    "qe_constrained": (c_int, [c_void_p, c_int]),
    "qe_start_mm": (c_int, [c_void_p, P_int, c_int, c_int, c_u64, c_int, c_void_p, c_int, P_int, c_int]),
    "qe_lookup_corpus": (None, [c_void_p, P_int, c_int]),
    "qe_set_steering": (c_int, [c_void_p, c_int, c_float]),
    "qe_set_steering_ex": (c_int, [c_void_p, c_int, c_float, c_int, c_int, c_int]),
    "qe_steer_info": (c_char_p, [c_void_p]),
    "qe_capture": (c_int, [c_void_p, P_int, c_int, c_int, ctypes.POINTER(c_float)]),
}
REQUIRED = ("qe_create", "qe_destroy", "qe_max_context", "qe_set_sampling", "qe_start3", "qe_spec_step",
            "qe_last_reused", "qe_last_slot", "qe_release")
# feature -> functions that must all exist for it
GROUPS = {
    "chunked": ("qe_start_begin", "qe_prefill_more", "qe_start_finish", "qe_prefill_more_slot",
                "qe_start_finish_slot", "qe_num_prefills", "qe_prefill_pos"),
    "batch": ("qe_step_batch", "qe_resume", "qe_set_concurrent"),
    "cache": ("qe_probe_cached", "qe_set_prio", "qe_num_pinned", "qe_last_copied", "qe_last_from_archive"),
    "archive": ("qe_archive_more", "qe_persist", "qe_num_archives", "qe_num_disk_archives"),
    "monitor": ("qe_slot_info", "qe_archive_info", "qe_pool_counters"),
    "stops": ("qe_watch_stops", "qe_set_no_ahead", "qe_drafts", "qe_trim"),
    "grammar": ("qe_mask_buffer", "qe_spec_io", "qe_constrain", "qe_constrained", "qe_vocab"),
    "vision": ("qe_start_mm", "qe_start_begin"),
}
SINGLE = ("qe_vocab", "qe_num_slots", "qe_pool_tokens", "qe_can_admit", "qe_set_limit", "qe_warmup",
          "qe_reset", "qe_set_lookahead")


def bind(lib):
    """Set the signatures of the functions the library has; returns the set of their names."""
    have = set()
    for name, (res, args) in SIGS.items():
        try:
            f = getattr(lib, name)
        except AttributeError:
            continue
        f.restype, f.argtypes = res, args
        have.add(name)
    return have


class Engine(SteeringMixin):  # + steer_info / set_steering / capture (server/steering.py)
    def __init__(self, model_dir, devices="0,1,2", max_t=262144, split="", lib_path=None, vocab=None):
        self.lib_path = lib_path or os.environ.get("QE_LIB") or os.path.join(_ROOT, "build", "libqengine.so")
        self.lib = ctypes.CDLL(self.lib_path)
        self._init(model_dir, devices, max_t, split, vocab)

    def _init(self, model_dir, devices, max_t, split, vocab):
        L = self.lib
        self.fns = bind(L)
        miss = [f for f in REQUIRED if f not in self.fns]
        if miss:
            raise RuntimeError(f"{self.lib_path}: required engine functions missing: {', '.join(miss)}")
        self.caps = {g: all(f in self.fns for f in fs) for g, fs in GROUPS.items()}
        self._vocab = vocab
        self._bout = (ctypes.c_int * (32 * 64))()  # kMaxSegs sequences per batched step
        self._bcnt = (ctypes.c_int * 32)()
        self._buf = (ctypes.c_int * 64)()
        self._dbuf = (ctypes.c_int * 64)()
        self._pending = {}   # emulated start_begin (no chunked API): slot -> start arguments / first token
        self._active = set()  # slots started and not released (can_admit emulation)
        self._masks, self._sio = {}, {}
        self.h = L.qe_create(model_dir.encode(), devices.encode(), max_t, (split or "").encode())
        if not self.h:
            raise RuntimeError(f"qe_create failed for {model_dir}")
        self.max_t = L.qe_max_context(self.h)
        self.pool_tokens = L.qe_pool_tokens(self.h) if "qe_pool_tokens" in self.fns else self.max_t
        n = L.qe_num_slots(self.h) if "qe_num_slots" in self.fns else 1
        if n > 1 and not (self.caps["chunked"] and self.caps["batch"]):
            raise RuntimeError(f"{self.lib_path}: {n} slots need the chunked-prefill and batch functions")
        # KV pool: every slot can grow up to max_t (regions are assigned on demand)
        self.slots = [self.max_t] * n

    # ------------------------------------------------------------------ capabilities
    def has(self, feature):
        """Feature group (GROUPS) or single function name available."""
        return self.caps.get(feature, feature in self.fns)

    def missing_optional(self):
        return [g for g, ok in self.caps.items() if not ok] + [f for f in SINGLE if f not in self.fns]

    # activation steering: steer_info / set_steering / capture come from server/steering.py (SteeringMixin)

    # ------------------------------------------------------------------ sequences
    def set_sampling(self, temperature, top_p=1.0, top_k=40, min_p=0.0, seed=0):
        self.lib.qe_set_sampling(self.h, float(temperature), float(top_p), int(top_k),
                                 float(min_p), int(seed))

    def start(self, ids, k=3, affinity=0, hint=0, mm=None):
        """Prefill (reusing cached prefixes) in the best sequence slot; affinity is a 64-bit
        conversation key (e.g. system-prompt hash). mm = (image embeddings [rows, hidden] bf16
        tensor, M-RoPE positions int32 [n, 3], position shift) for prompts with image rows.
        Returns (first token, reused, slot)."""
        arr = (ctypes.c_int * len(ids))(*ids)
        if mm is not None:
            if not self.caps["vision"]:
                raise RuntimeError("this engine has no image input (qe_start_mm)")
            emb, rope3, delta = mm
            emb = emb.contiguous()
            r3 = rope3.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            tok = self.lib.qe_start_mm(self.h, arr, len(ids), k, affinity & (2**64 - 1), int(hint),
                                       ctypes.c_void_p(emb.data_ptr()), emb.shape[0], r3, int(delta))
        else:
            tok = self.lib.qe_start3(self.h, arr, len(ids), k, affinity & (2**64 - 1), int(hint))
        if tok < 0:
            raise RuntimeError("KV pool full: no room for this prompt next to the running sequences")
        slot = self.lib.qe_last_slot(self.h)
        self._active.add(slot)
        return tok, self.lib.qe_last_reused(self.h), slot

    def start_begin(self, ids, k=3, affinity=0, hint=0, mm=None):
        """Prefill in steps: returns the slot (prefill_more until True, then start_finish).
        Without the chunked API (one slot): the whole prompt is prefilled by the first
        prefill_more (qe_start3), after the caller's set_sampling."""
        L = self.lib
        if not self.caps["chunked"]:
            if mm is not None:
                raise RuntimeError("this engine has no image input (qe_start_mm)")
            self._pending[0] = {"args": (list(ids), k, affinity, hint), "tok": None}
            return 0
        arr = (ctypes.c_int * len(ids))(*ids)
        if mm is not None:
            if not self.caps["vision"]:
                raise RuntimeError("this engine has no image input (qe_start_mm)")
            emb, rope3, delta = mm
            emb = emb.contiguous()
            self._mm_keep = (emb, rope3)  # alive until the prefill is done
            slot = L.qe_start_begin(self.h, arr, len(ids), k, affinity & (2**64 - 1), int(hint),
                                    ctypes.c_void_p(emb.data_ptr()), emb.shape[0],
                                    rope3.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), int(delta))
        else:
            slot = L.qe_start_begin(self.h, arr, len(ids), k, affinity & (2**64 - 1), int(hint),
                                    None, 0, None, 0)
        if slot < 0:
            raise RuntimeError("KV pool full: no room for this prompt next to the running sequences")
        self._active.add(slot)
        return slot

    def add_corpus(self, ids):
        """A finished reply (token ids) into the engine's global lookup corpus (drafts across requests)."""
        if "qe_lookup_corpus" in self.fns and len(ids):
            arr = (ctypes.c_int * len(ids))(*ids)
            self.lib.qe_lookup_corpus(self.h, arr, len(ids))

    def set_lookahead(self, on):
        if "qe_set_lookahead" in self.fns:
            self.lib.qe_set_lookahead(self.h, int(bool(on)))

    def prefill_more(self, budget, slot=None):
        """Next chunks (<= budget tokens) of the newest prefill, or of the given slot's prompt
        (up to 2 prompts may be in flight); True once that prompt is done."""
        if not self.caps["chunked"]:
            p = self._pending.get(0 if slot is None else slot)
            if p is not None and p["tok"] is None:
                p["tok"] = self.start(*p["args"])
            return True
        if slot is None:
            return bool(self.lib.qe_prefill_more(self.h, int(budget)))
        return bool(self.lib.qe_prefill_more_slot(self.h, int(slot), int(budget)))

    def start_finish(self, slot=None):
        """First token; (tok, reused, slot) like start()."""
        if not self.caps["chunked"]:
            p = self._pending.pop(0 if slot is None else slot)
            return p["tok"] if p["tok"] is not None else self.start(*p["args"])
        tok = self.lib.qe_start_finish(self.h) if slot is None else self.lib.qe_start_finish_slot(self.h, int(slot))
        self._mm_keep = None
        return tok, self.lib.qe_last_reused(self.h), self.lib.qe_last_slot(self.h)

    def num_prefills(self):
        return self.lib.qe_num_prefills(self.h) if self.caps["chunked"] else len(self._pending)

    def prefill_pos(self, slot):
        """Prompt rows of the slot prefilled so far (-1: no prefill of it in flight)."""
        if not self.caps["chunked"]:
            return 0 if slot in self._pending else -1
        return self.lib.qe_prefill_pos(self.h, int(slot))

    def step(self):
        """One decoding step; returns 1..k+1 new tokens."""
        n = self.lib.qe_spec_step(self.h, self._buf)
        return list(self._buf[:n])

    def step_batch(self, slots):
        """One speculative step for several active slots together; returns a token list each."""
        if not self.caps["batch"]:
            raise RuntimeError("this engine runs one sequence at a time (no qe_step_batch)")
        arr = (ctypes.c_int * len(slots))(*slots)
        self.lib.qe_step_batch(self.h, arr, len(slots), self._bout, self._bcnt)
        return [list(self._bout[i * 64:i * 64 + self._bcnt[i]]) for i in range(len(slots))]

    def release(self, slot):
        self._active.discard(slot)
        self._pending.pop(slot, None)
        self.lib.qe_release(self.h, slot)

    def resume(self, slot):
        """Make an active slot the current single sequence (for step())."""
        if "qe_resume" in self.fns:
            self.lib.qe_resume(self.h, slot)

    def set_concurrent(self, on=True):
        if "qe_set_concurrent" in self.fns:
            self.lib.qe_set_concurrent(self.h, int(on))

    def can_admit(self, n):
        """A new sequence of n tokens fits next to the running ones (free slot + pool room)."""
        if "qe_can_admit" in self.fns:
            return bool(self.lib.qe_can_admit(self.h, int(n)))
        return len(self._active) < len(self.slots) and n <= self.max_t

    def set_limit(self, slot, tokens):
        """The reply of the slot may emit at most `tokens` more tokens (max_tokens): its steps
        verify only the drafts that fit, so the slot ends exactly at the returned reply."""
        if "qe_set_limit" in self.fns:
            self.lib.qe_set_limit(self.h, int(slot), int(tokens))

    def warmup(self, greedy=False, sampling=True):
        """Pre-build the decode CUDA graphs of slot 0 (seconds spent)."""
        if "qe_warmup" not in self.fns:
            return 0.0
        return self.lib.qe_warmup(self.h, int(greedy) | (int(sampling) << 1))

    def reset(self):
        """Forget every cached prefix (slots, snapshots, archives)."""
        if "qe_reset" in self.fns:
            self.lib.qe_reset(self.h)

    def vocab(self):
        if "qe_vocab" in self.fns:
            return self.lib.qe_vocab(self.h)
        return self._vocab

    # ------------------------------------------------------------------ prompt cache / archive
    def last_copied(self):
        """Prompt tokens of the last start() whose KV was copied from another slot."""
        return self.lib.qe_last_copied(self.h) if "qe_last_copied" in self.fns else 0

    def last_from_archive(self):
        return bool(self.lib.qe_last_from_archive(self.h)) if "qe_last_from_archive" in self.fns else False

    def num_archives(self):
        return self.lib.qe_num_archives(self.h) if "qe_num_archives" in self.fns else 0

    def num_disk_archives(self):
        return self.lib.qe_num_disk_archives(self.h) if "qe_num_disk_archives" in self.fns else 0

    def persist(self):
        """Archive every live slot (memory + disk cache) and wait for the disk writes."""
        if "qe_persist" in self.fns:
            self.lib.qe_persist(self.h)

    def archive_more(self, budget=64 << 20):
        """Archive idle conversations to host memory, about `budget` bytes per GPU per call
        (call when nothing runs); True when nothing is left to archive."""
        if "qe_archive_more" not in self.fns:
            return True
        return bool(self.lib.qe_archive_more(self.h, int(budget)))

    def set_prio(self, slot, prio):
        """Eviction value of the slot's conversation: 0 finished, 1 normal, 2 main thread."""
        if "qe_set_prio" in self.fns:
            self.lib.qe_set_prio(self.h, int(slot), int(prio))

    def probe_cached(self, ids):
        """Prompt tokens a start() of ids would resume from now (no side effects)."""
        if "qe_probe_cached" not in self.fns:
            return 0
        arr = (ctypes.c_int * len(ids))(*ids)
        return self.lib.qe_probe_cached(self.h, arr, len(ids))

    def num_pinned(self):
        return self.lib.qe_num_pinned(self.h) if "qe_num_pinned" in self.fns else 0

    def pool_info(self):
        """Monitoring snapshot (call from the engine thread): slots, archives, pool events."""
        L, h = self.lib, self.h
        slots = []
        b = (ctypes.c_longlong * 6)()
        for i in range(len(self.slots)):
            if "qe_slot_info" in self.fns:
                L.qe_slot_info(h, i, b)
                slots.append({"off": b[0], "cap": b[1], "pos": b[2], "active": bool(b[3]), "used": b[4], "prio": b[5]})
            else:
                slots.append({"off": 0, "cap": self.max_t, "pos": 0, "active": i in self._active, "used": 0, "prio": 1})
        a = (ctypes.c_double * 6)()
        if "qe_archive_info" in self.fns:
            L.qe_archive_info(h, a)
        c = (ctypes.c_longlong * 3)()
        if "qe_pool_counters" in self.fns:
            L.qe_pool_counters(h, c)
        return {"pool_tokens": self.pool_tokens, "max_context": self.max_t, "slots": slots,
                "archives": {"ram": int(a[0]), "ram_bytes": a[1], "disk": int(a[2]), "disk_bytes": a[3],
                             "pinned_bytes": a[4], "pinned_used": a[5]},
                "events": {"grow": c[0], "move": c[1], "evict": c[2]}}

    # ------------------------------------------------------------------ stop sequences
    def watch_stops(self, slot, on=True):
        """The slot's reply has stop sequences: the engine keeps the drafts of its next step for
        drafts() and the undo of its last step for trim()."""
        if self.caps["stops"]:
            self.lib.qe_watch_stops(self.h, int(slot), int(on))

    def set_no_ahead(self, slot, on):
        """No step of the slot queued ahead of the one in flight (a stop may start in it)."""
        if self.caps["stops"]:
            self.lib.qe_set_no_ahead(self.h, int(slot), int(bool(on)))

    def drafts(self, slot):
        """The drafts the slot's step in flight / next step verifies (watch_stops)."""
        if not self.caps["stops"]:
            return []
        n = self.lib.qe_drafts(self.h, int(slot), self._dbuf)
        return list(self._dbuf[:n])

    def trim(self, slot, pos):
        """End the slot's tokens at position pos (within its last step): True if done."""
        if not self.caps["stops"]:
            return False
        return bool(self.lib.qe_trim(self.h, int(slot), int(pos)))

    # ------------------------------------------------------------------ structured output
    def mask_buffer(self, slot):
        """Pinned host bitmasks of a slot as a numpy view [8][words] (uint32, bit t = token t
        allowed; row j: after the pending token and planned drafts 1..j), see spec_io."""
        import numpy as np
        if slot not in self._masks:
            words = (self.vocab() + 31) // 32
            self._masks[slot] = np.ctypeslib.as_array(self.lib.qe_mask_buffer(self.h, int(slot)), shape=(8, words))
        return self._masks[slot]

    def spec_io(self, slot):
        """Pinned int32[32] of a constrained slot: [0] drafts the model made, [1..] the drafts;
        [16] drafts to verify next step, [17..] those tokens (written by the caller)."""
        import numpy as np
        if slot not in self._sio:
            self._sio[slot] = np.ctypeslib.as_array(self.lib.qe_spec_io(self.h, int(slot)), shape=(32,))
        return self._sio[slot]

    def constrain(self, slot, on=True):
        """Grammar-constrained decoding of a slot (after start_begin, mask_buffer(slot)[0] filled
        first): never queued ahead; every step verifies the drafts planned in spec_io(slot)
        with the per-row masks of mask_buffer(slot)."""
        self.mask_buffer(slot)
        self.lib.qe_constrain(self.h, int(slot), int(on))

    def constrained(self, slot):
        return bool(self.lib.qe_constrained(self.h, int(slot)))

    def close(self):
        if self.h:
            self.lib.qe_destroy(self.h)
            self.h = None
