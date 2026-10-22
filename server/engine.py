"""ctypes wrapper around libqengine (one engine per process, used from a single thread)."""
import ctypes
import os

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class Engine:
    def __init__(self, model_dir, devices="0,1,2", max_t=262144, split="", lib_path=None):
        self.lib = ctypes.CDLL(lib_path or os.environ.get("QE_LIB") or os.path.join(_ROOT, "build", "libqengine.so"))
        L = self.lib
        L.qe_create.restype = ctypes.c_void_p
        L.qe_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
        L.qe_start.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                               ctypes.c_int]
        L.qe_spec_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        L.qe_set_sampling.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
                                      ctypes.c_int, ctypes.c_float, ctypes.c_ulonglong]
        L.qe_last_reused.argtypes = [ctypes.c_void_p]
        L.qe_start2.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                ctypes.c_int, ctypes.c_ulonglong]
        L.qe_start3.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int]
        L.qe_start_mm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                  ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        for f in ("qe_last_slot", "qe_num_slots"):
            getattr(L, f).argtypes = [ctypes.c_void_p]
        L.qe_slot_capacity.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.qe_destroy.argtypes = [ctypes.c_void_p]
        L.qe_reset.argtypes = [ctypes.c_void_p]
        L.qe_release.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.qe_last_copied.argtypes = [ctypes.c_void_p]
        L.qe_warmup.restype = ctypes.c_double
        L.qe_warmup.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.qe_last_from_archive.argtypes = [ctypes.c_void_p]
        L.qe_num_archives.argtypes = [ctypes.c_void_p]
        L.qe_num_disk_archives.argtypes = [ctypes.c_void_p]
        L.qe_persist.argtypes = [ctypes.c_void_p]
        L.qe_resume.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.qe_set_concurrent.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.qe_step_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self._bout = (ctypes.c_int * (32 * 64))()  # kMaxSegs sequences per batched step
        self._bcnt = (ctypes.c_int * 32)()
        self.max_t = max_t
        self.h = L.qe_create(model_dir.encode(), devices.encode(), max_t, split.encode())
        self._buf = (ctypes.c_int * 16)()
        # KV pool: every slot can grow up to max_t (regions are assigned on demand)
        L.qe_pool_tokens.restype = ctypes.c_longlong
        L.qe_pool_tokens.argtypes = [ctypes.c_void_p]
        L.qe_max_context.argtypes = [ctypes.c_void_p]
        L.qe_can_admit.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.max_t = L.qe_max_context(self.h)
        self.pool_tokens = L.qe_pool_tokens(self.h)
        self.slots = [self.max_t] * L.qe_num_slots(self.h)

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
            emb, rope3, delta = mm
            emb = emb.contiguous()
            r3 = rope3.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            tok = self.lib.qe_start_mm(self.h, arr, len(ids), k, affinity & (2**64 - 1), int(hint),
                                       ctypes.c_void_p(emb.data_ptr()), emb.shape[0], r3, int(delta))
        else:
            tok = self.lib.qe_start3(self.h, arr, len(ids), k, affinity & (2**64 - 1), int(hint))
        if tok < 0:
            raise RuntimeError("KV pool full: no room for this prompt next to the running sequences")
        return tok, self.lib.qe_last_reused(self.h), self.lib.qe_last_slot(self.h)

    def start_begin(self, ids, k=3, affinity=0, hint=0, mm=None):
        """Prefill in steps: returns the slot (prefill_more until True, then start_finish)."""
        L = self.lib
        if not getattr(self, "_steps", False):
            L.qe_start_begin.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                         ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            L.qe_prefill_more.argtypes = [ctypes.c_void_p, ctypes.c_int]
            L.qe_start_finish.argtypes = [ctypes.c_void_p]
            L.qe_prefill_more_slot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            L.qe_start_finish_slot.argtypes = [ctypes.c_void_p, ctypes.c_int]
            L.qe_num_prefills.argtypes = [ctypes.c_void_p]
            L.qe_prefill_pos.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self._steps = True
        arr = (ctypes.c_int * len(ids))(*ids)
        if mm is not None:
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
        return slot

    def set_lookahead(self, on):
        self.lib.qe_set_lookahead.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.qe_set_lookahead(self.h, int(bool(on)))

    def prefill_more(self, budget, slot=None):
        """Next chunks (<= budget tokens) of the newest prefill, or of the given slot's prompt
        (up to 2 prompts may be in flight); True once that prompt is done."""
        if slot is None:
            return bool(self.lib.qe_prefill_more(self.h, int(budget)))
        return bool(self.lib.qe_prefill_more_slot(self.h, int(slot), int(budget)))

    def start_finish(self, slot=None):
        """First token; (tok, reused, slot) like start()."""
        tok = self.lib.qe_start_finish(self.h) if slot is None else self.lib.qe_start_finish_slot(self.h, int(slot))
        self._mm_keep = None
        return tok, self.lib.qe_last_reused(self.h), self.lib.qe_last_slot(self.h)

    def num_prefills(self):
        return self.lib.qe_num_prefills(self.h)

    def archive_more(self, budget=64 << 20):
        """Archive idle conversations to host memory, about `budget` bytes per GPU per call
        (call when nothing runs); True when nothing is left to archive."""
        if not getattr(self, "_bg", False):
            self.lib.qe_archive_more.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
            self._bg = True
        return bool(self.lib.qe_archive_more(self.h, int(budget)))

    def prefill_pos(self, slot):
        """Prompt rows of the slot prefilled so far (-1: no prefill of it in flight)."""
        return self.lib.qe_prefill_pos(self.h, int(slot))

    def step(self):
        """One decoding step; returns 1..k+1 new tokens."""
        n = self.lib.qe_spec_step(self.h, self._buf)
        return list(self._buf[:n])

    def warmup(self, greedy=False, sampling=True):
        """Pre-build the decode CUDA graphs of slot 0 (seconds spent)."""
        return self.lib.qe_warmup(self.h, int(greedy) | (int(sampling) << 1))

    def last_copied(self):
        """Prompt tokens of the last start() whose KV was copied from another slot."""
        return self.lib.qe_last_copied(self.h)

    def last_from_archive(self):
        return bool(self.lib.qe_last_from_archive(self.h))

    def num_archives(self):
        return self.lib.qe_num_archives(self.h)

    def num_disk_archives(self):
        return self.lib.qe_num_disk_archives(self.h)

    def persist(self):
        """Archive every live slot (memory + disk cache) and wait for the disk writes."""
        self.lib.qe_persist(self.h)

    def reset(self):
        """Forget every cached prefix (slots, snapshots, archives)."""
        self.lib.qe_reset(self.h)

    def set_concurrent(self, on=True):
        self.lib.qe_set_concurrent(self.h, int(on))

    def release(self, slot):
        self.lib.qe_release(self.h, slot)

    def resume(self, slot):
        """Make an active slot the current single sequence (for step())."""
        self.lib.qe_resume(self.h, slot)

    def pool_info(self):
        """Monitoring snapshot (call from the engine thread): slots, archives, pool events."""
        L, h = self.lib, self.h
        if not getattr(self, "_mon", False):
            L.qe_slot_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
            L.qe_archive_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
            L.qe_pool_counters.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong)]
            self._mon = True
        slots = []
        b = (ctypes.c_longlong * 6)()
        for i in range(len(self.slots)):
            L.qe_slot_info(h, i, b)
            slots.append({"off": b[0], "cap": b[1], "pos": b[2], "active": bool(b[3]), "used": b[4], "prio": b[5]})
        a = (ctypes.c_double * 6)()
        L.qe_archive_info(h, a)
        c = (ctypes.c_longlong * 3)()
        L.qe_pool_counters(h, c)
        return {"pool_tokens": self.pool_tokens, "max_context": self.max_t, "slots": slots,
                "archives": {"ram": int(a[0]), "ram_bytes": a[1], "disk": int(a[2]), "disk_bytes": a[3],
                             "pinned_bytes": a[4], "pinned_used": a[5]},
                "events": {"grow": c[0], "move": c[1], "evict": c[2]}}

    def can_admit(self, n):
        """A new sequence of n tokens fits next to the running ones (free slot + pool room)."""
        return bool(self.lib.qe_can_admit(self.h, int(n)))

    def set_prio(self, slot, prio):
        """Eviction value of the slot's conversation: 0 finished, 1 normal, 2 main thread."""
        if not getattr(self, "_prio", False):
            self.lib.qe_set_prio.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            self.lib.qe_probe_cached.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            self.lib.qe_num_pinned.argtypes = [ctypes.c_void_p]
            self._prio = True
        self.lib.qe_set_prio(self.h, int(slot), int(prio))

    def set_limit(self, slot, tokens):
        """The reply of the slot may emit at most `tokens` more tokens (max_tokens): its steps
        verify only the drafts that fit, so the slot ends exactly at the returned reply."""
        if not getattr(self, "_lim", False):
            self.lib.qe_set_limit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            self._lim = True
        self.lib.qe_set_limit(self.h, int(slot), int(tokens))

    def _stops_api(self):
        if not getattr(self, "_stp", False):
            L = self.lib
            for f in ("qe_watch_stops", "qe_set_no_ahead", "qe_trim"):
                getattr(L, f).argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            L.qe_drafts.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            self._dbuf = (ctypes.c_int * 64)()
            self._stp = True

    def watch_stops(self, slot, on=True):
        """The slot's reply has stop sequences: the engine keeps the drafts of its next step for
        drafts() and the undo of its last step for trim()."""
        self._stops_api()
        self.lib.qe_watch_stops(self.h, int(slot), int(on))

    def set_no_ahead(self, slot, on):
        """No step of the slot queued ahead of the one in flight (a stop may start in it)."""
        self._stops_api()
        self.lib.qe_set_no_ahead(self.h, int(slot), int(bool(on)))

    def drafts(self, slot):
        """The drafts the slot's step in flight / next step verifies (watch_stops)."""
        self._stops_api()
        n = self.lib.qe_drafts(self.h, int(slot), self._dbuf)
        return list(self._dbuf[:n])

    def trim(self, slot, pos):
        """End the slot's tokens at position pos (within its last step): True if done."""
        self._stops_api()
        return bool(self.lib.qe_trim(self.h, int(slot), int(pos)))

    def probe_cached(self, ids):
        """Prompt tokens a start() of ids would resume from now (no side effects)."""
        if not getattr(self, "_prio", False):
            self.set_prio(0, 1)
        arr = (ctypes.c_int * len(ids))(*ids)
        return self.lib.qe_probe_cached(self.h, arr, len(ids))

    def num_pinned(self):
        if not getattr(self, "_prio", False):
            self.set_prio(0, 1)
        return self.lib.qe_num_pinned(self.h)

    def step_batch(self, slots):
        """One speculative step for several active slots together; returns a token list each."""
        arr = (ctypes.c_int * len(slots))(*slots)
        self.lib.qe_step_batch(self.h, arr, len(slots), self._bout, self._bcnt)
        return [list(self._bout[i * 64:i * 64 + self._bcnt[i]]) for i in range(len(slots))]

    def vocab(self):
        self.lib.qe_vocab.argtypes = [ctypes.c_void_p]
        return self.lib.qe_vocab(self.h)

    def mask_buffer(self, slot):
        """Pinned host bitmasks of a slot as a numpy view [8][words] (uint32, bit t = token t
        allowed; row j: after the pending token and planned drafts 1..j), see spec_io."""
        import numpy as np
        L = self.lib
        if not getattr(self, "_mask", False):
            L.qe_mask_buffer.restype = ctypes.POINTER(ctypes.c_uint32)
            L.qe_mask_buffer.argtypes = [ctypes.c_void_p, ctypes.c_int]
            L.qe_constrain.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            self._mask = True
            self._masks = {}
        if slot not in self._masks:
            words = (self.vocab() + 31) // 32
            self._masks[slot] = np.ctypeslib.as_array(L.qe_mask_buffer(self.h, int(slot)), shape=(8, words))
        return self._masks[slot]

    def spec_io(self, slot):
        """Pinned int32[32] of a constrained slot: [0] drafts the model made, [1..] the drafts;
        [16] drafts to verify next step, [17..] those tokens (written by the caller)."""
        import numpy as np
        L = self.lib
        if not getattr(self, "_sio", None):
            L.qe_spec_io.restype = ctypes.POINTER(ctypes.c_int32)
            L.qe_spec_io.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self._sio = {}
        if slot not in self._sio:
            self._sio[slot] = np.ctypeslib.as_array(L.qe_spec_io(self.h, int(slot)), shape=(32,))
        return self._sio[slot]

    def constrain(self, slot, on=True):
        """Grammar-constrained decoding of a slot (after start_begin, mask_buffer(slot)[0] filled
        first): never queued ahead; every step verifies the drafts planned in spec_io(slot)
        with the per-row masks of mask_buffer(slot)."""
        self.mask_buffer(slot)
        self.lib.qe_constrain(self.h, int(slot), int(on))

    def constrained(self, slot):
        self.lib.qe_constrained.argtypes = [ctypes.c_void_p, ctypes.c_int]
        return bool(self.lib.qe_constrained(self.h, int(slot)))

    def close(self):
        if self.h:
            self.lib.qe_destroy(self.h)
            self.h = None
