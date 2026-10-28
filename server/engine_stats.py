"""Per-request engine statistics line in the formats benchmark log parsers know (both engines).

At the end of every request the server prints, unless QE_STATS_LOG=0:

  [stats] SpecDecoding metrics: Mean acceptance length: 3.21, Accepted throughput: 99.80 tokens/s, Drafted throughput:
          131.20 tokens/s, Accepted: 482 tokens, Drafted: 634 tokens, Avg Draft acceptance rate: 76.0%
          (MTP 401/517, lookup 81/117, 150 steps)
  [stats] prompt eval time =   412.31 ms /  8192 tokens (    0.05 ms per token, 19868.31 tokens per second)
  [stats]        eval time =  4829.10 ms /   632 tokens (    7.64 ms per token,   130.87 tokens per second)

(the SpecDecoding line is one line). The draft counters come from the engine (qe_draft_stats: steps, drafted and
accepted drafts of MTP and lookup per slot); "Mean acceptance length" is tokens per verify step (accepted drafts + 1).
The timing lines use the prefill time and the prompt tokens actually computed (cached tokens excluded), and the decode
time and generated tokens. Without drafts (no MTP head, or an engine library without qe_draft_stats) the SpecDecoding
line is left out.
"""
import ctypes
import os

ENABLED = os.environ.get("QE_STATS_LOG", "1") != "0"


class DraftStats:
    """Reads the engine's per-slot draft counters (None when the library does not have them)."""

    def __init__(self, engine):
        self.fn = None
        self.h = getattr(engine, "h", None)
        try:
            f = engine.lib.qe_draft_stats
        except AttributeError:
            return
        f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
        f.restype = None
        self.fn = f
        self.buf = (ctypes.c_longlong * 5)()

    def read(self, slot):
        """[steps, MTP drafted, MTP accepted, lookup drafted, lookup accepted] of the slot, or None."""
        if self.fn is None or slot is None or slot < 0:
            return None
        try:
            self.fn(self.h, int(slot), self.buf)
        except Exception:
            return None
        return list(self.buf)


def delta(a, b):
    """Counters over a request: b - a (a slot reset in between: b alone)."""
    if b is None:
        return None
    if a is None or any(y < x for x, y in zip(a, b)):
        return b
    return [y - x for x, y in zip(a, b)]


def lines(st, ds=None):
    """The stats lines of a finished request: st has prompt_tokens, cached_tokens, prefill_s, completion_tokens,
    decode_s; ds the draft counter delta (or None)."""
    out = []
    dec_s = max(st["decode_s"], 1e-9)
    gen = st["completion_tokens"]
    if ds and ds[0] > 0 and (ds[1] + ds[3]) > 0:
        steps, drafted, accepted = ds[0], ds[1] + ds[3], ds[2] + ds[4]
        out.append(f"[stats] SpecDecoding metrics: Mean acceptance length: {(accepted + steps) / steps:.2f}, "
                   f"Accepted throughput: {accepted / dec_s:.2f} tokens/s, "
                   f"Drafted throughput: {drafted / dec_s:.2f} tokens/s, "
                   f"Accepted: {accepted} tokens, Drafted: {drafted} tokens, "
                   f"Avg Draft acceptance rate: {100.0 * accepted / drafted:.1f}% "
                   f"(MTP {ds[2]}/{ds[1]}, lookup {ds[4]}/{ds[3]}, {steps} steps)")
    n_pf = st["prompt_tokens"] - st["cached_tokens"]
    pf_ms = 1000.0 * st["prefill_s"]
    if n_pf > 0 and pf_ms > 0:
        out.append(f"[stats] prompt eval time = {pf_ms:10.2f} ms / {n_pf:5d} tokens ({pf_ms / n_pf:8.2f} ms per token, "
                   f"{1000.0 * n_pf / pf_ms:8.2f} tokens per second)")
    if gen > 0:
        ms = 1000.0 * dec_s
        out.append(f"[stats]        eval time = {ms:10.2f} ms / {gen:5d} tokens ({ms / gen:8.2f} ms per token, "
                   f"{1000.0 * gen / ms:8.2f} tokens per second)")
    return out


def log(st, ds=None):
    if not ENABLED:
        return
    for ln in lines(st, ds):
        print(ln, flush=True)
