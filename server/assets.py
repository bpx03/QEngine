"""27B release assets on first start: the fine-tuned MTP head and the draft vocabulary ranking.

model_mtp_ft.safetensors and token_rank.bin are GitHub release assets of QEngine, not part of the RedHatAI
checkpoint nor of the Docker image. Without them the engine runs the checkpoint's own MTP head over the full
vocabulary (~8-9 % fewer accepted tokens). When the model dir lacks them (and no --mtp-file / QE_MTP_FILE is
given), the missing ones are fetched from the latest release into ~/.cache/qengine/assets (the Docker volume
qengine-cache), checked against the release's SHA256SUMS, renamed into place atomically and handed to the engine
through QE_MTP_FILE / QE_TOKEN_RANK_FILE. The model dir is never written (Docker mounts it read-only). Valid cached
files are reused without network. Any failure only logs a warning: the engine starts with the stock head.
INT8 checkpoints (quantization_config num_bits 8) keep their own MTP head: the fine-tuned one is neither fetched nor
set unless QE_INT8_MTP_FT=1 (use_mtp_ft).
QE_FETCH_ASSETS=0 turns the download off; QE_ASSETS_URL / QE_ASSETS_DIR override the source / cache dir.
Dependency-free (urllib, hashlib).
"""
import hashlib
import json
import os
import urllib.request

from qe_env import env_int, env_str

DEFAULT_URL = "https://github.com/bpx03/QEngine/releases/latest/download"
DEFAULT_DIR = "~/.cache/qengine/assets"
MTP_FT = "model_mtp_ft.safetensors"
TOKEN_RANK = "token_rank.bin"
TIMEOUT = 30  # seconds per socket operation (connect / each read), not for the whole file


def _log(msg):
    print(f"[server] assets: {msg}", flush=True)


def weight_bits(model_dir):
    """Bits of the checkpoint's quantized layer weights (compressed-tensors quantization_config), None if unknown."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            q = json.load(f).get("quantization_config") or {}
        bits = [g.get("weights", {}).get("num_bits") for g in (q.get("config_groups") or {}).values()]
        bits = [b for b in bits if isinstance(b, int)]
        return max(bits) if bits else None
    except (OSError, ValueError, AttributeError):
        return None


def use_mtp_ft(model_dir):
    """Whether the fine-tuned MTP head (model_mtp_ft.safetensors) applies to this checkpoint. It was tuned on the
    hidden states of the INT4 model; an INT8 checkpoint runs its own (stock BF16) head
    unless QE_INT8_MTP_FT=1. An explicit --mtp-file / QE_MTP_FILE always wins (callers check it first)."""
    if weight_bits(model_dir) == 8 and env_int("QE_INT8_MTP_FT", 0) == 0:
        return False
    return True


def _refuse_format(what):
    return (f"{what}. The 27B engine reads one model.safetensors in the compressed-tensors pack-quantized layout: INT4 or "
            "symmetric INT8, group 128, weight_packed int32 [N][K/8] or [N][K/4] with the stored value w + offset, "
            "weight_scale bf16 [N][K/128]. tools/quant_int8.py builds an INT8 one from the BF16 weights.")


def check_27b_format(model_dir, env=None):
    """Header check of a 27B checkpoint before the engine maps it: what the loader would misread stops the start here.
    Returns the reason (None: fine). A checkpoint without FP8 KV scales runs with a bf16 KV cache: the position split of
    the KV heads needs FP8, so this also sets QE_KV_FP8=0 for it."""
    env = os.environ if env is None else env
    path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(path):
        if any(f.endswith(".safetensors") and f.startswith("model-") or f == "model.safetensors.index.json"
               for f in os.listdir(model_dir)):
            return _refuse_format(f"{model_dir}: sharded checkpoint, no model.safetensors")
        return None  # (no weights at all: the engine's own error; tokenizer-only directories of the mock tests)
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n > os.path.getsize(path) - 8:
                return None  # (not a safetensors file: the engine says so)
            h = json.loads(f.read(n))
            data = 8 + n
            q8 = None
            for k, p in h.items():
                if not k.endswith(".weight_packed"):
                    continue
                b = k[: -len(".weight_packed")]
                if b + ".weight_zero_point" in h:
                    return _refuse_format(f"{b}: asymmetric weights (weight_zero_point)")
                s, sh = h.get(b + ".weight_scale"), h.get(b + ".weight_shape")
                if p["dtype"] != "I32" or len(p["shape"]) != 2:
                    return _refuse_format(f"{b}: weight_packed is {p['dtype']} {p['shape']}")
                if not s or s["dtype"] != "BF16" or len(s["shape"]) != 2:
                    return _refuse_format(f"{b}: weight_scale is {s and (s['dtype'], s['shape'])}")
                K = s["shape"][1] * 128
                if sh and sh["dtype"] == "I64":
                    f.seek(data + sh["data_offsets"][0])
                    Kt = int.from_bytes(f.read(16)[8:], "little")
                    if Kt != K:
                        return _refuse_format(f"{b}: group size {Kt / s['shape'][1]:g}, not 128")
                if p["shape"][1] * 32 not in (K * 4, K * 8):
                    return _refuse_format(f"{b}: packed width {p['shape'][1]} for {K} columns")
                if p["shape"][1] * 32 == K * 8 and q8 is None:
                    q8 = p
            if q8:  # stored value w + 128: most bytes sit around 128 (raw two's complement would sit around 0 / 255)
                f.seek(data + q8["data_offsets"][0])
                u = f.read(65536)
                if sum(64 <= c < 192 for c in u) * 2 < len(u):
                    return _refuse_format("INT8 weights are not stored as w + 128 (signed bytes?)")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as ex:
        _log(f"format check skipped: {ex}")
        return None
    if (not any(k.endswith("k_scale") for k in h) and not (env.get("QE_KV_SCALES") or "").strip()
            and not (env.get("QE_KV_FP8") or "").strip()):
        env["QE_KV_FP8"] = "0"
        _log("no FP8 KV scales in the checkpoint: bf16 KV cache, half the context pool, no KV position split "
             "(QE_KV_SCALES=<a 27B INT4 model.safetensors> uses its scales)")
    return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _mb(n):
    return f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def _get(url, dst, name, timeout):
    """Download url to dst (a temporary path) with a progress line every ~10 %; returns the byte count."""
    req = urllib.request.Request(url, headers={"User-Agent": "qengine-assets"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, step = 0, 1
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if total >= 10 << 20 and got * 10 >= step * total and got < total:
                _log(f"{name}: {got * 100 // total} % of {_mb(total)}")
                step = got * 10 // total + 1
    if total and got != total:
        raise IOError(f"{name}: got {got} of {total} bytes")
    return got


def _sums(base, cache, timeout):
    """The release's SHA256SUMS as {file: sha256}, fetched into the cache dir (atomic)."""
    path = os.path.join(cache, "SHA256SUMS")
    tmp = path + f".part{os.getpid()}"
    try:
        _get(f"{base}/SHA256SUMS", tmp, "SHA256SUMS", timeout)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    out = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                out[p[1].lstrip("*")] = p[0].lower()
    return out


def _warn(missing, model_dir, why):
    what = [w for n, w in ((MTP_FT, "the stock MTP head"), (TOKEN_RANK, "the full draft vocabulary")) if n in missing]
    _log(f"warning: {', '.join(missing)} not available ({why}): the engine runs with {' and '.join(what)} "
         f"(slower). Download manually:")
    _log(f"  for f in {' '.join(missing)} SHA256SUMS; do curl -LO {DEFAULT_URL}/$f; done")
    _log(f"  sha256sum -c --ignore-missing SHA256SUMS   # then put the files into {model_dir} "
         f"(or {DEFAULT_DIR})")


def ensure(model_dir, env=None, cache=None, base=None, timeout=TIMEOUT):
    """Find or fetch the missing 27B assets; sets QE_MTP_FILE / QE_TOKEN_RANK_FILE in `env` (default os.environ)
    for the ones taken from the cache. Returns {file: path used} for the cached / downloaded ones."""
    env = os.environ if env is None else env
    want = []
    ft_on = use_mtp_ft(model_dir)
    if not ft_on and not (env.get("QE_MTP_FILE") or "").strip():
        _log("INT8 checkpoint: the checkpoint's own MTP head (the fine-tuned head was trained on INT4 hidden states; "
             "QE_INT8_MTP_FT=1 uses it)")
    if (ft_on and not os.path.exists(os.path.join(model_dir, MTP_FT))
            and not (env.get("QE_MTP_FILE") or "").strip()):
        want.append((MTP_FT, "QE_MTP_FILE"))
    if not os.path.exists(os.path.join(model_dir, TOKEN_RANK)) and not (env.get("QE_TOKEN_RANK_FILE") or "").strip():
        want.append((TOKEN_RANK, "QE_TOKEN_RANK_FILE"))
    if not want:
        return {}
    cache = os.path.expanduser(cache or env_str("QE_ASSETS_DIR") or DEFAULT_DIR)
    base = (base or env_str("QE_ASSETS_URL") or DEFAULT_URL).rstrip("/")
    used = {}
    # cached files: checked against the SHA256SUMS kept next to them (no network)
    sums = {}
    try:
        with open(os.path.join(cache, "SHA256SUMS")) as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    sums[p[1].lstrip("*")] = p[0].lower()
    except OSError:
        pass
    for name, var in list(want):
        path = os.path.join(cache, name)
        if os.path.exists(path) and name in sums and _sha256(path) == sums[name]:
            env[var] = path
            used[name] = path
            want.remove((name, var))
            _log(f"using cached {name} ({_mb(os.path.getsize(path))}) from {cache}")
    if not want:
        return used
    missing = [n for n, _ in want]
    if env_int("QE_FETCH_ASSETS", 1) == 0:
        _warn(missing, model_dir, "QE_FETCH_ASSETS=0")
        return used
    # the download runs in a thread; the start waits at most QE_FETCH_WAIT_S (default 120 s). A slower link lets
    # the server start with the stock head now; the thread finishes the files into the cache for the next start.
    import threading
    late = threading.Event()
    local = {}
    t = threading.Thread(target=_fetch, args=(want, model_dir, cache, base, timeout, local, late), daemon=True)
    t.start()
    t.join(max(0, env_int("QE_FETCH_WAIT_S", 120)))
    late.set()
    if t.is_alive():
        _log(f"still downloading {', '.join(missing)}: this start uses the stock MTP head / full draft vocabulary; "
             f"the files are used from the next start ({cache})")
    for name, var in want:
        if name in local:
            env[var] = local[name]
            used[name] = local[name]
    return used


def _fetch(want, model_dir, cache, base, timeout, out, late):
    missing = [n for n, _ in want]
    try:
        os.makedirs(cache, exist_ok=True)
        sums = _sums(base, cache, timeout)
    except Exception as ex:
        _warn(missing, model_dir, f"{base}/SHA256SUMS: {ex}")
        return
    for name, var in want:
        path = os.path.join(cache, name)
        tmp = path + f".part{os.getpid()}"
        try:
            if name not in sums:
                raise ValueError("not listed in the release's SHA256SUMS")
            _log(f"downloading {name} from {base} ...")
            n = _get(f"{base}/{name}", tmp, name, timeout)
            sha = _sha256(tmp)
            if sha != sums[name]:
                raise ValueError(f"sha256 mismatch ({sha[:12]}... != {sums[name][:12]}...), file discarded")
            os.replace(tmp, path)
            if not late.is_set():
                out[name] = path
            _log(f"downloaded {name} ({_mb(n)}) to {path}, sha256 ok"
                 + (" (used from the next start)" if late.is_set() else ""))
        except Exception as ex:
            _warn([name], model_dir, str(ex))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
