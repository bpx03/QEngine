"""27B release assets on first start: the fine-tuned MTP head and the draft vocabulary ranking.

model_mtp_ft.safetensors and token_rank.bin are GitHub release assets of QEngine, not part of the RedHatAI
checkpoint nor of the Docker image. Without them the engine runs the checkpoint's own MTP head over the full
vocabulary (~8-9 % fewer accepted tokens). When the model dir lacks them (and no --mtp-file / QE_MTP_FILE is
given), the missing ones are fetched from the latest release into ~/.cache/qengine/assets (the Docker volume
qengine-cache), checked against the release's SHA256SUMS, renamed into place atomically and handed to the engine
through QE_MTP_FILE / QE_TOKEN_RANK_FILE. The model dir is never written (Docker mounts it read-only). Valid cached
files are reused without network. Any failure only logs a warning: the engine starts with the stock head.
QE_FETCH_ASSETS=0 turns the download off; QE_ASSETS_URL / QE_ASSETS_DIR override the source / cache dir.
Dependency-free (urllib, hashlib).
"""
import hashlib
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
    if not os.path.exists(os.path.join(model_dir, MTP_FT)) and not (env.get("QE_MTP_FILE") or "").strip():
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
