"""27B DFlash2 block drafter without manual steps: find or fetch the drafter checkpoint.

The drafter is the W4A16 build syvai/Qwen3.8-27B-DFlash2-W4A16 (Apache-2.0; GPTQ requantization of
incoai/Qwen3.8-27B-DFlash2, Apache-2.0), revision pinned below, one file (model.safetensors, 1.28 GB) checked against
its pinned sha256. The engine looks for it in this order: QE_DFLASH (a directory or the file), <model>/dflash2, a
sibling directory Qwen3.8-27B-DFlash2; when none exists the server fetches it once into $QE_DFLASH_DIR, else
$QE_DISK_CACHE/dflash2, else ~/.cache/qengine/dflash2 (Docker: the qengine-cache volume) and passes it as QE_DFLASH.

Never stops the start: QE_DRAFTER=mtp (nothing fetched), QE_FETCH_ASSETS=0, no network or a failed check logs one line
and the engine drafts with MTP only. The download runs in a thread; the start waits at most QE_FETCH_WAIT_S (120 s), a
later finish is used from the next start. QE_DFLASH_URL overrides the source URL (a mirror; the sha256 pin still
applies). Dependency-free (hashlib, urllib)."""
import hashlib
import json
import os
import threading
import time
import urllib.request

from qe_env import env_int, env_str

REPO = "syvai/Qwen3.8-27B-DFlash2-W4A16"
REVISION = "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279"
FILE = {"name": "model.safetensors", "size": 1280633960,
        "sha256": "ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7"}
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE['name']}"
LICENSE_NOTE = f"DFlash2 drafter {REPO}@{REVISION[:8]} (Apache-2.0; base incoai/Qwen3.8-27B-DFlash2, Apache-2.0)"
TIMEOUT = 30
SIBLING = "Qwen3.8-27B-DFlash2"


def _log(msg):
    print(f"[server] DFlash2: {msg}", flush=True)


def cache_dir(env=None):
    env = os.environ if env is None else env
    d = (env.get("QE_DFLASH_DIR") or "").strip()
    if not d:
        d = (env.get("QE_DISK_CACHE") or "").strip()
        d = os.path.join(d, "dflash2") if d else "~/.cache/qengine/dflash2"
    return os.path.abspath(os.path.expanduser(d))


def _local(model_dir):
    """The engine's own lookup (QE_DFLASH, <model>/dflash2, sibling): the first that exists, else None."""
    m = os.path.abspath(model_dir).rstrip("/")
    for c in [(env_str("QE_DFLASH") or "").strip(), os.path.join(m, "dflash2"), os.path.join(os.path.dirname(m), SIBLING)]:
        if not c:
            continue
        for f in (c, os.path.join(c, FILE["name"])):
            if os.path.isfile(f):
                return f
    return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_marker(path):
    """the sha256-ok marker, atomically; a read-only cache only costs a re-hash at the next start (review M6)"""
    try:
        st = os.stat(path)
        tmp = f"{path}.ok.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"sha256": FILE["sha256"], "size": st.st_size, "mtime": st.st_mtime}, f)
        os.replace(tmp, path + ".ok")
    except OSError:
        pass


def _verified(path):
    """The cached file passes its sha256 check (a marker with its size and mtime, so a start does not hash 1.28 GB)."""
    try:
        st = os.stat(path)
        d = json.load(open(path + ".ok"))
        if d.get("sha256") == FILE["sha256"] and d.get("size") == st.st_size == FILE["size"] and d.get("mtime") == st.st_mtime:
            return True
    except (OSError, ValueError):
        pass
    try:
        ok = os.path.getsize(path) == FILE["size"] and _sha256(path) == FILE["sha256"]
    except OSError:
        return False
    if ok:
        _write_marker(path)
    return ok


def _fetch(cache, out, res):
    try:
        url = (env_str("QE_DFLASH_URL") or "").strip() or URL
        os.makedirs(cache, exist_ok=True)
        tmp = os.path.join(cache, f".{FILE['name']}.{os.getpid()}.part")
        try:
            _log(f"downloading the drafter ({FILE['size'] / 1e9:.2f} GB) from {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "qengine-dflash2"})
            got, h = 0, hashlib.sha256()
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                if total and total != FILE["size"]:
                    raise IOError(f"{total} bytes offered, {FILE['size']} expected")
                while True:
                    b = r.read1(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    h.update(b)
                    got += len(b)
                    if got > FILE["size"]:
                        raise IOError(f"more than the {FILE['size']} bytes expected")
            if got != FILE["size"]:
                raise IOError(f"got {got} of {FILE['size']} bytes")
            if h.hexdigest() != FILE["sha256"]:
                raise ValueError(f"sha256 mismatch ({h.hexdigest()[:12]}... != {FILE['sha256'][:12]}...), file discarded")
            os.replace(tmp, out)
            _write_marker(out)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if res.get("late"):
            _log(f"downloaded to {out} (sha256 ok): used from the next start")
    except Exception as ex:
        res["err"] = ex
        if res.get("late"):
            _log(f"background download failed ({type(ex).__name__}: {ex})")


def resolve(model_dir):
    """Sets QE_DFLASH to a usable checkpoint when the engine would find none; never raises, never stops the start."""
    try:
        drafter = (env_str("QE_DRAFTER") or "auto").strip().lower()
        if drafter == "mtp":
            return None
        p = _local(model_dir)
        if p:
            return p
        user = (env_str("QE_DFLASH") or "").strip()
        if user:  # the user's path is kept (the engine reports it missing and drafts with MTP), never replaced (review L7)
            _log(f"QE_DFLASH={user} does not exist: MTP drafts (nothing fetched)")
            return None
        mtp = (env_str("QE_MTP_FILE") or "").strip() or os.path.join(model_dir, "model_mtp.safetensors")
        if not os.path.isfile(mtp):  # the drafter runs only next to an MTP head
            return None
        cache = cache_dir()
        out = os.path.join(cache, FILE["name"])
        if os.path.isfile(out):
            if _verified(out):
                os.environ["QE_DFLASH"] = out
                return out
            _log(f"cached {out} fails its sha256 check: fetched again")
            try:
                os.remove(out)
            except OSError:
                pass
        if env_int("QE_FETCH_ASSETS", 1) == 0 or (env_str("HF_HUB_OFFLINE") or "").strip().lower() in ("1", "true", "yes", "on"):
            _log(f"not in {cache} and downloads are off (QE_FETCH_ASSETS=0 / HF_HUB_OFFLINE): MTP drafts only")
            return None
        try:  # temporary files of a start killed mid-download (older than an hour)
            for f in os.listdir(cache):
                q = os.path.join(cache, f)
                if f.endswith(".part") and time.time() - os.path.getmtime(q) > 3600:
                    os.remove(q)
        except OSError:
            pass
        res = {}
        t = threading.Thread(target=_fetch, args=(cache, out, res), daemon=True)
        t.start()
        wait = max(0, env_int("QE_FETCH_WAIT_S", 120))
        t.join(wait)
        if t.is_alive():
            res["late"] = True
            _log(f"still downloading after {wait} s (QE_FETCH_WAIT_S): MTP drafts for this start, the drafter is used from "
                 f"the next start ({cache})")
            return None
        if res.get("err"):
            _log(f"fetch failed ({type(res['err']).__name__}: {res['err']}): MTP drafts only; by hand: "
                 f"curl -L -o {out} {URL} (sha256 {FILE['sha256'][:16]}...)")
            return None
        _log(f"{LICENSE_NOTE}: {out}")
        os.environ["QE_DFLASH"] = out
        return out
    except Exception as ex:  # never a stop
        _log(f"lookup failed ({type(ex).__name__}: {ex}): MTP drafts only")
        return None
