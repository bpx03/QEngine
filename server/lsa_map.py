"""27B --lsa without manual steps: find, fetch, convert and cache the D-LSA index map.

The map is the distilled index map released with the Latent Sparse Attention paper (arXiv 2610.42147): GitHub release
d-lsa-weights-v1 of Prefrontal-AI/latent-sparse-attention (Apache-2.0), file
qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt (6.3 MB). It is fetched from that pinned URL, checked against the
pinned sha256, converted to safetensors (the engine's format) without torch, checked against the pinned sha256 of the
converted file and cached in $QE_LSA_DIR, else $QE_DISK_CACHE/lsa, else ~/.cache/qengine/lsa (Docker: the
qengine-cache volume). A valid cached map is used without network.

Spec (--lsa / QE_LSA), documented forms: a bare --lsa or QE_LSA=auto: the map for this checkpoint, fetched once;
--lsa PATH / QE_LSA=PATH: that map. Silent aliases, kept working: --lsa auto, QE_LSA=1 / on / true / yes (= auto);
unset, '' or 0 / off / false / no / none: off. A path to a .safetensors map: used as is (unchanged). A path to a .pt map: converted
into the cache. Auto never stops the start: no known map, no network (QE_FETCH_ASSETS=0, offline) or a failed check
logs one line and the server starts with full attention. QE_LSA_URL overrides the source URL (a mirror; the sha256
pin still applies).

The download runs in a thread: the start waits at most QE_FETCH_WAIT_S (120 s); a later finish is used from the
next start. The .pt is a zip with a pickle; it is read by a restricted unpickler that resolves only the three names a tensor dict
needs (torch._utils._rebuild_tensor_v2, torch.FloatStorage, collections.OrderedDict) to local stand-ins: no other
global, no code from the file runs, torch is not imported. Dependency-free (zipfile, pickle, hashlib, urllib).
"""
import array
import hashlib
import io
import json
import os
import pickle
import tempfile
import threading
import time
import urllib.request
import zipfile

from qe_env import env_int, env_str

# the released map for Qwen3.8-27B (INT4 and INT8 checkpoints of the same base model): d2048, the learned one
# (0.806 vs 0.791 dense attention mass for the training-free pca-step0)
MAP_27B = {
    "name": "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt",
    "url": "https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/d-lsa-weights-v1/"
           "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt",
    "sha256": "24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b",
    "size": 6302911,
    "out": "qwen38_27b_d2048.safetensors",
    # sha256 of the converted file: byte for byte what tools/lsa/convert_map.py (torch + safetensors) writes
    "out_sha256": "8c44eab910b91816abd067bfaeb2b7aa13611728eb3e24cbb297933fc40c9d15",
}
LICENSE_NOTE = ("index map by Prefrontal AI (github.com/Prefrontal-AI/latent-sparse-attention, release d-lsa-weights-v1), "
                "Apache-2.0")
OFF = ("", "0", "off", "false", "no", "none")
AUTO = ("1", "auto", "on", "true", "yes")
TIMEOUT = 30          # seconds per socket operation
FULL_LAYERS = tuple(range(3, 64, 4))
ROWS, COLS = 768, 64
MAX_PKL = 1 << 20     # the map's pickle is ~10 KB; anything far larger is not a map


def _log(msg):
    print(f"[server] LSA: {msg}", flush=True)


def cache_dir(env=None):
    env = os.environ if env is None else env
    d = (env.get("QE_LSA_DIR") or "").strip()
    if not d:
        d = (env.get("QE_DISK_CACHE") or "").strip()
        d = os.path.join(d, "lsa") if d else "~/.cache/qengine/lsa"
    return os.path.abspath(os.path.expanduser(d))


def is_safetensors(path):
    try:
        with open(path, "rb") as f:
            head = f.read(9)
        n = int.from_bytes(head[:8], "little") if len(head) == 9 else 0
        return head[:2] != b"PK" and len(head) == 9 and head[8:9] == b"{" and n <= os.path.getsize(path) - 8
    except OSError:
        return False


def is_pt(path):
    """A torch zip archive (a released .pt map), not a safetensors file."""
    try:
        return os.path.isfile(path) and not is_safetensors(path) and zipfile.is_zipfile(path)
    except OSError:
        return False


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def map_for(model_dir):
    """The pinned map entry for this checkpoint, or (None, reason). Matches the Qwen3.8-27B attention shape: 64 layers,
    24 q / 4 kv heads of 256, full attention on layers 3, 7, ..., 63 (the layers the map has)."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
    except (OSError, ValueError) as ex:
        return None, f"config.json: {ex}"
    t = c.get("text_config") or c
    lt = t.get("layer_types") or []
    full = tuple(i for i, x in enumerate(lt) if x == "full_attention")
    if (t.get("num_hidden_layers"), t.get("num_attention_heads"), t.get("num_key_value_heads"), t.get("head_dim")) \
            != (64, 24, 4, 256) or (lt and full != FULL_LAYERS):
        return None, "no released index map for this checkpoint (it is for Qwen3.8-27B)"
    return MAP_27B, None


# ---------------------------------------------------------------- .pt reader (no torch, restricted unpickler)

class _Stand:  # no state from the pickle (BUILD) can be set on the stand-ins
    __slots__ = ()

    def __setstate__(self, state):
        raise pickle.UnpicklingError("state on a tensor object")


class _Storage(_Stand):
    __slots__ = ("key", "numel")

    def __init__(self, key, numel):
        self.key, self.numel = key, numel


class _FloatStorage:  # stand-in for torch.FloatStorage (only compared by identity)
    pass


def _ordered_dict():  # stand-in for collections.OrderedDict() (the tensor's backward hooks: always empty here)
    return {}


class _Tensor(_Stand):
    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, *rest):
    if not isinstance(storage, _Storage) or type(offset) is not int or not isinstance(size, tuple) \
            or not isinstance(stride, tuple):
        raise pickle.UnpicklingError("not a plain tensor")
    return _Tensor(storage, offset, size, stride)


class _MapUnpickler(pickle.Unpickler):
    ALLOWED = {("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
               ("torch", "FloatStorage"): _FloatStorage,
               ("collections", "OrderedDict"): _ordered_dict}

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed in an index map") from None

    def persistent_load(self, pid):
        if (not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage" or pid[1] is not _FloatStorage
                or not isinstance(pid[2], str) or len(pid[2]) > 16 or type(pid[4]) is not int):
            raise pickle.UnpicklingError(f"unexpected storage ({type(pid).__name__})")
        return _Storage(pid[2], pid[4])


def read_pt(path):
    """{name: bytes (fp32 little-endian, row-major [768, 64])} of a released map .pt; raises ValueError if it is not
    exactly the 32 tensors 3.W_Q_idx ... 63.W_K_idx of fp32 [768, 64]."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl = [n for n in names if n.endswith("/data.pkl") and n.count("/") == 1]
        if len(pkl) != 1:
            raise ValueError("not a torch zip archive (no single <name>/data.pkl)")
        root = pkl[0][: -len("data.pkl")]
        info = z.getinfo(pkl[0])
        if info.file_size > MAX_PKL:
            raise ValueError(f"data.pkl of {info.file_size} bytes")
        bo = root + "byteorder"
        if bo in names and (z.getinfo(bo).file_size > 16 or z.read(bo).strip() != b"little"):
            raise ValueError("big-endian storages")
        d = _MapUnpickler(io.BytesIO(z.read(pkl[0]))).load()
        if not isinstance(d, dict):
            raise ValueError(f"top level is {type(d).__name__}, not a dict")
        want = {f"{l}.W_{w}_idx" for l in FULL_LAYERS for w in "QK"}
        if not all(type(k) is str for k in d) or set(d) != want:
            raise ValueError(f"not the 32 tensors 3.W_Q_idx ... 63.W_K_idx ({len(d)} entries)")
        out = {}
        for k, t in d.items():
            if not isinstance(t, _Tensor) or t.size != (ROWS, COLS) or t.stride != (COLS, 1):
                raise ValueError(f"{k}: not a contiguous [768, 64] tensor")
            entry = f"{root}data/{t.storage.key}"
            if entry not in names:
                raise ValueError(f"{k}: storage {t.storage.key} missing")
            si = z.getinfo(entry)
            n = ROWS * COLS * 4
            # one storage per tensor, as released: nothing larger is ever read into memory
            if t.offset != 0 or t.storage.numel != ROWS * COLS or si.file_size != n:
                raise ValueError(f"{k}: storage of {si.file_size} bytes, offset {t.offset}")
            b = z.read(entry)
            if len(b) != n:
                raise ValueError(f"{k}: short storage")
            a = array.array("f")
            a.frombytes(b)
            if any(x != x or x in (float("inf"), float("-inf")) for x in a):
                raise ValueError(f"{k}: not finite")
            out[k] = b
    return out


def write_safetensors(tensors, dst, metadata):
    """The same bytes as safetensors' save_file for fp32 [768, 64] tensors: header JSON without spaces, __metadata__
    first, tensors in name order, padded with spaces to 8 bytes; data in the same order."""
    hdr, off = {"__metadata__": dict(sorted(metadata.items()))}, 0
    order = sorted(tensors)
    for k in order:
        n = len(tensors[k])
        hdr[k] = {"dtype": "F32", "shape": [ROWS, COLS], "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(dst, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for k in order:
            f.write(tensors[k])


def _tmp(d, name):
    fd, p = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=d)
    os.close(fd)
    return p


def convert(src, dst, source_name=None, want_sha=None):
    """Released .pt -> safetensors at dst: written to a unique temporary file, checked against want_sha (if given),
    then renamed into place. Any malformed input raises ValueError."""
    try:
        t = read_pt(src)
    except ValueError:
        raise
    except Exception as ex:  # (pickle / zip / type errors of a crafted file)
        raise ValueError(f"{type(ex).__name__}: {ex}"[:200]) from None
    tmp = _tmp(os.path.dirname(dst) or ".", os.path.basename(dst))
    try:
        write_safetensors(t, tmp, {"source": source_name or os.path.basename(src),
                                   "layout": "[768 = kv0..kv3 x dims 64..255, 64]"})
        if want_sha and _sha256(tmp) != want_sha:
            raise ValueError("the converted map differs from the pinned one")
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------- download

def _download(url, dst, size):
    req = urllib.request.Request(url, headers={"User-Agent": "qengine-lsa"})
    got = 0
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        if total and size and total != size:
            raise IOError(f"{total} bytes offered, {size} expected")
        while True:
            b = r.read1(1 << 16)
            if not b:
                break
            f.write(b)
            got += len(b)
            if size and got > size:
                raise IOError(f"more than the {size} bytes expected")
    if size and got != size:
        raise IOError(f"got {got} of {size} bytes")


def _fetch(m, cache, out, res):
    """Download, check, convert into the cache; res["err"] = the exception on failure (runs in a thread)."""
    try:
        url = (env_str("QE_LSA_URL") or "").strip() or m["url"]
        os.makedirs(cache, exist_ok=True)
        tmp = _tmp(cache, m["name"])
        try:
            _log(f"downloading the index map {m['name']} ({m['size'] / 1e6:.1f} MB) from {url}")
            _download(url, tmp, m["size"])
            sha = _sha256(tmp)
            if sha != m["sha256"]:
                raise ValueError(f"sha256 mismatch ({sha[:12]}... != {m['sha256'][:12]}...), file discarded")
            convert(tmp, out, m["name"], m["out_sha256"])
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if res.get("late"):
            _log(f"map {m['out']} downloaded to {out} (sha256 ok): used from the next start")
    except Exception as ex:
        res["err"] = ex
        if res.get("late"):
            _log(f"background download failed ({type(ex).__name__}: {ex})")


def _clean_parts(cache):
    """Temporary files left by a start that was killed mid-download (older than an hour)."""
    try:
        for f in os.listdir(cache):
            p = os.path.join(cache, f)
            if f.endswith(".part") and time.time() - os.path.getmtime(p) > 3600:
                os.remove(p)
    except OSError:
        pass


def _manual(m, cache):
    _log(f"  by hand: curl -L -o {os.path.join(cache, m['name'])} {m['url']}  (sha256 {m['sha256'][:16]}...), "
         f"then start again (the file in {cache} is converted without network)")


def resolve(spec, model_dir, env=None):
    """The safetensors path for QE_LSA, or None (LSA off). Never raises for 'auto'; an explicit path that does not exist
    or is neither safetensors nor a released .pt raises SystemExit (as before)."""
    env = os.environ if env is None else env
    s = (spec or "").strip()
    if s.lower() in OFF:
        return None
    if s.lower() in AUTO:
        return _auto(model_dir, env)
    if not os.path.isfile(s):
        raise SystemExit(f"--lsa: no such file {s} (a bare --lsa fetches the released map)")
    if is_safetensors(s):
        return os.path.abspath(s)
    if is_pt(s):  # a released .pt: convert once into the cache (keyed by its sha256)
        cache = cache_dir(env)
        sha = _sha256(s)
        out = os.path.join(cache, f"{os.path.splitext(os.path.basename(s))[0]}-{sha[:12]}.safetensors")
        if not is_safetensors(out):
            try:
                os.makedirs(cache, exist_ok=True)
                convert(s, out)
            except Exception as ex:  # (convert turns a malformed file into ValueError; OSError: the cache)
                raise SystemExit(f"--lsa: {s} is not a D-LSA index map ({ex})")
            _log(f"converted {s} to {out}")
        return out
    raise SystemExit(f"--lsa: {s} is neither a safetensors map nor a released .pt map (a bare --lsa fetches it)")


def _auto(model_dir, env):
    m, why = map_for(model_dir)
    if m is None:
        _log(f"off: {why}; the server starts with full attention")
        return None
    cache = cache_dir(env)
    out = os.path.join(cache, m["out"])
    _clean_parts(cache)
    try:
        if os.path.isfile(out) and _sha256(out) == m["out_sha256"]:
            _log(f"map {m['out']} (from {m['name']}; {LICENSE_NOTE}) from the cache {out}")
            return out
    except OSError:
        pass
    pt = os.path.join(cache, m["name"])  # a .pt fetched by hand into the cache: converted, no network
    if os.path.isfile(pt):
        try:
            if _sha256(pt) != m["sha256"]:
                raise ValueError(f"sha256 of {pt} is not the released one")
            convert(pt, out, m["name"], m["out_sha256"])
            _log(f"map {m['out']} converted from {pt} ({LICENSE_NOTE}), cached at {out}")
            return out
        except Exception as ex:
            _log(f"{pt} not used ({ex})")
    if env_int("QE_FETCH_ASSETS", 1) == 0:
        _log(f"off: the index map is not in {cache} and QE_FETCH_ASSETS=0; the server starts with full attention")
        _manual(m, cache)
        return None
    # the download runs in a thread; the start waits at most QE_FETCH_WAIT_S (default 120 s, as server/assets.py), so a
    # stalled link or DNS never holds the start; a late download is used from the next start
    res = {}
    t = threading.Thread(target=_fetch, args=(m, cache, out, res), daemon=True)
    t.start()
    t.join(max(0, env_int("QE_FETCH_WAIT_S", 120)))
    if t.is_alive():
        res["late"] = True
        _log(f"off for this start: the index map is still downloading after {max(0, env_int('QE_FETCH_WAIT_S', 120))} s "
             f"(QE_FETCH_WAIT_S); the server starts with full attention, the map is used from the next start ({cache})")
        return None
    if "err" in res:  # (offline, proxy, disk full, a changed upstream file: never a refused start)
        ex = res["err"]
        _log(f"off: could not get the index map ({type(ex).__name__}: {ex}); the server starts with full attention")
        _manual(m, cache)
        return None
    _log(f"map {m['out']} (from {m['name']}, sha256 ok; {LICENSE_NOTE}) cached at {out}")
    return out
