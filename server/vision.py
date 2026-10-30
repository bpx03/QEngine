"""Qwen3.5 vision tower (ViT, 27 blocks) in plain PyTorch, plus image preprocessing and the
multimodal RoPE positions of a prompt. Only torch, safetensors, numpy and PIL are needed (no
transformers), so the server does not depend on a transformers version that knows qwen3_5.

The engine consumes the encoder output as the prompt embeddings of the image rows; their token
ids are content-derived (>= IMG_TOKEN) so that the engine's prefix cache tells images apart."""
import base64
import hashlib
import io
import json
import math
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from qe_env import env_int, env_str

IMG_TOKEN = 1 << 30  # engine: ids >= this are image rows (kImgToken)


class VisionEncoder:
    def __init__(self, model_dir, device="cuda:0", max_pixels=4 * 1024 * 1024, res32=None):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        v = cfg["vision_config"]
        self.cfg = cfg
        # fp32 residual stream (GEMMs stay bf16): Qwen3.8-Flash-Next's tower is more sensitive to the bf16 residual
        # than the 27B one (vs HF fp32, test_screen.png: cos min 0.81 -> 0.88, mean 0.9981 -> 0.9992; 27B: already
        # 0.992 / 0.9997 in bf16, unchanged). Default: on for Flash-Next only (QE_VISION_RES32=0/1 overrides).
        if res32 is None:
            res32 = bool(env_int("QE_VISION_RES32", int(cfg.get("model_type", "").startswith("qwen4_exp"))))
        self.res32 = res32
        # GEMM dtype: fp16 for Flash-Next (vs HF fp32 on test_screen.png: cos min 0.998 / mean 0.99998 / rel err 0.008,
        # bf16 0.88 / 0.9992 / 0.050 at the same speed; the largest GEMM output, MLP fc2, stays ~10.6K < 65504 on
        # screenshots, photos, flat / noise images up to 4 MPix; a non-finite output falls back to bf16), bf16 for
        # the 27B (unchanged). QE_VISION_DTYPE=bf16|fp16 overrides.
        dt_def = "fp16" if cfg.get("model_type", "").startswith("qwen4_exp") else "bf16"
        dt = env_str("QE_VISION_DTYPE", dt_def).strip().lower()
        if dt not in ("fp16", "bf16"):
            print(f"[env] warning: QE_VISION_DTYPE={dt!r} is not fp16 or bf16, using the default", flush=True)
            dt = dt_def
        self.dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dt]
        self.dev = torch.device(device)
        self.patch, self.merge, self.tps = v["patch_size"], v["spatial_merge_size"], v["temporal_patch_size"]
        self.dim, self.heads, self.depth = v["hidden_size"], v["num_heads"], v["depth"]
        self.side = int(v["num_position_embeddings"] ** 0.5)
        self.image_token = cfg["image_token_id"]
        self.vstart, self.vend = cfg["vision_start_token_id"], cfg["vision_end_token_id"]
        pp = json.load(open(os.path.join(model_dir, "processor_config.json")))["image_processor"]
        self.mean = np.array(pp["image_mean"], np.float32)
        self.std = np.array(pp["image_std"], np.float32)
        self.min_pixels = pp["size"]["shortest_edge"]
        self.max_pixels = min(pp["size"]["longest_edge"], max_pixels)
        rp = (v.get("rope_parameters") or {})
        theta = rp.get("rope_theta", 10000.0)
        hd = self.dim // self.heads
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, hd // 2, 2, dtype=torch.float32) / (hd // 2)))
        self.inv_freq = self.inv_freq.to(self.dev)

        # QE_VISION_STREAM=1: the 27 blocks' weights stay in pinned host memory and are copied to the GPU block by
        # block (one block ahead, on a side stream) during each encode: less resident VRAM next to the Flash-Next
        # expert cache, ~30 MB copied per block
        self.stream = self.dev.type == "cuda" and bool(
            env_int("QE_VISION_STREAM", int(cfg.get("model_type", "").startswith("qwen4_exp"))))
        self.hw = None
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        names = [k for k in idx if k.startswith("model.visual.")]
        w = {}
        for f in sorted({idx[k] for k in names}):
            with safe_open(os.path.join(model_dir, f), "pt") as st:
                for k in names:
                    if idx[k] == f:
                        kk = k[len("model.visual."):]
                        host = self.stream and kk.startswith("blocks.")
                        w[kk] = st.get_tensor(k).to("cpu" if host else self.dev, self.dt)
        if self.stream:  # blocks: pinned host copies (moved off the GPU before anything else is allocated)
            self.hw = [{k: w.pop(k).pin_memory() for k in [k for k in w if k.startswith(f"blocks.{i}.")]}
                       for i in range(self.depth)]
            self.copy_stream = torch.cuda.Stream(self.dev)
            # two fixed device buffers (block i in buffer i % 2): no allocator churn across streams
            self.wbuf = [{k: torch.empty_like(t, device=self.dev) for k, t in self.hw[0].items()} for _ in range(2)]
            self.wfree = [None, None]  # event: the compute stream is done with the buffer
        self.w = w
        self.pe_w = w["patch_embed.proj.weight"].reshape(self.dim, -1)
        self.cache = {}  # image digest -> (embeddings on CPU, (t, h, w))
        self.cache_bytes, self.cache_limit = 0, env_int("QE_VISION_CACHE_MB", 2048) << 20
        # largest image once: PyTorch's pool then already holds the encoder's peak memory, so a
        # KV cache planned afterwards (engine start) leaves room for it
        side = int(math.sqrt(self.max_pixels))
        self.encode(*self.preprocess(Image.new("RGB", (side, side))))
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    # ------------------------------------------------------------------ preprocessing
    def smart_resize(self, h, w):
        f = self.patch * self.merge
        hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
        if hb * wb > self.max_pixels:
            beta = math.sqrt(h * w / self.max_pixels)
            hb, wb = max(f, math.floor(h / beta / f) * f), max(f, math.floor(w / beta / f) * f)
        elif hb * wb < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (h * w))
            hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
        return hb, wb

    def preprocess(self, img):
        """PIL image -> (patches [N, 3*2*16*16] float32, grid (t, h, w)) as Qwen2VLImageProcessor."""
        img = img.convert("RGB")
        H, W = self.smart_resize(img.height, img.width)
        img = img.resize((W, H), Image.BICUBIC)
        x = (np.asarray(img, np.float32) / 255.0 - self.mean) / self.std  # [H, W, 3]
        x = x.transpose(2, 0, 1)[None].repeat(self.tps, 0)  # [T=2, 3, H, W]
        p, m = self.patch, self.merge
        gh, gw = H // p, W // p
        x = x.reshape(1, self.tps, 3, gh // m, m, p, gw // m, m, p)
        x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return np.ascontiguousarray(x.reshape(gh * gw, 3 * self.tps * p * p)), (1, gh, gw)

    # ------------------------------------------------------------------ encoder
    def _pos_embed(self, gh, gw):
        m, side = self.merge, self.side
        n = gh * gw
        k = torch.arange(n, device=self.dev)
        bw = gw // m
        row = (k // (m * m * bw)) * m + (k // m) % m
        col = ((k // (m * m)) % bw) * m + k % m

        def taps(i, size):
            src = i.float() * (side - 1) / max(size - 1, 1)
            fl = torch.floor(src)
            t = (fl.long()[:, None] + torch.arange(2, device=self.dev)).clamp(0, side - 1)
            wt = (1 - (src[:, None] - fl[:, None] - torch.arange(2, device=self.dev)).abs()).clamp(min=0)
            return t, wt
        ht, hw_ = taps(row, gh)
        wt, ww = taps(col, gw)
        ind = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
        wgt = (hw_[:, :, None] * ww[:, None, :]).reshape(-1, 4)
        tab = self.w["pos_embed.weight"]
        e = tab[ind[:, 0]].float() * wgt[:, 0, None]
        for j in range(1, 4):  # tap by tap: no [n, 4, dim] fp32 temporary
            e += tab[ind[:, j]].float() * wgt[:, j, None]
        return e.to(self.dt), row, col

    @torch.no_grad()
    def encode(self, patches, grid):
        _, gh, gw = grid
        w = self.w
        x = torch.from_numpy(patches).to(self.dev, self.dt)
        x = F.linear(x, self.pe_w, w["patch_embed.proj.bias"])
        pe, row, col = self._pos_embed(gh, gw)
        x = x.float() + pe.float() if self.res32 else x + pe
        fr = torch.cat([row.float()[:, None] * self.inv_freq, col.float()[:, None] * self.inv_freq], -1)
        fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos()[:, None, :], fr.sin()[:, None, :]  # [n, 1, hd]
        n, hd = x.shape[0], self.dim // self.heads

        CH = 4096  # row chunks bound the temporaries (same math)

        def ln(t, b, w=w):  # LayerNorm -> GEMM input (in fp32 for the fp32 residual)
            if self.res32:
                return F.layer_norm(t, (self.dim,), w[b + "weight"].float(), w[b + "bias"].float(), 1e-6).to(self.dt)
            return F.layer_norm(t, (self.dim,), w[b + "weight"], w[b + "bias"], 1e-6)

        cur = torch.cuda.current_stream(self.dev) if self.stream else None

        def fetch(i):  # block i's weights -> device buffer i % 2 on the copy stream (streamed mode)
            buf = self.wbuf[i % 2]
            with torch.cuda.stream(self.copy_stream):
                if self.wfree[i % 2] is not None:
                    self.copy_stream.wait_event(self.wfree[i % 2])
                for k, t in self.hw[i].items():
                    buf[k.replace(f"blocks.{i}.", "blocks.0.")].copy_(t, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)
            return {k.replace("blocks.0.", f"blocks.{i}."): t for k, t in buf.items()}, ev
        nxt = fetch(0) if self.stream else None

        def rot(t):
            out = torch.empty_like(t)
            for s in range(0, n, CH):
                u = t[s:s + CH].float()
                u2 = torch.cat([-u[..., hd // 2:], u[..., :hd // 2]], -1)
                out[s:s + CH] = (u * cos[s:s + CH] + u2 * sin[s:s + CH]).to(self.dt)
            return out
        for i in range(self.depth):
            b = f"blocks.{i}."
            if self.stream:
                w, ev = nxt
                cur.wait_event(ev)
                if i + 1 < self.depth:
                    nxt = fetch(i + 1)
            h = ln(x, b + "norm1.", w)
            qkv = F.linear(h, w[b + "attn.qkv.weight"], w[b + "attn.qkv.bias"]).view(n, 3, self.heads, hd)
            del h
            q, k, v = rot(qkv[:, 0]), rot(qkv[:, 1]), qkv[:, 2]
            a = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                               v.transpose(0, 1)[None])
            a = a[0].transpose(0, 1).reshape(n, self.dim)
            del q, k, v, qkv
            x += F.linear(a, w[b + "attn.proj.weight"], w[b + "attn.proj.bias"])
            del a
            for s in range(0, n, CH):
                h = ln(x[s:s + CH], b + "norm2.", w)
                h = F.gelu(F.linear(h, w[b + "mlp.linear_fc1.weight"], w[b + "mlp.linear_fc1.bias"]),
                           approximate="tanh")
                x[s:s + CH] += F.linear(h, w[b + "mlp.linear_fc2.weight"], w[b + "mlp.linear_fc2.bias"])
            if self.stream:  # buffer i % 2 may take block i + 2 once this block's kernels are done
                self.wfree[i % 2] = torch.cuda.Event()
                self.wfree[i % 2].record(cur)
        w = self.w
        x = ln(x, "merger.norm.", w)
        x = x.view(-1, self.dim * self.merge ** 2)
        x = F.gelu(F.linear(x, w["merger.linear_fc1.weight"], w["merger.linear_fc1.bias"]))
        x = F.linear(x, w["merger.linear_fc2.weight"], w["merger.linear_fc2.bias"])
        return x.to(torch.bfloat16)  # [tokens, out_hidden] bf16 (the engine input)

    def embed(self, data):
        """Raw image bytes -> (digest, embeddings [tokens, hidden] bf16 CPU, grid (t, h, w))."""
        dg = hashlib.sha256(data).digest()
        hit = self.cache.get(dg)
        if hit is not None:
            self.cache[dg] = self.cache.pop(dg)  # LRU order
            return (dg,) + hit
        patches, grid = self.preprocess(Image.open(io.BytesIO(data)))
        e = self.encode(patches, grid)
        if self.dt == torch.float16 and not torch.isfinite(e).all():  # fp16 overflow: this image again in bf16
            print("vision: non-finite fp16 encoder output, image re-encoded in bf16", flush=True)
            dt, w0, hw0 = self.dt, self.w, self.hw
            self.dt, self.w = torch.bfloat16, {k: v.to(torch.bfloat16) for k, v in w0.items()}
            wb0 = getattr(self, "wbuf", None)
            if hw0:
                self.hw = [{k: v.to(torch.bfloat16).pin_memory() for k, v in d.items()} for d in hw0]
                self.wbuf = [{k: torch.empty_like(t, dtype=torch.bfloat16) for k, t in b.items()} for b in wb0]
            self.pe_w = self.w["patch_embed.proj.weight"].reshape(self.dim, -1)
            try:
                e = self.encode(patches, grid)
            finally:
                self.dt, self.w, self.hw = dt, w0, hw0
                if hw0:
                    self.wbuf = wb0
                self.pe_w = w0["patch_embed.proj.weight"].reshape(self.dim, -1)
        e = e.cpu()
        self.cache[dg] = (e, grid)
        self.cache_bytes += e.numel() * 2
        while self.cache_bytes > self.cache_limit and len(self.cache) > 1:
            old = next(iter(self.cache))
            self.cache_bytes -= self.cache.pop(old)[0].numel() * 2
        return dg, e, grid

    # ------------------------------------------------------------------ prompt assembly
    def expand(self, ids, images):
        """Replaces the i-th <|image_pad|> in ids with images[i]'s rows: content-derived ids
        (>= IMG_TOKEN), and computes the M-RoPE positions. images: list of raw bytes.
        Returns (ids, embeddings [rows, hidden] bf16 or None, rope3 int32 [n, 3] or None, delta)."""
        pads = [i for i, t in enumerate(ids) if t == self.image_token]
        if len(pads) != len(images):
            raise ValueError(f"{len(images)} images but {len(pads)} image placeholders")
        if not images:
            return ids, None, None, 0
        out, embs, pos = [], [], []
        cur, prev = 0, 0
        for p, data in zip(pads, images):
            text = ids[prev:p]
            out += text
            pos += [(cur + i,) * 3 for i in range(len(text))]
            cur += len(text)
            dg, e, (t, gh, gw) = self.embed(data)
            lh, lw = gh // self.merge, gw // self.merge
            base = int.from_bytes(dg[:4], "little") & (IMG_TOKEN - 1)
            out += [IMG_TOKEN | ((base + j) & (IMG_TOKEN - 1)) for j in range(lh * lw)]
            pos += [(cur, cur + r, cur + c) for r in range(lh) for c in range(lw)]
            cur += max(lh, lw)
            embs.append(e)
            prev = p + 1
        text = ids[prev:]
        out += text
        pos += [(cur + i,) * 3 for i in range(len(text))]
        cur += len(text)
        emb = torch.cat(embs).contiguous()
        rope3 = np.asarray(pos, np.int32)
        return out, emb, rope3, cur - len(out)


def load_image_bytes(src):
    """data: URL, base64 string or http(s) URL -> raw bytes."""
    if src.startswith("data:"):
        return base64.b64decode(src.split(",", 1)[1])
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src, timeout=30) as r:
            return r.read()
    return base64.b64decode(src)
