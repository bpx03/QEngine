"""Vision tower and preprocessing vs HF transformers (run with the reference venv):
  .venv-ref/bin/python tests/test_vision_ref.py <image> [model dir] [reference weights dir]
Model: Qwen3.8-27B-INT4 (default) or Qwen3.8-Flash-Next-QE-INT4 (qwen4_exp: the HF tower gets the weights of the
original checkpoint, default models/Qwen3.8-Flash-Next-W4A16-AutoRound, ours come from the converted dir)."""
import os, sys
import numpy as np, torch
from PIL import Image
from safetensors import safe_open
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from vision import VisionEncoder
from transformers import AutoProcessor

torch.backends.cudnn.enabled = False  # the patch conv as a plain GEMM (some venvs miss cuDNN sublibraries)

MODEL = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-27B-INT4")
FLASH = json.load(open(os.path.join(MODEL, "config.json"))).get("model_type", "").startswith("qwen4_exp")
if FLASH:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig as Cfg
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel as VisionModel
    REF = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-Flash-Next-W4A16-AutoRound")
else:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config as Cfg
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel as VisionModel
    REF = MODEL
img = Image.open(sys.argv[1])
dev = os.environ.get("QE_VISION_DEV", "cuda:0")
mine = VisionEncoder(MODEL, dev, max_pixels=16777216)
proc = AutoProcessor.from_pretrained(MODEL)
r = proc.image_processor(images=[img], return_tensors="pt")
pv, grid = r["pixel_values"], r["image_grid_thw"]
p, g = mine.preprocess(img)
print("grid HF", grid.tolist(), "mine", g)
d = (torch.from_numpy(p) - pv.float()).abs()
print(f"preprocess: max |diff| {d.max():.4f}, mean {d.mean():.5f}")

cfg = Cfg.from_pretrained(MODEL).vision_config
cfg._attn_implementation = "sdpa"
hf = VisionModel(cfg).to(dev, torch.bfloat16).eval()
idx = json.load(open(os.path.join(REF, "model.safetensors.index.json")))["weight_map"]
sd = {}
for f in sorted({v for k, v in idx.items() if k.startswith("model.visual.")}):
    with safe_open(os.path.join(REF, f), "pt") as st:
        for k in st.keys():
            if k.startswith("model.visual."):
                sd[k[len("model.visual."):]] = st.get_tensor(k)
missing = hf.load_state_dict(sd, strict=False)
print("missing/unexpected:", missing.missing_keys[:5], missing.unexpected_keys[:5])
hf = hf.to(dev, torch.bfloat16)
with torch.no_grad():
    ref = hf(pv.to(dev), grid_thw=grid.to(dev)).pooler_output.float()
    out = mine.encode(pv.numpy().astype(np.float32), tuple(grid[0].tolist())).float()
    out2 = mine.encode(p, g).float()
cos = torch.nn.functional.cosine_similarity
print(f"encoder (same pixels): cos min {cos(out, ref, dim=-1).min():.5f} mean {cos(out, ref, dim=-1).mean():.6f}, "
      f"rel err {((out - ref).norm() / ref.norm()):.4f}")
print(f"encoder (own preprocessing): cos min {cos(out2, ref, dim=-1).min():.5f} mean {cos(out2, ref, dim=-1).mean():.6f}")
print("tokens", ref.shape)

# noise floor: HF in fp32 as the gold standard
hf32 = VisionModel(cfg).to(dev).eval()
hf32.load_state_dict(sd, strict=False)
hf32 = hf32.float()
with torch.no_grad():
    gold = hf32(pv.to(dev).float(), grid_thw=grid.to(dev)).pooler_output.float()
for name, x in (("HF bf16", ref), ("mine", out), ("mine own preprocessing", out2)):
    c = cos(x, gold, dim=-1)
    print(f"{name} vs HF fp32: cos min {c.min():.5f} mean {c.mean():.6f} rel err {((x - gold).norm() / gold.norm()):.4f}")

# our weights (converted directory) == the reference checkpoint's
allw = dict(mine.w)
for d in mine.hw or []:  # streamed blocks (QE_VISION_STREAM): in pinned host memory
    allw.update(d)
same = all(torch.equal(allw[k].cpu().to(torch.bfloat16), sd[k].to(torch.bfloat16)) for k in sd if k in allw)
dmax = max((allw[k].cpu().float() - sd[k].float()).abs().max().item() for k in sd if k in allw)
print(f"weights: {len(sd)} reference tensors, {sum(k in allw for k in sd)} loaded by ours ({mine.dt}), "
      f"bit-identical: {same}, max |diff| {dmax:.2g}")
