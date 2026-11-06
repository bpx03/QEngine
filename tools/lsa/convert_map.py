"""Convert a released D-LSA index map (.pt, untrusted pickle) to safetensors, tensors only.
usage: python3 -I tools/lsa/convert_map.py <in.pt> <out.safetensors>
torch.load(weights_only=True) uses the restricted unpickler (no arbitrary code; torch >= 2.6, checked); the result must be a flat dict
"<layer>.W_Q_idx" / "<layer>.W_K_idx" of fp32 [768, 64] for the 16 full-attention layers 3, 7, ..., 63."""
import sys
import torch
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
# weights_only=True runs no pickled code only from torch 2.6 on (CVE-2025-32434 in <= 2.5.1)
if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
    sys.exit(f"convert_map.py needs torch >= 2.6 to load an untrusted .pt safely (this is {torch.__version__})")
d = torch.load(src, map_location="cpu", weights_only=True)
assert isinstance(d, dict), type(d)
want = {f"{l}.W_{w}_idx" for l in range(3, 64, 4) for w in "QK"}
assert set(d) == want, sorted(set(d) ^ want)
out = {}
for k, v in d.items():
    assert isinstance(v, torch.Tensor) and v.dtype == torch.float32 and tuple(v.shape) == (768, 64), (k, v.dtype, v.shape)
    assert torch.isfinite(v).all(), k
    out[k] = v.contiguous().clone()
save_file(out, dst, metadata={"source": src.rsplit("/", 1)[-1], "layout": "[768 = kv0..kv3 x dims 64..255, 64]"})
print(dst, len(out), "tensors")
