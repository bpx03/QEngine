"""Long-prompt prefill precision: run with QE_GEMM_H16=0 and =1 (separate processes) and compare.
Usage: python3 test_prefill_precision.py save <file.npz> | compare <a.npz> <b.npz>"""
import ctypes, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
if sys.argv[1] == "compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for name in a.files:
        if name.startswith("lg"):
            x, y = a[name].astype(np.float64), b[name].astype(np.float64)
            cos = (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())
            print(f"{name}: logits cos {cos:.6f}, argmax {x.argmax()} vs {y.argmax()}")
        else:
            ga, gb = a[name], b[name]
            d = next((i for i in range(len(ga)) if ga[i] != gb[i]), len(ga))
            print(f"{name}: greedy agrees for {d}/{len(ga)} tokens")
    sys.exit()
import common as T
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4/tokenizer.json"))
import transformers  # fixed text (not this repo's sources, which change between versions)
src = "".join(open(os.path.join(os.path.dirname(transformers.__file__), "generation", f)).read()
              for f in ("utils.py", "logits_process.py"))
e = T.lib.qe_create(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4").encode(), b"0,1,2", 32768, b"")
res = {}
for n in (1500, 6000):
    text = "<|im_start|>user\nExplain this code, then suggest improvements:\n" + src
    ids = tok.encode(text).ids[:n] + tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
    T.lib.qe_start(e, T.ints(ids), len(ids), 0)
    lg = np.zeros(248320, dtype=np.float32)
    T.lib.qe_logits(e, lg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
    res[f"lg{n}"] = lg
    res[f"gen{n}"] = np.array(T.generate(e, ids, 0, 200)[0])
np.savez(sys.argv[2], **res)
T.lib.qe_destroy(e)
