"""MTP == plain greedy at a long context (k = 7: verification groups > 32 rows, attn_dec8d MT=3)."""
import ctypes, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import common as T
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4/tokenizer.json"))
import transformers
d = os.path.dirname(transformers.__file__)
corpus = tok.encode("".join(open(f).read() for f in sorted(glob.glob(os.path.join(d, "generation/*.py"))))).ids
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
ids = tok.encode("<|im_start|>user\n").ids + corpus[:n] + tok.encode("\n\nExplain the code above.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
e = T.lib.qe_create(os.path.join(T.ROOT, "models/Qwen3.8-27B-INT4").encode(), b"0,1,2", n + 2048, b"")
plain, _, _ = T.generate(e, ids, 0, 300)
spec, rate, tps = T.generate(e, ids, 7, 300)
same = next((i for i in range(300) if plain[i] != spec[i]), 300)
print(f"ctx {len(ids)}: spec==plain for {same}/300 tokens ({tps:.2f} tok/step)")
print("PASS" if same == 300 else "FAIL")
T.lib.qe_destroy(e)
