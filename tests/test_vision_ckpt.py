"""CPU checks for the vision weight loader (server/vision.py visual_weight_map) over the three checkpoint
layouts: safetensors index (bf16/int4 conversions), INT8 sharded body without an index (qfn-int8 checkpoints)
and a single model.safetensors. Run: python3 tests/test_vision_ckpt.py"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from vision import visual_weight_map  # noqa: E402


def write_st(path, tensors):
    save_file({k: v for k, v in tensors.items()}, path)


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}", flush=True)
    if not ok:
        sys.exit(1)


with tempfile.TemporaryDirectory() as d:
    # sharded INT8 layout: no index, tensors spread over model-NNNNN-of-NNNNN.safetensors (+ an mtp file)
    write_st(os.path.join(d, "model-00001-of-00002.safetensors"),
             {"model.language_model.embed_tokens.weight": torch.zeros(2, 2),
              "model.visual.patch_embed.proj.weight": torch.ones(2, 2)})
    write_st(os.path.join(d, "model-00002-of-00002.safetensors"),
             {"model.visual.merger.norm.weight": torch.ones(2), "model.visual.blocks.0.norm1.weight": torch.ones(2)})
    write_st(os.path.join(d, "model_mtp.safetensors"), {"mtp.fc.weight": torch.ones(2, 2)})
    wm = visual_weight_map(d)
    check("sharded, no index", wm.get("model.visual.patch_embed.proj.weight") == "model-00001-of-00002.safetensors"
          and wm.get("model.visual.merger.norm.weight") == "model-00002-of-00002.safetensors"
          and wm.get("mtp.fc.weight") == "model_mtp.safetensors",
          f"({len(wm)} tensors)")

    # single-file layout (the pre-sharding quant_int8 output)
    with tempfile.TemporaryDirectory() as d1:
        write_st(os.path.join(d1, "model.safetensors"), {"model.visual.pos_embed.weight": torch.ones(2)})
        check("single file", visual_weight_map(d1).get("model.visual.pos_embed.weight") == "model.safetensors")
        # a directory without an index and without visual tensors names itself in the error
        with tempfile.TemporaryDirectory() as d2:
            open(os.path.join(d2, "config.json"), "w").write("{}")
            try:
                visual_weight_map(d2)
                check("empty dir refuses", False)
            except FileNotFoundError as ex:
                check("empty dir refuses", str(d2) in str(ex))

    # the index wins when present (a converted dir keeps the converter's map, not a re-scan)
    with tempfile.TemporaryDirectory() as d3:
        write_st(os.path.join(d3, "base.safetensors"), {"model.visual.patch_embed.proj.weight": torch.ones(2)})
        json.dump({"weight_map": {"model.visual.patch_embed.proj.weight": "base.safetensors",
                                  "gone.weight": "other.safetensors"}},
                  open(os.path.join(d3, "model.safetensors.index.json"), "w"))
        check("index preferred", visual_weight_map(d3) == json.load(
            open(os.path.join(d3, "model.safetensors.index.json")))["weight_map"])

print("vision weight loader: all layouts OK", flush=True)
