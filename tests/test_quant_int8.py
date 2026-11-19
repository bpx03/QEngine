"""Standalone CPU checks for QFN/27B INT8 checkpoint conversion. Run: python3 tests/test_quant_int8.py

QFN fixtures use the v2 schema: sharded body, self-contained n-gram table by default (external with
--external-ngram), source digest, manifest tensor->shard mapping. Set QENGINE_INT8_REFERENCE to a real
checkpoint directory to additionally run the byte-layout check against real safetensors bytes."""
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
import quant_int8 as quant  # noqa: E402

SCRIPT = os.path.join(TOOLS, "quant_int8.py")


def rand_bf16(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)


def expect_error(fn, contains=""):
    try:
        fn()
    except (ValueError, RuntimeError) as error:
        assert not contains or contains in str(error), f"expected {contains!r} in {error}"
        return
    raise AssertionError("expected conversion/check failure")


def cli(*args, ok=True, env=None):
    environ = dict(os.environ, **(env or {}))
    result = subprocess.run([sys.executable, SCRIPT, *map(str, args)], cwd=ROOT,
                            capture_output=True, text=True, env=environ)
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


def make_args(**kw):
    base = dict(src=None, out="", kv_from="", ngram_from=None, external_ngram=None, link_ngram=False,
                source_repo=None, source_revision=None, max_shard_bytes=quant.MAX_SHARD_BYTES,
                search=0, dev="cpu", check="", n=8)
    base.update(kw)
    return SimpleNamespace(**base)


def qfn_tensors():
    t = {}
    layer = "model.language_model.layers.0."
    mtp = "mtp.layers.0."
    names_2d = [
        layer + f"self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj", "o_proj")
    ] + [
        layer + f"linear_attn.{p}.weight" for p in ("in_proj_qkv", "in_proj_z", "out_proj")
    ] + [
        layer + f"mlp.shared_expert.{p}.weight" for p in ("gate_proj", "up_proj")
    ] + [layer + "mlp.shared_expert.down_proj.weight"] + [
        layer + f"ple.{p}.weight" for p in ("key_proj", "value_proj")
    ]
    for i, name in enumerate(names_2d):
        shape = (128, 256) if ".shared_expert.gate_proj." in name or ".shared_expert.up_proj." in name else (
            (256, 128) if ".shared_expert.down_proj." in name else (256, 256))
        t[name] = rand_bf16(shape, 10 + i)
    t[layer + "mlp.experts.gate_up_proj"] = rand_bf16((3, 256, 256), 30)
    t[layer + "mlp.experts.down_proj"] = rand_bf16((3, 256, 128), 31)
    for i, p in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
        t[mtp + f"self_attn.{p}.weight"] = rand_bf16((256, 256), 40 + i)
    for i, p in enumerate(("gate_proj", "up_proj")):
        t[mtp + f"mlp.shared_expert.{p}.weight"] = rand_bf16((128, 256), 50 + i)
    t[mtp + "mlp.shared_expert.down_proj.weight"] = rand_bf16((256, 128), 52)
    t[mtp + "mlp.experts.gate_up_proj"] = rand_bf16((3, 256, 256), 53)
    t[mtp + "mlp.experts.down_proj"] = rand_bf16((3, 256, 128), 54)  # [E, hidden, moe_intermediate] per text_config
    t["mtp.fc_embedding.weight"] = rand_bf16((256, 512), 55)
    t["mtp.fc_hidden.weight"] = rand_bf16((256, 256), 56)
    t.update({
        "model.language_model.embed_tokens.weight": rand_bf16((4, 256), 60),
        "model.language_model.lm_head.weight": rand_bf16((4, 256), 61),
        layer + "self_attn.q_norm.weight": rand_bf16((32,), 62),
        layer + "mlp.router.weight": rand_bf16((3, 256), 63),
        "model.visual.patch_embed.proj.weight": rand_bf16((2, 2), 64),
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": rand_bf16((3,), 65),
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets": torch.arange(8, dtype=torch.int64),
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes": torch.full((8,), 2, dtype=torch.int64),
        "mtp.layers.0.input_layernorm.weight": rand_bf16((256,), 66),
        "mtp.layers.0.hyper_connection_mixer": rand_bf16((4, 4), 67),
        "fixture.scalar": torch.tensor(13, dtype=torch.int64),
    })
    return t


def qfn_config():
    return {"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": {"model_type": "qwen4_exp", "hidden_size": 256,
                            "num_hidden_layers": 2, "num_experts": 3,
                            "moe_intermediate_size": 128}}

def test_config_derived_qfn_projection_shapes():
    cfg = qfn_config()
    text = cfg["text_config"]
    text.update({
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
        "linear_num_key_heads": 1, "linear_key_head_dim": 64,
        "linear_num_value_heads": 2, "linear_value_head_dim": 64,
        "hc_count": 4,
        "layer_types": ["linear_attention", "full_attention"],
    })
    dims = quant.qfn_dims(cfg)
    body = "model.language_model.layers."
    cases = [
        (body + "1.self_attn.q_proj.weight", (256, 256)),
        (body + "1.self_attn.k_proj.weight", (64, 256)),
        (body + "1.self_attn.v_proj.weight", (64, 256)),
        (body + "1.self_attn.o_proj.weight", (256, 128)),
        (body + "0.linear_attn.in_proj_qkv.weight", (256, 256)),
        (body + "0.linear_attn.in_proj_z.weight", (128, 256)),
        (body + "0.linear_attn.out_proj.weight", (256, 128)),
        (body + "1.ple.key_proj.weight", (1024, 256)),
        (body + "1.ple.value_proj.weight", (256, 256)),
        ("mtp.layers.0.self_attn.q_proj.weight", (256, 256)),
    ]
    for name, shape in cases:
        desc = quant.qfn_descriptor(name)
        quant.validate_qfn_matrix(desc, shape, "BF16", dims, cfg)
    desc = quant.qfn_descriptor(body + "0.linear_attn.out_proj.weight")
    expect_error(lambda: quant.validate_qfn_matrix(desc, (256, 256), "BF16", dims, cfg))
    text["layer_types"][1] = "linear_attention"
    desc = quant.qfn_descriptor(body + "1.self_attn.q_proj.weight")
    expect_error(lambda: quant.validate_qfn_matrix(desc, (256, 256), "BF16", dims, cfg))
    # shared kMtpFcEmbeddingKMultipliers rule: [hidden, hidden] and [hidden, 2*hidden] are the
    # accepted mtp.fc_embedding shapes; anything else must fail (both multipliers, never wider)
    for k in (256, 512):
        quant.validate_qfn_matrix(quant.qfn_descriptor("mtp.fc_embedding.weight"), (256, k), "BF16", dims, cfg)
    for k in (128, 768):
        expect_error(lambda k=k: quant.validate_qfn_matrix(
            quant.qfn_descriptor("mtp.fc_embedding.weight"), (256, k), "BF16", dims, cfg))


def qfn_table_names():
    pre = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
    return [pre + str(i) + ".weight" for i in range(128)]


def external_table(path, rows=2, width=162):
    pre = "model.language_model.layers.1.ple.ple_embedding.ngram_q8.shard_"
    save_file({pre + str(i): torch.zeros((rows, width), dtype=torch.uint8) for i in range(128)}, path)
    return path


def qfn_fixture(root, sharded=False, bad_k=False):
    src = os.path.join(root, "source")
    os.makedirs(src)
    data = qfn_tensors()
    if bad_k:
        data["model.language_model.layers.0.self_attn.q_proj.weight"] = rand_bf16((256, 192), 99)
    for i, name in enumerate(qfn_table_names()):
        data[name] = torch.zeros((2, 160), dtype=torch.bfloat16)
    with open(os.path.join(src, "config.json"), "w") as f:
        json.dump(qfn_config(), f)
    assets = {"tokenizer.json": '{"source":"qfn"}', "tokenizer_config.json": '{"source":"qfn"}',
              "generation_config.json": '{"source":"qfn"}', "chat_template.jinja": "qfn-template",
              "preprocessor_config.json": '{"source":"qfn"}'}
    for name, content in assets.items():
        with open(os.path.join(src, name), "w") as f:
            f.write(content)
    if sharded:
        mixed_name = "model-00001-of-00003.safetensors"
        core_name = "model-00002-of-00003.safetensors"
        table_name = "model-00003-of-00003.safetensors"
        table_keys = qfn_table_names()
        mixed = {"model.language_model.layers.0.self_attn.q_proj.weight":
                 data["model.language_model.layers.0.self_attn.q_proj.weight"], table_keys[0]: data[table_keys[0]]}
        core = {k: v for k, v in data.items() if k not in mixed and k not in table_keys}
        tables = {k: data[k] for k in table_keys[1:]}
        for filename, tensors in ((mixed_name, mixed), (core_name, core), (table_name, tables)):
            save_file(tensors, os.path.join(src, filename))
        weight_map = {key: filename for filename, tensors in
                      ((mixed_name, mixed), (core_name, core), (table_name, tables)) for key in tensors}
        with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": sum(os.path.getsize(os.path.join(src, n))
                                                         for n in (mixed_name, core_name, table_name))},
                       "weight_map": weight_map}, f)
    else:
        save_file(data, os.path.join(src, "model.safetensors"))
    ngram = external_table(os.path.join(root, "ngram_int8.safetensors"))
    return src, ngram, data


# ---------------------------------------------------------------- quantization correctness

def test_hand_computed_group():
    # One 128-value group with 32 hand-computed values (rest zero). absmax is 254 so the BF16 scale is
    # exactly 254/127 = 2.0; every code below is the nearest-even round of value/2 (halves go to even).
    values = [254, -254, 253, -253, 5, -5, 3, -3, 1, -1, 2, -2,
              127, -127, 126, -126, 4, -4, 7, -7, 6, -6, 125, -125,
              0, 0, 252, -252, 9, -9, 11, -11]
    assert len(values) == 32
    expected = [127, -127, 126, -126, 2, -2, 2, -2, 0, 0, 1, -1,
                64, -64, 63, -63, 2, -2, 4, -4, 3, -3, 62, -62,
                0, 0, 126, -126, 4, -4, 6, -6]
    w = torch.zeros((1, 128))
    w[0, :32] = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).float()
    q, s = quant.quant_int8(w.to(torch.bfloat16))
    assert float(s.reshape(-1)[0]) == 2.0
    assert s.reshape(-1)[0].view(torch.uint16).item() == 0x4000  # BF16 bit pattern of 2.0
    assert q[0, :32].tolist() == expected
    assert q[0, 32:].abs().max().item() == 0
    packed = quant.pack_int8(q)
    raw = packed.contiguous().view(torch.uint8).reshape(-1)[:32]
    assert raw.tolist() == [c + 128 for c in expected]  # compressed-tensors offset-0 u8 = code + 128
    # scale/offset invariant for the hand-computed rows: symmetric zero-point is the constant 0
    for value, code in zip(values, expected):
        assert abs(code * 2.0 - value) <= 1.0  # |dequant - w| <= scale/2


def test_quantization_invariant():
    # |dequant - w| <= scale/2 for every group: random inputs, all-zero groups, huge values,
    # rank-3 expert tensors, and adversarial rows pinned to group boundaries.
    cases = []
    for seed, shape in enumerate([(2, 256), (4, 384), (3, 5, 256), (1, 1, 128)]):
        cases.append((rand_bf16(shape, 200 + seed), f"random{seed}"))
    zeros = torch.zeros((3, 256), dtype=torch.bfloat16)
    cases.append((zeros, "zeros"))
    huge = torch.full((2, 128), 3.0e38, dtype=torch.bfloat16)  # near BF16 max
    cases.append((huge, "huge"))
    adversarial = torch.zeros((4, 256), dtype=torch.bfloat16)
    adversarial[0, :128] = 63.5           # every group member equals absmax -> code 127
    adversarial[1, :128] = -63.5
    adversarial[2, 0] = 4.0
    adversarial[2, 1:128] = 0.0           # single nonzero: scale from one entry
    adversarial[3, :128] = torch.linspace(-1.0, 1.0, 128).to(torch.bfloat16)
    cases.append((adversarial, "adversarial"))
    for w, label in cases:
        q, s = quant.quant_int8(w)
        shape = tuple(w.shape)
        k = shape[-1]
        groups = (*shape[:-1], k // 128)
        x = w.float().reshape(*groups, 128)
        deq = q.float().reshape(*groups, 128) * s.float().unsqueeze(-1)
        error = (deq - x).abs()
        bound = s.float().unsqueeze(-1) / 2.0
        # BF16 scale rounding keeps the bound within a 2^-9 relative margin
        assert bool((error <= bound * (1.0 + 1e-2) + 1e-30).all()), f"{label}: dequant error exceeds scale/2"
    expect_error(lambda: quant.quant_int8(torch.full((2, 128), float("nan"), dtype=torch.bfloat16),
                                           name="layer.weight"),
                 contains="layer.weight")
    expect_error(lambda: quant.quant_int8(torch.tensor([[float("inf")] + [0.0] * 127] * 2,
                                                       dtype=torch.bfloat16), name="x.mlp.weight"),
                 contains="x.mlp.weight")


def test_subnormal_scale_floor():
    # a group whose absmax is positive but whose absmax/127 underflows BF16 to 0 must keep nonzero
    # codes (the scale is floored at the smallest positive BF16, 2**-133); only true-zero groups
    # take the recipe's scale-1 fallback
    tiny = torch.full((1, 128), 2.0 ** -133, dtype=torch.bfloat16)
    assert tiny.view(torch.uint16).reshape(-1)[0].item() == 0x0001  # smallest positive BF16 (subnormal)
    assert (tiny.float() / 127.0).to(torch.bfloat16).float().abs().max().item() == 0.0  # scale underflows
    q, s = quant.quant_int8(tiny)
    assert float(s.reshape(-1)[0]) == 2.0 ** -133
    assert q.reshape(-1).min().item() == 1 and q.reshape(-1).max().item() == 1  # constant group -> code 1
    zero = torch.zeros((1, 256), dtype=torch.bfloat16)
    q0, s0 = quant.quant_int8(zero)
    assert torch.equal(s0, torch.ones((1, 2), dtype=torch.bfloat16))  # true-zero group: recipe scale 1, unchanged
    assert q0.abs().max().item() == 0
    mixed = torch.zeros((2, 128), dtype=torch.bfloat16)
    mixed[0] = tiny[0]
    qm, sm = quant.quant_int8(mixed)
    assert qm[0].abs().min().item() == 1 and qm[1].abs().max().item() == 0
    assert float(sm[0]) == 2.0 ** -133 and float(sm[1]) == 1.0


def test_rank3_and_bounded_writer():
    for shape in ((3, 4, 256), (3, 5, 128), (1, 1, 128)):
        w = rand_bf16(shape, sum(shape))
        q, scales = quant.quant_int8(w, 5)
        for expert in range(shape[0]):
            eq, es = quant.quant_int8(w[expert], 5)  # rank-3 must equal per-expert rank-2
            assert torch.equal(q[expert], eq) and torch.equal(scales[expert], es)
        assert tuple(scales.shape) == (*shape[:-1], shape[-1] // 128)
    with tempfile.TemporaryDirectory() as temp:
        source = os.path.join(temp, "source.safetensors")
        key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        tensor = rand_bf16((3, 4, 256), 77)
        save_file({key: tensor}, source)
        src = quant.Src(temp, "source.safetensors")
        desc = quant.qfn_descriptor(key)
        shape = tuple(tensor.shape)
        packed_name = desc["base"] + ".weight_packed"
        scale_name = desc["base"] + ".weight_scale"
        shape_name = desc["base"] + ".weight_shape"
        plan = [(packed_name, "I32", [3, 4, 64], 3 * 4 * 256),
                (scale_name, "BF16", [3, 4, 2], 3 * 4 * 2 * 2),
                (shape_name, "I64", [3], 24)]
        paths = []
        original_get = quant.Src.get
        def no_expert_get(self, name):
            assert ".mlp.experts." not in name, "whole expert tensor requested with get_tensor"
            return original_get(self, name)
        quant.Src.get = no_expert_get
        try:
            for block_rows in (1, 3):
                path = os.path.join(temp, f"block-{block_rows}.safetensors")
                writer = quant.Writer(path, plan)
                quant.write_quant_tensor(writer, src, desc, shape, 5, "cpu", block_rows)
                writer.close()
                paths.append(path)
        finally:
            quant.Src.get = original_get
        a, b = (load_file(path) for path in paths)
        assert all(torch.equal(a[k], b[k]) for k in a)
        assert tuple(a[shape_name].tolist()) == shape
    try:
        quant.quant_int8(torch.zeros((3, 4, 192), dtype=torch.bfloat16))
    except ValueError:
        pass
    else:
        raise AssertionError("K=192 was accepted")


# ---------------------------------------------------------------- QFN conversion (v2 schema)

def output_mapping(out):
    with open(os.path.join(out, "qfn-int8-manifest.json")) as f:
        manifest = json.load(f)
    mapping = {}
    for name, shard in manifest["tensors"]["preserved"]["body"].items():
        mapping[name] = shard
    for name in manifest["tensors"]["preserved"]["mtp"]:
        mapping[name] = manifest["outputs"]["mtp"]
    for record in manifest["tensors"]["converted"]:
        for suffix in (".weight_packed", ".weight_scale", ".weight_shape"):
            mapping[record["base"] + suffix] = record["file"]
    return manifest, mapping


def load_all_outputs(out, files=None):
    with open(os.path.join(out, "qfn-int8-manifest.json")) as f:
        manifest = json.load(f)
    tensors = {}
    for filename in (files or (manifest["outputs"]["body"] + [manifest["outputs"]["mtp"]])):
        for name, tensor in load_file(os.path.join(out, filename)).items():
            assert name not in tensors, f"{name} present in two output files"
            tensors[name] = tensor
    return tensors


def check_qfn_passthrough(out, source_data, excluded):
    converted = {name for name in source_data if quant.qfn_descriptor(name)}
    manifest, mapping = output_mapping(out)
    tensors = load_all_outputs(out)
    body, mtp = set(), set()
    for name, shard in mapping.items():
        (mtp if shard == "model_mtp.safetensors" else body).add(name)
    assert not any(name in body or name in mtp for name in excluded)
    assert not any(key.startswith("mtp.") for key in body)
    assert all(key.startswith("mtp.") for key in mtp)
    assert "model.visual.patch_embed.proj.weight" in body
    assert "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets" in body
    assert "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes" in body
    assert set(tensors) == set(mapping), "manifest mapping does not cover every output tensor exactly once"
    for name, tensor in source_data.items():
        if name in excluded or name in converted:
            continue
        assert torch.equal(tensor, tensors[name]), name
    for name in converted:
        base = name[:-len(".weight")] if name.endswith(".weight") else name
        assert base + ".weight_packed" in tensors and base + ".weight_scale" in tensors
        assert tuple(tensors[base + ".weight_shape"].tolist()) == tuple(source_data[name].shape)


def test_qfn_self_converted_sharded_and_single_file():
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, source_data = qfn_fixture(temp, sharded=True)
        out = os.path.join(temp, "indexed-output")
        original_open, original_get, original_block = quant.safe_open, quant.Src.get, quant.Src.block
        def guarded_open(path, *args, **kwargs):
            assert os.path.basename(path) != "model-00003-of-00003.safetensors", "table-only shard opened"
            return original_open(path, *args, **kwargs)
        def guarded_get(self, name):
            assert not quant.NGRAM_BF16.fullmatch(name), "excluded table payload accessed"
            assert ".mlp.experts." not in name, "whole expert tensor loaded"
            return original_get(self, name)
        def guarded_block(self, name, index):
            assert not quant.NGRAM_BF16.fullmatch(name), "excluded table payload sliced"
            return original_block(self, name, index)
        quant.safe_open, quant.Src.get, quant.Src.block = guarded_open, guarded_get, guarded_block
        try:
            quant.cmd_make(make_args(src=src, out=out, ngram_from=ngram))
        finally:
            quant.safe_open, quant.Src.get, quant.Src.block = original_open, original_get, original_block
        excluded = qfn_table_names()
        check_qfn_passthrough(out, source_data, excluded)
        manifest, mapping = output_mapping(out)
        # self-contained n-gram is the default: table materialized next to the weights, relative name only
        assert manifest["schema"] == "qfn-int8-v2"
        assert manifest["outputs"]["ngram"] == "ngram_int8.safetensors"
        copied = os.path.join(out, "ngram_int8.safetensors")
        assert os.path.isfile(copied) and os.path.getsize(copied) == os.path.getsize(ngram)
        assert not os.path.isabs(manifest["outputs"]["ngram"])
        assert "path" not in manifest["ngram"]
        assert manifest["ngram"]["sha256"] == quant.sha256_file(copied)
        assert manifest["runtime_supported"] is False
        assert "no runtime in this repository executes INT8 W8 checkpoints yet" in manifest["runtime_support_reason"]
        assert len(manifest["tensors"]["excluded"]) == 128
        assert len(manifest["tensors"]["converted"]) == 25
        assert "repo" not in manifest["source"] and "revision" not in manifest["source"]
        assert len(manifest["outputs"]["body"]) >= 1
        assert all(quant.sandbox_relative(n) for n in manifest["outputs"]["body"])

        single_src = os.path.join(temp, "single-source")
        os.makedirs(single_src)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                     "chat_template.jinja", "preprocessor_config.json"):
            with open(os.path.join(src, name), "rb") as inp, open(os.path.join(single_src, name), "wb") as dst:
                dst.write(inp.read())
        save_file(source_data, os.path.join(single_src, "model.safetensors"))
        single_out = os.path.join(temp, "single-output")
        cli("--src", single_src, "--out", single_out, "--ngram-from", ngram,
            "--kv-from", "", "--search", 0, "--dev", "cpu")
        left, right = load_all_outputs(out), load_all_outputs(single_out)
        assert left.keys() == right.keys()
        assert all(torch.equal(left[k], right[k]) for k in left)


def test_sharding_bounds_and_naming():
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp)
        out = os.path.join(temp, "sharded")
        # layer0 alone is ~1,000,240 plan bytes; with the header margin this cap fits layer0
        # but not layer0 + the next (2 KiB) group, forcing exactly the split we want to test
        cap = 1068000
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "",
            "--max-shard-bytes", cap)
        manifest, mapping = output_mapping(out)
        shards = manifest["outputs"]["body"]
        assert len(shards) > 1, "small cap must produce multiple shards"
        assert shards == [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
                          for i in range(len(shards))], "shard naming scheme changed"
        assert manifest["shard_limit_bytes"] == cap
        for shard in shards:
            assert os.path.getsize(os.path.join(out, shard)) <= cap, f"{shard} exceeds the cap"
        # every tensor in exactly one shard and the layer grouping keeps a layer intact
        per_shard = {}
        for name, shard in mapping.items():
            per_shard.setdefault(shard, set()).add(name)
        assert set(per_shard) == set(shards) | {manifest["outputs"]["mtp"]}, "a shard carries no manifest tensors"
        layer0 = {n for n in mapping if n.startswith("model.language_model.layers.0.")}
        shards_of_layer0 = {mapping[n] for n in layer0}
        assert len(shards_of_layer0) == 1, "a layer was split across shards"
        # PLE/n-gram-adjacent preserved tensors ride the final shard
        for name in mapping:
            if "ple_embedding." in name:
                assert mapping[name] == shards[-1], f"{name} not on the final shard"
        cli("--src", src, "--check", out)  # structural check without --src
        cli("--check", out, "--src", src)  # deep check with --src
        cli("--src", src, "--out", os.path.join(temp, "bad-cap"), "--ngram-from", ngram,
            "--kv-from", "", "--max-shard-bytes", 64, ok=False)


def test_external_ngram_identity_and_check():
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp)
        out = os.path.join(temp, "external")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "", "--external-ngram")
        manifest, _ = output_mapping(out)
        assert "ngram" not in manifest["outputs"]
        assert not os.path.exists(os.path.join(out, "ngram_int8.safetensors"))
        assert not os.path.islink(os.path.join(out, "ngram_int8.safetensors"))
        identity = manifest["ngram"]
        assert identity["sha256"] == quant.sha256_file(ngram)
        assert identity["width_bytes"] == 162 and identity["shards"] == 128
        assert identity["equivalence_verified"] is False
        cli("--src", src, "--check", out, "--external-ngram", ngram)
        cli("--src", src, "--check", out, env={"QENGINE_INT8_NGRAM_PATH": ngram})
        cli("--src", src, "--check", out, ok=False)  # external checkpoint, no table path configured
        cli("--src", src, "--check", out, "--external-ngram", os.path.join(temp, "missing.safetensors"), ok=False)
        # tampering the external table breaks the recorded sha256
        tampered = external_table(os.path.join(temp, "tampered.safetensors"), rows=2)
        with safe_open(tampered, "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        first = sorted(tensors)[0]
        view = tensors[first].view(-1)
        view[0] = 7
        save_file(tensors, tampered)
        cli("--src", src, "--check", out, "--external-ngram", tampered, ok=False)
        # self-contained checkpoints refuse --external-ngram at check time
        self_out = os.path.join(temp, "self")
        cli("--src", src, "--out", self_out, "--ngram-from", ngram, "--kv-from", "")
        cli("--src", src, "--check", self_out, "--external-ngram", ngram, ok=False)
        # conversion --external-ngram takes no path
        cli("--src", src, "--out", os.path.join(temp, "bad-external"), "--ngram-from", ngram,
            "--kv-from", "", "--external-ngram", ngram, ok=False)


def test_ngram_copied_by_default_hardlinked_on_request():
    # self-contained packaging defaults to an independent copy: a hard link would let a later write
    # through the source table name mutate the deployed checkpoint, so the shortcut is opt-in
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp)
        out = os.path.join(temp, "copied")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "")
        table = os.path.join(out, "ngram_int8.safetensors")
        assert os.stat(table).st_nlink == 1, "default packaging must copy the table, not share it"
        assert quant.sha256_file(table) == quant.sha256_file(ngram)
        assert os.stat(ngram).st_nlink == 1, "conversion must not link the source table itself"
        cli("--check", out)
        linked = os.path.join(temp, "linked")
        cli("--src", src, "--out", linked, "--ngram-from", ngram, "--kv-from", "", "--link-ngram")
        assert os.stat(os.path.join(linked, "ngram_int8.safetensors")).st_nlink == 2, "--link-ngram must hard link"
        cli("--check", linked)


def test_sandbox_relative_rejects():
    for bad in ("", "/abs/x.safetensors", "../x", "a/../../x", "C:/evil.safetensors",
                "c:\\evil.safetensors", "\\\\server\\share\\x", "~x", "x\0y"):
        assert not quant.sandbox_relative(bad), bad
    for good in ("model-00001-of-00002.safetensors", "sub/dir/table.safetensors", "ngram_int8.safetensors"):
        assert quant.sandbox_relative(good), good


def test_source_digest_and_honest_identity():
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp, sharded=True)  # indexed source -> index_json digest
        out = os.path.join(temp, "sourced")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "",
            "--source-repo", "acme/fork", "--source-revision", "abcdef0123")
        manifest, _ = output_mapping(out)
        assert manifest["source"]["repo"] == "acme/fork"
        assert manifest["source"]["revision"] == "abcdef0123"
        assert manifest["source"]["index_sha256_kind"] == "index_json"
        assert "index_sha256_note" not in manifest["source"]  # exact content digest needs no caveat
        assert manifest["source"]["index_sha256"] == quant.sha256_file(
            os.path.join(src, "model.safetensors.index.json"))
        cli("--src", src, "--check", out)
        # an unlisted source is never rejected: --check does not consult any pin
        single_src, single_ngram, single_data = qfn_fixture(os.path.join(temp, "single"))
        single_out = os.path.join(temp, "single-sourced")
        cli("--src", single_src, "--out", single_out, "--ngram-from", single_ngram, "--kv-from", "")
        manifest, _ = output_mapping(single_out)
        assert manifest["source"]["index_sha256_kind"] == "weight_files"
        assert "best-effort" in manifest["source"]["index_sha256_note"]
        assert "size-preserving" in manifest["source"]["index_sha256_note"]
        assert "repo" not in manifest["source"]
        cli("--src", single_src, "--check", single_out)
        # half-recorded identity is refused at conversion
        cli("--src", single_src, "--out", os.path.join(temp, "half"), "--ngram-from", single_ngram,
            "--kv-from", "", "--source-repo", "acme/fork", ok=False)
        # a rewritten source body changes the recorded digest -> --check --src fails
        corrupt_src = os.path.join(temp, "corrupt")
        os.makedirs(corrupt_src)
        for name in os.listdir(single_src):
            path = os.path.join(single_src, name)
            if os.path.isfile(path):
                with open(path, "rb") as inp, open(os.path.join(corrupt_src, name), "wb") as dst:
                    dst.write(inp.read())
        with open(os.path.join(corrupt_src, "model.safetensors"), "ab") as f:
            f.write(b"x")  # same tensors, different file size: the recorded inventory digest no longer matches
        cli("--src", corrupt_src, "--check", out, ok=False)
        cli("--src", corrupt_src, "--check", single_out, ok=False)


def test_legacy_manifest_clean_break():
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp)
        out = os.path.join(temp, "legacy")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "")
        manifest_path = os.path.join(out, "qfn-int8-manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        # rewrite the manifest the way the pre-sharding converter wrote it
        manifest.pop("schema")
        manifest["outputs"]["body"] = "model.safetensors"
        manifest["outputs"]["ngram"] = ngram  # old absolute external path style
        manifest["source"] = {"repo": "test/example-model",
                              "revision": "da29941aee972c7eeab497e05a292c569da1f1d6"}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        result = cli("--check", out, ok=False)
        assert "schema updated, reconvert" in result.stdout + result.stderr


def test_nan_in_source_refuses_conversion():
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, data = qfn_fixture(temp)
        poisoned = dict(data)
        key = "model.language_model.layers.0.self_attn.q_proj.weight"
        poisoned[key] = data[key].clone()
        poisoned[key][0, 0] = float("nan")
        os.remove(os.path.join(src, "model.safetensors"))
        save_file(poisoned, os.path.join(src, "model.safetensors"))
        out = os.path.join(temp, "nan-output")
        result = cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "", ok=False)
        text = result.stdout + result.stderr
        assert key in text and "NaN" in text, text
        assert not os.path.exists(os.path.join(out, "model-00001-of-00001.safetensors"))


def read_safetensors_header(path):
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    return header, length


def rewrite_safetensors_dtype(path, name, new_dtype):
    """Byte-preserving header rewrite: only the dtype tag changes, payload and offsets stay intact."""
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
        payload = f.read()
    header[name]["dtype"] = new_dtype
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob + payload)


def test_structural_check_verifies_converted_tensor_contracts():
    # the structural --check (no --src) must verify converted tensors' header dtype/shape against
    # the manifest records: a lying header or a lying record fails even without the source
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp)
        out = os.path.join(temp, "contract")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "")
        cli("--check", out)  # the untouched fixture passes the strict structural check
        manifest, mapping = output_mapping(out)
        record = manifest["tensors"]["converted"][0]
        packed = record["outputs"]["packed"]
        shard = os.path.join(out, mapping[packed])
        rewrite_safetensors_dtype(shard, packed, "F32")  # same payload length, corrupted header dtype
        result = cli("--check", out, ok=False)
        assert f"{packed}: header F32" in result.stdout + result.stderr, result.stdout + result.stderr
        rewrite_safetensors_dtype(shard, packed, "I32")
        cli("--check", out)  # the restored header passes again
        # a manifest record lying about the logical shape fails the same source-independent check
        with open(os.path.join(out, "qfn-int8-manifest.json")) as f:
            manifest = json.load(f)
        record = manifest["tensors"]["converted"][0]
        record["shape"] = [record["shape"][0], record["shape"][1] * 128]
        with open(os.path.join(out, "qfn-int8-manifest.json"), "w") as f:
            json.dump(manifest, f)
        result = cli("--check", out, ok=False)
        assert "expected I32" in result.stdout + result.stderr, result.stdout + result.stderr


def test_byte_layout_against_manifest():
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp)
        out = os.path.join(temp, "layout")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "")
        manifest, mapping = output_mapping(out)
        record = manifest["tensors"]["converted"][0]
        for name, expect_dtype, expect_shape in (
                (record["outputs"]["packed"], "I32", record["shape"][:-1] + [record["shape"][-1] // 4]),
                (record["outputs"]["scale"], "BF16", record["shape"][:-1] + [record["shape"][-1] // 128]),
                (record["outputs"]["shape"], "I64", [len(record["shape"])])):
            shard = os.path.join(out, mapping[name])
            header, header_len = read_safetensors_header(shard)
            entry = header[name]
            assert entry["dtype"] == expect_dtype, name
            assert entry["shape"] == expect_shape, name
            begin, end = entry["data_offsets"]
            size = {"I32": 4, "BF16": 2, "I64": 8}[expect_dtype]
            product = 1
            for dim in expect_shape:
                product *= dim
            assert end - begin == product * size, name
            payload_end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
            assert payload_end <= os.path.getsize(shard) - 8, name
            assert begin >= 0 and end <= payload_end, name
    real = os.environ.get("QENGINE_INT8_REFERENCE")
    if not real:
        return
    with open(os.path.join(real, "qfn-int8-manifest.json")) as f:
        manifest = json.load(f)
    body = manifest["outputs"]["body"]
    body = [body] if isinstance(body, str) else body
    record = manifest["tensors"]["converted"][0]
    header, _ = read_safetensors_header(os.path.join(real, body[0]))
    for suffix, dtype in ((".weight_packed", "I32"), (".weight_scale", "BF16"), (".weight_shape", "I64")):
        name = record["base"] + suffix
        if name not in header:
            for candidate in body:
                header, _ = read_safetensors_header(os.path.join(real, candidate))
                if name in header:
                    break
        entry = header[name]
        shape = record["shape"]
        assert entry["dtype"] == dtype
        if suffix == ".weight_shape":
            assert entry["shape"] == [len(shape)]
        elif suffix == ".weight_packed":
            assert entry["shape"] == shape[:-1] + [shape[-1] // 4]
        else:
            assert entry["shape"] == shape[:-1] + [shape[-1] // 128]
        begin, end = entry["data_offsets"]
        assert begin <= end


def write_checkpoint(directory, tensors, config, indexed=False, assets=None):
    os.makedirs(directory)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(config, f)
    if assets:
        for name, text in assets.items():
            with open(os.path.join(directory, name), "w") as f:
                f.write(text)
    if indexed:
        filename = "model-00001-of-00001.safetensors"
        save_file(tensors, os.path.join(directory, filename))
        with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {name: filename for name in tensors}}, f)
    else:
        save_file(tensors, os.path.join(directory, "model.safetensors"))


def test_27b_compatibility_and_family_precedence():
    with tempfile.TemporaryDirectory() as temp:
        weights = {}
        layer = "model.language_model.layers.0."
        names = ([layer + f"self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj", "o_proj")] +
                 [layer + f"linear_attn.{p}.weight" for p in ("in_proj_qkv", "in_proj_z", "out_proj")] +
                 [layer + f"mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")])
        for i, name in enumerate(names):
            weights[name] = rand_bf16((2, 128), 100 + i)
        weights[layer + "self_attn.q_norm.weight"] = rand_bf16((2,), 120)
        weights["mtp.layers.0.stock.weight"] = rand_bf16((2, 128), 121)
        weights["fixture.scalar"] = torch.tensor(9, dtype=torch.int64)
        kv_tensors = {layer + "self_attn.k_scale": torch.tensor([1.25], dtype=torch.float32),
                      layer + "self_attn.v_scale": torch.tensor([2.5], dtype=torch.float32)}
        src_config = {"model_type": "qwen3_5", "marker": "source"}
        kv_config = {"model_type": "qwen3_5", "marker": "kv-reference"}
        source_asset = {"tokenizer.json": "source-tokenizer", "generation_config.json": "source-gen"}
        kv_asset = {"tokenizer.json": "kv-tokenizer", "generation_config.json": "kv-gen"}
        single = os.path.join(temp, "27b-single")
        indexed = os.path.join(temp, "27b-indexed")
        kv = os.path.join(temp, "kv")
        write_checkpoint(single, weights, src_config, assets=source_asset)
        write_checkpoint(indexed, weights, src_config, indexed=True, assets=source_asset)
        write_checkpoint(kv, kv_tensors, kv_config, assets=kv_asset)
        outputs = []
        for i, source in enumerate((single, indexed)):
            out = os.path.join(temp, f"27b-out-{i}")
            cli("--src", source, "--out", out, "--kv-from", kv, "--search", 0)
            outputs.append(out)
            model = load_file(os.path.join(out, "model.safetensors"))
            assert len([key for key in model if key.endswith(".weight_packed")]) == 10
            for name in names:
                base = name[:-len(".weight")]
                q, scale = quant.quant_int8(weights[name], name=name)
                unpacked = model[base + ".weight_packed"].contiguous().view(torch.uint8).reshape(2, 128)
                assert torch.equal(unpacked.to(torch.int16) - 128, q.to(torch.int16))
                assert torch.equal(model[base + ".weight_scale"], scale)
                assert tuple(model[base + ".weight_shape"].tolist()) == (2, 128)
                deq = (unpacked.to(torch.int16) - 128).float().view(2, 1, 128) * scale.float().view(2, 1, 1)
                err = (deq - weights[name].float().view(2, 1, 128)).abs()
                bound = scale.float().view(2, 1, 1) / 2 * (1.0 + 1e-2) + 1e-30
                if not bool((err <= bound).all()):
                    idx = int(err.argmax())
                    flat_q = (unpacked.to(torch.int16) - 128).flatten()
                    raise AssertionError(
                        f"{name}: err={err.flatten()[idx]:.6g} bound={bound.flatten()[idx // 128]:.6g} "
                        f"w={weights[name].float().flatten()[idx]:.6g} q={int(flat_q[idx])} "
                        f"s={float(scale.float().flatten()[idx // 128]):.6g} "
                        f"deq={float(deq.flatten()[idx]):.6g} "
                        f"recheck={float(flat_q[idx]) * float(scale.float().flatten()[idx // 128]):.6g}")
            assert torch.equal(model["fixture.scalar"], weights["fixture.scalar"])
            assert torch.equal(model[layer + "self_attn.k_scale"], kv_tensors[layer + "self_attn.k_scale"])
            assert torch.equal(model[layer + "self_attn.v_scale"], kv_tensors[layer + "self_attn.v_scale"])
            with open(os.path.join(out, "config.json")) as f:
                assert json.load(f)["marker"] == "kv-reference"
            with open(os.path.join(out, "tokenizer.json")) as f:
                assert f.read() == "kv-tokenizer"
            mtp = load_file(os.path.join(out, "model_mtp.safetensors"))
            assert torch.equal(mtp["mtp.layers.0.stock.weight"], weights["mtp.layers.0.stock.weight"])
        a, b = (load_file(os.path.join(out, "model.safetensors")) for out in outputs)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        fallback_out = os.path.join(temp, "27b-default-kv-fallback")
        default_reference = os.path.abspath(os.path.join(TOOLS, "..", "models", "Qwen3.8-27B-INT4"))
        real_exists = quant.os.path.exists
        def hide_default_reference(path):
            return False if os.path.abspath(path) == default_reference else real_exists(path)
        fallback_args = make_args(src=single, out=fallback_out, kv_from=None)
        quant.os.path.exists = hide_default_reference
        try:
            quant.cmd_make_27b(fallback_args, src_config)
        finally:
            quant.os.path.exists = real_exists
        fallback_model = load_file(os.path.join(fallback_out, "model.safetensors"))
        assert not any(name.endswith(("k_scale", "v_scale")) for name in fallback_model)
        with open(os.path.join(fallback_out, "config.json")) as f:
            assert json.load(f)["marker"] == "source"
        with open(os.path.join(fallback_out, "tokenizer.json")) as f:
            assert f.read() == "source-tokenizer"


        wrong_kv = os.path.join(temp, "wrong-kv")
        write_checkpoint(wrong_kv, kv_tensors, qfn_config())
        rejected = os.path.join(temp, "27b-wrong-family")
        cli("--src", single, "--out", rejected, "--kv-from", wrong_kv, ok=False)
        assert not os.path.exists(os.path.join(rejected, "model.safetensors"))


def test_qfn_cli_check_corruption_and_early_failures():
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, source_data = qfn_fixture(temp)
        out = os.path.join(temp, "qfn-output")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "", "--search", 0)
        check_qfn_passthrough(out, source_data, qfn_table_names())
        cli("--check", out)
        cli("--src", src, "--check", out, "--search", 0)
        manifest, mapping = output_mapping(out)
        converted_name = next(name for name in mapping if name.endswith(".weight_packed")
                              and not name.startswith("mtp."))
        shard_file = os.path.join(out, mapping[converted_name])
        original = {name: tensor.clone() for name, tensor in load_file(shard_file).items()}
        corrupted = {name: tensor.clone() for name, tensor in original.items()}
        bytes_view = corrupted[converted_name].contiguous().view(torch.uint8).reshape(-1)
        bytes_view[0] = bytes_view[0] ^ 1
        save_file(corrupted, shard_file)
        cli("--src", src, "--check", out, "--search", 0, ok=False)
        corrupted = {name: tensor.clone() for name, tensor in original.items()}
        scale_name = converted_name[:-len("_packed")] + "_scale"
        corrupted[scale_name].reshape(-1)[0] += 1
        save_file(corrupted, shard_file)
        cli("--src", src, "--check", out, "--search", 0, ok=False)
        # renaming a manifest mapping target: the shard no longer matches the manifest inventory
        save_file(original, shard_file)
        cli("--src", src, "--check", out, "--search", 0)
        with open(os.path.join(out, "qfn-int8-manifest.json")) as f:
            manifest = json.load(f)
        record = manifest["tensors"]["converted"][0]
        other = [s for s in manifest["outputs"]["body"] if s != record["file"]]
        if other:
            record["file"] = other[0]  # tensor claimed by the wrong shard
            with open(os.path.join(out, "qfn-int8-manifest.json"), "w") as f:
                json.dump(manifest, f)
            cli("--check", out, ok=False)
            record["file"] = mapping[record["outputs"]["packed"]]
            with open(os.path.join(out, "qfn-int8-manifest.json"), "w") as f:
                json.dump(manifest, f)
            cli("--check", out)
        # absolute manifest path rejected by --check
        with open(os.path.join(out, "qfn-int8-manifest.json")) as f:
            manifest = json.load(f)
        manifest["outputs"]["body"][0] = os.path.join(out, manifest["outputs"]["body"][0])
        with open(os.path.join(out, "qfn-int8-manifest.json"), "w") as f:
            json.dump(manifest, f)
        cli("--check", out, ok=False)

        bad_k_src, valid_table, _ = qfn_fixture(os.path.join(temp, "bad-k"), bad_k=True)
        bad_k_out = os.path.join(temp, "bad-k-output")
        cli("--src", bad_k_src, "--out", bad_k_out, "--ngram-from", valid_table, "--kv-from", "", ok=False)
        if os.path.isdir(bad_k_out):
            assert not any(name.endswith(".safetensors") for name in os.listdir(bad_k_out))
        no_table_out = os.path.join(temp, "missing-table-output")
        cli("--src", src, "--out", no_table_out, "--kv-from", "", ok=False)
        bad_table = external_table(os.path.join(temp, "bad-layout.safetensors"), width=160)
        bad_table_out = os.path.join(temp, "bad-table-output")
        cli("--src", src, "--out", bad_table_out, "--ngram-from", bad_table,
            "--kv-from", "", ok=False)
        kv_out = os.path.join(temp, "kv-rejected-output")
        cli("--src", src, "--out", kv_out, "--ngram-from", valid_table,
            "--kv-from", os.path.join(temp, "nonempty-kv"), ok=False)


def test_manifest_runtime_flag_contract():
    # No QENGINE_INT8.md is emitted; the manifest carries a boolean runtime_supported and --check
    # validates presence and type. New QFN manifests report that no runtime is available yet.
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp)
        out = os.path.join(temp, "flagged")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "")
        path = os.path.join(out, "qfn-int8-manifest.json")
        assert not os.path.exists(os.path.join(out, "QENGINE_INT8.md")), "converter must not emit QENGINE_INT8.md"
        with open(path) as f:
            manifest = json.load(f)
        assert manifest["runtime_supported"] is False
        assert "no runtime in this repository executes INT8 W8 checkpoints yet" in manifest["runtime_support_reason"]
        cli("--check", out)
        manifest["runtime_supported"] = False
        with open(path, "w") as f:
            json.dump(manifest, f)
        cli("--check", out)  # the contract is presence+boolean, not a pinned value
        manifest["runtime_supported"] = "yes"
        with open(path, "w") as f:
            json.dump(manifest, f)
        result = cli("--check", out, ok=False)
        assert "boolean runtime_supported" in result.stdout + result.stderr


def test_constants_cannot_drift_from_checkpoint_h():
    # the maintainer's point, applied to constants: the python mirrors in tools/quant_int8.py are
    # comments-only copies of src/flash/checkpoint.h, so parse the header and pin them for real
    with open(os.path.join(ROOT, "src", "flash", "checkpoint.h")) as f:
        header = f.read()
    schema = re.search(r'inline const std::string kManifestSchema = "([^"]+)"', header)
    limit = re.search(r"inline constexpr uint64_t kMaxShardBytes = (\d+)ull << (\d+);", header)
    multipliers = re.search(r"inline constexpr std::array<int, \d+> kMtpFcEmbeddingKMultipliers\{([^}]*)\}", header)
    assert schema and limit and multipliers, "constant declarations moved out of the expected shape in checkpoint.h"
    assert schema.group(1) == quant.MANIFEST_SCHEMA
    assert int(limit.group(1)) << int(limit.group(2)) == quant.MAX_SHARD_BYTES
    assert tuple(int(part) for part in multipliers.group(1).split(",")) == quant.MTP_FC_EMBEDDING_K_MULTIPLIERS
    for name, value in (("kBodyMatrices", quant.BODY_MATRICES), ("kMtpMatrices", quant.MTP_MATRICES),
                        ("kLayerCount", quant.LAYER_COUNT)):
        declared = re.search(r"inline constexpr int " + name + r" = (\d+);", header)
        assert declared and int(declared.group(1)) == value, f"{name} drifted from the python mirror"
    rows = re.search(r"inline constexpr long long kNgramRowsPerShard = (\d+);", header)
    assert rows and int(rows.group(1)) == quant.NGRAM_ROWS_PER_SHARD, "kNgramRowsPerShard drifted"
    assert re.search(r"inline constexpr uint64_t kMaxManifestBytes = (\d+)ull << (\d+);", header), \
        "kMaxManifestBytes moved out of the expected shape (C++-only bound, parsed for future mirroring)"


def test_plan_shards_rejects_tensor_larger_than_cap():
    # a single tensor that exceeds --max-shard-bytes would become a lone shard BOTH readers reject:
    # the converter must refuse it at plan time, naming the tensor
    plan = [("model.language_model.embed_tokens.weight", None, None, 200000),
            ("model.language_model.norm.weight", None, None, 10)]
    try:
        quant.plan_shards(plan, 100000)
    except ValueError as error:
        assert "model.language_model.embed_tokens.weight" in str(error), str(error)
        assert "--max-shard-bytes" in str(error), str(error)
    else:
        raise AssertionError("plan_shards accepted a tensor larger than the shard cap")


def test_plan_shards_rejects_oversized_deferred_tensors():
    # the deferred PLE/n-gram-adjacent set rides the final shard as one block: an entry over the cap
    # must be refused naming the tensor, and so must a deferred set whose total exceeds the cap
    oversize = [("model.language_model.layers.1.ple.ple_embedding.huge.weight", None, None, 200000)]
    try:
        quant.plan_shards(oversize, 100000)
    except ValueError as error:
        assert "ple_embedding.huge.weight" in str(error), str(error)
        assert "--max-shard-bytes" in str(error), str(error)
    else:
        raise AssertionError("plan_shards deferred a tensor larger than the shard cap")
    together = [("model.language_model.layers.1.ple.ple_embedding.a.weight", None, None, 60000),
                ("model.language_model.layers.1.ple.ple_embedding.b.weight", None, None, 60000)]
    try:  # each entry fits; the shared final shard would not
        quant.plan_shards(together, 100000)
    except ValueError as error:
        assert "final shard" in str(error) and "--max-shard-bytes" in str(error), str(error)
    else:
        raise AssertionError("plan_shards deferred a set larger than the shard cap")


def test_duplicate_key_manifest_rejected():
    # the C++ loader parses the manifest strictly (Json::parse_strict rejects duplicate keys);
    # --check must refuse the same file instead of letting json.load silently keep one value
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp)
        out = os.path.join(temp, "dupkey")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "")
        path = os.path.join(out, "qfn-int8-manifest.json")
        with open(path) as f:
            raw = f.read()
        assert raw.count('"format_version": 1') == 1
        with open(path, "w") as f:
            f.write(raw.replace('"format_version": 1', '"format_version": 1,\n      "format_version": 1', 1))
        result = cli("--check", out, ok=False)
        assert "duplicate JSON key" in result.stdout + result.stderr
        with open(path, "w") as f:
            f.write(raw)
        cli("--check", out)  # the untouched manifest checks clean again


def test_symlinked_manifest_files_rejected():
    # the sandbox is lexical; both readers must reject a manifest-named regular file that was
    # replaced by a symlink to somewhere else instead of following it
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp, sharded=True)
        out = os.path.join(temp, "symlinked")
        # small cap forces a multi-shard body so the first shard is not the only one
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "", "--max-shard-bytes", 1068000)
        manifest, _ = output_mapping(out)
        assert len(manifest["outputs"]["body"]) > 1, "cap must produce multiple shards"
        for name in (manifest["outputs"]["body"][0], manifest["outputs"]["mtp"],
                     "ngram_int8.safetensors", "qfn-int8-manifest.json", "config.json"):
            path = os.path.join(out, name)
            real = path + ".real"
            os.rename(path, real)
            os.symlink(os.path.basename(real), path)
            try:
                result = cli("--check", out, ok=False)
                assert "symbolic link rejected" in result.stdout + result.stderr, f"{name}: wrong failure"
            finally:
                os.unlink(path)
                os.rename(real, path)
        cli("--check", out)  # the restored checkpoint passes again


def test_intermediate_symlink_escape_rejected():
    # the gate lstats every component below the checkpoint root: a manifest-named path that runs
    # through a symlinked directory must be rejected even though the final component is a regular
    # file (without the per-component walk it would smuggle files from outside the sandbox in)
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp, sharded=True)
        out = os.path.join(temp, "escaping")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "", "--max-shard-bytes", 1068000)
        manifest, _ = output_mapping(out)
        victim = manifest["outputs"]["body"][0]
        moved = f"sub/{victim}"
        outside = os.path.join(temp, "outside")
        os.makedirs(outside)
        os.rename(os.path.join(out, victim), os.path.join(outside, victim))
        os.symlink(outside, os.path.join(out, "sub"))

        def relocate(obj, old, new):  # every manifest reference to the shard goes through "sub"
            if isinstance(obj, str):
                return new if obj == old else obj
            if isinstance(obj, dict):
                return {key: relocate(value, old, new) for key, value in obj.items()}
            if isinstance(obj, list):
                return [relocate(value, old, new) for value in obj]
            return obj

        with open(os.path.join(out, "qfn-int8-manifest.json"), "w") as f:
            json.dump(relocate(manifest, victim, moved), f)
        result = cli("--check", out, ok=False)
        assert "symbolic link rejected" in result.stdout + result.stderr, result.stdout + result.stderr
        os.unlink(os.path.join(out, "sub"))
        os.rename(os.path.join(outside, victim), os.path.join(out, victim))
        with open(os.path.join(out, "qfn-int8-manifest.json")) as f:
            manifest = json.load(f)
        with open(os.path.join(out, "qfn-int8-manifest.json"), "w") as f:
            json.dump(relocate(manifest, moved, victim), f)
        cli("--check", out)  # the restored checkpoint passes again


def test_empty_shard_rejected():
    # the C++ reader fails a shard that maps zero tensors; --check must not pass one
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, _ = qfn_fixture(temp, sharded=True)
        out = os.path.join(temp, "emptied")
        # small cap forces a multi-shard body: victim and other must be different files
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "", "--max-shard-bytes", 1068000)
        manifest, mapping = output_mapping(out)
        shards = manifest["outputs"]["body"]
        assert len(shards) > 1, "cap must produce multiple shards"
        victim, other = shards[-1], shards[0]
        merged = load_file(os.path.join(out, other))
        merged.update(load_file(os.path.join(out, victim)))
        save_file(merged, os.path.join(out, other))
        with open(os.path.join(out, victim), "wb") as f:  # valid safetensors with an empty header
            f.write(len(b"{}      ").to_bytes(8, "little") + b"{}      ")
        with open(os.path.join(out, "qfn-int8-manifest.json")) as f:
            manifest = json.load(f)
        for record in manifest["tensors"]["converted"]:
            if record["file"] == victim:
                record["file"] = other
        preserved = manifest["tensors"]["preserved"]["body"]
        for name in preserved:
            if preserved[name] == victim:
                preserved[name] = other
        with open(os.path.join(out, "qfn-int8-manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        result = cli("--check", out, ok=False)
        assert "maps zero tensors" in result.stdout + result.stderr, result.stdout + result.stderr


def main():
    test_config_derived_qfn_projection_shapes()
    test_hand_computed_group()
    test_quantization_invariant()
    test_subnormal_scale_floor()
    test_rank3_and_bounded_writer()
    test_qfn_self_converted_sharded_and_single_file()
    test_sharding_bounds_and_naming()
    test_external_ngram_identity_and_check()
    test_ngram_copied_by_default_hardlinked_on_request()
    test_sandbox_relative_rejects()
    test_source_digest_and_honest_identity()
    test_legacy_manifest_clean_break()
    test_nan_in_source_refuses_conversion()
    test_byte_layout_against_manifest()
    test_27b_compatibility_and_family_precedence()
    test_qfn_cli_check_corruption_and_early_failures()
    test_constants_cannot_drift_from_checkpoint_h()
    test_manifest_runtime_flag_contract()
    test_plan_shards_rejects_tensor_larger_than_cap()
    test_plan_shards_rejects_oversized_deferred_tensors()
    test_duplicate_key_manifest_rejected()
    test_symlinked_manifest_files_rejected()
    test_intermediate_symlink_escape_rejected()
    test_empty_shard_rejected()
    test_structural_check_verifies_converted_tensor_contracts()
    print("test_quant_int8: ALL PASS")


if __name__ == "__main__":
    main()
