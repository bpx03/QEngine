"""Standalone CPU checks for QFN/27B INT8 checkpoint conversion. Run: python3 tests/test_quant_int8.py"""
import json
import os
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


def expect_error(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("expected conversion/check failure")


def cli(*args, ok=True):
    result = subprocess.run([sys.executable, SCRIPT, *map(str, args)], cwd=ROOT,
                            capture_output=True, text=True)
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


def reference_int8(w, search=0):
    shape = tuple(w.shape)
    n, k = shape[-2:]
    x = w.float().reshape(-1, n, k // 128, 128)
    maximum = x.abs().amax(dim=-1, keepdim=True)
    best_q = None
    best_s = None
    best_error = None
    for i in range(max(1, search)):
        ratio = 1.0 if i == 0 else 1.0 - 0.20 * i / max(1, search - 1)
        scale = (maximum * ratio / 127.0).to(torch.bfloat16).float()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round(x / scale).clamp(-127, 127)
        error = ((codes * scale - x) ** 2).sum(dim=-1, keepdim=True)
        if best_error is None:
            best_q, best_s, best_error = codes, scale, error
        else:
            choose = error < best_error
            best_q = torch.where(choose, codes, best_q)
            best_s = torch.where(choose, scale, best_s)
            best_error = torch.where(choose, error, best_error)
    return (best_q.reshape(shape).to(torch.int8),
            best_s.reshape(*shape[:-1], k // 128).to(torch.bfloat16))


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
    t[mtp + "mlp.experts.down_proj"] = rand_bf16((3, 256, 128), 54)
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


def test_2d_reference_and_search():
    w = torch.zeros((2, 256), dtype=torch.bfloat16)
    w[0, :128] = torch.linspace(-2, 3, 128).to(torch.bfloat16)
    w[0, 0] = -40
    w[0, 128:] = 0
    w[1, :128] = torch.linspace(-1, 1, 128).to(torch.bfloat16)
    w[1, 128:] = torch.linspace(-4, 2, 128).to(torch.bfloat16)
    q, s = quant.quant_int8(w)
    rq, rs = reference_int8(w)
    assert torch.equal(q, rq) and torch.equal(s, rs)
    packed = quant.pack_int8(torch.tensor([[-127, -1, 0, 1] + [0] * 124], dtype=torch.int8))
    raw = packed.contiguous().view(torch.uint8).reshape(-1)
    assert raw[:4].tolist() == [1, 127, 128, 129]
    baseline_q, baseline_s = quant.quant_int8(w, 1)
    search_q, search_s = quant.quant_int8(w, 9)
    base_error = ((baseline_q.float().view(2, 2, 128) * baseline_s.float().unsqueeze(-1) -
                   w.float().view(2, 2, 128)) ** 2).sum(-1)
    search_error = ((search_q.float().view(2, 2, 128) * search_s.float().unsqueeze(-1) -
                     w.float().view(2, 2, 128)) ** 2).sum(-1)
    assert torch.all(search_error <= base_error + 1e-7)


def test_rank3_and_bounded_writer():
    for shape in ((3, 4, 256), (3, 5, 128), (1, 1, 128)):
        w = rand_bf16(shape, sum(shape))
        q, scales = quant.quant_int8(w, 5)
        rq, rs = reference_int8(w, 5)
        assert torch.equal(q, rq) and torch.equal(scales, rs)
        unpacked = quant.pack_int8(q).contiguous().view(torch.uint8).reshape(shape)
        assert torch.equal(unpacked.to(torch.int16) - 128, q.to(torch.int16))
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


def check_qfn_passthrough(out_dir, source_data, excluded):
    converted = {name for name in source_data if quant.qfn_descriptor(name)}
    with safe_open(os.path.join(out_dir, "model.safetensors"), "pt") as body, \
            safe_open(os.path.join(out_dir, "model_mtp.safetensors"), "pt") as mtp:
        body_keys, mtp_keys = set(body.keys()), set(mtp.keys())
        assert not any(name in body_keys or name in mtp_keys for name in excluded)
        assert not any(key.startswith("mtp.") for key in body_keys)
        assert all(key.startswith("mtp.") for key in mtp_keys)
        assert "model.visual.patch_embed.proj.weight" in body_keys
        assert "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets" in body_keys
        assert "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes" in body_keys
        for name, tensor in source_data.items():
            if name in excluded or name in converted:
                continue
            target = mtp if name.startswith("mtp.") else body
            assert torch.equal(tensor, target.get_tensor(name)), name


def test_qfn_indexed_filtering_and_single_file():
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
            quant.cmd_make(SimpleNamespace(src=src, out=out, kv_from="", ngram_from=ngram,
                                           search=0, dev="cpu"))
        finally:
            quant.safe_open, quant.Src.get, quant.Src.block = original_open, original_get, original_block
        excluded = qfn_table_names()
        check_qfn_passthrough(out, source_data, excluded)
        assert not os.path.exists(os.path.join(out, "ngram_int8.safetensors"))
        assert not os.path.islink(os.path.join(out, "ngram_int8.safetensors"))
        with open(os.path.join(out, "qfn-int8-manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["runtime_supported"] is False
        assert manifest["ngram_external"]["path"] == os.path.abspath(ngram)
        assert manifest["ngram_external"]["equivalence_verified"] is False
        assert len(manifest["tensors"]["excluded"]) == 128
        assert len(manifest["tensors"]["converted"]) == 25

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
        for filename in ("model.safetensors", "model_mtp.safetensors"):
            left, right = (load_file(os.path.join(folder, filename)) for folder in (out, single_out))
            assert left.keys() == right.keys()
            assert all(torch.equal(left[k], right[k]) for k in left)


def test_qfn_cli_check_corruption_and_early_failures():
    with tempfile.TemporaryDirectory() as temp:
        src, ngram, source_data = qfn_fixture(temp)
        out = os.path.join(temp, "qfn-output")
        cli("--src", src, "--out", out, "--ngram-from", ngram, "--kv-from", "", "--search", 0)
        check_qfn_passthrough(out, source_data, qfn_table_names())
        cli("--src", src, "--check", out, "--search", 0)
        model_path = os.path.join(out, "model.safetensors")
        original = {name: tensor.clone() for name, tensor in load_file(model_path).items()}
        converted_name = next(name for name in original if name.endswith(".weight_packed"))
        corrupted = {name: tensor.clone() for name, tensor in original.items()}
        bytes_view = corrupted[converted_name].contiguous().view(torch.uint8).reshape(-1)
        bytes_view[0] = bytes_view[0] ^ 1
        save_file(corrupted, model_path)
        cli("--src", src, "--check", out, "--search", 0, ok=False)
        corrupted = {name: tensor.clone() for name, tensor in original.items()}
        scale_name = converted_name[:-len("_packed")] + "_scale"
        corrupted[scale_name].reshape(-1)[0] += 1
        save_file(corrupted, model_path)
        cli("--src", src, "--check", out, "--search", 0, ok=False)
        restored = model_path + ".restore"
        save_file(original, restored)
        os.replace(restored, model_path)
        cli("--src", src, "--check", out, "--search", 0)

        bad_k_src, valid_table, _ = qfn_fixture(os.path.join(temp, "bad-k"), bad_k=True)
        bad_k_out = os.path.join(temp, "bad-k-output")
        cli("--src", bad_k_src, "--out", bad_k_out, "--ngram-from", valid_table, "--kv-from", "", ok=False)
        assert not os.path.exists(os.path.join(bad_k_out, "model.safetensors"))
        no_table_out = os.path.join(temp, "missing-table-output")
        cli("--src", src, "--out", no_table_out, "--kv-from", "", ok=False)
        assert not os.path.exists(os.path.join(no_table_out, "model.safetensors"))
        bad_table = external_table(os.path.join(temp, "bad-layout.safetensors"), width=160)
        bad_table_out = os.path.join(temp, "bad-table-output")
        cli("--src", src, "--out", bad_table_out, "--ngram-from", bad_table,
            "--kv-from", "", ok=False)
        assert not os.path.exists(os.path.join(bad_table_out, "model.safetensors"))
        kv_out = os.path.join(temp, "kv-rejected-output")
        cli("--src", src, "--out", kv_out, "--ngram-from", valid_table,
            "--kv-from", os.path.join(temp, "nonempty-kv"), ok=False)
        assert not os.path.exists(os.path.join(kv_out, "model.safetensors"))


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
                q, scale = reference_int8(weights[name])
                unpacked = model[base + ".weight_packed"].contiguous().view(torch.uint8).reshape(2, 128)
                assert torch.equal(unpacked.to(torch.int16) - 128, q.to(torch.int16))
                assert torch.equal(model[base + ".weight_scale"], scale)
                assert tuple(model[base + ".weight_shape"].tolist()) == (2, 128)
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
        fallback_args = SimpleNamespace(src=single, out=fallback_out, kv_from=None, search=0, dev="cpu")
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


def main():
    test_2d_reference_and_search()
    test_rank3_and_bounded_writer()
    test_qfn_indexed_filtering_and_single_file()
    test_qfn_cli_check_corruption_and_early_failures()
    test_27b_compatibility_and_family_precedence()
    test_config_derived_qfn_projection_shapes()
    print("test_quant_int8: ALL PASS")


if __name__ == "__main__":
    main()
