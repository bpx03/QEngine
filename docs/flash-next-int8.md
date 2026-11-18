# Flash-Next INT8 (W8) checkpoints

This page is for users of Qwen3.8-Flash-Next: what the INT8 checkpoint format is, how to convert a
checkpoint, what the runtime needs from the machine, and which modes have been tested.

## The format

The converter (`tools/quant_int8.py`) writes a self-contained checkpoint directory with manifest schema
`qfn-int8-v2`:

- `model-00001-of-000NN.safetensors` body shards, each at most 20 GiB, grouped by transformer layer;
- converted matrices carry `.weight_packed` (offset-binary INT8 codes), `.weight_scale` (BF16, one scale
  per 128-wide group) and `.weight_shape`; every other tensor is stored byte-identical to the source;
- `qfn-int8-manifest.json` maps every tensor to its shard, records the source digests, and carries the
  n-gram table identity with a `sha256`;
- the INT8 n-gram table is copied into the checkpoint (`ngram_int8.safetensors`) by default; `--link-ngram`
  hard-links it instead, and `--external-ngram` records the identity only and expects the table to be
  mounted with `FLASH_NGRAM_PATH`.

The runtime detects the format from the checkpoint itself (manifest present, tensor shapes); nothing in the
image or the binary selects INT8, and INT4 checkpoints keep loading exactly as before.

## Converting

    python3 tools/quant_int8.py --src <bf16 checkpoint dir> --out <int8 dir> \
        --ngram-from <ngram_int8 dir> --kv-from <kv template dir>

Optional: `--link-ngram`, `--external-ngram`, `--max-shard-bytes`, `--search`, `--dev`. Afterwards verify
the output:

    python3 tools/quant_int8.py --check <int8 dir>            # shards, manifest, n-gram identity
    python3 tools/quant_int8.py --check <int8 dir> --src <dir>  # also re-quantizes and compares codes/scales

`--check` is the acceptance gate: a directory it rejects is not served.

## Serving

Serve the directory like any other checkpoint (engine config or `--model` path); the loader reads the
manifest, opens the shards and selects the W8 kernels per checkpoint. Useful overrides: `FLASH_NGRAM_PATH`
(external table), `FLASH_MTP_FILE` / `FLASH_MTP_K` (MTP weights and draft depth), `FLASH_NGRAM_PIN=1`
(force-pin the n-gram table).

## What the machine needs

INT8 expert groups are 998,400 B per 128-wide group (against 506,880 B at INT4), so all routed expert
groups together are roughly 123 GB. The cards keep only what fits in their free VRAM next to everything
else; the rest is pinned host RAM: about **92 GB on a 2-card host and about 72 GB on a 3-card host**, plus
the base the server itself needs (process, pinned embeddings, KV archive and pinned prefixes, and the
51.8 GB n-gram table while it stays pinned).

The experts are the one budget the start-up plan never reduces: when the host-backed experts do not fit,
the plan says so at boot (a warning naming the experts' GB against the host's free RAM) and the pin guard
refuses the load when the memory is really not there - the same rule INT4 follows. What does degrade is
the n-gram table: on a host where it does not fit next to the experts it is served from the page cache
instead of pinned, and the boot log says so (`n-gram table not pinned ... read from the page cache
instead`). The start-up line prints the plan either way; the full plan table is printed by the CPU unit test
`tests/host_budget_unit.cpp` when run with `QE_HOST_BUDGET_TABLE=1` (a test knob, not a server option). VRAM needs
are the same as INT4: resident groups are chosen from the free memory reported per card at boot.

## Tested modes and performance

Converted and served on a five-rank tensor-parallel RTX 3090 host (5 x 24 GB): decode and prefill
benchmarks, with and without MTP drafts (`FLASH_MTP_K` = 0, 1, 2), the INT8 n-gram table pinned. Other
parallel layouts exercise the same W8 code paths but have not been benchmarked on an INT8 checkpoint.

Speed and quality against INT4: to be measured.

See also the README's [QFN INT8 (W8) checkpoints](../README.md#qfn-int8-w8-checkpoints) section for the
conversion summary and the surrounding engine options.
