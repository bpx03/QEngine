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
- the INT8 n-gram table is materialized inside the checkpoint (`ngram_int8.safetensors`, hard link with a
  copy fallback) unless you convert with `--external-ngram`, which records identity only and expects the
  table to be mounted with `FLASH_NGRAM_PATH`.

The runtime detects the format from the checkpoint itself (manifest present, tensor shapes); nothing in the
image or the binary selects INT8, and INT4 checkpoints keep loading exactly as before.

## Converting

    python3 tools/quant_int8.py --src <bf16 checkpoint dir> --out <int8 dir> \
        --ngram-from <ngram_int8 dir> --kv-from <kv template dir>

Optional: `--external-ngram`, `--max-shard-bytes`, `--search`, `--jobs`. Afterwards verify the output:

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
groups together are roughly 123 GB. At boot the host plan sizes, in order: the host-backed share of the
experts (what the cards cannot keep resident), then the n-gram table, then the KV archive and pinned
prefixes. On a host where the n-gram table does not fit next to the experts it is served from the page
cache instead of being pinned, and the boot log says so (`n-gram table not pinned ... read from the page
cache instead`); nothing fails at load for host RAM. The start-up line prints the plan (`QE_HOST_BUDGET_TABLE=1`
prints the full table). VRAM needs are the same as INT4: resident groups are chosen from the free memory
reported per card at boot.

## Tested modes and performance

Converted and served on 2-card (PP2 x TP2) and 3-card RTX 3090 hosts, decode and prefill, with and without
MTP (`FLASH_MTP_K=0`, K=1, K=2), balanced draft head on, INT8 n-gram table pinned and in page-cache mode.
INT4 checkpoints on the same binaries are unaffected by everything on this page.

Speed and quality against INT4: to be measured. This PR does not claim any equivalence.
