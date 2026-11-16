# Flash-Next INT8 (W8) — format, runtime and review notes

## Checkpoint format: `qfn-int8-v2`

The converter (`tools/quant_int8.py`) and the loader (`src/flash/checkpoint.h`) speak manifest schema
`qfn-int8-v2` (`format_version` 1). Pre-v2 checkpoints are not read: `--check` and `Checkpoint::open` fail with
`schema updated, reconvert`.

- **Sharded body.** `outputs.body` is a list of `model-00001-of-000NN.safetensors` shards, each ≤ 20 GiB, grouped
  per transformer layer (a layer never spans shards; PLE/n-gram-adjacent tensors ride the final shard). Every
  tensor names its shard in the manifest; the loader opens all shards into one index and resolves tensors by name.
- **Self-contained n-gram by default.** The INT8 table is materialized next to the weights as
  `ngram_int8.safetensors` (hard link, copy fallback) and recorded as the relative `outputs.ngram`.
- **External mode (opt-in).** Conversion with `--external-ngram` records no path at all: the manifest keeps the
  table identity (`width_bytes=162`, `shards=128`, `size_bytes`, `rows_per_shard`, layout, keys) plus a `sha256`
  re-verified by `--check`. At runtime the external table comes from `FLASH_NGRAM_PATH` (absolute or
  checkpoint-directory-relative) or an explicit override; the argument and the environment variable resolve
  identically. Checkpoints recorded as self-contained always open their recorded `outputs.ngram` — a configured
  override is reported as ignored, never followed; `--check` uses `--external-ngram PATH` or `QENGINE_INT8_NGRAM_PATH`.
- **Source digest.** `source.index_sha256` is a traceability record of the source (index-file bytes, else a digest
  over sorted `(filename, size)`); integrity comes from `--check --src`, not from the digest. `source.repo`/
  `revision` exist only when `--source-repo/--source-revision` were given. An unlisted source is never a rejection.
- **Hardening.** All manifest paths are sandboxed relative paths; `config.json` is size- and type-gated before
  parsing; the manifest is parsed with a strict JSON grammar; NaN/Inf in any source matrix aborts conversion.
  INT8 vs INT4 detection is by stored logical shape against the packed width (`width*4==K` INT8, `width*8==K`
  INT4), not by `.weight_shape` presence alone — compressed-tensors INT4 carries the same companion. An
  `mtp_override` must be a byte-compatible replacement of the recorded MTP inventory: the header-vs-manifest match
  rejects extra, missing or altered tensors. Format selection never depends on `config.json`.

## Prefill tail-ownership review (2026-11-08)

## Scope and chronology

This report distinguishes the original November 8 review from the later upstream integration. Historical resource constraints below describe that review, not the current deployment.

The original candidate was `pr/flash-int8` at `21e30b8` plus uncommitted changes. No end-to-end performance measurements were taken during that review: the protected `qengine-flash` service occupied all three GPUs. A subsequent review covered runtime and loader changes, corrected build provenance and the description of main's execution order, and built a candidate without a staging-only workaround. Those historical builds and CPU checks did not exercise GPU execution.

## Tail ownership behavior

`Worker::prefill_borrow_tails(format, n, base, staged, extension_lent)` uses:

```cpp
n > base || (format == WeightFormat::Int8 && staged && extension_lent)
```

`prefill_batch` and `prefill_multi` determine staging before assigning `bt_want_` and calling `ec_tick`. When a staged, multichunk INT8 prefill needs the ring extension currently lent to the expert cache, the next tick reclaims the tails. `pfs_full_` can then use the full ring rather than the `pfs_cap0_` limit.

The INT4 predicate remains `n > base`. The ownership change did not alter `prefill_batch_mtp`, checks, `ec_tick`, the `pfs_full_` formula, graph keys, copy events, MTP, quantization, or ring capacities.

In the 1.3.6 integration, staging decisions use upstream's adaptive `pfs_min_rows()` threshold consistently; expert staging addresses and byte accounting retain the format-dependent `group_bytes_` stride.

## Original review evidence

| Check | Observed result |
| --- | --- |
| Isolated candidate build | Exit 0; tag `qengine:flash-int8-prefill-review-20261108T152059Z-r1`, image `sha256:8d439d355c80e72ef784bcb27da4e01282a1481201db6cb4fd3102649cddc316` |
| CPU checkpoint fixture | `flash_checkpoint_unit: ok`, exit 0 |
| INT8 CUDA fixture | Compiled, not executed during the original review |
| Independent prefill diff review | No blocking or should-fix findings; 40 predicate cases checked individually; job offsets, addresses and touches checked manually against `k_jobs_pf` |

The independent review excluded pre-existing loader changes. The later expanded review covered those changes and found no demonstrated runtime or loader defect in the 256-row buffers, cold scratch, or reclaim protocol. This is a review finding, not GPU execution evidence.

Historical staging: `/home/cristian/qe-staging/prefill-review-20261108T152059Z`. Candidate source SHA256 prefixes: `flash.h c1ecf1cf`, `flash.cu fe4895a2`, `flash_int8_unit.cu fa59b7e9`; pre-edit prefixes: `150909be`, `f8660f56`, `e6e8e636` respectively.

## Historical build-context defect and expanded review

The committed `.dockerignore` omitted `!tests/` although `Dockerfile.flash-int8` copied `tests`. The original build therefore required adding that exception in remote staging only. Its successful build must not be presented as a build of the unmodified repository.

The expanded review fixed `.dockerignore` in the repository and made the CUDA fixture's `require` fail immediately by throwing `std::runtime_error`. Invalid counts, descriptors and token indices can no longer continue into allocation, copies, kernels or CPU indexing. Placement equivalence is checked separately for deterministic and nondeterministic `dn_pf` and for `dn_q` tiles 1 and 2, starting each `dn_q` case with zeroed slots.

| Expanded-review check | Observed result |
| --- | --- |
| Throwaway host harness using the actual `require` macro | Before: all three invalid-input paths reached consumers. After: exception, exit 1, consumer unreachable |
| Exact 177-file snapshot | SHA256 verification passed for 177/177 files; archive `c0589335497acf5ba55da50ef6e90b358854c7040c4213a29fca11434bb20c70` |
| Build without remote staging patches | Exit 0; `COPY tests ./tests` executed; image `sha256:571ad83955641db14328b3e15722a0b8f85eea944629d7ef369d101bb7859d25`, tag `qengine:flash-int8-prefill-review-20261108T153717Z-r1` |
| CPU checkpoint fixture | `flash_checkpoint_unit: ok`, exit 0 |
| Protected service before and after | Container `c51d60c9a4f4` remained healthy, with unchanged three-GPU ownership |

Expanded-review staging: `/home/cristian/qe-staging/prefill-review-20261108T153717Z`. Exact source hashes:

- `tests/flash_int8_unit.cu`: `7e47f399da1f973d32321002a77cb52f06afae4569f897e725acfbf32cadc2b3`
- `.dockerignore`: `e015f6271e6bf12886fd10552df4a52df2a4d7bf8f286a0493c69a0d11efc565`
- `Dockerfile.flash-int8`: `023aaae79c940487090c6a78d28ea17653533d5ea953bf86f2adcc4f3f305`

Raw historical artifacts include `build-r1.log`, service-state snapshots, the snapshot archive and checksums, and `require-before.txt` / `require-after.txt` harness transcripts.

## Historical version context and remaining coverage

At the original review, main was 1.3.5 (`c07bfc9`), without the INT8 fixture. Main assigned `bt_want_ = n > pfbn_base_` before `ec_tick`, then decided staging afterward. The earlier report incorrectly reversed that order. Structural portability was not integration proof at that time.

The protected service used image `1.3.5.2-vision` with INT4 checkpoint `Qwen3.8-Flash-Next-QE-1.3.4.1`; the checkpoint suffix was not the engine version.

Original resource-blocked checks included GPU INT8 execution, paired INT8 baseline/candidate and INT4 performance, multi-GPU reclaim/restore and graph replay, loader boundaries 256/272/512/528 and cold boundaries 63/64/65, and the 1M-context/four-slot profile in the rollout handoff below. Later smoke tests or smaller-context campaigns do not establish that 1M/four-slot profile or a paired performance improvement.


## Upstream 1.3.6 integration

PR [#26](https://github.com/bpx03/QEngine/pull/26) incorporates upstream through merge `5f41ec5`, followed by
correction `0502d78`. The integration preserves native INT8 support alongside upstream KV4, packed down-projection,
n-gram worker and slab-set changes.

Observed checks for the merged code:

- `qflash`, `flash_run`, checkpoint and INT8 fixture targets compiled.
- CPU checkpoint and GPU INT8 fixtures printed `ok`.
- Runtime image `qengine:qfn-int8-20261115T161932Z-r4` built successfully with CUDA architecture 86 and vision disabled.
- GitHub reported head `0502d7883cf1c046a9a0f5f2a696c9671f5d21ca` as `MERGEABLE`, with merge state `CLEAN`.

No throughput or latency improvement is inferred from compilation, fixture results, or mergeability.

## Rollout handoff (2026-11-08)

## Estado

Trabajo pausado por el usuario. Todos los contenedores de inferencia quedaron detenidos y las tres RTX 3090 libres. Se conservaron los contenedores y logs; no se borraron modelos ni checkpoints.

- `qengine-flash-next-int8`: detenido; última configuración que sí llegó a `ready`, con contexto 32768 y 2 slots. No es la configuración INT4 que pidió el usuario.
- `qengine-flash-next-int8-1m-trial-partial-20261108`: detenido durante la carga con contexto 1M y 4 slots; no llegó a health/readiness.
- `qengine-flash-next-int8-failed-no-processor-config-20261108T124854Z`: primer intento detenido; `--vision 1` falló porque el modelo no contiene `processor_config.json`.
- `qengine-orca27b-int8` y `qengine-flash`: detenidos y preservados.
- Sin listeners en 8000/9393 al aparcar los contenedores; GPU 0/1/2 en uso residual de 46/15/15 MiB.

## Cambios INT8 implementados

- `src/flash/flash.cu`: prefetch de expertos INT8 reutilizando el camino existente; expertos fríos se repackean en scratch VRAM y se copian en bloques contiguos a RAM pinned; carga de pesos W8 packed por lotes con doble buffer y eventos, conservando códigos y escalas.
- `tests/flash_int8_unit.cu`: pruebas de repack/copia y conservación byte-a-byte, además del fixture de checkpoint.
- `Dockerfile.flash-int8`: compila y copia `flash_checkpoint_unit` y `flash_int8_unit` a la imagen runtime.
- Imagen construida en el host: `qengine:flash-int8-optimized`, ID `sha256:2cc497d2bc1e18a8e5dc3b29f8c27f0bc4f356dd2040d6c3a840a2a2aecd36dc`.
- Build staging remoto: `/home/cristian/qe-int8-optimized-20261108T122731Z`.

## Verificación completada

- `flash_checkpoint_unit`: `ok`.
- `flash_int8_unit`: `ok` con `INT8_MODEL=/Proyectos/QEngine/models/Qwen3.8-Flash-Next-Uncensored-INT8-QEngine`.
- Perfil INT8 de 32768 tokens: llegó a `ready` en 273 s; `/health` devolvió `{"status":"ok"}`; `/v1/models` identificó `qwen3.8-flash-next-qengine`.
- Dos respuestas SSE terminaron en `[DONE]`: cálculo `7*8` → `56`; respuesta de código produjo `def suma(a, b): return a + b`. Los logs mostraron MTP activo.
- Este smoke test usó contexto 32768, 2 slots, KV en VRAM, archivo/prefijos en 0 y margen 512/2048/512. No valida los parámetros 1M solicitados.
- En ese arranque INT8: tabla n-gram INT8 pinned 51.8 GB; expertos fríos host 24.14 + 21.76 + 24.11 = ~70.01 GB; MemAvailable después del arranque ~36 GB.

## Parámetros pedidos para la siguiente prueba

Aplicar al checkpoint INT8; conservar la ruta de modelo, 3 GPUs y puertos del perfil anterior:

```text
QE_MODEL=/Proyectos/QEngine/models/Qwen3.8-Flash-Next-Uncensored-INT8-QEngine
QE_GPUS=0,1,2
FLASH_MTP_K=3
FLASH_MAX_SEQS=4
FLASH_KV=ram
FLASH_KV_RAM_TOKENS=1M
FLASH_KV_ARCHIVE_GB=48
FLASH_PREFIX_GB=8
FLASH_OOM_RESERVE_MB=64
FLASH_PIN_VRAM_MB_PER_GB=2
FLASH_NGRAM_PIN=1
FLASH_EC_MARGIN_MB=512,512,512
FLASH_COMM=auto
QE_DECODE_COMM=auto
--max-ctx 1M
```

Puertos: `8000:8080` y `127.0.0.1:9393:9393`; `--ipc=host`; `memlock` ilimitado; modelo montado solo lectura. La prueba 1M usó además `--vision 0`: el checkpoint INT8 no tiene `processor_config.json`, así que no puede arrancar con `--vision 1` como el contenedor INT4. `/v1/models` reportó modalidad `text`.

## Intento 1M interrumpido

El contenedor con los parámetros pedidos llegó a asignar KV en RAM de 1.05M tokens (~14.2 GB pinned) y comenzó la carga de expertos. El health waiter expiró a los 15 minutos sin readiness; el usuario reportó que no avanzaba. El último progreso guardado alcanzaba la carga de `layer-23`; se detuvo para liberar las GPUs. **No se verificó** el startup completo, el presupuesto efectivo del archivo de 48 GB, el contexto servido de 1M ni generación con esta configuración.

No comparar los 273 s del perfil 32K con este intento parcial como benchmark: configuración y memoria distintas.

## Evidencia remota

Resultados completos: `/home/cristian/qe-int8-results-20261108T124444Z/`.

- `unit-tests.log`
- `new.startup.log`, `chat-math.sse`, `chat-code.sse`, `new.smoke.log`
- `corrected.final.log`, `corrected.final.inspect.json`
- `corrected.container-id`, `new.container-id`

Próxima sesión: reanudar desde los logs `corrected.final.log`; probar primero el arranque con los parámetros del bloque anterior y vigilar el progreso de carga y `MemAvailable`. No activar visión para este checkpoint hasta disponer de `processor_config.json` compatible.
