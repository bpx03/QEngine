# Flash-Next INT8 prefill — revisión de ownership de tails (2026-11-08)

Candidato revisado `pr/flash-int8` (HEAD `21e30b8` + cambios sin commitear). Este documento registra el cambio de comportamiento y la evidencia observada. **Ninguna medición de rendimiento end-to-end fue realizada**: las tres GPU quedaron ocupadas por el servicio protegido `qengine-flash` durante toda la sesión; véanse los gates bloqueados.

Una sesión posterior añadió la sección **Ampliación**: revisión ampliada (incluye runtime/loader, sin defecto demostrado), correcciones a este documento (alcance de la revisión histórica, procedencia del build histórico, orden real en main) y un candidato nuevo construido ya sin el workaround de staging. Sigue sin ejecutarse nada en GPU: build/checkpoint solo prueban compilación y validación CPU.

## Cambio de comportamiento

- `Worker::prefill_borrow_tails(format, n, base, staged, extension_lent)` (flash.h, junto a `pf_batch_base`): `n > base || (format == Int8 && staged && extension_lent)`.
- `prefill_batch` y `prefill_multi` calculan ahora `pfs_min`/`pfs_now_` **antes** de `bt_want_`/`ec_tick` y asignan `bt_want_` con ese predicado compartido. Un prefill INT8 que habilita staging (`FLASH_PF_STAGE_MIN`, multichunk) con la extensión del ring prestada a la caché de expertos reclama los tails en el tick siguiente, de modo que `pfs_full_` usa el ring completo en lugar del tope `pfs_cap0_`.
- Sin cambios: INT4 (el predicado se reduce a `n > base`), `prefill_batch_mtp`, CHECKs, `ec_tick`, fórmula de `pfs_full_`, graph keys, eventos de copia, MTP, cuantización, caps del ring.

## Verificado (evidencia observada)

| Verificación | Comando | Resultado |
| --- | --- | --- |
| Compilación del candidato | `docker build -f Dockerfile.flash-int8 --build-arg BASE_RUNTIME=ghcr.io/bpx03/qengine:1.3.5.2-vision` en staging aislado | RC=0 (`build-r1.log`), image ID `sha256:8d439d355c80e72ef784bcb27da4e01282a1481201db6cb4fd3102649cddc316`, tag `qengine:flash-int8-prefill-review-20261108T152059Z-r1`. **Procedencia: el árbol entonces fallaba `COPY tests` por el `.dockerignore` roto; este build exigió agregar `!tests/` solo en el staging remoto** (no en el repo) — ver "Defecto de herramienta" |
| Checkpoint INT8 (CPU-only) | `docker run --rm --entrypoint /app/build/flash_checkpoint_unit 8d439d355c80` | `flash_checkpoint_unit: ok`, exit 0 |
| `flash_int8_unit` | compilado en el build anterior | compila; **ejecución no realizada** (gate GPU) |
| Revisión read-only | reviewer independiente sobre el diff de prefill de esa sesión | sin bloqueantes ni should-fix; matrices de 40 filas del predicado verificadas una a una; expectativas de jobs (`off/n/aoff/ng`, direcciones, touches) reproducidas a mano contra `k_jobs_pf`. **Alcance real: dejó fuera los cambios preexistentes del loader**, que esa revisión no cubrió (la revisión ampliada los cubre: ver Ampliación) |

Snapshot de las tres fuentes candidatas (SHA256 en staging `/home/cristian/qe-staging/prefill-review-20261108T152059Z`): `flash.h c1ecf1cf…`, `flash.cu fe4895a2…`, `flash_int8_unit.cu fa59b7e9…`. Baseline pre-edición local: `150909be…`, `f8660f56…`, `e6e8e636…`.

## Gates bloqueados por recurso externo (no fallos)

- **`flash_int8_unit` en GPU**: GPU2 (`GPU-667bb12e-9274-48fd-d04a-fcd6bf7bc374`) ocupada por pid 12649 del contenedor protegido `qengine-flash` (23.6 GiB; consultado 15:14 y 15:25 UTC en la sesión histórica, y 15:37/15:41 UTC antes/después del build nuevo, igual; servicio `{"status":"ok"}` intacto; VRAM libre <2 GiB → BLOQUEADO, no intentado). La receta queda pendiente, ahora contra el candidato nuevo: `docker run --rm --gpus device=GPU-667bb12e-9274-48fd-d04a-fcd6bf7bc374 --entrypoint /app/build/flash_int8_unit --mount type=bind,src=/Proyectos/QEngine/models,dst=/Proyectos/QEngine/models,readonly -e INT8_MODEL=/Proyectos/QEngine/models/Qwen3.8-Flash-Next-Uncensored-INT8-QEngine sha256:571ad83955641db14328b3e15722a0b8f85eea944629d7ef369d101bb7859d25`, previa re-verificación de UUID/procesos/VRAM y del checkpoint `/Proyectos/QEngine/models/Qwen3.8-Flash-Next-Uncensored-INT8-QEngine`.
- **Benchmarks pareados INT8 baseline/candidato e INT4**: requieren GPU libre sin interrumpir `qengine-flash`. No se afirma ninguna mejora de throughput/TTFT/TPOT.
- **Regresión multiGPU de lifecycle** (reclaim/restauración de tablas, esperas de copias, `FLASH_EC_VERIFY` sin mismatches, capture/replay, decodes intercalados): requiere 2–3 GPU y no puede ejecutarse junto al servicio.
- **Validación del perfil INT8 1M/4slots** del handoff `docs/flash-next-int8-rollout-2026-11-08.md`: sigue pendiente, parámetros exactos allí; no sustituir por el smoke 32K.

## Defecto de herramienta observado (corregido después en el repo)

`Dockerfile.flash-int8` hace `COPY tests ./tests`, pero el `.dockerignore` commiteado (desde `e73ba5e`) no incluía `!tests/`, por lo que el build desde el árbol tal cual fallaba con `"/tests": not found`. La ejecución histórica aplicó `printf '!tests/\n' >> .dockerignore` **solo en el staging remoto** y no tocó el repo. El fix permanente (`!tests/` en `.dockerignore`, más el comentario inicial) se aplicó en la revisión ampliada y quedó probado: el candidato nuevo se construyó desde un staging exacto del árbol sin parchear nada (véase Ampliación).

## Contexto de versiones

`origin/main` es 1.3.5 (`c07bfc9`) y el PR INT8 sigue sin mergear (`tests/flash_int8_unit.cu` no existe en main). El patrón defectuoso existe en main (`flash.cu:6060/6350/6431`), pero en el orden real allí `bt_want_ = n > pfbn_base_` se asigna **antes** de `ec_tick` y la decisión de staging (`pfs_min`/`pfs_now_`) se toma **después** del tick: por eso `bt_want_` no puede contemplar el staging y `pfs_full_` queda capado. (Esta línea decía al revés el orden; corregido al volver a leer el código.) Esa coincidencia hace el cambio estructuralmente portable al rebase, pero la portabilidad **no está verificada por integración**: el PR no se ha rebaseado a main 1.3.5 ni la mezcla se ha compilado ni ejecutado. El servicio activo usa imagen `1.3.5.2-vision` con checkpoint INT4 `Qwen3.8-Flash-Next-QE-1.3.4.1` (el `1.3.4.1` es nombre del checkpoint, no del código).

## Reproducción

Staging nuevo con el snapshot actual (no histórico), tag `qengine:flash-int8-prefill-review-<UTC>-r<N>` nunca reciclado, `-j3` del Dockerfile, paralelismo una sola carga GPU y nunca sobre GPU ocupada. Artefactos crudos de esta sesión: `int8-prefill-evidence/` (host, fuera del repo): `protected-service-baseline.txt`, `gpu-and-service-state-after.txt`, `build-r1.log` en el staging.

## Ampliación (revisión ampliada, sesión posterior)

Cambios en esta sesión: `tests/flash_int8_unit.cu` (`require` ahora lanza `std::runtime_error` tras el diagnóstico, en vez de acumular y continuar — un count/descriptor/token inválido ya no alcanza `vector`/`cudaMemcpy`/kernels/índices CPU; la equivalencia entre placements se compara por variante dentro del bucle: dn_pf determinista y no determinista, dn_q tile 1 y tile 2, cada dn_q parte de slots en cero) y `.dockerignore` (`!tests/`). Runtime/loader revisados de nuevo: sin defecto demostrado en buffers de 256 filas, scratch cold ni protocolo de reclaim — `flash.cu`/`flash.h` quedan byte a byte idénticos al snapshot histórico (`fe4895a2…`/`c1ecf1cf…`).

| Verificación (sesión actual) | Resultado |
| --- | --- |
| Fail-fast del `require` (artesanía desechable fuera del repo, g++ 13.3 remoto, sin CUDA; macro REAL extraído del archivo) | antes: los 3 caminos (count inválido, descriptor inválido, token inválido) continuaban y alcanzaban su consumidor; después: excepción, exit 1, consumidor inalcanzable |
| Snapshot actual (177 archivos, sin `.git`) | tar `int8-review-snapshot.tar.gz` SHA256 `c0589335497acf5ba55da50ef6e90b358854c7040c4213a29fca11434bb20c70`; staging `/home/cristian/qe-staging/prefill-review-20261108T153717Z`; `sha256sum -c` 177/177 OK |
| Fuentes | `tests/flash_int8_unit.cu` `7e47f399da1f973d32321002a77cb52f06afae4569f897e725acfbf32cadc2b3`, `.dockerignore` `e015f6271e6bf12886fd10552df4a52df2a4d7bf8f286a0493c69a0d11efc565`, `Dockerfile.flash-int8` `023aaae79c940487090c6a78d28ea17653533d5ea953bf86f2adcc4f3ef3f305` |
| Build del candidato nuevo SIN parchear `.dockerignore` remoto | RC=0 (`build-r1.log` en staging), `COPY tests ./tests` ejecutado, image ID `sha256:571ad83955641db14328b3e15722a0b8f85eea944629d7ef369d101bb7859d25`, tag `qengine:flash-int8-prefill-review-20261108T153717Z-r1` |
| Checkpoint INT8 (CPU-only) | `docker run --rm --entrypoint /app/build/flash_checkpoint_unit 571ad8395564` → `flash_checkpoint_unit: ok`, exit 0 |
| Servicio protegido antes/después | contenedor `c51d60c9a4f4` (`ghcr.io/bpx03/qengine:1.3.5.2-vision`) arriba, `{"status":"ok"}` intacto; pid 12649 con las 3 GPU igual antes/después |

Build y checkpoint CPU no prueban ninguna ruta GPU. Pendientes de recursos aislados: `flash_int8_unit` en GPU, lifecycle multiGPU, límites loader 256/272/512/528 y cold 63/64/65, perfil 1M/4slots y rendimiento — la receta está en "Gates bloqueados"; no sustituir por el build ni afirmar mejora. Artefactos crudos de esta sesión (staging `/home/cristian/qe-staging/prefill-review-20261108T153717Z`): `build-r1.log`, `service-state-after-build.txt`, `evidence/` (`int8-review-snapshot.tar.gz` + `.sha256`, arneses del `require` y transcripciones `require-before.txt`/`require-after.txt`).
