# Flash-Next INT8 rollout handoff — 2026-11-08

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
