# Andar Help RAG MVP

Microservicio de ayuda RAG para evaluar una arquitectura de soporte de Andar.mx con corpus sintético, búsqueda lexical primero, fallback semántico con embeddings y UI de prueba.

## Estado Actual

- Corpus sintético: `15` documentos Markdown, `60` chunks.
- Inventario del corpus: `SYNTHETIC_CORPUS_INVENTORY.md`.
- Multimedia de prueba: `10` archivos desde `Files/Images` y `Files/Videos`.
- Retrieval recomendado: lexical primero; BGE-M3 como fallback/reranker semántico.
- Generación RAG: `ollama qwen2.5:1.5b` primero, `gemini-3.1-flash-lite` como fallback.
- Temperatura de generación: `0` para reducir variación y alucinaciones.
- UI: muestra por separado dispositivo de embeddings y hardware LLM/Ollama.

## Tamaños Medidos

| Parte | Tamaño |
|---|---:|
| Proyecto completo `Andar/` | `18 GB` |
| `services/help-rag/` | `8.7 GB` |
| Modelos Ollama | `2.6 GB` |
| Cache HuggingFace / embeddings | `6.1 GB` |
| `.local-bin` Ollama/binarios | `2.1 GB` |
| `Files/` multimedia prueba | `3.6 MB` |
| Corpus Markdown | `64 KB` |
| UI estática | `28 KB` |

## Uso De Recursos Medido

Medición local con demo activa, Ollama sirviendo `qwen2.5:1.5b` y embeddings configurados como `HELP_RAG_DEVICE=cpu`.

| Proceso | Uso |
|---|---:|
| `ollama serve` | ~48 MB RAM |
| `server.py` help-rag | ~27 MB RAM |
| `llama-server` Qwen | ~601 MB RAM |
| `llama-server` Qwen | ~1122 MB VRAM |
| Total actual del stack demo | ~676 MB RAM + ~1.1 GB VRAM |

Escenarios esperados:

| Escenario | RAM sistema | VRAM |
|---|---:|---:|
| Actual, sin BGE-M3 cargado | ~676 MB | ~1.1 GB |
| Con BGE-M3 cargado en CPU | ~3.2 GB aprox | ~1.1 GB |
| Ollama forzado a CPU | más RAM/CPU | ~0 GB |

Notas:

- `HELP_RAG_DEVICE=cpu` aplica a embeddings/retrieval, no al LLM.
- Ollama corre como proceso separado y puede usar GPU automáticamente si está disponible.
- Qwen es el modelo; Ollama es el runtime/servidor local que lo ejecuta.
- BGE-M3 se carga bajo demanda; cuando está cargado en CPU, se midió alrededor de `2.5 GB RSS`.

## Ejecutar Demo

```bash
cd services/help-rag
HELP_RAG_EMBED_MODEL=BAAI/bge-m3 HELP_RAG_DEVICE=cpu PORT=8000 ../../entorno-virtual/bin/python server.py
```

Abrir:

```text
http://localhost:8000/
```

Para modo ligero sin embeddings reales:

```bash
cd services/help-rag
PORT=8000 ../../entorno-virtual/bin/python server.py
```

## Endpoints

- `GET /health`
- `GET /topics`
- `GET /corpus`
- `POST /ask` con `{ "question": "¿Cómo agrego una oferta?", "session_id": "demo" }`
- `POST /feedback` con `{ "answer_id": "...", "feedback": "useful" }`
- `POST /ingest` con `{ "title": "...", "content": "..." }`
- `GET /sources/<id>/status`

## Límites De Chat Y Memoria

| Límite | Valor |
|---|---:|
| Máximo por pregunta | 500 caracteres |
| `top_k` máximo | 8 chunks |
| TTL de sesión | 30 minutos |
| Historial por sesión | 4 turnos |
| Resumen de sesión | 800 caracteres |
| Sesiones en RAM | 2000 |
| Caché `/ask` | 1000 entradas |
| Caché búsqueda | 3000 entradas |

La memoria de conversación vive en RAM y no es persistente. Sirve para resolver referencias cortas dentro de la sesión, no para recordar datos permanentes del usuario.

## Prompt Y Guardrails

El servidor usa un system prompt estricto con delimitadores para instrucciones, contexto recuperado y pregunta de usuario. Debe responder solo con contexto recuperado, no inventar políticas, precios, resultados de negocio, estrategias comerciales ni asesoría financiera/legal/fiscal. Si el corpus no contiene la respuesta, debe decir que no hay información suficiente.

Además hay un guardrail previo para consultas de ventas, utilidades, accionistas, finanzas o estrategia comercial cuando no piden una acción concreta de Andar.mx. Esas consultas se responden como falta de información y no disparan LLM.

También hay guardrail previo para solicitudes inválidas como revelar el prompt interno, claves, reglas del sistema, ignorar instrucciones o actuar como administrador. Esas solicitudes no llegan al LLM.

## Probar

```bash
cd services/help-rag
../../entorno-virtual/bin/python evaluate_system.py
../../entorno-virtual/bin/python regression_acceptance.py
CUDA_VISIBLE_DEVICES='' ../../entorno-virtual/bin/python benchmark_models.py
../../entorno-virtual/bin/python session_memory_benchmark.py
```

`regression_acceptance.py` es la suite rápida determinística: desactiva LLM/red, prueba `rag_core` directo y cubre corpus mínimo, fuentes, cache exacto, memoria ambigua, aclaración, prompt injection, handoff de negocio, multimedia opcional, límite de longitud y snapshot `/corpus`.

Artefactos principales:

- `artifacts/evaluation.json`
- `artifacts/model_benchmark.json`
- `artifacts/synthetic_corpus_eval_report.txt`
- `artifacts/session_memory_benchmark.json`
- `artifacts/session_cache_perf_report.txt`
- `artifacts/llm_eval_report.txt`

## Resultados Recientes

| Área | Resultado |
|---|---:|
| Evaluación directa | 12/12 |
| Intent accuracy | 1.000 |
| Groundedness | 1.000 |
| Rechazo fuera de dominio | 1.000 |
| Hard queries | 5/5 |
| Lexical avg | ~0.1 ms |
| BGE-M3 avg | ~235 ms |
| BGE-M3 RSS | ~2.56 GB |

Nota: `HELP_RAG_DEVICE=cpu` describe embeddings/retrieval. Ollama corre como proceso separado y puede usar GPU automáticamente si está disponible.

## Multimedia De Prueba

La demo lee archivos desde `Files/Images` y `Files/Videos`. Los metadatos se derivan del nombre del archivo y se usan para recomendar material opcional.

La respuesta no carga videos/imágenes automáticamente: muestra una sugerencia y la UI carga el archivo solo cuando el usuario presiona el botón o responde afirmativamente en la misma sesión. Los archivos se exponen bajo `/media-files/<ruta-relativa>` sin permitir salir de la carpeta `Files`.

En producción, los binarios deberían vivir fuera del contenedor o en un volumen/objeto de storage. La DB debería guardar metadata, rutas, hashes, OCR, captions, transcripciones, timestamps y embeddings del texto derivado.
