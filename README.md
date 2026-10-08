# Andar.mx Help RAG + LLM Demo

Microservicio RAG de ayuda para Andar.mx con corpus sintético, memoria corta de conversación, multimedia opcional y generación local con Qwen vía Ollama.

El objetivo de este repositorio es presentar la arquitectura, no un corpus final de producción.

## Qué Incluye

- Microservicio HTTP sin framework: `services/help-rag/server.py`.
- Motor RAG: `services/help-rag/rag_core.py`.
- UI web de prueba: `services/help-rag/static/`.
- Corpus sintético Markdown: `services/help-rag/content/`.
- Inventario editable del corpus: `services/help-rag/SYNTHETIC_CORPUS_INVENTORY.md`.
- Multimedia de prueba: `Files/Images` y `Files/Videos`.
- Suite de regresión determinística: `services/help-rag/regression_acceptance.py`.

## Arquitectura

```text
Usuario
  |
  v
UI de prueba
  |
  v
POST /ask
  |
  v
Guardrails + memoria de sesión
  |
  v
Retrieval lexical primero
  |
  v
Fallback semántico opcional con BGE-M3
  |
  v
Qwen local vía Ollama
  |
  v
Gemini opcional solo como fallback externo
```

## Modelo De Costos

El sistema puede operar sin pagar APIs:

- Qwen corre localmente con Ollama.
- Retrieval/embeddings corren localmente.
- Corpus y multimedia viven en el servidor.
- Gemini es opcional y puede desactivarse.

El costo real pasa a ser hardware: CPU, RAM, GPU/VRAM, disco y electricidad/hosting.

## Qwen vs Ollama

- **Qwen** es el modelo LLM.
- **Ollama** es el runtime/servidor local que carga y ejecuta el modelo.

Ollama expone una API local en `127.0.0.1:11434` y puede usar GPU automáticamente si hay disponible.

## Estado Actual Del Corpus

| Elemento | Cantidad |
|---|---:|
| Documentos Markdown | 15 |
| Chunks indexados | 60 |
| Archivos multimedia | 10 |
| Límite por pregunta | 500 caracteres |
| Memoria por sesión | 4 turnos |
| TTL de sesión/cache | 30 min |

## Uso De Recursos Medido

Medición local con demo activa, Ollama sirviendo `qwen2.5:1.5b` y embeddings configurados como CPU.

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
| Sin BGE-M3 cargado | ~676 MB | ~1.1 GB |
| Con BGE-M3 cargado en CPU | ~3.2 GB aprox | ~1.1 GB |
| Ollama forzado a CPU | más RAM/CPU | ~0 GB |

Tamaños en disco del proyecto original usado para preparar este repo:

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

Este repositorio no incluye modelos, cache de embeddings, binarios de Ollama ni claves API.

## Ejecutar Demo

Desde la raíz del repositorio:

```bash
cd services/help-rag
PORT=8000 python server.py
```

Abrir:

```text
http://localhost:8000/
```

Modo con embeddings BGE-M3 en CPU:

```bash
cd services/help-rag
HELP_RAG_EMBED_MODEL=BAAI/bge-m3 HELP_RAG_DEVICE=cpu PORT=8000 python server.py
```

Modo con LLM local Qwen vía Ollama:

```bash
cd services/help-rag
HELP_RAG_USE_LLM=1 HELP_RAG_OLLAMA_MODEL=qwen2.5:1.5b PORT=8000 python server.py
```

## Instalar Ollama Y Modelo Local

Este repo no instala Ollama ni descarga modelos. En una máquina con Ollama disponible:

```bash
ollama pull qwen2.5:1.5b
ollama serve
```

Modelos usados durante la evaluación:

| Modelo | Tamaño |
|---|---:|
| `qwen2.5:0.5b` | `397 MB` |
| `qwen2.5:1.5b` | `986 MB` |
| `llama3.2:1b` | `1.3 GB` |

## Endpoints

- `GET /health`
- `GET /topics`
- `GET /corpus`
- `GET /sources/<id>/status`
- `GET /media-files/<ruta-relativa>`
- `POST /ask`
- `POST /feedback`
- `POST /ingest`

Ejemplo:

```bash
curl -s http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"¿Cómo escaneo un menú desde una fotografía?","session_id":"demo"}'
```

## Guardrails

El sistema usa defensas por capas:

- Límite de pregunta: 500 caracteres.
- Contexto limitado a chunks recuperados.
- `top_k` máximo: 8.
- Prompt delimitado con secciones explícitas.
- Temperatura LLM: 0.
- Guardrail para prompt injection.
- Guardrail para ventas/utilidades/ticket/finanzas fuera del corpus.
- Respuestas de falta de información cuando no hay fuente suficiente.

Solicitudes como revelar prompt interno, claves, reglas del sistema o actuar como administrador no llegan al LLM.

## Multimedia Opcional

La demo lee archivos desde:

```text
Files/Images
Files/Videos
```

El sistema deriva metadatos desde el nombre del archivo. La respuesta no carga videos o imágenes automáticamente; primero sugiere material de ayuda y la UI lo carga solo si el usuario acepta.

Ejemplo:

```text
Usuario: ¿Cómo registrar una cuenta?
Asistente: [respuesta textual]
Asistente: ¿Te gustaría ver un video de ayuda: Crear cuenta?
Usuario: sí
Asistente: carga el video/imagen relacionado
```

## Pruebas

Suite rápida, determinística, sin red ni LLM:

```bash
cd services/help-rag
python regression_acceptance.py
```

Resultado esperado:

```json
{
  "passed": true,
  "failed": 0
}
```

Evaluación directa del RAG:

```bash
cd services/help-rag
python evaluate_system.py
```

La suite de regresión cubre:

- corpus mínimo
- retrieval normal con fuentes
- cache exacto
- memoria de sesión para preguntas ambiguas
- aclaración cuando falta contexto
- prompt injection
- preguntas de negocio/finanzas fuera de alcance
- multimedia opcional
- límite de longitud
- snapshot del corpus

## Estructura

```text
.
├── Files/
│   ├── Images/
│   └── Videos/
└── services/
    └── help-rag/
        ├── content/
        ├── static/
        ├── rag_core.py
        ├── server.py
        ├── regression_acceptance.py
        ├── evaluate_system.py
        └── README.md
```

## Notas De Producción

Este repo es una demo técnica. Para producción faltaría:

- Corpus real validado.
- Storage externo para multimedia.
- DB para metadatos y auditoría.
- Observabilidad de latencia/costos.
- Autenticación y permisos.
- Pruebas de carga reales.
- Evaluación humana de respuestas.
