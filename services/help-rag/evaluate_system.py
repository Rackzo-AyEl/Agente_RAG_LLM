"""Small direct evaluation for the Andar help RAG MVP."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HELP_RAG_USE_LLM", "0")

import rag_core


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

CASES = [
    {"question": "¿Cómo me registro con correo?", "intent": "registro", "source": "registro-cuenta"},
    {"question": "¿Cómo creo mi negocio?", "intent": "crear_negocio", "source": "crear-negocio"},
    {"question": "Necesito editar la dirección de mi tienda", "intent": "editar_tienda", "source": "editar-tienda"},
    {"question": "¿Cómo publico una promoción?", "intent": "agregar_oferta", "source": "ofertas-promociones"},
    {"question": "¿Cómo subo la información de mi menú en mi cuenta?", "intent": "menu_catalogo", "source": "menu-catalogo"},
    {"question": "¿Cómo actualizo mi perfil?", "intent": "perfil", "source": "perfil-usuario"},
    {"question": "¿Cómo cambio mi número de contacto?", "intent": "contacto_soporte", "source": "contacto-ubicacion"},
    {"question": "¿Cómo cambio mis coordenadas?", "intent": "ubicacion", "source": "contacto-ubicacion"},
    {"question": "¿Para qué sirve Andar?", "intent": "descubrir", "source": "que-es-andar"},
    {"question": "¿Qué beneficios tiene el plan Plus?", "intent": "plan_plus", "source": "plan-plus"},
    {"question": "¿Cómo me comunico con soporte?", "intent": "contacto_soporte", "source": "contacto-soporte"},
    {"question": "¿Dónde busco negocios en el mapa?", "intent": "descubrir", "source": "mapa-busqueda"},
]

OUT_OF_DOMAIN = [
    "¿Cuál es el clima de mañana?",
    "Dame una receta de mole",
]


def has_source(answer, expected):
    return any(source.get("id") == expected for source in answer.get("sources", []))


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    ingest = rag_core.ingest()
    rows = []
    correct_intent = 0
    grounded = 0
    start_all = time.perf_counter()
    for case in CASES:
        start = time.perf_counter()
        answer = rag_core.ask(case["question"])
        latency_ms = (time.perf_counter() - start) * 1000
        intent_ok = answer.get("intent") == case["intent"]
        source_ok = has_source(answer, case["source"])
        correct_intent += int(intent_ok)
        grounded += int(source_ok and bool(answer.get("sources")))
        rows.append({
            "question": case["question"],
            "expected_intent": case["intent"],
            "actual_intent": answer.get("intent"),
            "expected_source": case["source"],
            "source_ok": source_ok,
            "intent_ok": intent_ok,
            "latency_ms": round(latency_ms, 3),
        })

    rejections = 0
    for question in OUT_OF_DOMAIN:
        answer = rag_core.ask(question)
        rejected = not answer.get("sources") or answer.get("response_type") == "handoff"
        rejections += int(rejected)
        rows.append({"question": question, "out_of_domain": True, "rejected": rejected, "sources": answer.get("sources")})

    first = rag_core.ask(CASES[0]["question"])
    second = rag_core.ask(CASES[0]["question"])
    report = {
        "ingest": ingest,
        "cases": len(CASES),
        "intent_accuracy": round(correct_intent / len(CASES), 3),
        "groundedness": round(grounded / len(CASES), 3),
        "out_of_domain_rejection": round(rejections / len(OUT_OF_DOMAIN), 3),
        "cache_ok": bool(second.get("metadata", {}).get("cache_hit")) and first.get("answer_id") == second.get("answer_id"),
        "total_latency_ms": round((time.perf_counter() - start_all) * 1000, 3),
        "rows": rows,
    }
    out = ARTIFACTS / "evaluation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
