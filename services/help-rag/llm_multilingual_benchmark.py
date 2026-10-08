"""Benchmark multilingual LLM providers for the Andar help RAG use case."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
ARTIFACTS = ROOT / "artifacts"
API_KEY_PATH = PROJECT_ROOT / "API_KEY.txt"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

OLLAMA_MODELS = ["qwen2.5:0.5b", "qwen2.5:1.5b", "llama3.2:1b"]
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]

CONTEXT = """
Registro: Para usar Andar.mx, la persona comerciante crea una cuenta con teléfono móvil o correo electrónico, confirma un código de verificación y acepta términos.
Crear negocio: Desde el panel principal se elige Crear negocio. El formulario solicita nombre comercial, categoría, dirección, municipio, teléfono de contacto y descripción breve.
Editar tienda: En Mi tienda se puede cambiar nombre visible, descripción, categoría, teléfono, enlaces y dirección. Después se revisa la vista pública.
Ofertas: En el panel de la tienda se abre Ofertas y Nueva oferta. Se captura título, descripción, precio o beneficio, vigencia y condiciones.
Fotos: En Fotos se suben imágenes del local, productos, menú o servicios. Se recomiendan JPG o PNG con buena luz.
Horarios: En Horarios se definen días abiertos, apertura, cierre, días cerrados y excepciones por vacaciones o feriados.
""".strip()

CASES = [
    {
        "id": "es_crear_negocio",
        "lang": "es",
        "question": "¿Cómo creo mi negocio en Andar.mx?",
        "must": ["Crear negocio", "nombre", "categoría", "dirección"],
        "forbid": ["clima", "receta"],
    },
    {
        "id": "en_create_business",
        "lang": "en",
        "question": "How do I create my business on Andar.mx?",
        "must": ["Create business", "name", "category", "address"],
        "forbid": ["weather", "recipe"],
    },
    {
        "id": "es_editar_tienda",
        "lang": "es",
        "question": "Cambié mi teléfono, ¿dónde edito mi tienda?",
        "must": ["Mi tienda", "teléfono", "vista pública"],
        "forbid": ["clima", "receta"],
    },
    {
        "id": "en_add_promo",
        "lang": "en",
        "question": "How do I add a promotion with an expiration date?",
        "must": ["Offers", "New offer", "validity"],
        "forbid": ["weather", "recipe"],
    },
    {
        "id": "spanglish_photos",
        "lang": "es",
        "question": "Quiero upload fotos del menú y productos, ¿dónde va eso?",
        "must": ["Fotos", "menú", "productos"],
        "forbid": ["clima", "receta"],
    },
    {
        "id": "out_weather",
        "lang": "es",
        "question": "¿Cuál es el clima de mañana?",
        "must": ["No tengo información"],
        "forbid": ["soleado", "lluvia", "temperatura"],
        "out_of_domain": True,
    },
]


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((pct / 100) * (len(values) - 1))))
    return round(values[index], 3)


def post_json(url, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def build_prompt(case):
    language = "Spanish" if case["lang"] == "es" else "English"
    return f"""You are the Andar.mx help assistant for business owners.
Answer in {language}.
Use only the provided context. If the answer is not in the context, say you do not have enough information.
Be concise, respectful, and use 3 to 5 steps when possible.

Context:
{CONTEXT}

Question: {case['question']}
Answer:"""


def call_ollama(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 180},
    }
    start = time.perf_counter()
    data = post_json(OLLAMA_URL, payload, timeout=120)
    latency = (time.perf_counter() - start) * 1000
    text = data.get("response", "")
    eval_count = data.get("eval_count") or 0
    eval_duration = data.get("eval_duration") or 0
    tps = (eval_count / (eval_duration / 1_000_000_000)) if eval_count and eval_duration else None
    return {"text": text, "latency_ms": latency, "tokens_per_second": tps, "raw": {"eval_count": eval_count}}


def gemini_key():
    if not API_KEY_PATH.exists():
        return None
    raw = API_KEY_PATH.read_text(encoding="utf-8").strip()
    if ":" in raw:
        raw = raw.rsplit(":", 1)[-1].strip()
    return raw


def call_gemini(model, prompt):
    key = gemini_key()
    if not key:
        raise RuntimeError("missing API_KEY.txt")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(key)}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 180},
    }
    start = time.perf_counter()
    data = post_json(url, payload, timeout=120)
    latency = (time.perf_counter() - start) * 1000
    candidates = data.get("candidates") or []
    text = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts)
    usage = data.get("usageMetadata", {})
    return {"text": text, "latency_ms": latency, "tokens_per_second": None, "raw": {"usage": usage}}


def language_match(text, lang):
    lowered = text.lower()
    if lang == "es":
        return any(word in lowered for word in ["negocio", "tienda", "oferta", "fotos", "información", "paso"])
    return any(word in lowered for word in ["business", "store", "offer", "photos", "information", "step"])


def score_answer(case, text):
    lowered = text.lower()
    must_hits = sum(1 for token in case.get("must", []) if token.lower() in lowered)
    forbid_hits = sum(1 for token in case.get("forbid", []) if token.lower() in lowered)
    lang_ok = language_match(text, case["lang"])
    grounded = forbid_hits == 0 and must_hits >= max(1, min(2, len(case.get("must", []))))
    if case.get("out_of_domain"):
        grounded = must_hits >= 1 and forbid_hits == 0
    return {
        "must_hits": must_hits,
        "forbid_hits": forbid_hits,
        "language_match": lang_ok,
        "grounded": grounded,
        "score": round((int(lang_ok) + int(grounded)) / 2, 3),
    }


def call_provider(provider, model, prompt):
    if provider == "ollama":
        return call_ollama(model, prompt)
    if provider == "gemini":
        return call_gemini(model, prompt)
    raise ValueError(provider)


def evaluate_model(provider, model):
    rows = []
    for case in CASES:
        prompt = build_prompt(case)
        try:
            result = call_provider(provider, model, prompt)
            score = score_answer(case, result["text"])
            rows.append({
                "case_id": case["id"],
                "lang": case["lang"],
                "latency_ms": round(result["latency_ms"], 3),
                "tokens_per_second": round(result["tokens_per_second"], 3) if result["tokens_per_second"] else None,
                "score": score,
                "sample": result["text"][:500],
                "ok": True,
            })
        except Exception as exc:
            detail = str(exc)
            if "key=" in detail:
                detail = detail.split("key=", 1)[0] + "key=<redacted>"
            rows.append({"case_id": case["id"], "ok": False, "error": exc.__class__.__name__, "detail": detail[:500]})
    ok_rows = [row for row in rows if row.get("ok")]
    latencies = [row["latency_ms"] for row in ok_rows]
    scores = [row["score"]["score"] for row in ok_rows]
    grounded = [row["score"]["grounded"] for row in ok_rows]
    lang = [row["score"]["language_match"] for row in ok_rows]
    return {
        "provider": provider,
        "model": model,
        "cases": len(rows),
        "successes": len(ok_rows),
        "quality_score": round(statistics.mean(scores), 3) if scores else 0,
        "groundedness": round(sum(grounded) / len(grounded), 3) if grounded else 0,
        "language_match": round(sum(lang) / len(lang), 3) if lang else 0,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "avg_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "rows": rows,
    }


def rpm_test(provider, model, rpm, total_requests):
    workload = [CASES[i % len(CASES)] for i in range(total_requests)]
    interval = 60.0 / rpm
    start = time.perf_counter()
    futures = []
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, min(8, total_requests))) as pool:
        for idx, case in enumerate(workload):
            target = start + idx * interval
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            prompt = build_prompt(case)
            futures.append(pool.submit(call_provider, provider, model, prompt))
        for future in as_completed(futures):
            try:
                result = future.result(timeout=180)
                rows.append({"ok": True, "latency_ms": result["latency_ms"], "tokens_per_second": result.get("tokens_per_second")})
            except Exception as exc:
                detail = str(exc)
                if "key=" in detail:
                    detail = detail.split("key=", 1)[0] + "key=<redacted>"
                rows.append({"ok": False, "error": exc.__class__.__name__, "detail": detail[:300]})
    total_s = time.perf_counter() - start
    ok_rows = [row for row in rows if row.get("ok")]
    latencies = [row["latency_ms"] for row in ok_rows]
    return {
        "provider": provider,
        "model": model,
        "target_rpm": rpm,
        "requests": total_requests,
        "successes": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "observed_rpm": round((len(ok_rows) / total_s) * 60, 3) if total_s else None,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "avg_ms": round(statistics.mean(latencies), 3) if latencies else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--ollama-model", action="append", default=[])
    parser.add_argument("--gemini-model", default="gemini-2.0-flash")
    parser.add_argument("--rpm", type=int, nargs="+", default=[10, 30, 60])
    parser.add_argument("--requests-per-rpm", type=int, default=6)
    parser.add_argument("--output", default="llm_multilingual_benchmark.json")
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    models = []
    if not args.skip_ollama:
        for model in args.ollama_model or OLLAMA_MODELS:
            models.append(("ollama", model))
    if not args.skip_gemini:
        models.append(("gemini", args.gemini_model))

    evaluations = []
    rpm_results = []
    for provider, model in models:
        evaluation = evaluate_model(provider, model)
        evaluations.append(evaluation)
        rpm_levels = args.rpm if provider == "ollama" else [rpm for rpm in args.rpm if rpm <= 30]
        for rpm in rpm_levels:
            rpm_results.append(rpm_test(provider, model, rpm, args.requests_per_rpm))

    ok_evals = [item for item in evaluations if item["successes"]]
    best_main = sorted(ok_evals, key=lambda x: (x["quality_score"], x["groundedness"], -x["p95_ms"]), reverse=True)[0] if ok_evals else None
    report = {
        "cases": [{"id": c["id"], "lang": c["lang"], "question": c["question"]} for c in CASES],
        "evaluations": evaluations,
        "rpm_results": rpm_results,
        "best_main_candidate": best_main,
    }
    out = ARTIFACTS / args.output
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
