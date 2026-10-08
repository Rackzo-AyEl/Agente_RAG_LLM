"""CPU-only stress benchmark for BAAI/bge-m3 on the Andar help RAG shape."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_CACHE = ROOT / "model-cache"
ARTIFACTS = ROOT / "artifacts"
MODEL_NAME = "BAAI/bge-m3"

TOPICS = {
    "registro": ["registro", "cuenta", "correo", "telefono", "codigo", "verificacion"],
    "crear_negocio": ["crear", "negocio", "alta", "categoria", "direccion", "municipio"],
    "editar_tienda": ["editar", "tienda", "descripcion", "telefono", "direccion", "vista publica"],
    "ofertas": ["oferta", "promocion", "precio", "vigencia", "condiciones", "beneficio"],
    "fotos": ["foto", "imagen", "menu", "producto", "jpg", "png"],
    "horarios": ["horario", "apertura", "cierre", "feriado", "vacaciones", "dias cerrados"],
}

QUERIES = [
    ("No me llega el codigo para registrar mi cuenta con correo", "registro"),
    ("Quiero dar de alta mi negocio y poner direccion municipal", "crear_negocio"),
    ("Cambie el telefono y quiero revisar como se ve mi tienda", "editar_tienda"),
    ("Necesito publicar una promocion con precio y vigencia", "ofertas"),
    ("Quiero subir fotos del menu y productos", "fotos"),
    ("Mis horarios cambian por vacaciones y feriados", "horarios"),
    ("Soy nuevo, primero cuenta o negocio", "registro"),
    ("La promo termina domingo y tiene condiciones", "ofertas"),
    ("La direccion de mi local cambio", "editar_tienda"),
    ("Que imagen principal vera el cliente", "fotos"),
]


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((pct / 100) * (len(values) - 1))))
    return round(values[index], 3)


def memory_snapshot():
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        return {"rss_mb": round(proc.memory_info().rss / 1024 / 1024, 1)}
    except Exception as exc:
        return {"error": exc.__class__.__name__}


def synthetic_corpus(size):
    topics = list(TOPICS.items())
    chunks = []
    for i in range(size):
        topic, words = topics[i % len(topics)]
        noise_topic, noise_words = topics[(i + 2) % len(topics)]
        text = (
            f"Documento {i} para {topic}. "
            f"Esta guia explica {' '.join(words)} para dueños de negocios en Andar.mx. "
            f"Pasos: revisar panel, guardar cambios, confirmar vista publica. "
            f"Contexto secundario controlado: {noise_words[0]} {noise_words[-1]}."
        )
        chunks.append({"id": f"chunk-{i}", "topic": topic, "text": text})
    return chunks


def load_model(threads):
    import torch  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu", cache_folder=str(MODEL_CACHE))
    return model, round((time.perf_counter() - start) * 1000, 3)


def encode(model, texts, batch_size):
    start = time.perf_counter()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vectors, round((time.perf_counter() - start) * 1000, 3)


def query_once(model, chunk_vectors, chunks, query, expected, batch_size):
    start = time.perf_counter()
    query_vector = model.encode([query], batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    scores = chunk_vectors @ query_vector
    ranked = sorted(range(len(chunks)), key=lambda idx: float(scores[idx]), reverse=True)[:5]
    topics = [chunks[idx]["topic"] for idx in ranked]
    return {
        "latency_ms": (time.perf_counter() - start) * 1000,
        "hit": expected in topics,
        "top_topic": topics[0] if topics else None,
    }


def run_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE / "hub"))

    chunks = synthetic_corpus(args.corpus_size)
    model, load_ms = load_model(args.threads)
    chunk_vectors, index_ms = encode(model, [c["text"] for c in chunks], args.batch_size)

    simple_latencies, simple_hits = [], 0
    for query, expected in QUERIES:
        row = query_once(model, chunk_vectors, chunks, query, expected, args.batch_size)
        simple_latencies.append(row["latency_ms"])
        simple_hits += int(row["hit"])

    workload = [QUERIES[i % len(QUERIES)] for i in range(args.requests)]
    start = time.perf_counter()
    stress_latencies, stress_hits, errors = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(query_once, model, chunk_vectors, chunks, q, expected, args.batch_size) for q, expected in workload]
        for future in as_completed(futures):
            try:
                row = future.result()
                stress_latencies.append(row["latency_ms"])
                stress_hits += int(row["hit"])
            except Exception:
                errors += 1
    total_s = time.perf_counter() - start

    result = {
        "model": MODEL_NAME,
        "device": "cpu",
        "threads": args.threads,
        "corpus_size": args.corpus_size,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "load_ms": load_ms,
        "index_ms": index_ms,
        "index_chunks_per_s": round(args.corpus_size / (index_ms / 1000), 3) if index_ms else None,
        "simple": {
            "queries": len(simple_latencies),
            "hit_rate": round(simple_hits / len(simple_latencies), 3),
            "p50_ms": percentile(simple_latencies, 50),
            "p95_ms": percentile(simple_latencies, 95),
            "avg_ms": round(statistics.mean(simple_latencies), 3),
        },
        "stress": {
            "hit_rate": round(stress_hits / max(1, len(stress_latencies)), 3),
            "errors": errors,
            "total_s": round(total_s, 3),
            "qps": round(len(stress_latencies) / total_s, 3) if total_s else None,
            "p50_ms": percentile(stress_latencies, 50),
            "p95_ms": percentile(stress_latencies, 95),
            "p99_ms": percentile(stress_latencies, 99),
            "avg_ms": round(statistics.mean(stress_latencies), 3) if stress_latencies else None,
        },
        "memory": memory_snapshot(),
    }
    print(json.dumps(result, ensure_ascii=False))


def run_parent(args):
    ARTIFACTS.mkdir(exist_ok=True)
    MODEL_CACHE.mkdir(exist_ok=True)
    results = []
    for corpus_size in args.corpus_sizes:
        for threads in args.threads:
            for concurrency in args.concurrencies:
                requests = max(args.requests, concurrency * 4)
                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": "",
                    "HF_HOME": str(MODEL_CACHE),
                    "TRANSFORMERS_CACHE": str(MODEL_CACHE),
                    "HF_HUB_CACHE": str(MODEL_CACHE / "hub"),
                    "OMP_NUM_THREADS": str(threads),
                    "MKL_NUM_THREADS": str(threads),
                    "OPENBLAS_NUM_THREADS": str(threads),
                    "TOKENIZERS_PARALLELISM": "false",
                })
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--threads", str(threads),
                    "--corpus-size", str(corpus_size),
                    "--batch-size", str(args.batch_size),
                    "--concurrency", str(concurrency),
                    "--requests", str(requests),
                ]
                start = time.perf_counter()
                proc = subprocess.run(command, cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=args.timeout)
                elapsed = round(time.perf_counter() - start, 3)
                if proc.returncode != 0:
                    results.append({
                        "threads": threads,
                        "corpus_size": corpus_size,
                        "concurrency": concurrency,
                        "status": "failed",
                        "elapsed_s": elapsed,
                        "stderr": proc.stderr[-2000:],
                        "stdout": proc.stdout[-2000:],
                    })
                    continue
                lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
                row = json.loads(lines[-1]) if lines else {"status": "no-json", "stdout": proc.stdout[-2000:]}
                row["status"] = "ok"
                row["elapsed_s"] = elapsed
                results.append(row)
                print(json.dumps(row, ensure_ascii=False, indent=2))

    ok = [r for r in results if r.get("status") == "ok"]
    best = None
    if ok:
        best = sorted(ok, key=lambda r: (r["stress"]["errors"] == 0, r["stress"]["hit_rate"], r["stress"]["qps"], -r["stress"]["p95_ms"]), reverse=True)[0]
    report = {
        "hardware": {
            "logical_cpus": os.cpu_count(),
            "cpu_only": True,
        },
        "model": MODEL_NAME,
        "results": results,
        "best_balance": best,
    }
    out = ARTIFACTS / "cpu_bge_m3_benchmark.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(out), "best_balance": best}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--corpus-sizes", type=int, nargs="+", default=[120, 1000])
    parser.add_argument("--corpus-size", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=None)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        if isinstance(parsed.threads, list):
            parsed.threads = parsed.threads[0]
        run_worker(parsed)
    else:
        if parsed.concurrencies is None:
            parsed.concurrencies = [parsed.concurrency]
        run_parent(parsed)
