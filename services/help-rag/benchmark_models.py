"""Model/provider benchmark for the Andar help RAG MVP.

Uses optional ML dependencies when installed, but keeps lexical/hash baselines.
All Hugging Face cache defaults to ./model-cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HELP_RAG_USE_LLM", "0")

import rag_core


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_CACHE = ROOT / "model-cache"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE / "hub"))

QUESTIONS = [
    ("¿Cómo registro mi cuenta?", "registro-cuenta"),
    ("Quiero crear mi negocio en Andar", "crear-negocio"),
    ("¿Cómo cambio el teléfono de mi tienda?", "contacto-ubicacion"),
    ("¿Dónde agrego una oferta con vigencia?", "ofertas-promociones"),
    ("¿Cómo subo mi menú?", "menu-catalogo"),
    ("¿Cómo actualizo mi perfil?", "perfil-usuario"),
    ("¿Qué beneficios tiene el plan Plus?", "plan-plus"),
    ("¿Dónde está el mapa para buscar negocios?", "mapa-busqueda"),
]

HARD_QUESTIONS = [
    {"question": "Soy dueño de una fonda y no sé si primero creo cuenta o negocio. ¿Qué hago?", "source": "registro-cuenta"},
    {"question": "Cambié de teléfono y también quiero revisar cómo lo ve el cliente", "source": "contacto-ubicacion"},
    {"question": "Quiero subir mi menú y una promo que termina el domingo", "source": "menu-catalogo"},
    {"question": "¿Puedo poner una receta de mole en mi tienda?", "source": None},
    {"question": "No sé si Plus me conviene o si la tienda básica basta", "source": "plan-plus"},
]

EMBEDDING_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "BAAI/bge-m3",
]

LLM_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]


def tokens(text):
    return rag_core._tokens(text)


def hash_embed(items, dims):
    vec = [0.0] * dims
    for item in items:
        digest = hashlib.blake2b(item.encode(), digest_size=8).digest()
        n = int.from_bytes(digest, "big")
        vec[n % dims] += 1.0 if n & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def rank_lexical(question, chunks, top_k=3):
    q = set(tokens(question))
    scored = []
    for chunk in chunks:
        score = len(q & chunk["token_set"]) / (len(q) or 1)
        if score > 0:
            scored.append((score, chunk))
    return [chunk for _, chunk in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]]


def rank_hash(question, chunks, dims, top_k=3):
    q_tokens = tokens(question)
    q_vec = hash_embed(q_tokens, dims)
    q_set = set(q_tokens)
    scored = []
    for chunk in chunks:
        lexical = len(q_set & chunk["token_set"]) / (len(q_set) or 1)
        semantic = (cosine(q_vec, hash_embed(chunk["tokens"], dims)) + 1.0) / 2.0
        if lexical > 0:
            scored.append((0.7 * lexical + 0.3 * semantic, chunk))
    return [chunk for _, chunk in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]]


def detect_gpu():
    info = {"nvidia_smi": False, "torch": False, "cuda": False, "device": None}
    if shutil.which("nvidia-smi"):
        info["nvidia_smi"] = True
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
                timeout=5,
            ).strip()
            info["device"] = out
        except Exception as exc:  # pragma: no cover - environment-specific
            info["device_error"] = str(exc)
    try:
        import torch  # type: ignore

        info["torch"] = True
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = exc.__class__.__name__
    return info


def optional_providers():
    providers = []
    try:
        import sentence_transformers  # type: ignore  # noqa: F401

        providers.append({"name": "sentence-transformers", "status": "available"})
    except Exception as exc:
        providers.append({"name": "sentence-transformers", "status": "skipped", "reason": exc.__class__.__name__})

    providers.append({
        "name": "ollama",
        "status": "available" if shutil.which("ollama") else "skipped",
        "reason": None if shutil.which("ollama") else "command not found",
    })
    return providers


def memory_snapshot():
    data = {}
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        data["rss_mb"] = round(process.memory_info().rss / 1024 / 1024, 1)
    except Exception as exc:
        data["rss_error"] = exc.__class__.__name__
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            ).strip()
            used, total = [int(x.strip()) for x in out.split(",")[:2]]
            data["gpu_used_mb"] = used
            data["gpu_total_mb"] = total
        except Exception as exc:  # pragma: no cover
            data["gpu_error"] = exc.__class__.__name__
    return data


def evaluate_ranker(name, ranker):
    chunks = rag_core.INDEX["chunks"]
    latencies = []
    hits = 0
    mrr = 0.0
    for question, expected_doc in QUESTIONS:
        start = time.perf_counter()
        results = ranker(question, chunks)
        latencies.append((time.perf_counter() - start) * 1000)
        docs = [chunk["doc_id"] for chunk in results]
        if expected_doc in docs:
            hits += 1
            mrr += 1.0 / (docs.index(expected_doc) + 1)
    return {
        "name": name,
        "device": "cpu",
        "status": "ok",
        "questions": len(QUESTIONS),
        "recall_at_3": round(hits / len(QUESTIONS), 3),
        "mrr": round(mrr / len(QUESTIONS), 3),
        "latency_ms_avg": round(sum(latencies) / len(latencies), 3),
    }


def evaluate_sentence_transformer(model_name, device):
    start_load = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name, device=device, cache_folder=str(MODEL_CACHE))
    except Exception as exc:
        return {"name": model_name, "device": device, "status": "skipped", "reason": exc.__class__.__name__, "detail": str(exc)[:240]}

    chunks = rag_core.INDEX["chunks"]
    texts = [chunk["heading"] + "\n" + chunk["text"] for chunk in chunks]
    try:
        chunk_embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        latencies = []
        hits = 0
        mrr = 0.0
        for question, expected_doc in QUESTIONS:
            query = "query: " + question if "e5" in model_name else question
            start = time.perf_counter()
            query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
            scores = chunk_embeddings @ query_embedding
            ranked = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)[:3]
            latencies.append((time.perf_counter() - start) * 1000)
            docs = [chunks[i]["doc_id"] for i in ranked]
            if expected_doc in docs:
                hits += 1
                mrr += 1.0 / (docs.index(expected_doc) + 1)
        return {
            "name": model_name,
            "device": device,
            "status": "ok",
            "questions": len(QUESTIONS),
            "recall_at_3": round(hits / len(QUESTIONS), 3),
            "mrr": round(mrr / len(QUESTIONS), 3),
            "load_ms": round((time.perf_counter() - start_load) * 1000, 1),
            "latency_ms_avg": round(sum(latencies) / len(latencies), 3),
            "memory": memory_snapshot(),
        }
    except Exception as exc:
        return {"name": model_name, "device": device, "status": "failed", "reason": exc.__class__.__name__, "detail": str(exc)[:240]}


def evaluate_llm(model_name, device):
    start_load = time.perf_counter()
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE))
        kwargs = {"cache_dir": str(MODEL_CACHE)}
        if device == "cuda":
            kwargs.update({"torch_dtype": torch.float16, "device_map": {"": 0}})
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device == "cpu":
            model.to("cpu")
    except Exception as exc:
        return {"name": model_name, "device": device, "status": "skipped", "reason": exc.__class__.__name__, "detail": str(exc)[:240]}

    prompts = [
        "Responde en español, breve y con pasos: ¿Cómo registro mi negocio en Andar.mx?",
        "Responde con cuidado y sin inventar: ¿Cuál es el clima de mañana?",
    ]
    rows = []
    try:
        for prompt in prompts:
            start = time.perf_counter()
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            output = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            new_tokens = max(1, int(output.shape[-1] - inputs["input_ids"].shape[-1]))
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            rows.append({
                "prompt": prompt,
                "new_tokens": new_tokens,
                "seconds": round(elapsed, 3),
                "tokens_per_second": round(new_tokens / elapsed, 2),
                "sample": text[-500:],
            })
        return {
            "name": model_name,
            "device": device,
            "status": "ok",
            "load_ms": round((time.perf_counter() - start_load) * 1000, 1),
            "rows": rows,
            "memory": memory_snapshot(),
        }
    except Exception as exc:
        return {"name": model_name, "device": device, "status": "failed", "reason": exc.__class__.__name__, "detail": str(exc)[:240], "memory": memory_snapshot()}


def evaluate_hard_queries():
    rows = []
    ok = 0
    for item in HARD_QUESTIONS:
        start = time.perf_counter()
        answer = rag_core.ask(item["question"])
        elapsed = (time.perf_counter() - start) * 1000
        sources = [source.get("id") for source in answer.get("sources", [])]
        expected = item["source"]
        passed = (expected in sources) if expected else not sources
        ok += int(passed)
        rows.append({
            "question": item["question"],
            "expected_source": expected,
            "actual_sources": sources,
            "intent": answer.get("intent"),
            "response_type": answer.get("response_type"),
            "title": answer.get("title"),
            "latency_ms": round(elapsed, 3),
            "passed": passed,
            "metadata": answer.get("metadata", {}),
        })
    return {"total": len(rows), "passed": ok, "score": round(ok / len(rows), 3), "rows": rows}


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    MODEL_CACHE.mkdir(exist_ok=True)
    ingest = rag_core.ingest()
    gpu = detect_gpu()
    results = [
        evaluate_ranker("lexical", rank_lexical),
        evaluate_ranker("hash-128", lambda q, c: rank_hash(q, c, 128)),
        evaluate_ranker("hash-512", lambda q, c: rank_hash(q, c, 512)),
    ]
    embedding_results = []
    devices = ["cpu"] + (["cuda"] if gpu.get("cuda") else [])
    for model_name in EMBEDDING_MODELS:
        for device in devices:
            embedding_results.append(evaluate_sentence_transformer(model_name, device))
    llm_results = []
    for model_name in LLM_MODELS:
        for device in devices:
            llm_results.append(evaluate_llm(model_name, device))
    available = [result for result in results if result["status"] == "ok"]
    available += [result for result in embedding_results if result.get("status") == "ok"]
    best = sorted(available, key=lambda r: (r["recall_at_3"], r["mrr"], -r["latency_ms_avg"]), reverse=True)[0]
    report = {
        "ingest": ingest,
        "gpu": gpu,
        "optional_providers": optional_providers(),
        "baseline_results": results,
        "embedding_results": embedding_results,
        "llm_results": llm_results,
        "best_available": best,
        "hard_query_results": evaluate_hard_queries(),
    }
    out = ARTIFACTS / "model_benchmark.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
