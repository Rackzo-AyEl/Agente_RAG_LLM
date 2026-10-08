"""Session/cache memory smoke benchmark for the help RAG service."""

from __future__ import annotations

import json
import os
import statistics
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("HELP_RAG_USE_LLM", "0")

import rag_core


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
QUESTIONS = [
    "¿Cómo me registro?",
    "¿Cómo creo mi negocio?",
    "¿Cómo agrego menú y ofertas?",
    "¿Cómo cambio mi número de contacto?",
]


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, round((len(values) - 1) * pct))
    return values[idx]


def run(users):
    rag_core.ASK_CACHE.clear()
    rag_core.SEARCH_CACHE.clear()
    rag_core.SESSIONS.clear()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    latencies = []
    for user in range(users):
        session_id = f"bench-{users}-{user}"
        for question in QUESTIONS:
            start = time.perf_counter()
            rag_core.ask(question, session_id=session_id)
            latencies.append((time.perf_counter() - start) * 1000)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    delta = sum(stat.size_diff for stat in after.compare_to(before, "filename"))
    summaries = [len(session.get("summary", "")) for session in rag_core.SESSIONS.values()]
    return {
        "users": users,
        "turns_per_user": len(QUESTIONS),
        "requests": users * len(QUESTIONS),
        "sessions": len(rag_core.SESSIONS),
        "ask_cache_entries": len(rag_core.ASK_CACHE),
        "search_cache_entries": len(rag_core.SEARCH_CACHE),
        "avg_ms": round(statistics.mean(latencies), 3),
        "p50_ms": round(percentile(latencies, 0.50), 3),
        "p95_ms": round(percentile(latencies, 0.95), 3),
        "heap_delta_kb": round(delta / 1024, 1),
        "heap_per_user_kb": round((delta / 1024) / users, 3),
        "avg_summary_chars": round(statistics.mean(summaries), 1) if summaries else 0,
        "max_summary_chars": max(summaries) if summaries else 0,
    }


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    ingest = rag_core.ingest()
    rows = [run(users) for users in (100, 500, 1000)]
    report = {
        "ingest": ingest,
        "limits": {
            "question_max_chars": rag_core.QUESTION_MAX_CHARS,
            "session_ttl_seconds": rag_core.CACHE_TTL_SECONDS,
            "session_max_turns": rag_core.SESSION_MAX_TURNS,
            "session_summary_max_chars": rag_core.SESSION_SUMMARY_MAX_CHARS,
            "ask_cache_max_entries": rag_core.ASK_CACHE_MAX_ENTRIES,
            "search_cache_max_entries": rag_core.SEARCH_CACHE_MAX_ENTRIES,
        },
        "rows": rows,
    }
    (ARTIFACTS / "session_memory_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "Session Memory / Cache Benchmark",
        "================================",
        "",
        f"Corpus: {ingest['docs']} docs, {ingest['chunks']} chunks",
        "TTL: 30 minutes; history: 4 turns; summary cap: 800 chars; question cap: 500 chars.",
        "",
        "+-------+----------+---------+--------+--------+------------+-------------+--------------+",
        "| Users | Requests | Avg ms  | p95 ms | Cache  | Heap KB    | KB / user   | Max summary  |",
        "+-------+----------+---------+--------+--------+------------+-------------+--------------+",
    ]
    for row in rows:
        lines.append(
            f"| {row['users']:<5} | {row['requests']:<8} | {row['avg_ms']:<7} | {row['p95_ms']:<6} | "
            f"{row['ask_cache_entries']:<6} | {row['heap_delta_kb']:<10} | {row['heap_per_user_kb']:<11} | {row['max_summary_chars']:<12} |"
        )
    lines.extend([
        "+-------+----------+---------+--------+--------+------------+-------------+--------------+",
        "",
        "Reading:",
        "- Per-user session memory is small; model RAM dominates when BGE-M3 is loaded.",
        "- ASK cache is capped at 1000 entries, so at 1000 users x 4 turns it keeps recent answers only.",
        "- SEARCH cache stays tiny here because the test repeats four canonical questions.",
    ])
    (ARTIFACTS / "session_cache_perf_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
