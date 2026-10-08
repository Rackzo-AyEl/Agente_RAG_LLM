from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
import json
import mimetypes
import os
import re
import uuid

try:
    import rag_core
except ImportError:
    rag_core = None


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
FILES_DIR = ROOT.parents[1] / "Files"
QUESTION_MAX_CHARS = 500
TOP_K_MAX = 8
SOURCES = {}
ANSWERS = {}
FEEDBACK = []


def core_call(names, payload=None, **kwargs):
    if rag_core is None:
        return None
    for name in names:
        fn = getattr(rag_core, name, None)
        if not callable(fn):
            continue
        try:
            return fn(payload) if payload is not None else fn(**kwargs)
        except TypeError:
            try:
                return fn(**kwargs)
            except TypeError:
                return fn()
    return None


def as_dict(value, fallback):
    if value is None:
        return fallback
    return value if isinstance(value, dict) else {"result": value}


def core_docs():
    if rag_core is None:
        return []
    if not getattr(rag_core, "INDEX", {}).get("docs") and callable(getattr(rag_core, "ingest", None)):
        rag_core.ingest()
    return getattr(rag_core, "INDEX", {}).get("docs", [])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def json(self, data, status=200):
        self.send_headers(status)
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def body(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(size) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return None

    def do_OPTIONS(self):
        self.send_headers(204)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return self.json({"ok": True, "rag_core": rag_core is not None})
        if path == "/topics":
            topics = core_call(("topics", "get_topics", "list_topics"))
            if topics is None:
                docs = core_docs()
                topics = {
                    d.get("metadata", {}).get("topic") or d.get("metadata", {}).get("category") or d.get("title")
                    for d in docs
                } | {
                    s.get("metadata", {}).get("topic") or s.get("metadata", {}).get("category") or s.get("type") or s.get("title")
                    for s in SOURCES.values()
                }
                topics = sorted(t for t in topics if t)
            return self.json({"topics": topics})
        if path == "/corpus":
            snapshot = core_call(("corpus_snapshot", "corpus", "get_corpus"))
            return self.json(as_dict(snapshot, {"docs": core_docs()}))
        if path.startswith("/sources/") and path.endswith("/status"):
            source_id = unquote(path.removeprefix("/sources/").removesuffix("/status").strip("/"))
            status = core_call(("source_status", "get_source_status"), source_id=source_id)
            if status is None:
                doc = next((d for d in core_docs() if d.get("doc_id") == source_id), None)
                if doc:
                    return self.json({"source_id": source_id, "status": "indexed", "hash": doc.get("hash")})
                source = SOURCES.get(source_id)
                if source is None:
                    return self.json({"error": "source not found"}, 404)
                status = {"source_id": source_id, "status": source.get("status", "indexed")}
            return self.json(as_dict(status, {}))
        if path.startswith("/media-files/"):
            return self.serve_media(path)
        if self.serve_static(path):
            return
        self.json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        data = self.body()
        if not isinstance(data, dict):
            return self.json({"error": "invalid json"}, 400)
        if path == "/ask":
            return self.ask(data)
        if path == "/feedback":
            return self.feedback(data)
        if path == "/ingest":
            return self.ingest(data)
        self.json({"error": "not found"}, 404)

    def ask(self, data):
        question = str(data.get("question", "")).strip()
        if not question:
            return self.json({"error": "question required"}, 400)
        if len(question) > QUESTION_MAX_CHARS:
            return self.json({"error": f"question must be {QUESTION_MAX_CHARS} characters or less"}, 400)
        session_id = str(data.get("session_id") or data.get("conversation_id") or "").strip() or None
        try:
            top_k = int(data.get("top_k", 4) or 4)
        except (TypeError, ValueError):
            top_k = 4
        top_k = max(1, min(top_k, TOP_K_MAX))
        try:
            result = core_call(("ask", "answer", "query"), question=question, top_k=top_k, session_id=session_id)
        except ValueError as exc:
            return self.json({"error": str(exc)}, 400)
        if result is not None:
            return self.json(as_dict(result, {}))

        words = {w.lower() for w in question.split() if len(w) > 3}
        hits = [s for s in SOURCES.values() if words & set(s.get("content", "").lower().split())]
        answer = "No tengo información suficiente para responder." if not hits else hits[0].get("content", "")[:1500]
        answer_id = "ans_" + uuid.uuid4().hex[:12]
        response = {
            "answer_id": answer_id,
            "answer": answer,
            "confidence": 0.0 if not hits else 0.5,
            "citations": [{"source_id": h["id"], "title": h.get("title"), "url": h.get("url")} for h in hits[:3]],
            "needs_human": not hits,
        }
        ANSWERS[answer_id] = response
        self.json(response)

    def feedback(self, data):
        if not data.get("answer_id"):
            return self.json({"error": "answer_id required"}, 400)
        result = core_call(("feedback", "record_feedback"), data)
        if result is not None:
            return self.json(as_dict(result, {"ok": True}))
        FEEDBACK.append(data)
        self.json({"ok": True})

    def ingest(self, data):
        source_id = data.get("id") or data.get("source_id") or "src_" + uuid.uuid4().hex[:12]
        if rag_core is not None and data.get("content"):
            source_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", source_id).strip("-") or "src_" + uuid.uuid4().hex[:12]
            content_dir = ROOT / "content"
            content_dir.mkdir(exist_ok=True)
            meta = {"id": source_id, "title": data.get("title", source_id), "url": data.get("url", ""), "type": data.get("type", "help_article")}
            body = "---\n" + "\n".join(f"{k}: {v}" for k, v in meta.items() if v) + "\n---\n\n" + data["content"]
            (content_dir / f"{source_id}.md").write_text(body, encoding="utf-8")
            return self.json({"source_id": source_id, "status": "indexed", "ingest": rag_core.ingest()}, 201)
        result = core_call(("ingest", "ingest_document", "add_source"))
        if result is not None:
            return self.json(as_dict(result, {}), 202)
        SOURCES[source_id] = {**data, "id": source_id, "status": "indexed"}
        self.json({"source_id": source_id, "status": "indexed"}, 201)

    def serve_static(self, path):
        if not STATIC_DIR.is_dir():
            return False
        rel = "index.html" if path == "/" else path.removeprefix("/static/").lstrip("/")
        file_path = (STATIC_DIR / unquote(rel)).resolve()
        if STATIC_DIR.resolve() not in file_path.parents or not file_path.is_file():
            return False
        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        self.send_headers(200, content_type)
        self.wfile.write(file_path.read_bytes())
        return True

    def serve_media(self, path):
        if not FILES_DIR.is_dir():
            return self.json({"error": "media directory not found"}, 404)
        rel = unquote(path.removeprefix("/media-files/").lstrip("/"))
        file_path = (FILES_DIR / rel).resolve()
        if FILES_DIR.resolve() not in file_path.parents or not file_path.is_file():
            return self.json({"error": "media not found"}, 404)
        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        self.send_headers(200, content_type)
        self.wfile.write(file_path.read_bytes())
        return True


def main():
    port = int(os.environ.get("PORT", "8000"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
