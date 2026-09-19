"""API 网关：把内部链路暴露成稳定的 HTTP 契约。

作者: 晨星

单一职责：协议转换与传输（SSE 流式）。业务一律下沉到 pipeline。
端点契约固定，前端与 CLI 都只认这一层。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import AUTHOR, PROFILES, Settings
from .generator import AnswerGenerator
from .pipeline import RAGPipeline
from .version import __version__

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="Starlight AI Stack", version=__version__, description=f"本地优先 RAG 系统 · 作者 {AUTHOR}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipe: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipe
    if _pipe is None:
        _pipe = RAGPipeline()
        _pipe.prepare()
    return _pipe


class IngestRequest(BaseModel):
    path: str = Field(..., description="本地文件或目录绝对路径")


class IngestTextRequest(BaseModel):
    text: str
    source: str = "inline"


class ChatRequest(BaseModel):
    query: str
    k: int | None = None
    stream: bool = False


class SearchResponseItem(BaseModel):
    chunk_id: str
    score: float
    dense_score: float
    sparse_score: float
    source: str
    text: str


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    html = WEB_DIR / "index.html"
    if not html.exists():
        raise HTTPException(status_code=404, detail="缺少 web/index.html")
    return FileResponse(html)


@app.get("/health")
def health() -> dict[str, Any]:
    pipe = get_pipeline()
    try:
        info = pipe.health()
        info["version"] = __version__
        info["author"] = AUTHOR
        info["profiles"] = {k: {"llm": p.llm, "embed": p.embed, "note": p.note} for k, p in PROFILES.items()}
        return info
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"链路未就绪: {exc}") from exc


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, Any]:
    pipe = get_pipeline()
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"路径不存在: {req.path}")
    report = pipe.ingest(str(path))
    return report.__dict__


@app.post("/ingest_text")
def ingest_text(req: IngestTextRequest) -> dict[str, Any]:
    pipe = get_pipeline()
    return {"added_chunks": pipe.ingest_text(req.text, req.source)}


@app.get("/search")
def search(q: str, k: int = 5, debug: bool = False) -> dict[str, Any]:
    pipe = get_pipeline()
    if debug:
        fused = pipe.search_debug(q, k)
        return {
            "query": q,
            "dense": fused["dense"],
            "sparse": fused["sparse"],
            "results": [_item(s) for s in fused["fused"]],
        }
    return {"query": q, "results": [_item(s) for s in pipe.search(q, k)]}


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    pipe = get_pipeline()
    answer = pipe.ask(req.query, req.k)
    return {
        "answer": answer.text,
        "citations": [c.__dict__ for c in answer.citations],
        "usage": answer.usage,
        "model": answer.model,
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    pipe = get_pipeline()
    ctx, gen = pipe.ask_stream(req.query, req.k)

    def events() -> Iterator[str]:
        head = {
            "type": "context",
            "results": [_item(s) for s in ctx],
        }
        yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"
        buf = ""
        for piece in gen:
            buf += piece
            yield f"data: {json.dumps({'type': 'delta', 'text': piece}, ensure_ascii=False)}\n\n"
        citations = AnswerGenerator.parse_citations(buf, ctx)
        tail = {"type": "done", "citations": [c.__dict__ for c in citations], "text": buf}
        yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, Any]:
    pipe = get_pipeline()
    return {"doc_id": doc_id, "removed_chunks": pipe.store.delete_document(doc_id)}


@app.get("/stats")
def stats() -> dict[str, Any]:
    return get_pipeline().store.stats()


def _item(s) -> dict[str, Any]:
    return {
        "chunk_id": s.chunk.id,
        "score": round(s.score, 6),
        "dense_score": round(s.dense_score, 6),
        "sparse_score": round(s.sparse_score, 6),
        "dense_rank": s.dense_rank,
        "sparse_rank": s.sparse_rank,
        "source": s.chunk.meta.get("source", ""),
        "text": s.chunk.text,
    }
