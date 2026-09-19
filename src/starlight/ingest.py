"""摄取模块：文件 -> 文档 -> 切块 -> 入库。

作者: 晨星

单一职责：只负责"把外部资料变成库里的块"。不检索、不生成。
幂等性是硬要求：同一个文件再摄取一次，必须 0 新增，否则重复摄取会污染索引。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .config import Settings
from .contracts import Chunk, Document, IngestReport
from .embedder import TextEmbedder
from .store import SqliteStore
from .text import chunk_text, normalize

SUPPORTED = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".json", ".csv", ".log"}
_TAG_STRIP = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG_ALL = re.compile(r"<[^>]+>")


def parse_file(path: Path) -> str:
    """按扩展名选择解析器。PDF 用 pypdf（业界标准纯 Python 实现）。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return normalize("\n".join((page.extract_text() or "") for page in reader.pages))
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        text = _TAG_STRIP.sub(" ", text)
        text = _TAG_ALL.sub(" ", text)
    if suffix == ".json":
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=1)
        except json.JSONDecodeError:
            pass
    return normalize(text)


def doc_id_for(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


class Ingestor:
    """摄取器：切块与入库由它编排，向量化交给注入的 Embedder。"""

    def __init__(self, store: SqliteStore, embedder: TextEmbedder, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def build_chunks(self, doc: Document) -> list[Chunk]:
        pieces = chunk_text(doc.text, self.settings.chunk_size, self.settings.chunk_overlap)
        return [
            Chunk(
                id=f"{doc.id}#{i}",
                doc_id=doc.id,
                seq=i,
                text=piece,
                meta={"source": doc.source, "seq": i, **(doc.meta or {})},
            )
            for i, piece in enumerate(pieces)
        ]

    def ingest_path(self, path: str | Path) -> IngestReport:
        root = Path(path)
        report = IngestReport()
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
        report.scanned = len(files)
        for f in files:
            report.files.append(str(f))
            try:
                added = self.ingest_file(f)
                if added == 0:
                    report.skipped += 1
                else:
                    report.added += 1
                    report.chunks += added
            except Exception:
                report.failed += 1
        return report

    def ingest_file(self, path: Path) -> int:
        """返回新增块数；内容未变返回 0。"""
        source = str(path.resolve())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        doc_id = doc_id_for(source)
        if self.store.doc_hash(doc_id) == digest:
            return 0
        text = parse_file(path)
        if not text.strip():
            return 0
        doc = Document(id=doc_id, source=source, text=text,
                       meta={"hash": digest, "name": path.name, "bytes": path.stat().st_size})
        chunks = self.build_chunks(doc)
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        return self.store.upsert_document(doc, chunks, vectors)

    def ingest_text(self, text: str, source: str = "inline") -> int:
        """把一段内联文本入库，便于测试与 API 调用。"""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = doc_id_for(source)
        if self.store.doc_hash(doc_id) == digest:
            return 0
        doc = Document(id=doc_id, source=source, text=normalize(text),
                       meta={"hash": digest, "name": source, "bytes": len(text)})
        chunks = self.build_chunks(doc)
        vectors = self.embedder.embed([c.text for c in chunks])
        return self.store.upsert_document(doc, chunks, vectors)
