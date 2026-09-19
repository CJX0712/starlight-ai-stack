"""模块契约（接口定义）。

作者: 晨星

设计原则：上层只依赖这里的 Protocol 和数据类，不依赖任何具体实现。
因此 Ollama / llama.cpp / vLLM / OpenAI 之间可以整层替换而不改一行编排代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable


@dataclass
class Document:
    """一篇被解析后的原始文档。"""

    id: str
    source: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """最小检索单元。rowid 由存储层分配，入库后回填。"""

    id: str
    doc_id: str
    seq: int
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    rowid: int | None = None


@dataclass
class ScoredChunk:
    """一次检索命中的结果，保留两路分数便于诊断融合效果。"""

    chunk: Chunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass
class Citation:
    index: int
    chunk_id: str
    source: str
    excerpt: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""


@dataclass
class IngestReport:
    scanned: int = 0
    added: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    files: list[str] = field(default_factory=list)


@runtime_checkable
class ModelProvider(Protocol):
    """模型服务：对话 + 向量化。"""

    def health(self) -> bool: ...
    def list_models(self) -> list[str]: ...
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Iterator[str]: ...
    def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...


@runtime_checkable
class Embedder(Protocol):
    """向量化：文本 -> 定长向量。"""

    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """存储：稠密向量 + 稀疏倒排 + 原文。"""

    def upsert_document(self, doc: Document, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...
    def delete_document(self, doc_id: str) -> int: ...
    def vector_search(self, vector: list[float], k: int) -> list[tuple[int, float]]: ...
    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]: ...
    def get_chunks(self, rowids: Iterable[int]) -> dict[int, Chunk]: ...
    def stats(self) -> dict[str, Any]: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]: ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, query: str, context: list[ScoredChunk], *, stream: bool = False) -> Answer | Iterator[str]: ...
