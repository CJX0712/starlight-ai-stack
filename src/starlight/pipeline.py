"""编排层：把各模块按依赖装配成一条可运行链路。

作者: 晨星

这一层只做装配与流程编排，不承载任何算法。
每个子模块都可以被单独替换（换后端、换存储、换检索策略），编排层代码不动。
"""

from __future__ import annotations

from typing import Any, Iterator

from .config import Settings, resolve_profile
from .contracts import Answer, IngestReport, ScoredChunk
from .embedder import TextEmbedder
from .generator import AnswerGenerator
from .ingest import Ingestor
from .models import OllamaProvider, build_provider
from .retriever import HybridRetriever
from .store import DimensionMismatch, SqliteStore


class RAGPipeline:
    """端到端链路：摄取 -> 检索 -> 生成。"""

    def __init__(self, settings: Settings | None = None, backend: str = "ollama") -> None:
        self.settings = settings or Settings()
        self.provider = build_provider(backend, self.settings.ollama_base_url, self.settings.timeout_s)
        self.profile, self.notes = self._resolve()
        self.embedder = TextEmbedder(self.provider, self.profile.embed)
        self.store = SqliteStore(self.settings.db_path)
        self.ingestor = Ingestor(self.store, self.embedder, self.settings)
        self.retriever = HybridRetriever(
            self.store,
            self.embedder,
            candidate_k=self.settings.candidate_k,
            rrf_k=self.settings.rrf_k,
        )
        self.generator = AnswerGenerator(self.provider, self.profile)

    def _resolve(self):
        """按本机实际可用模型修正档位，缺失模型自动降级而不是直接崩。"""
        try:
            info = self.provider.models_info()
        except Exception:
            info = {}
        if not info:
            return self.settings.model, ["模型服务不可达，未做模型降级解析"]
        loaded = getattr(self.provider, "loaded_models", lambda: [])()
        return resolve_profile(info, self.settings.model, loaded_models=loaded)

    def prepare(self) -> None:
        """确保向量表维度与当前嵌入模型一致；不一致则原地重建（原文不丢）。"""
        try:
            self.store.ensure_vec_table(self.embedder.dim)
        except DimensionMismatch:
            self.store.rebuild_vectors(self.embedder)

    def health(self) -> dict[str, Any]:
        models: list[str] = []
        try:
            models = self.provider.list_models()
        except Exception:
            pass
        return {
            "backend_up": self.provider.health(),
            "profile": self.profile.name,
            "requested": {"llm": self.settings.model.llm, "embed": self.settings.model.embed},
            "llm": self.profile.llm,
            "embed": self.profile.embed,
            "llm_ready": self.profile.llm in models,
            "embed_ready": self.profile.embed in models,
            "available_models": models,
            "notes": self.notes,
            "stats": self.store.stats(),
        }

    def ingest(self, path: str) -> IngestReport:
        self.prepare()
        return self.ingestor.ingest_path(path)

    def ingest_text(self, text: str, source: str = "inline") -> int:
        self.prepare()
        return self.ingestor.ingest_text(text, source)

    def search(self, query: str, k: int | None = None) -> list[ScoredChunk]:
        return self.retriever.retrieve(query, k or self.settings.top_k)

    def search_debug(self, query: str, k: int | None = None) -> dict:
        return self.retriever.fuse(query, k or self.settings.top_k)

    def ask(self, query: str, k: int | None = None) -> Answer:
        ctx = self.search(query, k)
        if not ctx:
            return Answer(text="知识库里没有检索到相关内容，请先摄取资料。", model=self.profile.llm)
        return self.generator.generate(query, ctx)

    def ask_stream(self, query: str, k: int | None = None) -> tuple[list[ScoredChunk], Iterator[str]]:
        ctx = self.search(query, k)
        if not ctx:
            def _empty() -> Iterator[str]:
                yield "知识库里没有检索到相关内容，请先摄取资料。"
            return [], _empty()
        return ctx, self.generator.stream(query, ctx)

    def close(self) -> None:
        self.store.close()
