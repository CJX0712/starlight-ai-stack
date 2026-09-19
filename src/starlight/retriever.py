"""检索模块：稠密 + 稀疏双路召回，RRF 融合。

作者: 晨星

为什么用 RRF 而不是"重排器直接定序"：
交叉编码器重排在查询语言不匹配时会无人制衡地压掉正确答案（实测中文场景掉点明显）。
本机没有可用的中文 cross-encoder（bge-reranker 的 GGUF 在本机网络下拿不到），
因此采用 RRF 融合——两路互相制衡，任一路失手另一路仍能兜住。
rerank 位留了接口，将来接上 reranker 也只影响这一层。
"""

from __future__ import annotations

from .contracts import ScoredChunk
from .embedder import TextEmbedder
from .store import SqliteStore


class HybridRetriever:
    """单一职责：给定查询，返回融合排序后的候选块。"""

    def __init__(
        self,
        store: SqliteStore,
        embedder: TextEmbedder,
        candidate_k: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]:
        return self.fuse(query, k)["fused"]

    def fuse(self, query: str, k: int = 5) -> dict:
        """返回融合结果 + 两路原始排序，便于诊断"到底是哪一路把结果捞出来的"。"""
        qvec = self.embedder.embed_one(query)
        dense = self.store.vector_search(qvec, self.candidate_k)
        sparse = self.store.sparse_search(query, self.candidate_k)

        dense_rank = {rid: i for i, (rid, _) in enumerate(dense)}
        sparse_rank = {rid: i for i, (rid, _) in enumerate(sparse)}
        dense_score = {rid: 1.0 / (1.0 + d) for rid, d in dense}
        sparse_score = dict(sparse)

        rids = set(dense_rank) | set(sparse_rank)
        scored: list[ScoredChunk] = []
        chunks = self.store.get_chunks(rids)
        for rid in rids:
            ch = chunks.get(rid)
            if ch is None:
                continue
            score = 0.0
            if rid in dense_rank:
                score += self.dense_weight / (self.rrf_k + dense_rank[rid] + 1)
            if rid in sparse_rank:
                score += self.sparse_weight / (self.rrf_k + sparse_rank[rid] + 1)
            scored.append(
                ScoredChunk(
                    chunk=ch,
                    score=score,
                    dense_score=dense_score.get(rid, 0.0),
                    sparse_score=sparse_score.get(rid, 0.0),
                    dense_rank=dense_rank.get(rid),
                    sparse_rank=sparse_rank.get(rid),
                )
            )
        scored.sort(key=lambda s: -s.score)
        return {
            "fused": scored[:k],
            "dense": [(rid, round(d, 5)) for rid, d in dense[:k]],
            "sparse": [(rid, round(s, 5)) for rid, s in sparse[:k]],
            "query": query,
        }
