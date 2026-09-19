"""向量化模块。

作者: 晨星

单一职责：把文本变成定长向量，其他一概不管。
维度在首次调用时探测并缓存，因为不同嵌入模型维度不同（bge-m3=1024，nomic=768），
存储层的向量表必须知道维度才能建表。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import ModelProvider


class EmbedError(RuntimeError):
    pass


class TextEmbedder:
    """批量向量化 + 归一化。"""

    def __init__(self, provider: ModelProvider, model: str, batch_size: int = 32) -> None:
        self.provider = provider
        self.model = model
        self.batch_size = max(1, batch_size)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            probe = self.embed(["维度探测"])
            self._dim = len(probe[0])
        return self._dim

    def set_dim(self, dim: int) -> None:
        self._dim = int(dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            try:
                vecs = self.provider.embed(batch, model=self.model)
            except Exception as exc:  # 后端异常统一转成领域异常
                raise EmbedError(f"嵌入失败（模型 {self.model}）: {exc}") from exc
            if len(vecs) != len(batch):
                raise EmbedError(f"嵌入返回 {len(vecs)} 条，期望 {len(batch)} 条")
            out.extend(self._normalize(v) for v in vecs)
        widths = {len(v) for v in out}
        if len(widths) != 1:
            raise EmbedError(f"嵌入维度不一致: {sorted(widths)}")
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @staticmethod
    def _normalize(vec: Any) -> list[float]:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.astype(float).tolist()
