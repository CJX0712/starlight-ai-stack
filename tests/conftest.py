"""测试公共夹具。

作者: 晨星

关键设计：用一个「确定性哈希嵌入器」替代真实嵌入模型。
这样存储、检索这些模块的单元测试不依赖模型服务，断网、无 GPU 也能跑。
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class HashEmbedder:
    """把文本映射成确定性伪向量：相同文本必得相同向量，语义无关但几何性质稳定。"""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            v = [(h[i % len(h)] / 255.0) * 2 - 1 for i in range(self._dim)]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
