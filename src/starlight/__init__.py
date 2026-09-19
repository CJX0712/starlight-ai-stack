"""Starlight AI Stack —— 本地优先的检索增强生成系统。

作者: 晨星

模块划分（每层只依赖下一层的接口，不依赖实现）：
    config     环境与档位
    contracts  接口契约（数据类 + Protocol）
    models     模型服务适配（Ollama / OpenAI 兼容）
    embedder   向量化
    store      存储（sqlite-vec 稠密 + BM25 稀疏）
    ingest     摄取与切块
    retriever  混合检索与融合
    generator  生成与引用解析
    pipeline   编排
    server     API 网关
    selftest   逐模块自检
"""

__version__ = "1.0.0"
__author__ = "晨星"

from .config import Settings, PROFILES  # noqa: F401
from .contracts import Answer, Chunk, Document, ScoredChunk  # noqa: F401

__all__ = ["__version__", "__author__", "Settings", "PROFILES", "Answer", "Chunk", "Document", "ScoredChunk"]
