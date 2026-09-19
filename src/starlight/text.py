"""文本切分与分词（中文友好）。

作者: 晨星

两个决定，都是被实测逼出来的：
1. 不做 SQLite FTS5：trigram 分词器对 2 字中文词（如"检索"）命中 0 条，
   unicode61 分词器对中文几乎完全失效。所以稀疏检索自己实现 BM25 倒排。
2. 中文按"单字 + 相邻二字组"建索引：既能命中长词，也能命中 2 字短查询。
"""

from __future__ import annotations

import re

CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_CJK_RUN = re.compile(rf"[{CJK}]+")
_LATIN_RUN = re.compile(r"[a-zA-Z0-9_]+")
_SENT_SPLIT = re.compile(rf"(?<=[{CJK}。！？；!?;\n])")


def normalize(text: str) -> str:
    """归一化：统一空白，去掉控制字符。"""
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_cjk(ch: str) -> bool:
    return bool(_CJK_RUN.match(ch))


def tokenize(text: str) -> list[str]:
    """中英混排分词：中文取单字与相邻二字组，英文数字取整词。"""
    text = text.lower()
    tokens: list[str] = []
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i] for i in range(len(run)))
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    for run in _LATIN_RUN.findall(text):
        tokens.append(run)
    return tokens


def split_sentences(text: str) -> list[str]:
    """按中英文句读切句，用于让切块尽量落在自然边界上。"""
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, size: int = 420, overlap: int = 80) -> list[str]:
    """按句子聚合切块，块长尽量贴近 size，块间保留 overlap 个字符的重叠。

    重叠的意义：跨块的语义（一个概念写在两句之间）不会因切断而丢失。
    """
    if size <= 0:
        raise ValueError("size 必须为正")
    if overlap >= size:
        raise ValueError("overlap 必须小于 size")

    text = normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sentences = split_sentences(text) or [text]
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
            continue
        if len(buf) + len(s) + 1 <= size:
            buf = f"{buf}{s}" if buf.endswith(("\n", "。", "！", "？")) else f"{buf} {s}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail} {s}".strip() if tail else s
        while len(buf) > size:
            chunks.append(buf[:size])
            buf = buf[size - overlap :] if overlap else buf[size:]
    if buf:
        chunks.append(buf)

    return [c for c in (x.strip() for x in chunks) if c]
