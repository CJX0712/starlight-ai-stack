"""生成模块：带引用约束的问答生成。

作者: 晨星

单一职责：把「问题 + 候选块」变成「答案 + 可追溯引用」。
引用不是装饰：答案里出现的每个 [n] 必须能映射回真实块，否则判定为幻觉，
解析层会把它剔除，宁可少引用也不编造。
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from .config import ModelProfile
from .contracts import Answer, Citation, ModelProvider, ScoredChunk

SYSTEM_PROMPT = (
    "你是本地知识库助手。只依据提供的【参考资料】回答，不得使用资料之外的知识。\n"
    "规则：\n"
    "1. 每个关键结论后用方括号标注来源编号，例如 [1] 或 [2][3]。\n"
    "2. 若资料不足以回答，直接说明资料中没有相关信息，不要猜测。\n"
    "3. 用简体中文回答，条理清晰，不超过 300 字。\n"
    "4. 输出必须以「答案：」这四个字开头，其后直接写结论，不要写分析过程或开场白。\n"
)

ANSWER_MARK = "答案："


def _is_head_mark(idx: int) -> bool:
    """标记必须真的在开头才算合规。

    实测踩坑：思考型模型会写出「我应该以答案：开头……」这类分析，
    标记位置在 5，若放宽到 8 就会把分析段后半截当成答案切出来。宁可不切，不可误切。
    """
    return 0 <= idx <= 1

_CITE = re.compile(r"\[(\d+)\]")


class AnswerGenerator:
    """生成 + 引用解析。"""

    def __init__(self, provider: ModelProvider, profile: ModelProfile) -> None:
        self.provider = provider
        self.profile = profile

    @staticmethod
    def build_context(context: list[ScoredChunk]) -> str:
        blocks = []
        for i, sc in enumerate(context, start=1):
            src = sc.chunk.meta.get("source", "")
            blocks.append(f"[{i}] 来源: {src}\n{sc.chunk.text}")
        return "\n\n".join(blocks)

    def build_messages(self, query: str, context: list[ScoredChunk]) -> list[dict[str, Any]]:
        material = self.build_context(context)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考资料】\n{material}\n\n【问题】\n{query}"},
        ]

    def options(self, max_tokens: int | None = None) -> dict[str, Any]:
        p = self.profile
        return {
            "num_ctx": p.ctx,
            "num_thread": p.num_thread,
            "temperature": p.temperature,
            "num_predict": max_tokens or p.max_tokens,
        }

    def generate(self, query: str, context: list[ScoredChunk], *, max_tokens: int | None = None) -> Answer:
        messages = self.build_messages(query, context)
        result = self.provider.chat(messages, model=self.profile.llm, options=self.options(max_tokens))
        raw = (result.get("content") or "").strip() if isinstance(result, dict) else str(result)
        text = self.strip_preamble(raw)
        return Answer(
            text=text,
            citations=self.parse_citations(text, context),
            usage=result.get("usage", {}) if isinstance(result, dict) else {},
            model=self.profile.llm,
        )

    def stream(self, query: str, context: list[ScoredChunk], *, max_tokens: int | None = None) -> Iterator[str]:
        """流式输出：思考型模型会先吐一段分析，这里缓冲到「答案：」才开始下发。

        兜底：若 400 字内没等到标记（模型没遵守格式），则原样放行，避免界面一直空白。
        """
        messages = self.build_messages(query, context)
        gen = self.provider.chat(
            messages, model=self.profile.llm, stream=True, options=self.options(max_tokens)
        )
        if isinstance(gen, dict):  # 后端不支持流式的兜底
            yield self.strip_preamble(gen.get("content") or "")
            return
        buf = ""
        started = False
        for piece in gen:
            if started:
                yield piece
                continue
            buf += piece
            idx = buf.find(ANSWER_MARK)
            if _is_head_mark(idx):
                started = True
                yield buf[idx + len(ANSWER_MARK):]
            elif len(buf) > 400:
                started = True
                yield buf

    @staticmethod
    def strip_preamble(text: str) -> str:
        """只认「开头就是标记」的合规输出。

        思考型模型会把标记写进自己的分析里（"我应该以答案：开头"），
        若不加位置判定就会把分析段的后半截当成答案切出来。
        """
        t = text.strip()
        idx = t.find(ANSWER_MARK)
        if _is_head_mark(idx):
            body = t[idx + len(ANSWER_MARK):].strip()
            if len(body) >= 8:
                return body
        return t

    @staticmethod
    def parse_citations(text: str, context: list[ScoredChunk]) -> list[Citation]:
        """抽取答案中的 [n]，映射回真实块；越界编号直接丢弃。"""
        out: list[Citation] = []
        seen: set[int] = set()
        for m in _CITE.finditer(text):
            idx = int(m.group(1))
            if idx < 1 or idx > len(context) or idx in seen:
                continue
            seen.add(idx)
            sc = context[idx - 1]
            out.append(
                Citation(
                    index=idx,
                    chunk_id=sc.chunk.id,
                    source=sc.chunk.meta.get("source", ""),
                    excerpt=sc.chunk.text[:120],
                )
            )
        return out
