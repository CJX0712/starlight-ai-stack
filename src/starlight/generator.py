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
    "5. 不得补充资料中未出现的信息，包括擅自展开缩写、补充定义或举例。\n"
    "   资料里出现缩写就照原样使用（例如资料写 RRF 就写 RRF），不要自行解释成全称。\n"
)

ANSWER_MARK = "答案："


def _is_head_mark(idx: int) -> bool:
    """标记必须真的在开头才算合规。

    实测踩坑：思考型模型会写出「我应该以答案：开头……」这类分析，
    标记位置在 5，若放宽到 8 就会把分析段后半截当成答案切出来。宁可不切，不可误切。
    """
    return 0 <= idx <= 1

_CITE = re.compile(r"\[(\d+)\]")

# 缩写 + 括号展开，例如 RRF（Reciprocal Rank Fusion）
_ACRONYM_EXPANSION = re.compile(r"\b([A-Z]{2,8})[（(]([A-Za-z][^）)]{1,80})[）)]")


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
            # 助手预填充：把「答案：」先写进对话，模型只能从这里续写。
            # 实测（qwen3:4b，本机）：同一问题从 29.3s 的思考废话变成 1.7s 的直接答案，
            # 且不再把 token 预算烧在推理过程上导致答案被截断。
            {"role": "assistant", "content": ANSWER_MARK},
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
        text = self.enforce_grounding(self.strip_preamble(raw), context)
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
        # 已有预填充，模型是从「答案：」之后开始续写的，直接下发即可，
        # 再缓冲等标记只会白白增加首字延迟。
        for piece in gen:
            yield piece

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
    def enforce_grounding(text: str, context: list[ScoredChunk]) -> str:
        """剔除资料中无依据的缩写展开。

        实测踩坑：资料只写「RRF 融合排序」，qwen2.5:7B 自行补成
        「RRF（Relevance Function Fusion）」——括号里的全称是编造的。
        提示词约束对它无效，所以这里做确定性校验：把括号展开的每个英文词
        拿去资料里核对，核对不上就整段删掉，只留缩写本身。
        """
        material = " ".join(sc.chunk.text for sc in context).lower()
        if not material:
            return text

        def repl(m: re.Match) -> str:
            acronym, inner = m.group(1), m.group(2)
            if acronym.lower() not in material:
                return m.group(0)  # 缩写本身就不是资料里的，不处理
            words = re.findall(r"[A-Za-z]{2,}", inner)
            if words and all(w.lower() in material for w in words):
                return m.group(0)  # 展开能在资料里找到依据，保留
            return acronym

        return _ACRONYM_EXPANSION.sub(repl, text)

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
