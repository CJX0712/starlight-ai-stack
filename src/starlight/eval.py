"""评测模块：金标集 + 指标 + 门禁。

作者: 晨星

三条设计原则：
1. 指标全部可离线计算（纯函数），所以指标本身的正确性可以用单元测试锁死
2. 中文匹配先做「去空白归一化」，否则 "3 个工作日" 与 "3个工作日" 会被判成不匹配
3. 门禁是硬阈值：不达标就非零退出，而不是打印一句"建议优化"
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Mode = Literal["rag", "agent"]

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """小写 + 去掉所有空白。中英混排文本做包含判断前必须先归一化。"""
    return _WS.sub("", (text or "").lower())


def contains_all(text: str, keywords: list[str]) -> bool:
    """所有关键词都要出现，避免"碰对一个词就算通过"。"""
    if not keywords:
        return False
    t = normalize(text)
    return all(normalize(k) in t for k in keywords)


def first_hit_rank(candidates: list[str], keywords: list[str]) -> int | None:
    """第一个命中的候选排名（1 起）。命中定义：该候选含全部关键词。"""
    for i, text in enumerate(candidates, start=1):
        if contains_all(text, keywords):
            return i
    return None


def rank_to_rr(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


@dataclass
class EvalCase:
    id: str
    question: str
    keywords: list[str]
    source: str = ""


@dataclass
class CaseResult:
    case_id: str
    question: str
    ctx_hit: bool = False
    top1_hit: bool = False
    rank: int | None = None
    answer_hit: bool = False
    citation_valid: bool = True
    n_ctx: int = 0
    latency_s: float = 0.0
    answer: str = ""
    error: str = ""

    @property
    def rr(self) -> float:
        return rank_to_rr(self.rank)


@dataclass
class EvalReport:
    mode: str
    profile: str
    model: str
    results: list[CaseResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "profile": self.profile,
            "model": self.model,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "passed": self.passed,
            "failures": self.failures,
            "results": [asdict(r) | {"rr": r.rr} for r in self.results],
        }


def aggregate(results: list[CaseResult]) -> dict[str, float]:
    if not results:
        return {"cases": 0}
    n = len(results)
    return {
        "cases": float(n),
        "ctx_hit": sum(1 for r in results if r.ctx_hit) / n,
        "top1_hit": sum(1 for r in results if r.top1_hit) / n,
        "mrr": sum(r.rr for r in results) / n,
        "answer_hit": sum(1 for r in results if r.answer_hit) / n,
        "citation_valid": sum(1 for r in results if r.citation_valid) / n,
        "mean_latency_s": sum(r.latency_s for r in results) / n,
    }


def gate(metrics: dict[str, float], thresholds: dict[str, float], results: list[CaseResult] | None = None) -> list[str]:
    """返回未达标项列表；空列表表示通过。"""
    failures: list[str] = []
    for key, floor in thresholds.items():
        actual = metrics.get(key)
        if actual is None:
            continue
        if actual + 1e-9 < floor:
            failures.append(f"{key} {actual:.3f} < 阈值 {floor:.3f}")
    for r in results or []:
        if r.error:
            failures.append(f"{r.case_id} 执行异常: {r.error}")
    return failures


def load_golden(path: str | Path) -> tuple[list[EvalCase], dict[str, float], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [
        EvalCase(id=c["id"], question=c["question"], keywords=list(c["keywords"]), source=c.get("source", ""))
        for c in data.get("cases", [])
    ]
    return cases, dict(data.get("thresholds", {})), str(data.get("version", ""))


class Evaluator:
    """对一条已装配好的链路做评测。"""

    def __init__(self, pipeline, cases: list[EvalCase], top_k: int = 5, mode: Mode = "rag") -> None:
        self.pipeline = pipeline
        self.cases = cases
        self.top_k = top_k
        self.mode = mode

    def run_case(self, case: EvalCase) -> CaseResult:
        r = CaseResult(case_id=case.id, question=case.question)
        t0 = time.time()
        try:
            if self.mode == "agent":
                runtime = self.pipeline.build_agent(max_steps=3)
                trace = runtime.run(case.question)
                texts = [sc.chunk.text for sc in runtime.collected]
                answer, citations = trace.answer, trace.citations
            else:
                hits = self.pipeline.search(case.question, self.top_k)
                texts = [sc.chunk.text for sc in hits]
                ans = self.pipeline.ask(case.question, self.top_k)
                answer, citations = ans.text, ans.citations
        except Exception as exc:
            r.error = f"{type(exc).__name__}: {exc}"
            r.latency_s = time.time() - t0
            return r

        r.latency_s = time.time() - t0
        r.answer = answer
        r.n_ctx = len(texts)
        joined = "\n".join(texts)
        r.ctx_hit = contains_all(joined, case.keywords)
        r.top1_hit = bool(texts) and contains_all(texts[0], case.keywords)
        r.rank = first_hit_rank(texts, case.keywords)
        r.answer_hit = contains_all(answer, case.keywords)
        r.citation_valid = all(1 <= c.index <= len(texts) for c in citations)
        return r

    def run(self, limit: int | None = None) -> EvalReport:
        cases = self.cases[:limit] if limit else self.cases
        profile = self.pipeline.profile
        results = [self.run_case(c) for c in cases]
        return EvalReport(
            mode=self.mode,
            profile=profile.name,
            model=profile.llm,
            results=results,
            metrics=aggregate(results),
        )
