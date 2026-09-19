"""Agent 编排：带护栏的工具调用循环（ReAct 风格）。

作者: 晨星

单一职责：决定"要不要调工具、调哪个、什么时候停"。
工具本身在 tools.py，模型在 models.py，检索在 retriever.py —— 这里只做循环与护栏。

四道护栏（缺一个都会在生产上咬人）：
1. 步数上限：模型可能无限调工具，硬性截断
2. 单次工具超时：检索卡住不能拖死整个请求
3. 工具异常收敛：任何失败都变成观察结果回喂，而不是 500
4. 引用编号全局唯一：多次检索的 [n] 不能互相撞号，否则引用会指错块
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .config import ModelProfile
from .contracts import Answer, Citation, ModelProvider, ScoredChunk
from .generator import ANSWER_MARK, AnswerGenerator
from .tools import ToolError, ToolRegistry

AGENT_SYSTEM_PROMPT = (
    "你是本地知识库智能体，可以调用工具获取事实。\n"
    "规则：\n"
    "1. 需要事实性信息时必须调用 knowledge_search，不要凭记忆回答。\n"
    "2. 数学计算必须调用 calculator。\n"
    "3. 最终回答必须以「答案：」开头，并用方括号标注来源编号，编号沿用检索结果里的编号。\n"
    "4. 不得补充工具返回内容之外的信息。\n"
    "5. 工具调用轮数有限，够用就尽快给出结论。\n"
)


@dataclass
class AgentStep:
    index: int
    tool: str
    args: dict[str, Any]
    observation: str
    ok: bool
    elapsed_s: float


@dataclass
class AgentTrace:
    goal: str
    steps: list[AgentStep] = field(default_factory=list)
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    stopped_reason: str = ""
    model: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "answer": self.answer,
            "citations": [c.__dict__ for c in self.citations],
            "stopped_reason": self.stopped_reason,
            "model": self.model,
            "elapsed_s": round(self.elapsed_s, 2),
            "steps": [
                {
                    "index": s.index, "tool": s.tool, "args": s.args, "ok": s.ok,
                    "elapsed_s": round(s.elapsed_s, 2),
                    "observation": s.observation[:300],
                }
                for s in self.steps
            ],
        }


class AgentRuntime:
    """工具调用循环 + 护栏。"""

    def __init__(
        self,
        provider: ModelProvider,
        profile: ModelProfile,
        registry: ToolRegistry,
        max_steps: int = 4,
        tool_timeout_s: float = 60.0,
        collector: list[ScoredChunk] | None = None,
    ) -> None:
        self.provider = provider
        self.profile = profile
        self.registry = registry
        self.max_steps = max(1, max_steps)
        self.tool_timeout_s = tool_timeout_s
        # 与 tools 的 on_chunks 共用同一个列表对象，保证引用编号与块顺序一致
        self._collected = collector if collector is not None else []

    # ---------- 内部 ----------

    @property
    def collected(self) -> list[ScoredChunk]:
        """本轮运行中工具检索到的候选块（引用编号与它一一对应）。"""
        return list(self._collected)

    def _options(self) -> dict[str, Any]:
        p = self.profile
        return {
            "num_ctx": p.ctx,
            "num_thread": p.num_thread,
            "temperature": p.temperature,
            "num_predict": p.max_tokens,
        }

    def _call_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool, float]:
        """带超时地执行工具；任何失败都转成观察文本，绝不向上抛。"""
        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                obs = pool.submit(self.registry.call, name, args).result(timeout=self.tool_timeout_s)
            return obs, True, time.time() - t0
        except FutureTimeout:
            return f"工具 {name} 执行超时（>{self.tool_timeout_s:.0f}s）", False, time.time() - t0
        except ToolError as exc:
            return f"工具调用失败：{exc}", False, time.time() - t0
        except Exception as exc:  # 兜底：任何异常都不能让链路 500
            return f"工具异常：{type(exc).__name__}: {exc}", False, time.time() - t0

    @staticmethod
    def _normalize_calls(raw: list[Any]) -> list[tuple[str, dict[str, Any]]]:
        """把不同后端的工具调用格式统一成 (name, args) 列表。"""
        out: list[tuple[str, dict[str, Any]]] = []
        for c in raw or []:
            fn = c.get("function", c) if isinstance(c, dict) else {}
            name = fn.get("name") or c.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if name:
                out.append((str(name), args if isinstance(args, dict) else {"value": args}))
        return out

    # ---------- 主流程 ----------

    def run_iter(self, goal: str) -> Iterator[dict[str, Any]]:
        """逐步产出事件：step（工具）、answer（最终答案）。"""
        self._collected.clear()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]
        trace = AgentTrace(goal=goal, model=self.profile.llm)
        t0 = time.time()
        step_index = 0
        answer = ""

        for _ in range(self.max_steps):
            resp = self.provider.chat(
                messages, model=self.profile.llm,
                tools=self.registry.specs(), options=self._options(),
            )
            if not isinstance(resp, dict):
                answer = str(resp)
                trace.stopped_reason = "answered"
                break
            calls = self._normalize_calls(resp.get("tool_calls") or [])
            if not calls:
                answer = (resp.get("content") or "").strip()
                trace.stopped_reason = "answered"
                break

            messages.append(
                {"role": "assistant", "content": resp.get("content") or "", "tool_calls": resp.get("tool_calls")}
            )
            for name, args in calls:
                obs, ok, cost = self._call_tool(name, args)
                step_index += 1
                step = AgentStep(step_index, name, args, obs, ok, cost)
                trace.steps.append(step)
                messages.append({"role": "tool", "content": obs, "tool_name": name})
                yield {"type": "step", "step": step}
        else:
            # 步数用尽：强制收口，再要一次纯回答（不再给工具）
            messages.append({"role": "user", "content": "请基于以上观察直接给出结论，不要再调用工具。"})
            resp = self.provider.chat(messages, model=self.profile.llm, options=self._options())
            answer = ((resp.get("content") or "") if isinstance(resp, dict) else str(resp)).strip()
            if not answer:
                # 兜底：模型仍然返回工具调用而没给正文时，不能让用户拿到空答案
                tail = [s.observation[:200] for s in trace.steps[-2:]]
                answer = "未能生成结论，以下是工具返回的原始信息：\n" + "\n".join(tail)
            trace.stopped_reason = "max_steps_forced_answer"

        answer = AnswerGenerator.strip_preamble(answer)
        answer = AnswerGenerator.enforce_grounding(answer, self._collected)
        trace.answer = answer
        trace.citations = AnswerGenerator.parse_citations(answer, self._collected)
        trace.elapsed_s = time.time() - t0
        yield {"type": "answer", "trace": trace}

    def run(self, goal: str, on_step: Callable[[AgentStep], None] | None = None) -> AgentTrace:
        trace: AgentTrace | None = None
        for event in self.run_iter(goal):
            if event["type"] == "step" and on_step:
                on_step(event["step"])
            elif event["type"] == "answer":
                trace = event["trace"]
        assert trace is not None
        return trace

    def answer(self, goal: str) -> Answer:
        t = self.run(goal)
        return Answer(text=t.answer, citations=t.citations, model=t.model,
                      usage={"steps": len(t.steps), "stopped_reason": t.stopped_reason})
