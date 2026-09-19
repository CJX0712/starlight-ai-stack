"""工具层：受控工具注册表。

作者: 晨星

设计要点：
- 工具是「数据」而不是「代码分支」：新增工具只需注册一个 ToolSpec，Agent 循环不用改
- 一切外部输入都验证：参数缺失、类型不符、未知工具、执行超时、输出过长，全部转成
  可读的观察结果喂回模型，而不是抛异常炸掉整条链路
- 计算器不用 eval()，而是走 AST 白名单求值，避免任意代码执行
"""

from __future__ import annotations

import ast
import math
import operator
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

MAX_OBSERVATION_CHARS = 1200

_ALLOWED_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "exp": math.exp}


class ToolError(RuntimeError):
    """工具调用失败（参数、超时、执行异常等），调用方应把消息作为观察结果回喂模型。"""


def calculate(expression: str) -> float:
    """AST 白名单算术求值。只允许数字、四则运算、幂、取模与少量数学函数。"""
    if not expression or len(expression) > 200:
        raise ToolError("表达式为空或过长")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ToolError("只允许数值常量")
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN:
            return _ALLOWED_BIN[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UN:
            return _ALLOWED_UN[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
                return float(_ALLOWED_FUNCS[node.func.id](*[_eval(a) for a in node.args]))
            raise ToolError("不允许的函数调用")
        raise ToolError(f"不支持的语法: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"表达式语法错误: {exc.msg}") from exc
    value = _eval(tree)
    if not math.isfinite(value):
        raise ToolError("结果不是有限数")
    return value


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的完整定义：给模型看的元数据 + 真正执行的函数。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """注册表 + 统一调用入口（含参数校验与输出截断）。"""

    def __init__(self, max_output: int = MAX_OBSERVATION_CHARS) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.max_output = max_output

    def register(self, spec: ToolSpec, override: bool = False) -> None:
        if spec.name in self._tools and not override:
            raise ToolError(f"工具 {spec.name} 已注册")
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.to_openai() for t in self._tools.values()]

    def describe(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def call(self, name: str, args: dict[str, Any] | None) -> str:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"未知工具 {name}，可用工具: {', '.join(self.names())}")
        args = args or {}
        if not isinstance(args, dict):
            raise ToolError("参数必须是 JSON 对象")
        required = spec.parameters.get("required", [])
        missing = [k for k in required if k not in args or args[k] in (None, "")]
        if missing:
            raise ToolError(f"缺少必填参数: {', '.join(missing)}")
        try:
            out = spec.handler(args)
        except ToolError:
            raise
        except Exception as exc:  # 工具内部异常同样转成可读观察结果
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        text = str(out)
        if len(text) > self.max_output:
            text = text[: self.max_output] + f"\n...(已截断，共 {len(text)} 字)"
        return text


def build_default_tools(pipeline, on_chunks: Callable[[list], None] | None = None) -> ToolRegistry:
    """按当前链路装配默认工具集。

    on_chunks：knowledge_search 命中时把候选块回传给编排层，
    用于最终答案的引用映射（工具只负责返回文本，引用由上层统一处理）。
    """
    reg = ToolRegistry()
    counter = {"n": 0}  # 运行内全局编号：多次检索的 [n] 不撞号，引用才不会指错块

    def knowledge_search(args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        k = int(args.get("k", 5) or 5)
        k = max(1, min(k, 10))
        hits = pipeline.search(query, k)
        if not hits:
            return "知识库中没有检索到相关内容。"
        if on_chunks is not None:
            on_chunks(hits)
        base = counter["n"]
        counter["n"] = base + len(hits)
        lines = []
        for i, sc in enumerate(hits, start=1):
            lines.append(f"[{base + i}] 来源: {sc.chunk.meta.get('source', '')}\n{sc.chunk.text}")
        return "\n\n".join(lines)

    def list_documents(_args: dict[str, Any]) -> str:
        stats = pipeline.store.stats()
        return f"知识库现有文档 {stats['docs']} 篇，切块 {stats['chunks']} 个，词项 {stats['lexicon_terms']} 个。"

    def calculator(args: dict[str, Any]) -> str:
        expression = str(args.get("expression", "")).strip()
        value = calculate(expression)
        return f"{expression} = {value:g}"

    def current_time(_args: dict[str, Any]) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reg.register(ToolSpec(
        name="knowledge_search",
        description="在本地知识库中做混合检索（BM25 + 向量）。需要任何事实性信息时先用它。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题，用自然语言"},
                "k": {"type": "integer", "description": "返回条数，默认 5，最多 10"},
            },
            "required": ["query"],
        },
        handler=knowledge_search,
    ))
    reg.register(ToolSpec(
        name="calculator",
        description="计算算术表达式，支持 + - * / ** % 与 sqrt/abs/round/log/exp。",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "例如 (3+4)*2/7"}},
            "required": ["expression"],
        },
        handler=calculator,
    ))
    reg.register(ToolSpec(
        name="list_documents",
        description="查看知识库当前规模（文档数、切块数）。",
        parameters={"type": "object", "properties": {}},
        handler=list_documents,
    ))
    reg.register(ToolSpec(
        name="current_time",
        description="获取本机当前时间。",
        parameters={"type": "object", "properties": {}},
        handler=current_time,
    ))
    return reg
