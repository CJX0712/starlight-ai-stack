"""Agent 循环测试：用脚本化假模型验证护栏，不依赖真实模型。

作者: 晨星

为什么必须用假模型测循环：真模型行为不确定，测不出"步数上限是否生效""超时是否收敛"。
这里把模型的每次回复写成脚本，循环的每个分支都能被精确触发。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import HashEmbedder  # noqa: E402

from starlight.agent import AgentRuntime  # noqa: E402
from starlight.config import PROFILES  # noqa: E402
from starlight.contracts import Chunk, ScoredChunk  # noqa: E402
from starlight.tools import ToolRegistry, ToolSpec, calculate  # noqa: E402


class ScriptedProvider:
    """按脚本依次返回回复的假模型。"""

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, *, model, tools=None, stream=False, options=None):
        self.calls.append({"messages": messages, "tools": tools})
        if not self.script:
            return {"content": "答案：无更多脚本", "tool_calls": []}
        return self.script.pop(0)


def _reg(output_max: int = 1200) -> ToolRegistry:
    reg = ToolRegistry(max_output=output_max)
    reg.register(ToolSpec(name="calculator", description="算数", parameters={
        "type": "object", "properties": {"expression": {"type": "string"}},
        "required": ["expression"]}, handler=lambda a: f"{a['expression']} = {calculate(a['expression']):g}"))
    return reg


def _tool_call(name: str, args: dict) -> dict:
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_single_tool_then_answer():
    provider = ScriptedProvider([
        _tool_call("calculator", {"expression": "2+3*4"}),
        {"content": "答案：结果是 14", "tool_calls": []},
    ])
    rt = AgentRuntime(provider, PROFILES["balanced"], _reg())
    trace = rt.run("算一下 2+3*4")
    assert len(trace.steps) == 1
    assert trace.steps[0].ok is True
    assert "14" in trace.steps[0].observation
    assert trace.answer == "结果是 14"
    assert trace.stopped_reason == "answered"
    # 工具结果必须以 tool 角色回喂给模型
    assert provider.calls[1]["messages"][-1]["role"] == "tool"
    assert provider.calls[0]["tools"][0]["function"]["name"] == "calculator"


def test_unknown_tool_does_not_break_loop():
    provider = ScriptedProvider([
        _tool_call("not_exists", {}),
        {"content": "答案：改用直接回答", "tool_calls": []},
    ])
    rt = AgentRuntime(provider, PROFILES["balanced"], _reg())
    trace = rt.run("随便问问")
    assert trace.steps[0].ok is False
    assert "未知工具" in trace.steps[0].observation
    assert trace.stopped_reason == "answered"


def test_max_steps_forces_answer():
    script = [_tool_call("calculator", {"expression": "1+1"}) for _ in range(5)]
    provider = ScriptedProvider(script + [{"content": "答案：强制收口", "tool_calls": []}])
    rt = AgentRuntime(provider, PROFILES["balanced"], _reg(), max_steps=2)
    trace = rt.run("无限调用工具试试")
    assert len(trace.steps) == 2, "步数上限必须截断循环"
    assert trace.stopped_reason == "max_steps_forced_answer"
    assert trace.answer


def test_tool_timeout_is_captured():
    import time as _t

    reg = ToolRegistry()
    reg.register(ToolSpec(name="slow", description="慢工具", parameters={},
                          handler=lambda a: (_t.sleep(3), "done")[1]))
    provider = ScriptedProvider([
        _tool_call("slow", {}),
        {"content": "答案：超时了", "tool_calls": []},
    ])
    rt = AgentRuntime(provider, PROFILES["balanced"], reg, tool_timeout_s=0.3)
    trace = rt.run("调个慢工具")
    assert trace.steps[0].ok is False
    assert "超时" in trace.steps[0].observation


def test_citation_numbering_is_global_across_searches():
    """两次检索的编号不能撞号，否则引用会指错块。"""
    collected: list[ScoredChunk] = []

    def search(args):
        base = len(collected)
        for i in range(2):
            collected.append(ScoredChunk(
                chunk=Chunk(id=f"c{base+i}", doc_id=f"d{base+i}", seq=i,
                            text=f"第{base+i}块内容", meta={"source": f"f{base+i}"}),
                score=0.5))
        return "\n".join(f"[{base+i+1}] 第{base+i}块内容" for i in range(2))

    reg = ToolRegistry()
    reg.register(ToolSpec(name="knowledge_search", description="检索", parameters={}, handler=search))
    provider = ScriptedProvider([
        _tool_call("knowledge_search", {}),
        _tool_call("knowledge_search", {}),
        {"content": "答案：见 [1] 与 [3]", "tool_calls": []},
    ])
    rt = AgentRuntime(provider, PROFILES["balanced"], reg, collector=collected)
    trace = rt.run("查两次")
    assert len(collected) == 4
    assert [c.index for c in trace.citations] == [1, 3]
    assert trace.citations[1].source == "f2"
