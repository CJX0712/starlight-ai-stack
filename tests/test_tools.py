"""工具层测试：算术白名单、参数校验、输出截断。作者: 晨星"""

import pytest

from starlight.tools import ToolError, ToolRegistry, ToolSpec, calculate


@pytest.mark.parametrize("expr,expected", [
    ("2+3*4", 14),
    ("(3+4)*2/7", 2.0),
    ("2**10", 1024),
    ("17%5", 2),
    ("sqrt(16)", 4),
    ("-3+1", -2),
])
def test_calculate_supported(expr, expected):
    assert calculate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('dir')",
    "open('x','w')",
    "1 if 1 else 2",
    "[1,2][0]",
    "lambda: 1",
    "x+1",
])
def test_calculate_rejects_non_arithmetic(expr):
    with pytest.raises(ToolError):
        calculate(expr)


def test_calculate_rejects_overlong_expression():
    with pytest.raises(ToolError):
        calculate("1+" * 200 + "1")


def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry(max_output=20)
    reg.register(ToolSpec(
        name="echo", description="回显", parameters={"type": "object",
        "properties": {"text": {"type": "string"}}, "required": ["text"]},
        handler=lambda a: a["text"],
    ))
    return reg


def test_unknown_tool_reports_available():
    with pytest.raises(ToolError) as exc:
        _echo_registry().call("nope", {})
    assert "可用工具" in str(exc.value)


def test_missing_required_argument():
    with pytest.raises(ToolError) as exc:
        _echo_registry().call("echo", {})
    assert "缺少必填参数" in str(exc.value)


def test_output_is_truncated():
    out = _echo_registry().call("echo", {"text": "a" * 200})
    assert "已截断" in out and len(out) < 60


def test_duplicate_registration_rejected():
    reg = _echo_registry()
    with pytest.raises(ToolError):
        reg.register(ToolSpec(name="echo", description="x", parameters={}, handler=lambda a: ""))


def test_handler_exception_becomes_tool_error():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="boom", description="x", parameters={},
                          handler=lambda a: 1 / 0))
    with pytest.raises(ToolError):
        reg.call("boom", {})


def test_specs_are_openai_shape():
    spec = _echo_registry().specs()[0]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "echo"
    assert "parameters" in spec["function"]
