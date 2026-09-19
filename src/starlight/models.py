"""模型服务适配层。

作者: 晨星

两个实现，同一个契约：
- OllamaProvider：走 Ollama 原生 /api/chat 与 /api/embed，可关掉 thinking、可锁线程数
- OpenAICompatProvider：走 OpenAI 官方 SDK，指向任意 OpenAI 兼容端点（llama.cpp / vLLM / 云端）

本机没有 C/C++ 编译器，所以不自己编译推理引擎，而是把 Ollama 当作本地推理后端。
这层是唯一知道"用哪个后端"的地方，上层对后端完全无感。
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx


class ModelError(RuntimeError):
    """模型服务不可用或返回异常。"""


class OllamaProvider:
    """Ollama 原生接口适配器。

    关键点：
    - trust_env=False：本机全局 HTTP 代理会把 127.0.0.1 请求也代理掉，导致 502
    - think=False：Qwen3 默认会输出思考过程，白白吃掉 token 预算
    - num_thread：小量化模型在 CPU 上线程开太多反而更慢，实测锁 4 线程最快
    """

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, trust_env=False)

    def health(self) -> bool:
        try:
            r = self._client.get("/api/tags")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        return list(self.models_info().keys())

    def models_info(self) -> dict[str, list[str]]:
        """返回 模型名 -> 能力列表（completion / embedding / tools ...）。"""
        r = self._client.get("/api/tags")
        r.raise_for_status()
        return {
            m["name"]: list(m.get("capabilities") or []) for m in r.json().get("models", [])
        }

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        r = self._client.post("/api/embed", json={"model": model, "input": texts})
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise ModelError(f"嵌入数量不匹配: 期望 {len(texts)}，实际 {len(embeddings)}")
        return [[float(x) for x in v] for v in embeddings]

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Iterator[str]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": False,
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options

        if stream:
            return self._stream(payload)
        r = self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        msg = r.json().get("message", {})
        return {
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls", []) or [],
            "usage": {
                "prompt_tokens": r.json().get("prompt_eval_count", 0),
                "completion_tokens": r.json().get("eval_count", 0),
            },
        }

    def _stream(self, payload: dict[str, Any]) -> Iterator[str]:
        with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (d.get("message") or {}).get("content")
                if piece:
                    yield piece
                if d.get("done"):
                    break


class OpenAICompatProvider:
    """OpenAI 兼容端点适配器（llama.cpp server / vLLM / 云端 API 均可）。

    复用官方 openai SDK，不手写协议。用于把本机链路无缝迁到别的推理后端。
    """

    def __init__(self, base_url: str, api_key: str = "ollama", timeout: float = 300.0) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.base_url = base_url

    def health(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        return [m.id for m in self._client.models.list().data]

    def models_info(self) -> dict[str, list[str]]:
        """OpenAI 的 /v1/models 不暴露能力标签，统一按可对话处理。"""
        return {m: ["completion"] for m in self.list_models()}

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        resp = self._client.embeddings.create(model=model, input=texts)
        return [list(map(float, d.embedding)) for d in resp.data]

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Iterator[str]:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if options:
            for k in ("temperature", "max_tokens", "num_ctx"):
                if k in options:
                    kwargs[k if k != "num_ctx" else "max_tokens"] = options[k]
        if stream:
            def _gen() -> Iterator[str]:
                for chunk in self._client.chat.completions.create(stream=True, **kwargs):
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
            return _gen()
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        return {
            "content": msg.content or "",
            "tool_calls": [t.model_dump() for t in (msg.tool_calls or [])],
            "usage": (resp.usage.model_dump() if resp.usage else {}),
        }


def build_provider(backend: str, base_url: str, timeout: float = 300.0):
    """工厂：按配置返回满足 ModelProvider 契约的对象。"""
    if backend == "ollama":
        return OllamaProvider(base_url, timeout=timeout)
    if backend == "openai":
        return OpenAICompatProvider(base_url, timeout=timeout)
    raise ValueError(f"未知后端 {backend!r}，可选: ollama / openai")
