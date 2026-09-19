"""模型基准测试：测真实生成速度与内存占用。

作者: 晨星

用法：
    python scripts/bench.py                        # 测当前档位的 LLM
    python scripts/bench.py --model qwen2.5:7b-instruct-q4_K_M --threads 4

为什么要单独测：CPU 推理速度受内存带宽与是否换页支配，
必须在目标机器上实测，任何"理论上应该多快"的判断都不可信。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def free_memory_gb() -> float:
    from starlight.config import free_memory_gb as f

    return f()


def bench(model: str, prompt: str, threads: int, predict: int, ctx: int) -> dict:
    payload = {
        "model": model,
        "stream": True,
        "think": False,
        "options": {"num_predict": predict, "num_thread": threads, "num_ctx": ctx, "temperature": 0.2},
        "messages": [
            {"role": "system", "content": "直接回答，不要解释。"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "答案："},
        ],
    }
    base = os.getenv("STARLIGHT_OLLAMA_URL", "http://127.0.0.1:11434")
    req = urllib.request.Request(base + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    first = None
    text = ""
    info: dict = {}
    with _opener.open(req, timeout=900) as resp:
        for line in resp:
            if not line.strip():
                continue
            d = json.loads(line)
            piece = (d.get("message") or {}).get("content") or ""
            if piece:
                if first is None:
                    first = time.time() - t0
                text += piece
            if d.get("done"):
                ec = d.get("eval_count", 0)
                ed = d.get("eval_duration", 0) / 1e9
                info = {
                    "load_s": round(d.get("load_duration", 0) / 1e9, 2),
                    "ttft_s": round(first or 0, 2),
                    "tokens": ec,
                    "tok_s": round(ec / ed, 2) if ed else 0.0,
                    "total_s": round(time.time() - t0, 2),
                }
                break
    info["text"] = text[:80]
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Starlight 模型基准测试")
    ap.add_argument("--model", default=None)
    ap.add_argument("--profile", default=os.getenv("STARLIGHT_PROFILE", "high"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--predict", type=int, default=80)
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()

    os.environ["STARLIGHT_PROFILE"] = args.profile
    from starlight.config import PROFILES

    model = args.model or PROFILES[args.profile].llm
    print(f"可用内存: {free_memory_gb():.1f} GB | 目标模型: {model} | 线程: {args.threads}")
    r = bench(model, "用一句话说明什么是检索增强生成。", args.threads, args.predict, args.ctx)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r.get("tokens") else 1


if __name__ == "__main__":
    raise SystemExit(main())
