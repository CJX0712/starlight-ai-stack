"""一键启动：拉起模型服务（若未运行）+ 启动 API 网关 + 打开控制台。

作者: 晨星

用法：
    python scripts/start.py [--profile balanced|high|lite] [--port 8787] [--no-browser]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402


def ollama_up(base: str) -> bool:
    try:
        return httpx.get(base + "/api/tags", timeout=3, trust_env=False).status_code == 200
    except Exception:
        return False


def ensure_ollama(base: str, exe: str | None = None) -> bool:
    if ollama_up(base):
        print(f"[ok] 模型服务已在运行: {base}")
        return True
    exe = exe or "ollama"
    print(f"[..] 尝试启动模型服务: {exe} serve")
    try:
        creation = 0x00000008  # DETACHED_PROCESS
        subprocess.Popen([exe, "serve"], creationflags=creation,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[!!] 未找到 ollama 可执行文件，请先安装 Ollama 或改用 OpenAI 兼容后端")
        return False
    for _ in range(40):
        time.sleep(1)
        if ollama_up(base):
            print(f"[ok] 模型服务已就绪: {base}")
            return True
    print("[!!] 模型服务启动超时")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="启动 Starlight AI Stack")
    ap.add_argument("--profile", default=os.getenv("STARLIGHT_PROFILE", "balanced"))
    ap.add_argument("--port", type=int, default=int(os.getenv("STARLIGHT_API_PORT", "8787")))
    ap.add_argument("--host", default=os.getenv("STARLIGHT_API_HOST", "127.0.0.1"))
    ap.add_argument("--ollama", default=os.getenv("STARLIGHT_OLLAMA_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    os.environ["STARLIGHT_PROFILE"] = args.profile
    os.environ["STARLIGHT_OLLAMA_URL"] = args.ollama

    ensure_ollama(args.ollama)

    import uvicorn

    url = f"http://{args.host}:{args.port}/"
    print(f"[..] 启动 API 网关: {url}")
    print(f"[..] 档位: {args.profile}")
    if not args.no_browser:
        import threading

        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("starlight.server:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
