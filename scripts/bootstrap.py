"""干净环境一键复现：装依赖 -> 准备模型 -> 跑自检。

作者: 晨星

用法：
    python scripts/bootstrap.py                  # 用当前解释器装依赖并自检
    python scripts/bootstrap.py --profile high   # 指定档位
    python scripts/bootstrap.py --skip-install   # 依赖已装，只做模型与自检

约束：本机没有 C/C++ 编译器，依赖一律以「仅二进制 wheel」安装，
任何需要本地编译的包都会在这里被明确拒绝并给出替代建议，而不是装一半失败。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MIN_PY = (3, 10)


def check_python() -> bool:
    v = sys.version_info
    ok = v >= MIN_PY
    print(f"[{'ok' if ok else '!!'}] Python {v.major}.{v.minor}.{v.micro} "
          f"(要求 >={MIN_PY[0]}.{MIN_PY[1]})")
    return ok


def install_deps() -> bool:
    req = ROOT / "requirements.txt"
    if not req.exists():
        print("[!!] 缺少 requirements.txt")
        return False
    cmd = [sys.executable, "-m", "pip", "install", "--only-binary=:all:", "-r", str(req)]
    print(f"[..] {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[!!] 依赖安装失败：存在需要本地编译的包，或网络不可达")
        return False
    print("[ok] 依赖安装完成")
    return True


def ensure_ollama(profile: str) -> bool:
    exe = shutil.which("ollama")
    base = os.getenv("STARLIGHT_OLLAMA_URL", "http://127.0.0.1:11434")
    if not exe:
        print(f"[!!] 未检测到 Ollama，请先安装后执行: ollama serve")
        print(f"     或改用 OpenAI 兼容后端: RAGPipeline(backend='openai')")
        return False
    print(f"[ok] Ollama 可执行文件: {exe}")
    try:
        import httpx

        models = httpx.get(base + "/api/tags", timeout=5, trust_env=False).json().get("models", [])
        names = [m["name"] for m in models]
        print(f"[ok] 模型服务可达，已有模型: {names}")
    except Exception:
        print(f"[!!] 模型服务未运行，请执行: ollama serve")
        return False

    from starlight.config import PROFILES

    p = PROFILES[profile]
    missing = [m for m in (p.llm, p.embed) if m not in names]
    for m in missing:
        print(f"[..] 缺少模型 {m}，执行: ollama pull {m}")
        r = subprocess.run([exe, "pull", m])
        if r.returncode != 0:
            print(f"[!!] {m} 拉取失败，系统会自动降级到本机已有的同类模型")
    return True


def run_selftest(modules: list[str] | None) -> int:
    from starlight.selftest import run

    return run(modules)


def main() -> int:
    ap = argparse.ArgumentParser(description="Starlight 环境引导")
    ap.add_argument("--profile", default=os.getenv("STARLIGHT_PROFILE", "balanced"))
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--module", default=None, help="逗号分隔，只自检指定模块")
    args = ap.parse_args()

    os.environ["STARLIGHT_PROFILE"] = args.profile
    print(f"Starlight bootstrap · 档位 {args.profile} · 作者 晨星")
    print("=" * 62)
    ok = check_python()
    if not args.skip_install:
        ok &= install_deps()
    ensure_ollama(args.profile)
    print("=" * 62)
    mods = args.module.split(",") if args.module else None
    return run_selftest(mods)


if __name__ == "__main__":
    raise SystemExit(main())
