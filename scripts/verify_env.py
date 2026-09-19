"""干净环境复现验证。

作者: 晨星

做的事：在一个全新的虚拟环境里，用与 Dockerfile 完全相同的安装命令
（pip install --only-binary=:all: -r requirements.txt）把依赖装一遍，
然后在新环境里跑单元测试。

为什么需要它：README 里写"干净环境可一键复现"是一句承诺，
这个脚本把承诺变成可执行的证据。任何一次依赖变更后都应该跑它。

用法：
    python scripts/verify_env.py            # 验证并清理
    python scripts/verify_env.py --keep     # 保留临时环境便于排查
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None, capture: bool = True) -> tuple[int, str]:
    """默认捕获输出；capture=False 时直接继承父进程 stdio，让进度实时可见。

    教训：早期版本一律捕获输出，结果 pip 装到一半时控制台长时间毫无动静，
    被误判为卡死而中断，验证结论也随之失真。长耗时的安装步骤必须让输出流出去。
    """
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return r.returncode, ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Starlight 干净环境复现验证")
    ap.add_argument("--keep", action="store_true", help="保留临时环境")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    base = ROOT / "data" / "_cleanenv"
    base.mkdir(parents=True, exist_ok=True)
    # 先清理历史残留，避免半装的包污染结论
    for stale in base.glob("venv-*"):
        shutil.rmtree(stale, ignore_errors=True)
    target = Path(tempfile.mkdtemp(dir=base, prefix="venv-"))
    log(f"临时环境: {target}")
    log(f"基础解释器: {args.python}")
    log("=" * 68)

    t0 = time.time()
    rc, out = run([args.python, "-m", "venv", str(target)])
    if rc != 0:
        log("[失败] 创建虚拟环境失败\n" + out[-800:])
        return 1
    py = target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    log(f"[通过] 虚拟环境创建 ({time.time() - t0:.1f}s)")

    t0 = time.time()
    log("[..] 安装依赖（仅预编译 wheel，以下为 pip 原始输出）")
    rc, _ = run([str(py), "-m", "pip", "install", "--only-binary=:all:",
                 "-r", str(ROOT / "requirements.txt")], capture=False)
    if rc != 0:
        log("[失败] 依赖安装失败（可能存在需要本地编译的包，或网络不可达）")
        return 1
    log(f"[通过] 依赖安装完成，全程无编译动作 ({time.time() - t0:.1f}s)")

    rc, out = run([str(py), "-m", "pip", "list", "--format=freeze"])
    log(f"[通过] 已安装包数: {len(out.strip().splitlines())}")

    rc, out = run([str(py), "-c", "import starlight, sqlite_vec, fastapi, httpx, numpy, pypdf, openai;"
                                 "print(starlight.__version__, starlight.__author__)"])
    if rc != 0:
        log("[失败] 新环境导入核心模块失败")
        log(out[-1000:])
        return 1
    log(f"[通过] 核心模块导入正常，版本与作者: {out.strip().splitlines()[-1]}")

    rc, out = run([str(py), "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT)
    tail = "\n".join(out.strip().splitlines()[-3:])
    if rc != 0:
        log("[失败] 新环境单元测试未通过")
        log(tail)
        return 1
    log(f"[通过] 新环境单元测试: {tail}")

    log("=" * 68)
    log("结论: 干净环境可一键复现")
    if args.keep:
        log(f"（已保留 {target}）")
    else:
        shutil.rmtree(target, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
