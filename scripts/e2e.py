"""端到端验收：真实语料进库 -> 提问 -> 检查答案与引用，输出可判定的报告。

作者: 晨星

用法：
    python scripts/e2e.py                 # 用内置语料跑一遍
    python scripts/e2e.py --corpus D:/docs # 用你自己的目录

判定口径（机器可判定，不靠肉眼）：
    1. 摄取：新增块数 > 0，且二次摄取新增 = 0（幂等）
    2. 检索：每个问题的 top1 命中预期关键词
    3. 生成：答案非空，且引用编号全部落在提供的上下文范围内
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CORPUS = {
    "报销制度.txt": (
        "报销制度：员工应在每月 25 日前提交发票与报销单。\n"
        "财务部门在收到材料后 3 个工作日内完成打款。\n"
        "差旅报销需额外附上出差申请单编号。\n"
    ),
    "运维手册.txt": (
        "运维手册：服务健康检查接口为 /health，返回 200 表示进程存活。\n"
        "日志文件存放在 /var/log/starlight 目录，按天切分。\n"
        "出现 5xx 错误时应先查看数据库连接池是否耗尽。\n"
    ),
    "向量检索说明.txt": (
        "本系统采用混合检索：稀疏检索用 BM25，稠密检索用向量近邻。\n"
        "两路结果通过 RRF 融合排序，避免单一路召回失手。\n"
        "中文分词采用单字与相邻二字组，兼顾长词与二字短查询。\n"
    ),
}

QUESTIONS = [
    ("报销需要在几号前提交？", "25 日"),
    ("健康检查接口是哪个？", "/health"),
    ("混合检索是怎么融合两路结果的？", "RRF"),
]


def prepare_corpus(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, text in CORPUS.items():
        (dirpath / name).write_text(text, encoding="utf-8")
    return dirpath


def main() -> int:
    ap = argparse.ArgumentParser(description="Starlight 端到端验收")
    ap.add_argument("--profile", default=os.getenv("STARLIGHT_PROFILE", "balanced"))
    ap.add_argument("--corpus", default=None)
    args = ap.parse_args()
    os.environ["STARLIGHT_PROFILE"] = args.profile

    from starlight.pipeline import RAGPipeline

    corpus = Path(args.corpus) if args.corpus else prepare_corpus(ROOT / "data" / "corpus")
    pipe = RAGPipeline()
    if not pipe.provider.health():
        print("[!!] 模型服务不可达，先执行 ollama serve")
        return 1
    pipe.prepare()
    print(f"生效档位: {pipe.profile.name} | LLM={pipe.profile.llm} | 嵌入={pipe.profile.embed}")
    if pipe.notes:
        for n in pipe.notes:
            print(f"  降级说明: {n}")
    print("=" * 70)

    t0 = time.time()
    r1 = pipe.ingest(str(corpus))
    t_ingest = time.time() - t0
    t0 = time.time()
    r2 = pipe.ingest(str(corpus))
    t_re = time.time() - t0
    ok_ingest = r1.chunks > 0 and r2.chunks == 0
    print(f"摄取: 首次新增块={r1.chunks} 文件={r1.added} 耗时={t_ingest:.2f}s")
    print(f"幂等: 二次新增块={r2.chunks}（必须为 0）耗时={t_re:.2f}s")
    print("-" * 70)

    fails = 0
    for q, expect in QUESTIONS:
        t0 = time.time()
        ans = pipe.ask(q)
        dt = time.time() - t0
        ctx = pipe.search(q, 5)
        hit = bool(ctx) and expect in ctx[0].chunk.text
        cite_ok = all(1 <= c.index <= len(ctx) for c in ans.citations)
        good = hit and cite_ok and bool(ans.text.strip())
        fails += 0 if good else 1
        mark = "通过" if good else "失败"
        print(f"[{mark}] {q}")
        print(f"        top1 命中关键词={hit} 预期={expect!r}")
        print(f"        引用数={len(ans.citations)} 引用合法={cite_ok} 耗时={dt:.2f}s")
        print(f"        答案: {ans.text[:90]}")
    print("=" * 70)
    print(f"统计: {pipe.store.stats()}")
    print(f"结论: {'全部通过' if ok_ingest and fails == 0 else '存在失败项'}")
    pipe.close()
    return 0 if (ok_ingest and fails == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
