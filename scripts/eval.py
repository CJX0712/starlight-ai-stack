"""评测门禁：跑金标集，指标不达标就非零退出。

作者: 晨星

用法：
    python scripts/eval.py                       # 默认 rag 模式，全部用例
    python scripts/eval.py --mode agent          # 评测智能体（含工具调用）
    python scripts/eval.py --mode rag --limit 3  # 冒烟
    python scripts/eval.py --profile balanced    # 换档位对比

退出码：0 通过，1 未达标，2 环境问题（模型服务不可达 / 金标集缺失）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Starlight 评测门禁")
    ap.add_argument("--profile", default=os.getenv("STARLIGHT_PROFILE", "high"))
    ap.add_argument("--mode", default="rag", choices=["rag", "agent", "both"])
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden_set.json"))
    ap.add_argument("--corpus", default=str(ROOT / "eval" / "corpus"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    os.environ["STARLIGHT_PROFILE"] = args.profile

    from starlight.eval import Evaluator, gate, load_golden
    from starlight.pipeline import RAGPipeline

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"[!!] 金标集缺失: {golden_path}")
        return 2
    cases, thresholds, version = load_golden(golden_path)
    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"[!!] 语料目录缺失: {corpus}")
        return 2

    pipe = RAGPipeline()
    if not pipe.provider.health():
        print("[!!] 模型服务不可达，先执行: ollama serve")
        return 2
    pipe.prepare()
    rep = pipe.ingest(str(corpus))
    print(f"评测 · 金标集 v{version} · 档位 {pipe.profile.name} · 模型 {pipe.profile.llm}")
    print(f"语料: {corpus.name} 扫描 {rep.scanned} 文件，新增 {rep.added}，跳过 {rep.skipped}")
    print(f"库内统计: {pipe.store.stats()}")
    print("=" * 74)

    modes = ["rag", "agent"] if args.mode == "both" else [args.mode]
    all_reports = []
    exit_code = 0
    for mode in modes:
        ev = Evaluator(pipe, cases, top_k=args.top_k, mode=mode)
        report = ev.run(args.limit or None)
        report.thresholds = thresholds
        report.failures = gate(report.metrics, thresholds, report.results)
        all_reports.append(report)

        print(f"模式 {mode.upper()} · 用例 {len(report.results)}")
        print(f"{'ID':5s} {'ctx':4s} {'top1':5s} {'rank':4s} {'答案':4s} {'引用':4s} {'耗时':>7s}  问题")
        for r in report.results:
            print(
                f"{r.case_id:5s} {'Y' if r.ctx_hit else 'N':4s} {'Y' if r.top1_hit else 'N':5s} "
                f"{(r.rank if r.rank else '-'):>4} {'Y' if r.answer_hit else 'N':4s} "
                f"{'Y' if r.citation_valid else 'N':4s} {r.latency_s:7.1f}s  {r.question}"
            )
        m = report.metrics
        print("-" * 74)
        print(
            f"指标: ctx_hit@k={m['ctx_hit']:.3f} top1={m['top1_hit']:.3f} "
            f"MRR={m['mrr']:.3f} answer={m['answer_hit']:.3f} "
            f"citation={m['citation_valid']:.3f} 平均耗时={m['mean_latency_s']:.1f}s"
        )
        if report.failures:
            print("门禁: 未通过")
            for f in report.failures:
                print(f"   - {f}")
            exit_code = 1
        else:
            print("门禁: 通过")
        print("=" * 74)
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )

    pipe.close()
    if args.json_out:
        print(f"报告已写入 {args.json_out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
