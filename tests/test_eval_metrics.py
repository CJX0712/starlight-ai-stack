"""评测指标测试：全部离线，锁死指标定义与门禁行为。作者: 晨星"""

from starlight.eval import (
    CaseResult,
    EvalCase,
    aggregate,
    contains_all,
    first_hit_rank,
    gate,
    normalize,
    rank_to_rr,
)


def test_normalize_makes_whitespace_irrelevant():
    assert normalize("3 个工作日") == normalize("3个工作日")
    assert normalize("RRF 融合") == "rrf融合"


def test_contains_all_requires_every_keyword():
    text = "稀疏检索用 BM25，稠密检索用向量近邻。"
    assert contains_all(text, ["BM25", "向量"])
    assert not contains_all(text, ["BM25", "RRF"])


def test_contains_all_matches_across_candidate_join():
    joined = "第一条讲 /health\n第二条讲 /var/log"
    assert contains_all(joined, ["/health"])


def test_first_hit_rank_is_one_based():
    cands = ["无关内容", "这里提到 /health 接口", "也提到 /health"]
    assert first_hit_rank(cands, ["/health"]) == 2
    assert first_hit_rank(cands, ["不存在"]) is None
    assert rank_to_rr(2) == 0.5
    assert rank_to_rr(None) == 0.0


def test_aggregate_metrics_math():
    results = [
        CaseResult("a", "q1", ctx_hit=True, top1_hit=True, rank=1, answer_hit=True, latency_s=2.0),
        CaseResult("b", "q2", ctx_hit=True, top1_hit=False, rank=3, answer_hit=False, latency_s=4.0),
    ]
    m = aggregate(results)
    assert m["cases"] == 2
    assert m["ctx_hit"] == 1.0
    assert m["top1_hit"] == 0.5
    assert m["mrr"] == (1.0 + 1 / 3) / 2
    assert m["answer_hit"] == 0.5
    assert m["mean_latency_s"] == 3.0


def test_gate_reports_each_miss():
    metrics = {"ctx_hit": 0.5, "mrr": 0.9}
    failures = gate(metrics, {"ctx_hit": 0.9, "mrr": 0.7})
    assert len(failures) == 1
    assert "ctx_hit" in failures[0]


def test_gate_fails_on_case_error():
    results = [CaseResult("x", "q", error="ModelError: boom")]
    assert gate({"ctx_hit": 1.0}, {"ctx_hit": 0.9}, results)


def test_gate_passes_when_all_above_threshold():
    assert gate({"ctx_hit": 0.95, "mrr": 0.8}, {"ctx_hit": 0.9, "mrr": 0.7}) == []


def test_golden_file_is_well_formed():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "eval" / "golden_set.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cases"], "金标集不能为空"
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids)), "用例 ID 不能重复"
    for c in data["cases"]:
        assert c["keywords"], f"{c['id']} 缺少关键词判据"
    assert set(data["thresholds"]) >= {"ctx_hit", "mrr", "answer_hit", "citation_valid"}


def test_golden_corpus_covers_every_source():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    corpus = {p.name for p in (root / "eval" / "corpus").glob("*")}
    import json

    data = json.loads((root / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    for c in data["cases"]:
        assert c["source"] in corpus, f"{c['id']} 指向的语料文件不存在: {c['source']}"


def test_eval_case_dataclass_defaults():
    c = EvalCase(id="q", question="问题", keywords=["x"])
    assert c.source == ""
