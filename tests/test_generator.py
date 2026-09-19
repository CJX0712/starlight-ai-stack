"""生成模块测试：引用解析与答案清洗。作者: 晨星"""

from starlight.contracts import Chunk, ScoredChunk
from starlight.generator import AnswerGenerator


def _ctx():
    return [
        ScoredChunk(chunk=Chunk(id="a#0", doc_id="a", seq=0, text="报销需在每月 25 日前提交。", meta={"source": "policy"}), score=0.9),
        ScoredChunk(chunk=Chunk(id="a#1", doc_id="a", seq=1, text="财务在 3 个工作日内完成打款。", meta={"source": "policy"}), score=0.8),
    ]


def test_citations_only_map_to_real_chunks():
    ctx = _ctx()
    cites = AnswerGenerator.parse_citations("25 日前提交 [1]，3 个工作日打款 [2]，另见 [7]。", ctx)
    assert [c.index for c in cites] == [1, 2]
    assert cites[0].source == "policy"


def test_duplicate_citation_deduped():
    ctx = _ctx()
    cites = AnswerGenerator.parse_citations("见 [1] 和 [1]", ctx)
    assert len(cites) == 1


def test_strip_preamble_only_when_mark_leads():
    assert AnswerGenerator.strip_preamble("答案：报销在 25 日前提交") == "报销在 25 日前提交"
    # 标记出现在分析段里（思考型模型复述指令）时，不能误切
    messy = "我应该以答案：开头，所以现在给出结论。真正的答案是 25 日。"
    assert AnswerGenerator.strip_preamble(messy) == messy


def test_context_block_contains_numbered_sources():
    ctx = _ctx()
    block = AnswerGenerator.build_context(ctx)
    assert "[1]" in block and "[2]" in block


def test_messages_end_with_assistant_prefill():
    """预填充是让思考型模型直接给结论的关键，不能被后续改动悄悄删掉。"""
    from starlight.config import PROFILES
    gen = AnswerGenerator(provider=None, profile=PROFILES["balanced"])
    msgs = gen.build_messages("报销几号提交？", _ctx())
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "答案："
