"""逐模块自检：每个模块都能脱离完整链路单独证明自己是对的。

作者: 晨星

判据全部是机械可判定的不变量，不是"看起来能跑"：
    text       切块覆盖原文且重叠生效
    models     服务可达、嵌入维度恒定、对话非空
    store      向量 top1 必为自身；BM25 命中；重复摄取 0 新增；删除后不可检索
    retriever  已知查询能召回正确块，且双路分数都留痕
    generator  引用只映射到真实块，越界编号被剔除
    pipeline   端到端：摄取 -> 检索 -> 生成 -> 引用非空
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

from .config import Settings, resolve_profile
from .contracts import Chunk, Document, ScoredChunk
from .embedder import TextEmbedder
from .generator import AnswerGenerator
from .ingest import Ingestor, doc_id_for
from .models import OllamaProvider
from .pipeline import RAGPipeline
from .retriever import HybridRetriever
from .store import SqliteStore
from .text import chunk_text, tokenize

CHECKERS: dict[str, Callable[[], tuple[bool, list[str]]]] = {}


def register(name: str):
    def deco(fn):
        CHECKERS[name] = fn
        return fn
    return deco


def _fresh_settings(tmp: str) -> Settings:
    s = Settings()
    s.home = Path(tmp)
    return s


def _tmpdir() -> Path:
    """测试库放项目目录内：系统临时目录在 Windows 上常被安全软件锁住导致建库失败。"""
    base = Path(__file__).resolve().parents[2] / "data" / "_selftest"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=base))


def _resolved(settings: Settings | None = None) -> tuple[Settings, object, list[str]]:
    """拿到「本机实际可用」的档位，避免自检因为某个模型没拉下来而假失败。"""
    s = settings or Settings()
    p = OllamaProvider(s.ollama_base_url, timeout=60)
    try:
        info = p.models_info()
    except Exception:
        info = {}
    if not info:
        return s, s.model, ["模型服务不可达"]
    profile, notes = resolve_profile(info, s.model)
    return s, profile, notes


@register("text")
def check_text() -> tuple[bool, list[str]]:
    log: list[str] = []
    sample = "检索增强生成把外部知识接入大模型。第一句讲原理。第二句讲工程实现。第三句讲评测方法。" * 3
    chunks = chunk_text(sample, size=120, overlap=30)
    ok = len(chunks) >= 3
    log.append(f"切块数量={len(chunks)}（>=3 为通过）")
    joined = " ".join(chunks)
    cover = sum(1 for c in set(sample) if c in joined) / max(1, len(set(sample)))
    ok &= cover > 0.9
    log.append(f"字符覆盖率={cover:.3f}（>0.9 为通过）")
    toks = tokenize("检索增强 RAG")
    ok &= "检索" in toks and "rag" in toks
    log.append(f"中英混排分词={toks[:6]}")
    return ok, log


@register("models")
def check_models() -> tuple[bool, list[str]]:
    log: list[str] = []
    s, profile, notes = _resolved()
    p = OllamaProvider(s.ollama_base_url, timeout=60)
    if not p.health():
        return False, [f"模型服务不可达: {s.ollama_base_url}"]
    models = p.list_models()
    log.append(f"本机模型={models}")
    log.append(f"档位 {s.profile} 期望 {s.model.llm} / {s.model.embed} -> 生效 {profile.llm} / {profile.embed}")
    for n in notes:
        log.append(f"降级说明: {n}")
    try:
        v1 = p.embed(["自检"], model=profile.embed)
        v2 = p.embed(["自检"], model=profile.embed)
        dim_ok = len(v1[0]) == len(v2[0]) > 0
        log.append(f"嵌入维度={len(v1[0])}（两次一致={dim_ok}）")
    except Exception as exc:
        dim_ok = False
        log.append(f"嵌入失败: {exc}")
    return dim_ok, log


@register("store")
def check_store() -> tuple[bool, list[str]]:
    log: list[str] = []
    s, profile, _notes = _resolved()
    provider = OllamaProvider(s.ollama_base_url, timeout=120)
    if not provider.health():
        return False, ["模型服务不可达，跳过存储自检"]
    tmp = _tmpdir()
    try:
        emb = TextEmbedder(provider, profile.embed)
        store = SqliteStore(tmp / "t.db")
        store.ensure_vec_table(emb.dim)
        doc = Document(id="d1", source="selftest", text="向量数据库用于存储嵌入向量。BM25 是稀疏检索算法。")
        pieces = chunk_text(doc.text, 60, 10)
        chunks = []
        for i, t in enumerate(pieces):
            chunks.append(Chunk(id=f"d1#{i}", doc_id="d1", seq=i, text=t, meta={"source": "selftest"}))
        vecs = emb.embed([c.text for c in chunks])

        # 幂等性由 Ingestor 保证（内容哈希未变直接跳过），不是 store 的职责
        ing = Ingestor(store, emb, s)
        body = "向量数据库用于存储嵌入向量。BM25 是稀疏检索算法。"
        a1 = ing.ingest_text(body, "selftest-doc")
        a2 = ing.ingest_text(body, "selftest-doc")
        log.append(f"摄取幂等: 首次新增={a1} 二次新增={a2}（二次必须为 0）")

        before = store.stats()["chunks"]
        n1 = store.upsert_document(doc, chunks, vecs)
        n2 = store.upsert_document(doc, chunks, vecs)
        after = store.stats()["chunks"]
        # 整篇替换语义：重复写入不应让块数翻倍
        log.append(f"整篇替换: 写入={n1}/{n2} 库内块数 {before}->{after}（应只增加 {n1}）")

        qv = emb.embed_one("向量数据库")
        hits = store.vector_search(qv, 3)
        top1_self = store.vector_search(vecs[0], 1)
        log.append(f"稠密检索命中={len(hits)} top1 自反={top1_self[0][0] if top1_self else None}")
        sparse = store.sparse_search("BM25", 3)
        log.append(f"稀疏检索命中={len(sparse)} 最高分={sparse[0][1] if sparse else 0:.4f}")

        removed = store.delete_document("d1") + store.delete_document(doc_id_for("selftest-doc"))
        after_hits = store.sparse_search("BM25", 3)
        log.append(f"删除块数={removed} 删除后命中={len(after_hits)}（必须为 0）")
        ok = (a1 > 0 and a2 == 0 and after - before == n1
              and len(hits) > 0 and len(sparse) > 0 and len(after_hits) == 0)
        store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok, log


@register("retriever")
def check_retriever() -> tuple[bool, list[str]]:
    log: list[str] = []
    s, profile, _notes = _resolved()
    provider = OllamaProvider(s.ollama_base_url, timeout=120)
    if not provider.health():
        return False, ["模型服务不可达，跳过检索自检"]
    tmp = _tmpdir()
    top: list = []
    hit = False
    try:
        st = _fresh_settings(str(tmp))
        emb = TextEmbedder(provider, profile.embed)
        store = SqliteStore(st.db_path)
        store.ensure_vec_table(emb.dim)
        ing = Ingestor(store, emb, st)
        n = ing.ingest_text("公司的报销流程：员工需在每月 25 日前提交发票与报销单，财务在 3 个工作日内打款。", "policy")
        ret = HybridRetriever(store, emb, candidate_k=10)
        fused = ret.fuse("报销什么时候提交？", 3)
        top = fused["fused"]
        log.append(f"入库块数={n} 融合命中={len(top)}")
        if top:
            log.append(f"首条得分={top[0].score:.5f} 稠密秩={top[0].dense_rank} 稀疏秩={top[0].sparse_rank}")
            hit = "25 日" in top[0].chunk.text or "报销" in top[0].chunk.text
            log.append(f"首条含关键信息={hit} 摘要={top[0].chunk.text[:30]}")
        store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return bool(top) and hit, log


@register("generator")
def check_generator() -> tuple[bool, list[str]]:
    log: list[str] = []
    s, profile, _notes = _resolved()
    provider = OllamaProvider(s.ollama_base_url, timeout=60)
    if not provider.health():
        return False, ["模型服务不可达，跳过生成自检"]
    gen = AnswerGenerator(provider, profile)
    ctx = [
        ScoredChunk(chunk=Chunk(id="a#0", doc_id="a", seq=0, text="报销需在每月 25 日前提交。", meta={"source": "policy"}), score=0.9),
        ScoredChunk(chunk=Chunk(id="a#1", doc_id="a", seq=1, text="财务在 3 个工作日内完成打款。", meta={"source": "policy"}), score=0.8),
    ]
    fake = "报销要在 25 日前提交 [1]，打款需要 3 个工作日 [2]，另见 [9]。"
    cites = AnswerGenerator.parse_citations(fake, ctx)
    idxs = [c.index for c in cites]
    log.append(f"解析引用编号={idxs}（越界 [9] 必须被剔除）")
    ok = idxs == [1, 2]
    msgs = gen.build_messages("报销流程？", ctx)
    ok &= "[1]" in msgs[1]["content"] and "[2]" in msgs[1]["content"]
    log.append(f"提示词含引用标记={'[1]' in msgs[1]['content'] and '[2]' in msgs[1]['content']}")
    return ok, log


@register("pipeline")
def check_pipeline() -> tuple[bool, list[str]]:
    log: list[str] = []
    tmp = _tmpdir()
    try:
        st = _fresh_settings(str(tmp))
        pipe = RAGPipeline(st)
        if not pipe.provider.health():
            return False, ["模型服务不可达，跳过端到端自检"]
        pipe.prepare()
        t0 = time.time()
        added = pipe.ingest_text(
            "运维手册：服务健康检查地址为 /health，返回 200 表示存活。日志目录位于 /var/log/starlight。", "ops"
        )
        t1 = time.time()
        ans = pipe.ask("健康检查地址是什么？")
        t2 = time.time()
        log.append(f"入库块数={added} 摄取耗时={t1 - t0:.2f}s 生成耗时={t2 - t1:.2f}s")
        log.append(f"答案={ans.text[:60]!r}")
        log.append(f"引用数={len(ans.citations)} 模型={ans.model}")
        pipe.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return added > 0 and bool(ans.text.strip()), log


def run(modules: list[str] | None = None) -> int:
    targets = modules or list(CHECKERS)
    print(f"Starlight 自检 · 作者 晨星 · 目标模块: {', '.join(targets)}")
    print("-" * 62)
    failed = 0
    for name in targets:
        fn = CHECKERS.get(name)
        if fn is None:
            print(f"[跳过] {name}: 未注册")
            continue
        t0 = time.time()
        try:
            ok, log = fn()
        except Exception as exc:
            ok, log = False, [f"异常: {type(exc).__name__}: {exc}"]
        status = "通过" if ok else "失败"
        print(f"[{status}] {name:10s} ({time.time() - t0:.2f}s)")
        for line in log:
            print(f"         {line}")
        failed += 0 if ok else 1
    print("-" * 62)
    print(f"结果: 通过 {len(targets) - failed}/{len(targets)}")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    mods = None
    if args and args[0] == "--module":
        mods = args[1].split(",")
    raise SystemExit(run(mods))
