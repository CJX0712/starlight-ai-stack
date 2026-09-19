"""存储模块测试：不依赖模型服务，用哈希嵌入器验证不变量。作者: 晨星"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import HashEmbedder  # noqa: E402

from starlight.contracts import Chunk, Document  # noqa: E402
from starlight.store import SqliteStore  # noqa: E402


def _fixture(tmp_path):
    emb = HashEmbedder(dim=32)
    store = SqliteStore(tmp_path / "t.db")
    store.ensure_vec_table(emb.dim)
    body = "向量数据库用于存储嵌入向量，支持近似最近邻检索。BM25 是经典稀疏检索算法。"
    doc = Document(id="d1", source="test", text=body, meta={"hash": "h1"})
    chunks = [Chunk(id=f"d1#{i}", doc_id="d1", seq=i, text=t, meta={"source": "test"})
              for i, t in enumerate([body[:20], body[20:]])]
    return store, emb, doc, chunks


def test_upsert_then_vector_search_top1_is_self(tmp_path):
    store, emb, doc, chunks = _fixture(tmp_path)
    vecs = emb.embed([c.text for c in chunks])
    n = store.upsert_document(doc, chunks, vecs)
    assert n == 2
    hits = store.vector_search(vecs[0], 2)
    assert hits[0][0] == chunks[0].rowid
    store.close()


def test_replacement_does_not_duplicate(tmp_path):
    store, emb, doc, chunks = _fixture(tmp_path)
    vecs = emb.embed([c.text for c in chunks])
    store.upsert_document(doc, chunks, vecs)
    before = store.stats()["chunks"]
    store.upsert_document(doc, chunks, vecs)
    assert store.stats()["chunks"] == before
    store.close()


def test_sparse_bm25_hits_chinese_two_char_query(tmp_path):
    store, emb, doc, chunks = _fixture(tmp_path)
    vecs = emb.embed([c.text for c in chunks])
    store.upsert_document(doc, chunks, vecs)
    hits = store.sparse_search("稀疏", 5)
    assert hits, "FTS5 trigram 对二字查询会失效，自建倒排必须能命中"
    store.close()


def test_delete_removes_index(tmp_path):
    store, emb, doc, chunks = _fixture(tmp_path)
    vecs = emb.embed([c.text for c in chunks])
    store.upsert_document(doc, chunks, vecs)
    removed = store.delete_document("d1")
    assert removed == 2
    assert store.sparse_search("稀疏", 5) == []
    assert store.stats()["chunks"] == 0
    store.close()
