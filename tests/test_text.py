"""文本模块测试：切块与分词。作者: 晨星"""

from starlight.text import chunk_text, normalize, tokenize


def test_chunk_covers_original_text():
    sample = "第一段落讲向量检索。第二段落讲稀疏检索。第三段落讲融合排序。" * 12
    chunks = chunk_text(sample, size=120, overlap=30)
    assert len(chunks) >= 3
    joined = "".join(chunks)
    for ch in set(sample.replace(" ", "")):
        assert ch in joined


def test_chunk_respects_size_bound():
    sample = "中文测试数据。" * 200
    chunks = chunk_text(sample, size=100, overlap=20)
    assert all(len(c) <= 130 for c in chunks)


def test_tokenize_cjk_unigram_and_bigram():
    toks = tokenize("检索增强")
    assert "检索" in toks
    assert "检" in toks
    assert "索" in toks


def test_tokenize_latin_kept_as_word():
    toks = tokenize("BM25 与 RAG 对比")
    assert "bm25" in toks
    assert "rag" in toks


def test_normalize_collapses_blank():
    assert normalize("a   b\n\n\nc") == "a b\n\nc"
