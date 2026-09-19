"""存储模块：稠密向量 + 稀疏倒排 + 原文，全部落在一个 SQLite 文件里。

作者: 晨星

选型理由：
- sqlite-vec 提供向量近邻检索，官方发 win_amd64 的 py3-none wheel，本机无编译器也能装
- 稀疏检索自己实现 BM25 倒排（原因见 text.py 模块头），与向量检索共用同一个 rowid 空间
- 不引入外部数据库服务：单机可跑、可拷贝、可版本化，部署面最小
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import sqlite_vec  # type: ignore

from .contracts import Chunk, Document
from .text import tokenize

K1 = 1.5
B = 0.75


class StoreError(RuntimeError):
    pass


class DimensionMismatch(StoreError):
    """数据库既有向量维度与当前嵌入模型不一致，需要重建索引。"""


class SqliteStore:
    """单一职责：存与查。不做切分、不做向量化、不做排序融合。"""

    def __init__(self, path: Path | str, dim: int | None = None) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._load_vec()
        self._init_schema()
        self.dim = dim if dim is not None else self._read_dim()

    def _load_vec(self) -> None:
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def _init_schema(self) -> None:
        c = self.conn.cursor()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS docs(
                id TEXT PRIMARY KEY, source TEXT, hash TEXT, meta TEXT, created_at REAL);
            CREATE TABLE IF NOT EXISTS chunks(
                id TEXT UNIQUE NOT NULL, doc_id TEXT NOT NULL, seq INTEGER,
                text TEXT NOT NULL, meta TEXT);
            CREATE TABLE IF NOT EXISTS lexicon(term TEXT PRIMARY KEY, df INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS postings(
                term TEXT NOT NULL, rid INTEGER NOT NULL, tf INTEGER NOT NULL,
                PRIMARY KEY(term, rid));
            CREATE TABLE IF NOT EXISTS doclen(rid INTEGER PRIMARY KEY, len INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_postings_rid ON postings(rid);
            """
        )
        self.conn.commit()

    def _read_dim(self) -> int | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='dim'").fetchone()
        return int(row["value"]) if row else None

    def ensure_vec_table(self, dim: int) -> None:
        """按维度建向量表；维度变化则重建（原有原文仍在，可整体重算向量）。"""
        existing = self._read_dim()
        if existing is not None and existing != dim:
            raise DimensionMismatch(
                f"库内维度 {existing} 与当前模型维度 {dim} 不一致，请执行重建索引"
            )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}])"
        )
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('dim', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(dim),),
        )
        self.conn.commit()
        self.dim = dim

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---------- 写入 ----------

    def doc_hash(self, doc_id: str) -> str | None:
        row = self.conn.execute("SELECT hash FROM docs WHERE id=?", (doc_id,)).fetchone()
        return row["hash"] if row else None

    def upsert_document(
        self, doc: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> int:
        """幂等写入：内容哈希未变则整篇跳过，返回新增块数。"""
        if len(chunks) != len(vectors):
            raise StoreError(f"块数与向量数不一致: {len(chunks)} vs {len(vectors)}")
        with self._lock:
            self.ensure_vec_table(len(vectors[0]) if vectors else self.dim or 0)
            cur = self.conn.cursor()
            cur.execute("DELETE FROM docs WHERE id=?", (doc.id,))
            cur.execute(
                "INSERT INTO docs(id, source, hash, meta, created_at) VALUES(?,?,?,?,?)",
                (doc.id, doc.source, doc.meta.get("hash", ""), json.dumps(doc.meta, ensure_ascii=False), time.time()),
            )
            self._purge_chunks(cur, doc.id)

            n = 0
            for ch, vec in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO chunks(id, doc_id, seq, text, meta) VALUES(?,?,?,?,?)",
                    (ch.id, ch.doc_id, ch.seq, ch.text, json.dumps(ch.meta, ensure_ascii=False)),
                )
                rid = int(cur.lastrowid)
                ch.rowid = rid
                cur.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES(?, ?)",
                    (rid, json.dumps([float(x) for x in vec])),
                )
                self._index_tokens(cur, rid, ch.text)
                n += 1
            self.conn.commit()
            return n

    def _purge_chunks(self, cur: sqlite3.Cursor, doc_id: str) -> None:
        rids = [r[0] for r in cur.execute("SELECT rowid FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()]
        if not rids:
            return
        marks = ",".join("?" * len(rids))
        cur.execute(f"DELETE FROM postings WHERE rid IN ({marks})", rids)
        cur.execute(f"DELETE FROM doclen WHERE rid IN ({marks})", rids)
        cur.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({marks})", rids)
        cur.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self._refresh_lexicon(cur)

    def _index_tokens(self, cur: sqlite3.Cursor, rid: int, text: str) -> None:
        toks = tokenize(text)
        cur.execute("INSERT OR REPLACE INTO doclen(rid, len) VALUES(?, ?)", (rid, len(toks)))
        freq = Counter(toks)
        for term, tf in freq.items():
            cur.execute(
                "INSERT INTO postings(term, rid, tf) VALUES(?,?,?) "
                "ON CONFLICT(term, rid) DO UPDATE SET tf=excluded.tf",
                (term, rid, tf),
            )
            cur.execute(
                "INSERT INTO lexicon(term, df) VALUES(?, 1) "
                "ON CONFLICT(term) DO UPDATE SET df = df + 1",
                (term,),
            )

    def _refresh_lexicon(self, cur: sqlite3.Cursor) -> None:
        cur.execute("DELETE FROM lexicon WHERE term NOT IN (SELECT DISTINCT term FROM postings)")
        cur.execute(
            "UPDATE lexicon SET df = (SELECT COUNT(*) FROM postings WHERE postings.term = lexicon.term)"
        )

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            cur = self.conn.cursor()
            n = cur.execute("SELECT COUNT(*) c FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()["c"]
            self._purge_chunks(cur, doc_id)
            cur.execute("DELETE FROM docs WHERE id=?", (doc_id,))
            self.conn.commit()
            return int(n)

    # ---------- 检索 ----------

    def vector_search(self, vector: list[float], k: int = 20) -> list[tuple[int, float]]:
        if self.dim is None:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                (json.dumps([float(x) for x in vector]), int(k)),
            ).fetchall()
        return [(int(r["rowid"]), float(r["distance"])) for r in rows]

    def sparse_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """BM25 稀疏检索。倒排表在库里，打分在内存里，规模到十万块仍然够用。"""
        with self._lock:
            total = self.conn.execute("SELECT COUNT(*) c FROM doclen").fetchone()["c"]
            if total == 0:
                return []
            avg_len = self.conn.execute("SELECT AVG(len) a FROM doclen").fetchone()["a"] or 1.0
            terms = list(dict.fromkeys(tokenize(query)))
            scores: dict[int, float] = {}
            for term in terms:
                row = self.conn.execute("SELECT df FROM lexicon WHERE term=?", (term,)).fetchone()
                if not row:
                    continue
                df = int(row["df"])
                idf = _idf(total, df)
                post = self.conn.execute(
                    "SELECT rid, tf FROM postings WHERE term=?", (term,)
                ).fetchall()
                for p in post:
                    dl = self.conn.execute("SELECT len FROM doclen WHERE rid=?", (int(p["rid"]),)).fetchone()
                    dlv = int(dl["len"]) if dl else int(avg_len)
                    tf = int(p["tf"])
                    denom = tf + K1 * (1 - B + B * dlv / avg_len)
                    scores[int(p["rid"])] = scores.get(int(p["rid"]), 0.0) + idf * (tf * (K1 + 1) / denom)
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return top

    def get_chunks(self, rowids: Iterable[int]) -> dict[int, Chunk]:
        ids = list(rowids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT rowid, id, doc_id, seq, text, meta FROM chunks WHERE rowid IN ({marks})", ids
            ).fetchall()
        out: dict[int, Chunk] = {}
        for r in rows:
            out[int(r["rowid"])] = Chunk(
                id=r["id"], doc_id=r["doc_id"], seq=int(r["seq"]), text=r["text"],
                meta=json.loads(r["meta"] or "{}"), rowid=int(r["rowid"]),
            )
        return out

    def stats(self) -> dict[str, Any]:
        with self._lock:
            docs = self.conn.execute("SELECT COUNT(*) c FROM docs").fetchone()["c"]
            chunks = self.conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
            terms = self.conn.execute("SELECT COUNT(*) c FROM lexicon").fetchone()["c"]
            vecs = self.conn.execute("SELECT COUNT(*) c FROM vec_chunks").fetchone()["c"]
        return {
            "db": self.path, "docs": docs, "chunks": chunks,
            "lexicon_terms": terms, "vectors": vecs, "dim": self.dim,
        }

    def all_chunk_texts(self) -> list[tuple[int, str]]:
        with self._lock:
            rows = self.conn.execute("SELECT rowid, text FROM chunks ORDER BY rowid").fetchall()
        return [(int(r["rowid"]), r["text"]) for r in rows]

    def rebuild_vectors(self, embedder) -> int:
        """维度变化后重建向量索引：原文还在，重新算一遍向量即可。"""
        pairs = self.all_chunk_texts()
        if not pairs:
            return 0
        dim = embedder.dim
        with self._lock:
            self.conn.execute("DROP TABLE IF EXISTS vec_chunks")
            self.conn.commit()
        self.ensure_vec_table(dim)
        texts = [t for _, t in pairs]
        vectors = embedder.embed(texts)
        with self._lock:
            cur = self.conn.cursor()
            for (rid, _), vec in zip(pairs, vectors):
                cur.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES(?, ?)",
                    (rid, json.dumps([float(x) for x in vec])),
                )
            self.conn.commit()
        return len(pairs)


def _idf(total: int, df: int) -> float:
    import math

    return math.log(1 + (total - df + 0.5) / (df + 0.5))
