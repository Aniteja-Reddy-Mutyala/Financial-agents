"""Unit tests for the dense Retriever's own machinery (sorting, per-section
de-duplication, top_k handling, shape, the no-Bedrock-call short-circuit on
empty input, and Retriever.build()'s file-integrity checks) — all against a
small synthetic corpus with a fake embedder, so these run with no network/
AWS calls and no dependency on data/embeddings.npz existing.

Actual domain relevance (does "what's the CTR threshold" really surface
31 CFR 1010.311 out of the real 215-section KB) requires a real Bedrock
embedding call to embed the query, so those assertions live in
tests/test_live_rag.py instead, gated the same way as test_live_bedrock.py.
See write-compliance-tests and bedrock-rag-retrieval skills.
"""
import json

import numpy as np
import pytest

import retriever as retriever_mod
from retriever import Chunk, Retriever


class _FakeEmbedder:
    """Returns a fixed or scripted vector per call instead of calling
    Bedrock. `queries` records every query passed to embed_query, so tests
    can assert the empty/whitespace short-circuit never calls it at all."""

    def __init__(self, vector=None, by_query=None):
        self.vector = vector if vector is not None else [1.0, 0.0, 0.0]
        self.by_query = by_query or {}
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return self.by_query.get(text, self.vector)


def _chunk(source_id, suffix="0", title="Some section", text="Some regulatory text."):
    return Chunk(chunk_id=f"{source_id}#{suffix}", source_id=source_id, title=title, text=text)


def _unit_matrix(rows):
    """rows: list of raw vectors -> stacked, L2-normalized float32 matrix,
    matching how build_embeddings.py stores normalized Titan vectors."""
    m = np.array(rows, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return m / norms


def _source_ids(hits):
    return [h["source_id"] for h in hits]


# --- Core retrieval machinery ------------------------------------------------


def test_hits_are_sorted_by_score_descending():
    # Vectors must be non-collinear with the query — collinear vectors all
    # collapse to the same direction once L2-normalized, erasing any
    # magnitude difference between them.
    chunks = [_chunk("A"), _chunk("B"), _chunk("C")]
    matrix = _unit_matrix([[0.3, 0.95, 0], [0.95, 0.3, 0], [0.7, 0.7, 0]])
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search("query", top_k=3)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert _source_ids(hits) == ["B", "C", "A"]


def test_results_never_repeat_the_same_section():
    # Section "A" has two chunks; only its best-scoring one should appear.
    chunks = [_chunk("A", "0"), _chunk("A", "1"), _chunk("B", "0")]
    matrix = _unit_matrix([[0.3, 0.95, 0], [0.95, 0.3, 0], [0.7, 0.7, 0]])
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search("query", top_k=5)
    source_ids = _source_ids(hits)
    assert len(source_ids) == len(set(source_ids))
    assert source_ids == ["A", "B"]  # A's best chunk (A#1) outscores B's only chunk


def test_top_k_zero_returns_empty():
    chunks = [_chunk("A")]
    matrix = _unit_matrix([[1, 0, 0]])
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    assert r.search("query", top_k=0) == []


def test_top_k_larger_than_corpus_does_not_crash():
    chunks = [_chunk("A"), _chunk("B")]
    matrix = _unit_matrix([[1, 0, 0], [0, 1, 0]])
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search("query", top_k=10_000)
    assert isinstance(hits, list) and len(hits) <= 2


def test_non_positive_similarity_is_filtered_out():
    chunks = [_chunk("A"), _chunk("B")]
    matrix = _unit_matrix([[1, 0, 0], [-1, 0, 0]])  # B is anti-correlated -> negative score
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search("query", top_k=5)
    assert _source_ids(hits) == ["A"]
    assert all(h["score"] > 0 for h in hits)


def test_hit_shape():
    chunks = [_chunk("31 CFR 1020.320", title="Reports by banks of suspicious transactions.")]
    matrix = _unit_matrix([[1, 0, 0]])
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search("suspicious activity report", top_k=1)
    assert hits
    hit = hits[0]
    for key in ("chunk_id", "source_id", "title", "text", "score"):
        assert key in hit
    assert hit["source_id"] == "31 CFR 1020.320"
    assert isinstance(hit["score"], float)


# --- The no-Bedrock-call short-circuit ---------------------------------------


def test_empty_query_returns_no_crash_and_skips_embedding_call():
    fake = _FakeEmbedder()
    chunks = [_chunk("A")]
    matrix = _unit_matrix([[1, 0, 0]])
    r = Retriever(chunks, matrix, fake)
    assert r.search("", top_k=3) == []
    assert fake.queries == [], "empty query must not spend a real Bedrock call"


def test_whitespace_only_query_skips_embedding_call():
    fake = _FakeEmbedder()
    chunks = [_chunk("A")]
    matrix = _unit_matrix([[1, 0, 0]])
    r = Retriever(chunks, matrix, fake)
    assert r.search("   \n\t  ", top_k=3) == []
    assert fake.queries == []


# --- Edge cases: weird input doesn't crash the plumbing ---------------------


@pytest.mark.parametrize(
    "query",
    [
        "xyzzy qwerty zzzzplorp blah blah nonsense",
        "suspicious activity report filing " * 300,
        "¿Necesitamos presentar un informe de actividad sospechosa?",
        "SAR? $10,000++ (urgent)!! @#%^&*",
    ],
)
def test_odd_queries_do_not_crash(query):
    chunks = [_chunk("A"), _chunk("B")]
    matrix = _unit_matrix([[1, 0, 0], [0, 1, 0]])
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search(query, top_k=3)
    assert isinstance(hits, list)


# --- Retriever.build(): file-integrity checks --------------------------------


def test_build_raises_file_not_found_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever_mod, "EMBEDDINGS_PATH", tmp_path / "missing.npz")
    monkeypatch.setattr(retriever_mod, "META_PATH", tmp_path / "missing_meta.json")
    with pytest.raises(FileNotFoundError, match="build_embeddings.py"):
        Retriever.build()


def test_build_raises_value_error_on_mismatched_lengths(tmp_path, monkeypatch):
    embeddings_path = tmp_path / "embeddings.npz"
    meta_path = tmp_path / "embeddings_meta.json"
    np.savez_compressed(embeddings_path, matrix=np.zeros((2, 3), dtype=np.float32))
    meta_path.write_text(json.dumps([{"chunk_id": "A#0", "source_id": "A", "title": "t", "text": "x"}]))
    monkeypatch.setattr(retriever_mod, "EMBEDDINGS_PATH", embeddings_path)
    monkeypatch.setattr(retriever_mod, "META_PATH", meta_path)
    with pytest.raises(ValueError, match="must match"):
        Retriever.build()
