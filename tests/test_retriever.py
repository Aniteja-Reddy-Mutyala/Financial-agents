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
from retriever import CHUNK_WORDS, Chunk, Retriever, _chunk_text


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


# --- Bounded neighbor-chunk expansion ----------------------------------------
#
# Regression tests for GitHub issue #1's "best-chunk-only" finding:
# search() used to return only a section's single best-scoring chunk, so a
# decisive clause living in a different (lower-scoring) chunk of the same
# section could never reach the LLM. The fix adds the immediately
# preceding/following same-section chunk's text, bounded to at most 2 extra
# chunks per hit regardless of how long the section is (deliberately not
# "give the whole section" — that was tried before and reverted for not
# generalizing, and is unbounded for a long section).


def test_search_includes_preceding_and_following_same_section_chunks():
    # Section "A" has 3 chunks; the middle one (A#1) scores best. Both its
    # neighbors should be merged into the returned text.
    chunks = [_chunk("A", "0", text="first part"), _chunk("A", "1", text="middle part"), _chunk("A", "2", text="last part")]
    matrix = _unit_matrix([[0.5, 0.5, 0], [1, 0, 0], [0.5, 0, 0.5]])
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search("query", top_k=1)
    assert len(hits) == 1
    assert hits[0]["source_id"] == "A"
    assert "first part" in hits[0]["text"]
    assert "middle part" in hits[0]["text"]
    assert "last part" in hits[0]["text"]


def test_search_does_not_pull_in_a_different_sections_chunk_as_neighbor():
    # Section "A" has one chunk, immediately followed (in self.chunks
    # order) by section "B"'s only chunk. "A"'s hit must not absorb "B"'s
    # text just because it's adjacent in the list — only same-source_id
    # neighbors count.
    chunks = [_chunk("A", "0", text="section A text"), _chunk("B", "0", text="section B text")]
    matrix = _unit_matrix([[1, 0, 0], [0, 1, 0]])
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search("query", top_k=2)
    a_hit = next(h for h in hits if h["source_id"] == "A")
    assert "section B text" not in a_hit["text"]


def test_search_neighbor_expansion_bounded_regardless_of_section_length():
    # A section with many chunks (simulating a long real section, e.g. a
    # 5,000+ word SOX section) must still only ever contribute at most 3
    # chunks' worth of text to one hit (best + 1 before + 1 after) — never
    # the whole section.
    chunks = [_chunk("A", str(i), text=f"chunk{i}text") for i in range(20)]
    # Make chunk 10 score highest; the rest point in an orthogonal
    # direction so they don't tie with it once L2-normalized.
    rows = [[0, 1, 0]] * 20
    rows[10] = [1, 0, 0]
    matrix = _unit_matrix(rows)
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search("query", top_k=1)
    assert len(hits) == 1
    included = sum(1 for i in range(20) if f"chunk{i}text" in hits[0]["text"])
    assert included == 3  # chunk 9, 10, 11 only


def test_search_neighbor_expansion_handles_edge_chunk_with_no_following_neighbor():
    # The best-scoring chunk being the very first or last chunk of a
    # section (no "before" or "after" neighbor to borrow from) must not
    # crash or go out of bounds.
    chunks = [_chunk("A", "0", text="only neighbor-less chunk")]
    matrix = _unit_matrix([[1, 0, 0]])
    r = Retriever(chunks, matrix, _FakeEmbedder(vector=[1, 0, 0]))
    hits = r.search("query", top_k=1)
    assert hits[0]["text"] == "only neighbor-less chunk"


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


# --- _chunk_text: sentence-boundary chunking ---------------------------------
#
# Regression tests for GitHub issue #1's chunking-truncation root cause: the
# old fixed-word-count sliding window could cut a chunk off mid-sentence,
# silently dropping a regulation's decisive clause from the single
# best-scoring chunk (measured for 21 CFR 11.100's identity-verification
# requirement). The fix packs whole sentences into ~CHUNK_WORDS windows
# instead, so a chunk boundary never falls inside a sentence.


def test_chunk_text_never_splits_a_sentence_in_half():
    # One short section's worth of sentences, well under CHUNK_WORDS on
    # their own — every sentence should land whole inside some chunk.
    text = (
        "Each electronic signature shall be unique to one individual. "
        "The organization shall verify the identity of the individual. "
        "Persons using electronic signatures shall certify to the agency."
    )
    chunks = list(_chunk_text(text))
    assert len(chunks) == 1
    for sentence in [
        "Each electronic signature shall be unique to one individual.",
        "The organization shall verify the identity of the individual.",
        "Persons using electronic signatures shall certify to the agency.",
    ]:
        assert sentence in chunks[0]


def test_chunk_text_packs_multiple_sentences_up_to_size():
    # A long section (several sentences each ~15 words) should split into
    # more than one chunk once the running word count exceeds CHUNK_WORDS,
    # but every individual sentence still appears intact in some chunk.
    sentence = "The regulated entity shall maintain complete records of every transaction for audit purposes. "
    text = sentence * 20  # ~280 words total, well past CHUNK_WORDS=120
    chunks = list(_chunk_text(text))
    assert len(chunks) > 1
    for c in chunks:
        assert c.strip().endswith(".")  # never cut off mid-sentence
        assert len(c.split()) <= CHUNK_WORDS + len(sentence.split())  # no runaway growth


def test_chunk_text_overlap_shares_sentences_between_consecutive_chunks():
    sentence = "The regulated entity shall maintain complete records of every transaction for audit purposes. "
    text = sentence * 20
    chunks = list(_chunk_text(text))
    assert len(chunks) > 1
    # Consecutive chunks should share at least the trailing/leading sentence
    # (the whole point of CHUNK_OVERLAP > 0) rather than picking up exactly
    # where the previous one left off.
    assert chunks[0].split(". ")[-1].strip().rstrip(".") in chunks[1]


def test_chunk_text_handles_single_sentence_longer_than_chunk_size():
    # A sentence with no internal '.', ';', or ':' longer than CHUNK_WORDS
    # has nowhere to legally split — it must still come back as one whole
    # chunk (not truncated, not an infinite loop) rather than crash.
    long_sentence = "shall " * (CHUNK_WORDS + 50) + "comply"
    chunks = list(_chunk_text(long_sentence))
    assert len(chunks) == 1
    assert chunks[0].split()[-1] == "comply"


def test_chunk_text_empty_input_yields_no_chunks():
    assert list(_chunk_text("")) == []
    assert list(_chunk_text("   ")) == []


def test_load_chunks_builds_from_real_kb_without_crashing():
    chunks = retriever_mod.load_chunks()
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    # Every chunk_id is unique and traces back to a real source_id#index.
    assert len(chunks) == len({c.chunk_id for c in chunks})


def test_load_chunks_21_cfr_11_100_keeps_identity_verification_sentence_whole():
    # The exact section from GitHub issue #1 — assert the decisive clause
    # ("the organization shall verify the identity of the individual")
    # survives intact in at least one chunk, not cut off mid-sentence.
    chunks = [c for c in retriever_mod.load_chunks() if c.source_id == "21 CFR 11.100"]
    assert chunks, "21 CFR 11.100 missing from data/kb.json"
    assert any("verify the identity of the individual" in c.text for c in chunks)
    assert any(
        c.text.strip().count("shall verify the identity of the individual.") == 1
        and "Before an organization establishes" in c.text
        for c in chunks
    ), "the identity-verification sentence should appear whole, with its own sentence intact"
