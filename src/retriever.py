"""Dense RAG retriever: chunk eCFR sections, embed with Amazon Bedrock Titan
embeddings, retrieve by cosine similarity.

Stands in for the PGVector/Pinecone/ChromaDB RAG pipelines described in the
original project — this is the real dense-embedding version of it, not an
approximation. Replaces an earlier TF-IDF implementation (see README's
"Honest limitations" for that fix history): dense embeddings understand
semantic relationships natively (SAR ~ "suspicious activity report", cash ~
currency) without a hand-maintained synonym list, which TF-IDF needed.

The embedding matrix is precomputed offline by build_embeddings.py and
cached to data/embeddings.npz + data/embeddings_meta.json — building a
Retriever here only reads those cached files, no Bedrock call. The one
Bedrock call per `search()` is to embed the query itself; see
bedrock-rag-retrieval skill before changing this file.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bedrock_embeddings import BedrockEmbedder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_PATH = DATA_DIR / "kb.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"
META_PATH = DATA_DIR / "embeddings_meta.json"
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30

# Splits after a sentence-ending punctuation mark followed by whitespace.
# CFR prose is heavily sub-clausal (";", ":" introducing a list) as well as
# ".", so all three count as a sentence boundary here — see _chunk_text.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;:])\s+")


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    text: str


def _split_sentences(text):
    return [s for s in _SENTENCE_BOUNDARY_RE.split(text.strip()) if s]


def _chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Pack whole sentences into ~`size`-word windows so a chunk never cuts
    a sentence in half.

    The previous version sliced raw `words[start:start+size]` with no regard
    for sentence boundaries — measured (GitHub issue #1) to silently drop a
    regulation's decisive clause when it happened to fall right after a
    120-word cut (21 CFR 11.100's identity-verification requirement started
    just after one such cut, so the best-scoring chunk never contained it).
    Packing by sentence trades a little precision in chunk size (a chunk can
    run over `size` words if a single sentence is long) for never silently
    truncating mid-sentence.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return
    sentence_words = [s.split() for s in sentences]
    n = len(sentences)
    i = 0
    while i < n:
        chunk_start = i
        current, current_len = [], 0
        while i < n and (current_len == 0 or current_len + len(sentence_words[i]) <= size):
            current.append(sentences[i])
            current_len += len(sentence_words[i])
            i += 1
        yield " ".join(current)
        if i >= n:
            break
        # Overlap: back up by whole sentences until we've covered at least
        # `overlap` words, so the next chunk re-starts a few sentences
        # before this one ended rather than at an arbitrary word offset.
        # Never back up past chunk_start + 1 — a lone sentence longer than
        # `size` words (current_len==0 branch above) would otherwise back
        # up onto itself and loop forever re-yielding the same chunk.
        overlap_words, back = 0, 0
        j = i - 1
        while j > chunk_start and overlap_words < overlap:
            overlap_words += len(sentence_words[j])
            back += 1
            j -= 1
        i -= back


def load_chunks():
    records = json.loads(KB_PATH.read_text())
    chunks = []
    for rec in records:
        for i, piece in enumerate(_chunk_text(rec["text"])):
            chunks.append(
                Chunk(
                    chunk_id=f"{rec['id']}#{i}",
                    source_id=rec["id"],
                    title=rec["title"],
                    text=piece,
                )
            )
    return chunks


class Retriever:
    """Dense-embedding retrieval with per-section de-duplication.

    `search()` returns at most one chunk per section, ranked by that
    section's best-scoring chunk — without that, a single long section
    split into many overlapping chunks could occupy every `top_k` slot
    and crowd out shorter, more on-point sections (same reasoning as the
    old TF-IDF retriever had, independent of how similarity is computed).
    """

    def __init__(self, chunks, matrix, embedder: BedrockEmbedder):
        self.chunks = chunks
        self.matrix = matrix  # shape (n_chunks, dims), L2-normalized rows
        self.embedder = embedder

    @classmethod
    def build(cls, embedder: BedrockEmbedder = None):
        if not EMBEDDINGS_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                f"{EMBEDDINGS_PATH.name} / {META_PATH.name} not found in "
                f"{DATA_DIR} — run `python3 src/build_embeddings.py` first "
                "to build the dense index from data/kb.json."
            )
        meta = json.loads(META_PATH.read_text())
        chunks = [Chunk(**m) for m in meta]
        matrix = np.load(EMBEDDINGS_PATH)["matrix"]
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"{EMBEDDINGS_PATH.name} has {matrix.shape[0]} vectors but "
                f"{META_PATH.name} has {len(chunks)} chunks — they must "
                "match. Rebuild both together with build_embeddings.py."
            )
        return cls(chunks, matrix, embedder or BedrockEmbedder())

    def _text_with_neighbors(self, idx):
        """Best-scoring chunk's text, plus its immediate same-section
        neighbor chunks (one before, one after, if they exist) for extra
        context — bounded growth: at most 2 extra ~CHUNK_WORDS-word chunks
        per hit, regardless of how long the section itself is.

        GitHub issue #1's deeper finding: `search()` returns only a
        section's single best-scoring chunk, so a decisive clause can live
        in a *different*, lower-scoring chunk of the same section and
        never reach the LLM — confirmed for 21 CFR 11.100, where the
        identity-verification clause lives in the chunk right before the
        one that scores best for a signature-validation query. Since
        `self.chunks` preserves load_chunks()'s per-section sequential
        order (see build_embeddings.py — embeddings/meta are written in
        that same order), the chunk immediately before/after the best one
        is usually the next most relevant content in that section. This
        is deliberately NOT "give the whole section" (tried before,
        reverted for not generalizing, and unbounded for a 5,000+ word
        section) — it's a small, fixed-size expansion either way.
        """
        c = self.chunks[idx]
        parts = [c.text]
        if idx > 0 and self.chunks[idx - 1].source_id == c.source_id:
            parts.insert(0, self.chunks[idx - 1].text)
        if idx < len(self.chunks) - 1 and self.chunks[idx + 1].source_id == c.source_id:
            parts.append(self.chunks[idx + 1].text)
        return " ".join(parts)

    def search(self, query, top_k=3):
        if not query or not query.strip():
            return []  # short-circuit: don't spend a Bedrock call on nothing
        query_vec = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        sims = self.matrix @ query_vec  # dot product == cosine sim (both L2-normalized)
        ranked = np.argsort(sims)[::-1]
        seen_sources = set()
        results = []
        for idx in ranked:
            if len(results) >= top_k:
                break
            score = float(sims[idx])
            if score <= 0:
                continue
            c = self.chunks[idx]
            if c.source_id in seen_sources:
                continue
            seen_sources.add(c.source_id)
            results.append(
                {
                    "chunk_id": c.chunk_id,
                    "source_id": c.source_id,
                    "title": c.title,
                    "text": self._text_with_neighbors(idx),
                    "score": score,
                }
            )
        return results


if __name__ == "__main__":
    r = Retriever.build()
    for hit in r.search("suspicious activity report filing deadline for banks"):
        print(f"[{hit['score']:.3f}] {hit['source_id']} — {hit['title']}")
        print("   ", hit["text"][:160], "...")
