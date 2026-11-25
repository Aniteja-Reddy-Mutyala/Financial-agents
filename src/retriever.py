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


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    text: str


def _chunk_text(words, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if not piece:
            continue
        yield " ".join(piece)
        if start + size >= len(words):
            break


def load_chunks():
    records = json.loads(KB_PATH.read_text())
    chunks = []
    for rec in records:
        words = rec["text"].split()
        for i, piece in enumerate(_chunk_text(words)):
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
                    "text": c.text,
                    "score": score,
                }
            )
        return results


if __name__ == "__main__":
    r = Retriever.build()
    for hit in r.search("suspicious activity report filing deadline for banks"):
        print(f"[{hit['score']:.3f}] {hit['source_id']} — {hit['title']}")
        print("   ", hit["text"][:160], "...")
