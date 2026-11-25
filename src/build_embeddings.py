"""One-time batch job: embed every chunk in data/kb.json via Amazon Bedrock
Titan Embed Text v2 and cache the resulting vectors to data/embeddings.npz
+ data/embeddings_meta.json.

Run this whenever data/kb.json changes (new regulation added, chunking
params changed) — NOT on every process start. retriever.py's Retriever
only reads the cached files; it never calls Bedrock to index, only to
embed a query at search time.

Cost/time note (read before running): Titan embeddings aren't batchable
in one Bedrock API call (unlike Cohere's embedding models) — each chunk
is a separate billed call. At ~1,420 chunks and Titan Embed Text v2
pricing (~$0.02 / 1M input tokens, ~160 tokens/chunk), this is roughly
$0.004-0.005 total. This account's actual Bedrock embedding throughput
quota turned out to be low enough that naive 10-way concurrency hit
ThrottlingException immediately — this script runs with modest
concurrency, retries throttled chunks with backoff, and checkpoints
progress to disk every CHECKPOINT_EVERY chunks so a crash or interrupt
loses at most a few minutes of work, not the whole run. Re-running this
script resumes from the last checkpoint instead of starting over.
"""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bedrock_embeddings import BedrockEmbedder, BedrockEmbeddingUnavailable
from retriever import load_chunks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"
META_PATH = DATA_DIR / "embeddings_meta.json"
CHECKPOINT_PATH = DATA_DIR / ".embeddings_checkpoint.json"

MAX_WORKERS = 3
CHECKPOINT_EVERY = 50
MAX_RETRIES_PER_CHUNK = 6


def _embed_with_backoff(embedder: BedrockEmbedder, text: str) -> list:
    delay = 1.0
    last_exc = None
    for attempt in range(MAX_RETRIES_PER_CHUNK):
        try:
            return embedder.embed_query(text)
        except BedrockEmbeddingUnavailable as exc:
            last_exc = exc
            if "Throttling" not in str(exc) and "Too many requests" not in str(exc):
                raise  # a real failure, not just throttling — don't mask it
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 20)
    raise last_exc


def _load_checkpoint():
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        return {int(k): v for k, v in data.items()}
    return {}


def build(embedder: BedrockEmbedder = None):
    embedder = embedder or BedrockEmbedder()
    chunks = load_chunks()
    done = _load_checkpoint()
    if done:
        print(f"Resuming from checkpoint: {len(done)}/{len(chunks)} chunks already embedded.")

    remaining = {i: c for i, c in enumerate(chunks) if i not in done}
    print(f"Embedding {len(remaining)} remaining chunks via {embedder.model_id} ({embedder.dimensions}d), "
          f"{MAX_WORKERS}-way concurrency...")

    t0 = time.time()
    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_embed_with_backoff, embedder, f"{c.title}\n\n{c.text}"): i
            for i, c in remaining.items()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            done[idx] = future.result()
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= CHECKPOINT_EVERY:
                CHECKPOINT_PATH.write_text(json.dumps(done))
                completed_since_checkpoint = 0
            if len(done) % 100 == 0 or len(done) == len(chunks):
                print(f"  {len(done)}/{len(chunks)} embedded ({time.time() - t0:.0f}s elapsed this run)")

    CHECKPOINT_PATH.write_text(json.dumps(done))  # final checkpoint before finalizing

    vectors = [done[i] for i in range(len(chunks))]
    matrix = np.array(vectors, dtype=np.float32)
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(EMBEDDINGS_PATH, matrix=matrix)
    meta = [
        {"chunk_id": c.chunk_id, "source_id": c.source_id, "title": c.title, "text": c.text}
        for c in chunks
    ]
    META_PATH.write_text(json.dumps(meta))
    CHECKPOINT_PATH.unlink(missing_ok=True)  # no longer needed once the final files exist

    print(f"\nWrote {matrix.shape} matrix to {EMBEDDINGS_PATH} ({EMBEDDINGS_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"Wrote metadata for {len(meta)} chunks to {META_PATH}")
    print(f"Total time this run: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    build()
