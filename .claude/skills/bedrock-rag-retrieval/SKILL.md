---
name: bedrock-rag-retrieval
description: Use before changing src/retriever.py, src/bedrock_embeddings.py, or src/build_embeddings.py — the dense-embedding RAG retrieval pipeline (replaced an earlier TF-IDF implementation). Covers the two-stage design (offline batch embedding vs. online query embedding), the one-time indexing job's cost/concurrency/checkpointing, and why MIN_SCORE in agents.py needs recalibrating if you touch the similarity metric.
---

## Two-stage design — don't blur these

1. **Offline, one-time**: `build_embeddings.py` reads every chunk from
   `data/kb.json`, calls Bedrock (`BedrockEmbedder.embed_query`, one real
   call per chunk — Titan embeddings aren't batchable in one API call) for
   each, and caches the result to `data/embeddings.npz` (the vectors) +
   `data/embeddings_meta.json` (the matching chunk metadata, same order).
   Run this manually whenever `data/kb.json` changes. It is **not** run as
   part of the app starting up.
2. **Online, per query**: `Retriever.build()` only reads the two cached
   files — zero Bedrock calls, zero cost, same as the old TF-IDF retriever
   building its index. `Retriever.search(query)` makes exactly **one**
   real Bedrock call (to embed the query itself) and then does a plain
   numpy dot product against the precomputed matrix — fast, in-memory,
   free beyond that one call.

If you ever see `Retriever.build()` or anything at import time making a
network call, something has leaked stage 1 into stage 2 — that's a bug.

## The precomputed index is a cache that can go stale

`data/embeddings.npz`/`embeddings_meta.json` must always describe exactly
the chunks in `data/kb.json` — nothing more, nothing less.
`Retriever.build()` checks this (`matrix.shape[0] != len(chunks)` raises
`ValueError`), but it can't detect "same count, different content" (e.g.
you edited a section's text in `kb.json` without touching its chunk
count) — that silently serves a stale vector until you rebuild. See
`add-ecfr-regulation`'s "rebuild the dense embedding index" step: this is
not optional after any `kb.json` change, including edits to existing
sections, not just additions.

## Cost and concurrency — read before re-running build_embeddings.py

Titan Embed Text v2 is cheap per call (~$0.02/1M input tokens — the full
1,420-chunk KB costs roughly $0.004-0.005 total), but it is **not
batchable** — `embed_documents()` falls through to one API call per text
(see `BedrockEmbeddings._iteratively_embed_documents` in langchain_aws;
only Cohere's embedding models get real batching). This account's actual
Bedrock embedding throughput quota turned out to be low enough that naive
10-way concurrency hit `ThrottlingException` immediately during initial
development. `build_embeddings.py` therefore:

- Uses modest concurrency (`MAX_WORKERS = 3`), not 10+.
- Retries throttled calls with exponential backoff
  (`_embed_with_backoff`), separate from and on top of boto3's own
  built-in retries.
- Checkpoints progress to `data/.embeddings_checkpoint.json` every 50
  completions, and resumes from it on a re-run instead of starting over.
  This file is deleted automatically once the final `.npz`/`.json` are
  written; if you see it lying around, a previous run didn't finish.

Don't "fix" slowness by cranking `MAX_WORKERS` back up without checking
current throttling behavior first — that's what caused the original
failure (a run that died mid-way with a bare, uncaught exception and zero
progress saved, before checkpointing existed). If you do increase
concurrency, test it on a handful of chunks before committing to a full
rebuild, and keep the backoff/checkpoint logic regardless.

## MIN_SCORE in agents.py's verifier_node is metric-dependent — recalibrate it

`verifier_node`'s grounding decision (`MIN_SCORE = 0.08`) was tuned for
TF-IDF cosine similarity's score distribution. Dense Titan-embedding
cosine similarity (computed here as a plain dot product of L2-normalized
vectors, see `Retriever.search`) has a **different** typical range for
"relevant" vs. "irrelevant" — don't assume the old threshold still means
the same thing. After the embedding migration, run a handful of known
grounded and known off-topic queries through the real retriever and check
where their scores actually land before trusting `MIN_SCORE` unchanged;
adjust it if the distributions don't match what `verifier_node` expects.
This is exactly the kind of small-threshold change that has caused real
regressions elsewhere in this project (see the classifier training-data
regression in git history) — change it deliberately, with real queries
checked before and after, not by guessing a new number.

## Testing — mirrors the LLM's fake/live split

Same shape as `bedrock-llm-integration`'s fallback testing: a fake
embedder client for free unit tests, one gated live suite for the real
thing.

- `tests/test_bedrock_embeddings.py` — mocked `BedrockEmbedder`
  (`_FakeEmbeddingsClient`/`_FailingEmbeddingsClient`), mirrors
  `test_bedrock_llm.py`'s pattern exactly.
- `tests/test_retriever.py` — retrieval **machinery** only (sorting,
  per-section dedup, top_k handling, the empty/whitespace short-circuit
  that must never call the embedder) against a small synthetic corpus
  with a `_FakeEmbedder`. Free, no dependency on `data/embeddings.npz`
  existing.
- `tests/test_agents_integration.py` — full-graph plumbing (routing,
  citation passthrough, Bedrock-failure fallback) using a
  `_FixedVectorEmbedder`-backed synthetic retriever that's deterministically
  always "grounded." Free.
- `tests/test_live_rag.py` — the real domain-relevance checks (does "CTR
  threshold" really find 31 CFR 1010.311 in the real 215-section KB) —
  gated by `RUN_LIVE_BEDROCK_TESTS=1` same as `test_live_bedrock.py`,
  since it needs a real embedding call and a built `data/embeddings.npz`.

Never add a domain-relevance assertion to `test_retriever.py` or
`test_agents_integration.py` expecting it to pass for free — it can't,
since real relevance requires a real embedding call. It belongs in
`test_live_rag.py`.
