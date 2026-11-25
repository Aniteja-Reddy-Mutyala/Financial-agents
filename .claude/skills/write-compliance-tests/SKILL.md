---
name: write-compliance-tests
description: Use before adding or updating any test in this repo, and whenever a new feature (a new node, intent/domain, regulatory part, CLI flag, or module) needs test coverage as part of being "done." Covers which test file owns what, the mocking conventions that avoid real AWS calls, the three-tier coverage pattern, and how to run a subset fast.
---

## Test file ownership — don't duplicate across files

| File | Owns |
|---|---|
| `tests/test_classifier.py` | `IntentClassifier` — one row in `CLEAR_CASES` per intent, plus shared ambiguous/edge-case tests that already iterate `VALID_LABELS` generically. (No chitchat-specific tests — greetings aren't special-cased in the classifier at all; see `edit-compliance-graph`'s "escalate_node" section.) |
| `tests/test_retriever.py` | `Retriever`'s **machinery** — sorting, per-section dedup, top_k handling, the empty/whitespace no-embedding-call short-circuit, `Retriever.build()`'s file-integrity checks, sentence-boundary `_chunk_text` behavior, and the bounded neighbor-chunk expansion (`_text_with_neighbors`) — against a small synthetic corpus with a fake embedder. Free, no dependency on `data/embeddings.npz`. Real relevance does NOT belong here (see `test_live_rag.py`). |
| `tests/test_bedrock_embeddings.py` | `BedrockEmbedder` — mocked embeddings-client classes, success/failure, model/dimensions regression guard. |
| `tests/test_routing.py` | `verifier_node`, `route_after_verify`, `escalate_node` (both the LLM-reply path via `respond_ungrounded()` and the static-fallback path on `SummarizerUnavailable`), `_unique_evidence`, `planner_node`'s `has_signal`/`NO_PAD_INTENTS` padding decision, and `retriever_node`'s raw/augmented-query merge — unit-tested against **hand-built state dicts and scripted fakes** (a fake retriever, a fake ungrounded-capable summarizer), not live retrieval or a live LLM, so these never flake on retrieval scores or real model output. |
| `tests/test_bedrock_llm.py` | `BedrockSummarizer` — mocked chat model classes, success/failure/fallback for both `.summarize()` and `.respond_ungrounded()`, message shape (including that `.respond_ungrounded()` uses `UNGROUNDED_SYSTEM_PROMPT`, not `SYSTEM_PROMPT`, and is never passed evidence). Also owns `.chat()` (the generic method the Research agent uses) — success, failure, and that it passes the caller's own prompts, not `SYSTEM_PROMPT`/`build_user_message`. |
| `tests/test_groq_llm.py` | `GroqSummarizer` — mocked Groq client classes, success/failure/fallback for both `.summarize()` and `.respond_ungrounded()`, message shape, and the empty-content-on-exhausted-reasoning-budget case for each. Mirrors `test_bedrock_llm.py`'s structure, including `.chat()` coverage. |
| `tests/test_agents_integration.py` | The whole `ComplianceAgentGraph`'s **plumbing** (routing, citation passthrough, LLM-failure fallback, crash-resistance, and that a greeting goes through the normal plan/retrieve/verify path and still gets a conversational reply via `escalate_node`, not a special bypass) using a synthetic, deterministically-"grounded"/"ungrounded" fake retriever — not real relevance. Also owns `_default_llm`'s `LLM_PROVIDER` selection logic (bedrock/groq/unknown, case-insensitivity, explicit `llm=` override precedence, fresh-read-not-cached) — the implementation now lives in `agent_common.py`, but this file still owns the coverage for how `agents.py` uses it. |
| `tests/test_web_search.py` | `WebSearcher` (the Research agent's web-search client) — a fake `ddgs` client object, success/shape, the empty-query short-circuit, failure → `WebSearchUnavailable`. |
| `tests/test_document_parser.py` | `parse_document` (the Research agent's PDF/text extraction) — real `pypdf` against small in-test-built fixtures (not a network boundary, so no fake needed), unsupported extension / corrupt-file / empty-text failure modes, `MAX_EXTRACTED_CHARS` truncation. |
| `tests/test_research_agent.py` | The whole `ResearchAgentGraph`'s **plumbing** — mirrors `test_agents_integration.py`'s approach (fake `.chat()`-capable summarizer, fake web searcher, `ResearchAgentGraph(llm=..., web_searcher=..., document_parser=...)` overrides): intake routing, the `MAX_TOPICS` cap, per-topic search-failure isolation, the all-empty-results template fallback vs. the different LLM-failure template fallback, document-parse-failure degrading to query-only. Also owns its own copy of the `_default_llm`/`LLM_PROVIDER` coverage (same reasoning as `test_agents_integration.py`'s row above — one owning test file per *module*'s usage, even though the implementation is shared). |
| `tests/test_live_bedrock.py` | The one real, billed Bedrock LLM call — skipped unless `RUN_LIVE_BEDROCK_TESTS=1`. Don't add more tests here; add exactly one assertion to the existing test if you must. |
| `tests/test_live_rag.py` | Real domain-relevance checks (does "CTR threshold" really find 31 CFR 1010.311 in the real KB) against the real `Retriever.build()` — also skipped unless `RUN_LIVE_BEDROCK_TESTS=1`, since every search is a real, billed embedding call. This is the only place relevance assertions belong now; see `bedrock-rag-retrieval`. |
| `tests/test_live_research.py` | The Research agent's real `ddgs` call and a real end-to-end `.handle()` call (Bedrock or Groq, per `LLM_PROVIDER`) — skipped unless `RUN_LIVE_RESEARCH_TESTS=1` (its own gate, separate from `RUN_LIVE_BEDROCK_TESTS` — see `edit-research-agent`). Confirms the pipeline runs end-to-end for real, not that any given answer's claims are accurate. |

Before writing a new test, check whether an existing parametrized list
(`CLEAR_CASES`, `AMBIGUOUS_QUERIES`, the edge-case queries in
`test_agents_integration.py`) is the right place to add one row instead of
writing a whole new test function — it's cheaper to read, cheaper to
write, and keeps the generic assertions (shape, no-crash, valid label)
from being re-derived per case.

## The three-tier coverage pattern

Every "real" behavior in this repo (an intent, a retrieval domain, a graph
branch) gets tested at three tiers — this is what "test coverage" means
here, not just a happy-path test:

1. **Clear-cut** — one obviously-correct input per behavior, asserting the
   exact expected output (label, source_id substring, branch taken).
2. **Ambiguous** (classifier/routing only) — input that plausibly spans
   two behaviors; assert it still returns something *well-formed* (valid
   label, confidence in `[0, 1]`, not overconfident), not a specific
   answer — asserting a specific label here overfits the test to the
   model's current quirks.
3. **Edge case** — empty string, whitespace-only, punctuation-only, very
   long input, non-English, special characters, `top_k=0`, `top_k` larger
   than the corpus. Assert "doesn't crash, returns well-formed output,"
   not correctness — these inputs have no well-defined correct answer.

When you add a new intent/domain, add at least one clear-cut case; add an
edge case too if the new code path has new failure modes (e.g. a new
regex, a new data shape) that the existing edge-case tests wouldn't catch.

## Mocking conventions — no real AWS calls outside the two gated live files

- **Fake the LLM**, never the network layer: a `_FakeSummarizer`/`_FakeChatModel`
  class with a `model_id` attribute and a `summarize(...)`/`invoke(...)`
  method matching `BedrockSummarizer`'s interface. See
  `test_agents_integration.py::_FakeSummarizer` and
  `test_bedrock_llm.py::_FakeChatModel`/`_FailingChatModel` for the exact
  shape — reuse that shape for a new fake rather than inventing one.
- **Fake the embedder the same way**, never the network layer: a class
  with an `embed_query(text)` method, matching `BedrockEmbedder`'s
  interface. See `test_bedrock_embeddings.py::_FakeEmbeddingsClient`,
  `test_retriever.py::_FakeEmbedder` (records every query it was asked to
  embed, so tests can assert a short-circuit path never called it), and
  `test_agents_integration.py::_FixedVectorEmbedder` (always returns the
  same vector, used to make a synthetic retriever deterministically
  "grounded").
- **Fake the retriever to force a branch deterministically, or to test
  graph plumbing** — e.g. `_EmptyRetriever.search()` always returning `[]`
  to force the escalate branch, or a small synthetic `Retriever` built
  from 2-3 hand-written `Chunk`s + a `_FixedVectorEmbedder` to force the
  grounded branch. **Never use the real `Retriever.build()` in a test that
  should run for free** — every `search()` call against it makes a real,
  billed Bedrock embedding call now (this changed when retrieval moved
  from TF-IDF to dense embeddings; see `bedrock-rag-retrieval`). Real
  relevance assertions belong only in `test_live_rag.py`.
- **`ComplianceAgentGraph(llm=..., retriever=...)` takes both as
  constructor overrides** specifically so integration tests can swap
  either without touching real AWS — use that, don't monkeypatch
  internals.
- A test that needs a real, billed Bedrock call (LLM or embedding) belongs
  only in `test_live_bedrock.py` or `test_live_rag.py`, gated exactly like
  their existing skip condition (`os.environ.get("RUN_LIVE_BEDROCK_TESTS")
  != "1"`). Never add a third Bedrock test file under that same gate, and
  never add an ungated one. A *different* live-network boundary (e.g.
  `test_live_research.py`'s real `ddgs` + LLM calls for the Research
  agent) gets its **own** gate (`RUN_LIVE_RESEARCH_TESTS`) instead of
  reusing `RUN_LIVE_BEDROCK_TESTS` — different failure modes should stay
  isolatable, not whether a file exists at all.

## xfail policy

`xfail(strict=True)` is for a **real, root-caused bug you're deliberately
not fixing yet** — not a flaky test, not a placeholder. Strict mode means
an unexpected pass fails loudly (catching exactly the situation where the
bug gets fixed incidentally and the test goes stale) — see the
BSA/AML retrieval-gap fix history in `tests/test_retriever.py` /
README's "Honest limitations" for a worked example of the full lifecycle
(found → `xfail`'d with a comment explaining the cause → fixed → converted
to a real regression test, never just deleted).

## Running a subset fast

```bash
python3 -m pytest tests/test_classifier.py -v              # one file
python3 -m pytest -k "hipaa or sox"                          # by keyword across files
python3 -m pytest -q                                         # whole suite, mocked, no AWS cost
RUN_LIVE_BEDROCK_TESTS=1 python3 -m pytest tests/test_live_bedrock.py tests/test_live_rag.py -v
```

Prefer `-k` over re-reading a whole test file to find the right test name
when you already know roughly what you're looking for — it's cheaper than
scanning the file.
