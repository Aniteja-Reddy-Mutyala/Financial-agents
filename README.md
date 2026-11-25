# Mini BSA/AML Compliance Agent

A small, locally-runnable recreation of the agentic AI / RAG compliance
work described in the Northern Trust (Senior Gen AI Engineer) role —
scaled down to run with no cloud services, no API keys, and no GPU.

## What it does

Given a free-text compliance question (e.g. *"do we need to file a SAR for
this wire transfer?"* or *"what does HIPAA require for PHI access logs?"*),
the agent:

1. **Classifies intent** — SAR filing, CTR filing, recordkeeping, AML
   program requirements, inter-institution info sharing, FDA
   electronic-records/signature requirements, HIPAA privacy/security, or
   SOX internal-control/certification requirements.
2. **Plans retrieval** — picks a retrieval query and how many passages to
   pull based on the intent.
3. **Retrieves evidence** — searches a real, multi-domain regulatory
   knowledge base (BSA/AML, FDA, HIPAA, SOX — see Dataset below) via
   **dense Amazon Bedrock Titan embeddings** + cosine similarity over
   chunked sections (precomputed offline; see Stack below).
4. **Verifies grounding** — a Compliance Verifier agent checks retrieval
   confidence before allowing an answer to ship. A **LangGraph conditional
   edge** routes ungrounded queries to a human-escalation node instead of
   letting an LLM guess.
5. **Summarizes** — calls **Amazon Bedrock (Nova Micro)** via
   **langchain_aws**, the cheapest Amazon text model, to draft a cited
   answer from the verified regulatory text. Falls back to a templated
   draft if Bedrock is unreachable.

## Example queries

Factual lookups — "what does the regulation say":

- *"what is the CTR threshold for a cash deposit"*
- *"how long must we retain records of wire transfers"*
- *"what safeguards does HIPAA require for protected health information"*
- *"what must the CEO certify about internal controls each quarter"*
- *"what audit trail does the FDA require for electronic signatures"*

Scenario-based — "given this situation, what applies":

- *"a customer made three $4,000 cash deposits over two days, seemingly to
  avoid the $10,000 reporting threshold — what should we do?"*
- *"we noticed a customer wiring $50,000 to an offshore account with no
  clear business purpose — what are our obligations?"*
- *"an employee accessed a patient's medical records without
  authorization — what are our obligations now?"*

These aren't keyword lookups — "three $4,000 deposits" gets classified as
a SAR-filing question (structuring), not a CTR question, even though no
single deposit crosses the CTR's $10,000 line. The system reports what the
regulation requires and cites the specific section(s); it doesn't replace
the judgment call a human compliance officer makes on an ambiguous,
full-context fact pattern. Try any of these via `python3 demo.py
"<question>"` or in the Streamlit UI (`streamlit run app_ui.py`).

## How it maps to the original project

| Original (Northern Trust)                                   | This mini version                                              |
|---------------------------------------------------------------|------------------------------------------------------------------|
| LangGraph multi-agent orchestration (Orchestrator, Verifier, Planner) | A real `langgraph.graph.StateGraph` with conditional routing in [`src/agents.py`](src/agents.py) |
| Node roles: Orchestrator, Planner, Verifier, Summarizer/Compliance agent | Graph nodes `classify_intent`, `plan`, `retrieve`, `verify`, `summarize`/`escalate` |
| Hybrid LLM inference (Claude 3.5/4, Claude Instant v1) via LangChain | **Model-agnostic final-answer step**: `langchain_aws` `ChatBedrockConverse` ([`src/bedrock_llm.py`](src/bedrock_llm.py), default) or the `groq` SDK ([`src/groq_llm.py`](src/groq_llm.py)), switchable via `LLM_PROVIDER` — see "Stack" below |
| RAG pipelines (PGVector, Redis, Chroma, Pinecone)              | Dense-embedding retriever in [`src/retriever.py`](src/retriever.py), backed by **Amazon Bedrock Titan embeddings** ([`src/bedrock_embeddings.py`](src/bedrock_embeddings.py)) precomputed offline by [`src/build_embeddings.py`](src/build_embeddings.py) |
| Intent classification / financial query understanding          | `LogisticRegression` + TF-IDF in [`src/classifier.py`](src/classifier.py) |
| Regulatory compliance data (SOC2, GDPR, HIPAA, SOX, 21 CFR 11)  | Real text pulled from the eCFR API: BSA/AML (31 CFR 1010/1020), FDA e-records (21 CFR 11), HIPAA (45 CFR 160/164), SOX-implementing SEC rules (17 CFR). GDPR and SOC2 are intentionally **not** included — see Honest limitations |
| Data governance / auditability for AI systems in finance        | Every graph node appends to `state["trace"]`; answers carry citations |
| Error recovery / consensus building                            | LangGraph conditional edge: `verify` routes to `escalate` (human-in-the-loop) when retrieval isn't grounded, instead of to `summarize` |

## Dataset

Regulatory knowledge base spanning four domains, all real text from the
eCFR API (215 sections total):

- **31 CFR Chapter X** (FinCEN — Bank Secrecy Act regulations)
  - **Part 1010** — General Provisions (definitions, recordkeeping, SAR
    confidentiality, info sharing)
  - **Part 1020** — Rules for Banks (SAR/CTR filing obligations,
    recordkeeping for funds transfers, AML program requirements)
- **21 CFR Part 11** — FDA Electronic Records; Electronic Signatures
  (validation, audit trails, signature/record linking)
- **45 CFR Parts 160 & 164** — HIPAA Privacy, Security, and Breach
  Notification Rules (administrative/physical/technical safeguards,
  permitted uses and disclosures of PHI)
- **17 CFR** (Sarbanes-Oxley implementing SEC rules) — a hand-picked set
  of sections rather than whole parts, since the parts they live in
  (Reg S-K, Exchange Act rules) are mostly unrelated to SOX:
  `229.308` (management's internal-control report), `240.13a-14`/
  `240.15d-14` (CEO/CFO certification), `240.13a-15` (disclosure
  controls and procedures), `240.10A-3` (audit committee independence)

Pulled live from the free [eCFR API](https://www.ecfr.gov/developers/documentation/api/v1)
via [`src/parse_ecfr.py`](src/parse_ecfr.py) — real, current, authoritative
regulatory text, not synthetic data. Raw files are named
`title<N>_<label>.xml` so the parser can recover which CFR title each
section belongs to (see `add-ecfr-regulation` skill for the fetch
pattern).

## Stack

- **LangGraph** (`langgraph.graph.StateGraph`) for multi-agent orchestration
  — a real graph with nodes and a conditional edge, not hand-rolled control
  flow.
- **The final-answer LLM is model-agnostic — Amazon Bedrock or Groq,
  your choice.** Both implement the same `Summarizer` interface and
  share one prompt ([`src/llm_prompts.py`](src/llm_prompts.py)), selected
  via the `LLM_PROVIDER` env var (`bedrock`, the default, or `groq`) —
  see [`src/agents.py`](src/agents.py)'s `_default_llm()`. An explicit
  `llm=` override to `ComplianceAgentGraph` always wins over the env var.
  - **`bedrock` (default): `langchain_aws` `ChatBedrockConverse` with
    `amazon.nova-micro-v1:0`** — Amazon's cheapest on-demand text model —
    using your existing AWS credentials (`~/.aws/credentials`, region
    `us-east-1` by default, override with `AWS_REGION`). No extra setup
    needed if your account already has Bedrock model access, which it
    does in this environment.
  - **`groq`: the `groq` SDK with `openai/gpt-oss-20b`** — an open-weight
    reasoning model, no AWS cost for this step. Needs `GROQ_API_KEY` (see
    `.env.example`); the `groq` package is an opt-in extra, not in
    `requirements.txt`. `gpt-oss-20b` spends a variable share of its
    token budget on internal reasoning before the visible answer, so
    `GroqSummarizer` uses a generous `max_tokens` and treats empty output
    as a provider failure (triggering the same template fallback a
    Bedrock outage would) rather than returning a blank answer.
  - **Retrieval always uses Bedrock regardless of this choice** — Groq
    has no embeddings API, so picking `groq` only changes who answers,
    not who retrieves; a `groq` query still makes one real Bedrock call
    (embed) plus one real Groq call (summarize).
- **Amazon Bedrock, `amazon.titan-embed-text-v2:0`** (512 dimensions,
  L2-normalized) — for retrieval. Chunk embeddings are precomputed once
  by `src/build_embeddings.py` and cached to `data/embeddings.npz`; each
  live query makes exactly one more embedding call (always Bedrock,
  regardless of `LLM_PROVIDER`) plus one summarization call (Bedrock or
  Groq, depending on `LLM_PROVIDER`) — so **a live query costs two real
  API calls, not one**, at least one of which is always Bedrock. See
  "Running it" below — the cached index must be built before the app can
  answer anything live.
- **scikit-learn** TF-IDF + `LogisticRegression` for intent classification
  only (retrieval no longer uses TF-IDF — see above) — no GPU needed for
  either model.

To use a bigger Amazon model (e.g. for harder questions), change
`DEFAULT_MODEL_ID` in `src/bedrock_llm.py` to `amazon.nova-lite-v1:0` or
`amazon.nova-pro-v1:0` — same code, higher cost/quality. To switch the
final answer to Groq instead, set `LLM_PROVIDER=groq` (needs
`GROQ_API_KEY`) — no code change needed.

## Running it

```bash
# 0. install dependencies (langgraph, langchain-aws, boto3, scikit-learn, numpy, streamlit)
python3 -m pip install -r requirements.txt

# 1. (re)build the knowledge base from eCFR (already done — data/kb.json exists)
python3 src/parse_ecfr.py

# 2. build the dense embedding index — REQUIRED before anything below will
#    find real evidence (data/embeddings.npz doesn't exist until you run
#    this). One-time, real + billed (~$0.004 total, ~1,420 Bedrock calls),
#    resumable if interrupted — see bedrock-rag-retrieval skill.
python3 src/build_embeddings.py

# 3a. CLI: run the demo with built-in sample queries (calls Bedrock live)
python3 demo.py

# 3b. CLI: or ask your own question
python3 demo.py "what safeguards does HIPAA require for protected health information"

# 3c. Web UI: interactive chat interface (see "Web UI" below)
streamlit run app_ui.py
```

## Web UI

`streamlit run app_ui.py` opens a chat interface in your browser at
`http://localhost:8501`: type a free-text compliance question or click one
of five example questions, and see the detected intent, a grounded/
escalated badge, the answer, its citations, and an expandable reasoning
trace for every turn. A sidebar toggle switches the **summarization**
step only, between live Bedrock (a real, billed Nova Micro call) and a
free template fallback (the same one the pipeline uses on a real Bedrock
outage) — **retrieval always makes one real, billed Titan embedding call
regardless of this toggle**, since that's how dense-embedding search
works; there is no free retrieval mode. The UI is a thin presentation
layer over the same `ComplianceAgentGraph` the CLI uses; see `app_ui.py`
and the `edit-compliance-ui` skill before changing it.

## Testing

```bash
python3 -m pytest                 # fully mocked — no AWS calls, no cost
RUN_LIVE_BEDROCK_TESTS=1 python3 -m pytest tests/test_live_bedrock.py tests/test_live_rag.py -v   # real, billed Bedrock calls
```

Covers: intent classification (clear-cut, ambiguous, and edge-case inputs —
empty/whitespace/punctuation-only, very long, non-English, off-topic —
across all eight intents/domains), retrieval **machinery** (sorting,
per-section de-duplication, top_k handling, the empty-query short-circuit
that must never spend a Bedrock call) against a synthetic corpus with a
fake embedder, the verifier/routing logic that decides `summarize` vs.
`escalate` (unit-tested directly against hand-built state, independent of
whatever the retriever happens to score), and both Bedrock clients — LLM
and embeddings (mocked success, failure → fallback, message/vector shape).

Real domain relevance (does "CTR threshold" actually find 31 CFR 1010.311
in the real 215-section KB) and the one real LLM call are **not** free —
each needs a real Bedrock call, so both live in `tests/test_live_rag.py`
and `tests/test_live_bedrock.py` respectively, gated behind
`RUN_LIVE_BEDROCK_TESTS=1` and skipped by default. See
`write-compliance-tests`/`bedrock-rag-retrieval` for why relevance
assertions moved out of the always-on suite when retrieval switched from
TF-IDF to dense embeddings.

## Project layout

```
data/raw/              raw eCFR XML, one file per CFR title/part/section-bundle
data/kb.json            parsed, chunkable regulatory sections (215, 4 domains)
data/embeddings.npz     precomputed Titan embedding vectors (built, not checked in blindly — see below)
data/embeddings_meta.json  chunk metadata matching embeddings.npz, same order
src/parse_ecfr.py       eCFR XML (any title) -> flat section records
src/build_embeddings.py one-time batch job: kb.json chunks -> embeddings.npz (real, billed, resumable)
src/bedrock_embeddings.py  langchain_aws BedrockEmbeddings client (Titan Embed Text v2)
src/retriever.py        chunking + dense-embedding cosine-similarity retrieval
src/classifier.py       intent classifier for compliance queries
src/llm_prompts.py      shared prompt text for the Summarizer step (both providers)
src/bedrock_llm.py      langchain_aws ChatBedrockConverse client (Nova Micro) — Summarizer provider
src/groq_llm.py         groq SDK client (openai/gpt-oss-20b) — alternate Summarizer provider
src/agents.py           LangGraph StateGraph: Orchestrator/Planner/Verifier/Summarizer;
                        LLM_PROVIDER env var picks bedrock_llm vs groq_llm
demo.py                 CLI entry point
app_ui.py               Streamlit web UI (chat interface over the same agent)
tests/                  pytest suite (mocked) + two opt-in live-Bedrock test files
eval/                   one-off 1,416-case eval suite + Groq LLM-judge — see run-evaluation-suite skill
requirements.txt        langgraph, langchain-aws, langchain-core, boto3, scikit-learn, numpy, pytest, streamlit
                        (groq is a separate opt-in extra, only if LLM_PROVIDER=groq or using eval/'s judge)
.env.example            template for GROQ_API_KEY / GITHUB_TOKEN — copy to .env (gitignored), never commit real values
CLAUDE.md               project guide for Claude Code
```

## Honest limitations (vs. the real project)

- **(Open — tracked as [GitHub issue #1](https://github.com/Aniteja-Reddy-Mutyala/Financial-agents/issues/1), not yet fixed)**
  Citation-domain-correctness (99.9% in `eval/REPORT.md`'s 1,415-case
  structural eval) does not mean the generated answer is actually
  correct. A separate semantic LLM-judge pass (`eval/JUDGE_REPORT.md`,
  120-case sample, open-weight model via Groq, zero extra Bedrock cost)
  found HIPAA and FDA — the two domains that looked *strongest*
  structurally — were the *weakest* under real semantic review: only
  30%/46% of their answers were substantively correct, despite near-
  perfect citation accuracy. Two root causes found: chunking truncation
  (a decisive sentence silently cut out of the single best-scoring
  120-word chunk), and `planner_node`'s fixed, one-size-fits-all intent-
  description padding skewing FDA retrieval toward "signature"
  vocabulary regardless of the actual sub-topic asked (confirmed: a
  direct "audit trail" question never surfaces `21 CFR 11.10`, the
  section that actually covers it). A fix (full-section-text evidence +
  a stricter anti-hallucination prompt) was built, confirmed to close
  the two hand-diagnosed cases, but measured against a 32-case before/
  after re-judge to not generalize (40.6% → 37.5%, noise-level, trading
  one failure mode for another) — reverted rather than kept for an
  unproven benefit. See the issue and `eval/JUDGE_REPORT.md` for the
  full writeup, and the `run-evaluation-suite` skill before re-attempting
  a fix here.
- **(Fixed, previously a real bug found via the UI)** "Tell me the format
  of SAR" used to misclassify as `sox_internal_controls` at ~15%
  confidence — not a tie-breaker between plausible options, but
  effectively random: the classifier's `TfidfVectorizer` never stripped
  stop words, so "of"/"the" were treated as real signal and diluted the
  one real word ("sar") to noise. Because `planner_node` always appends
  the (here, wrong) intent's description to the retrieval query, that
  alone pulled an irrelevant SOX section into the "grounded" evidence
  alongside the correct SAR section, confusing the generated answer.
  Fixed at two levels: `stop_words="english"` added to the classifier's
  vectorizer, and `retriever_node` now also searches the raw user query
  and merges it with the intent-augmented query by best score per
  section, so a genuinely relevant section can't be crowded out by one
  that only matched because of a wrong intent description. See
  `tests/test_classifier.py::test_clear_intent_classification` (the new
  "Tell me the format of SAR" case) and
  `tests/test_live_rag.py::test_wrong_intent_does_not_pollute_evidence_with_other_domain`.
  At the time, this was a mitigation, not a full fix — a *truly*
  off-topic query still got the guaranteed-overlap treatment from the
  intent description. See the next two bullets for how that was actually
  closed, and what still isn't.
- **`MIN_SCORE` had to be recalibrated when retrieval moved to dense
  embeddings, and the old value was silently wrong for a while.**
  `agents.MIN_SCORE` (the verifier's grounding threshold) was left at
  `0.08` — correct for TF-IDF's scale, where irrelevant text usually
  scored near zero. Measured against the real dense-embedding index:
  genuinely relevant queries scored **0.37-0.70** top-hit; genuinely
  off-topic queries ("best pizza topping in Chicago", "capital of
  France," etc.) scored **0.13-0.16** — dense embeddings have a much
  higher noise floor, since unrelated English text still shares general
  vocabulary/structure. At `0.08`, *every* query cleared the bar — the
  escalate-to-human branch was effectively dead for any input, not just
  edge cases. Recalibrated to `0.25` (clear margin on both sides of the
  measured gap); see
  `tests/test_live_rag.py::test_min_score_separates_relevant_from_off_topic_queries`
  for the regression guard.
- **(Fixed) The "known rough edge" — intent-description padding forcing
  `grounded=True` for a genuinely off-topic query — is now fixed, via a
  different mechanism than first attempted.** Two cheaper approaches were
  tried and empirically ruled out first: a raw-query-only score floor
  (rejected — a real, vague-but-legitimate query, "what is the
  threshold," scored 0.176 raw-alone, inside the same 0.13-0.16 range as
  genuinely off-topic queries — no clean threshold separates them), and a
  classifier-confidence floor (rejected — this classifier legitimately
  produces 15-30% confidence on many *correctly* classified queries too,
  so a confidence floor would reject good answers along with bad ones).
  The actual fix: `IntentClassifier.predict()` now returns `has_signal` —
  whether the query shared *any* vocabulary with the tiny training set at
  all (`vectorizer.transform(query).nnz > 0`), a different and more
  reliable signal than confidence. A genuinely off-topic query shares
  zero training vocabulary, so its top "intent" is pure statistical
  noise regardless of its confidence number. `planner_node` now only pads
  the retrieval query with the intent description when `has_signal` is
  true; otherwise it searches the user's raw words alone — which, for a
  real off-topic query, correctly scores below `MIN_SCORE` on its own
  (measured: "what's the best pizza topping in Chicago" scores 0.146
  unpadded, vs. 0.335 when it used to get padded with a wrong intent's
  description). See `tests/test_classifier.py` (`has_signal` tests),
  `tests/test_routing.py` (`planner_node`/`retriever_node` unit tests),
  and
  `tests/test_live_rag.py::test_fully_off_topic_query_now_correctly_escalates`
  (the regression test that used to be `xfail(strict=True)`, exactly the
  same fix lifecycle as the earlier CTR-phrasing gap).
- **What the fix above does NOT close — a different, deeper limitation
  found while testing it.** A query that's *stylistically* plausible as a
  compliance question but about a genuinely uncovered topic (e.g. "what
  are the legal requirements for starting a business," "what is the
  penalty for violating a contract") can still score well above
  `MIN_SCORE` on dense-embedding similarity alone, with no padding
  involved at all — dense embeddings correctly recognize "this sounds
  like formal/legal register" even when the specific subject matter
  isn't in this project's 4 domains. This is a fundamentally different
  problem from the padding mechanism just fixed (retrieval itself giving
  a plausible score to the wrong subject matter, not an artifact of
  intent-description padding), and would need real semantic judgment —
  e.g. an LLM-judge second call checking "do these excerpts actually
  answer this question" — to close, which was deliberately not added
  here without discussing the cost first: it would make every grounded
  query 3 Bedrock calls instead of 2, not just a tuning change. Left
  open and documented, not silently patched around; see
  `edit-compliance-graph`'s "known rough edge" section.
- **`app_ui.py` has no automated test coverage** — the pytest suite's
  "every new feature gets tests" rule doesn't extend to Streamlit
  rendering itself; verification is manual (see `edit-compliance-ui`
  skill). Everything it calls into (`ComplianceAgentGraph` and below) is
  fully covered — the gap is specifically the UI layer's own code
  (caching, toggle wiring, citation de-duplication).
- **(Superseded — retrieval is no longer TF-IDF)** This project originally
  used TF-IDF instead of dense embeddings (no sentence-transformers or
  network vector store available in the original environment), with the
  documented cost that lexical retrieval couldn't bridge true synonymy it
  hadn't been explicitly told about, via a hand-maintained
  `_DOMAIN_SYNONYMS` list. `src/retriever.py` now uses real dense
  embeddings (Amazon Bedrock Titan Embed Text v2 — see Stack above),
  which closes that specific gap: no synonym list, no manual stemming —
  semantic relationships like "SAR" ~ "suspicious activity report" and
  "cash" ~ "currency" are handled natively. The real cost of *this*
  choice: every query now makes a second billed Bedrock call (one to
  embed the query, one to summarize), and the one-time indexing step
  (`src/build_embeddings.py`) must be re-run after any `data/kb.json`
  change or new content is silently unsearchable — see
  `bedrock-rag-retrieval`.
- **A real stress test of the old TF-IDF retriever at 80x scale** (17,338
  sections, mixing in 3 unrelated large CFR titles as distractor content)
  found precision dropping on generic-vocabulary queries (e.g. "CTR
  threshold," "wire transfer recordkeeping") once real topically-similar
  neighbor regulations existed to compete for the same words — this was
  the concrete, measured version of the synonymy limitation above, not a
  new one. **This has not been re-run against the new dense-embedding
  retriever** — dense embeddings should degrade more gracefully under the
  same stress (that's the whole point of moving past lexical overlap),
  but that is an expectation, not yet a verified result for this project.
- Bare natural-language CTR phrasing ("currency transaction report
  threshold cash deposit") used to miss the correct section (31 CFR
  1010.311) entirely under the old TF-IDF retriever — found, `xfail`'d
  with the exact cause documented, then fixed (stemming, domain-synonym
  expansion, title weighting, per-section de-duplication), all now moot
  under dense embeddings, which don't need any of those lexical
  workarounds. Real-relevance regression coverage for this and the other
  domain-specific queries now lives in `tests/test_live_rag.py` (gated,
  since it needs a real embedding call) rather than the always-on suite.
- Every Bedrock call costs a small amount against your AWS account. A
  live query now makes **two** real calls, not one: one Titan embedding
  call (~$0.02 / 1M input tokens) to embed the query, and one Nova Micro
  call (~$0.035 / 1M input tokens) to summarize — still a few hundredths
  of a cent total per query, but real, billed API calls, not mocks. The
  one-time index build (`src/build_embeddings.py`) is a separate ~$0.004
  cost for the whole 1,420-chunk KB.
- Observed once in testing: the model cited a section number (1020.420)
  that wasn't in the retrieved excerpts, likely drawn from its own
  pretraining rather than strictly the provided text. The system prompt
  instructs it to answer only from the excerpts, but nothing here enforces
  that mechanically — a production version would need citation
  verification (e.g. regex-check cited section IDs against the retrieved
  set) before trusting the output.
- Four regulatory domains (BSA/AML, FDA e-records, HIPAA, SOX) instead of
  the multilingual, multi-domain financial corpus (Refinitiv/Bloomberg/
  MarketPsych, ESG, 50+ languages) from the original — still a single
  jurisdiction (U.S. federal) and a single language.
- **GDPR and SOC2 are not included**, deliberately, not as an oversight:
  GDPR is EU law and isn't on eCFR at all (would need EUR-Lex or a similar
  non-U.S. source and its own parser); SOC2 is a proprietary AICPA Trust
  Services Criteria framework with no free, authoritative public text to
  pull — including it here would mean fabricating regulatory-sounding
  content, which this project avoids on principle.
- The SOX section of the knowledge base is five hand-picked sections of
  17 CFR (the SEC rules that implement SOX's certification/internal-control
  provisions), not a whole CFR part — SOX itself is mostly codified in the
  U.S. Code, not eCFR, and the parts these sections live in (Reg S-K,
  general Exchange Act rules) are mostly unrelated to SOX. This means SOX
  coverage here is deliberately narrow and would need more sections
  (e.g. whistleblower protection, auditor independence) for broader
  SOX questions.
