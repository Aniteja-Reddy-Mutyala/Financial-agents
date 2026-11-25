# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

## What this project is

A small, locally-runnable recreation of a multi-domain compliance agent
(BSA/AML, FDA e-records, HIPAA, SOX), built to mirror (at demo scale) the
Gen AI Engineer work described for the Northern Trust project: LangGraph
multi-agent orchestration, a dense-embedding RAG pipeline over regulatory
text (always Amazon Bedrock Titan embeddings, not TF-IDF — Groq has no
embeddings API), intent classification, and an LLM call for the final
answer — **model-agnostic between Amazon Bedrock and Groq** (an
open-weight model), selected via `LLM_PROVIDER`. See
[README.md](README.md) for the full architecture and how each piece maps
back to the original project.

## Running things

```bash
python3 -m pip install -r requirements.txt   # langgraph, langchain-aws, boto3, scikit-learn, numpy, pytest, streamlit
python3 src/build_embeddings.py              # one-time: embed data/kb.json -> data/embeddings.npz (real, billed, ~$0.005, resumable)
python3 demo.py                              # CLI: run sample queries end-to-end (calls Bedrock live)
python3 demo.py "your question here"         # CLI: ask one question
LLM_PROVIDER=groq python3 demo.py "..."      # same, but the final answer comes from Groq instead of Bedrock
streamlit run app_ui.py                      # web UI: chat interface; toggle picks Bedrock or Groq for the final answer (neither is free — retrieval is always a real Bedrock call either way)

python3 -m pytest                            # full test suite — no AWS calls, no cost
RUN_LIVE_BEDROCK_TESTS=1 python3 -m pytest tests/test_live_bedrock.py tests/test_live_rag.py -v   # real, billed Bedrock calls

# One-off quality eval (not part of pytest) — see run-evaluation-suite skill
python3 eval/run_eval.py --estimate-only     # cost/time estimate before committing
python3 eval/run_eval.py --sample 50         # pilot a subset first
python3 eval/generate_report.py              # precision/recall/F1 report from eval/results.json
```

`GROQ_API_KEY` (needed whenever `LLM_PROVIDER=groq`, and by
`eval/run_judge.py`) lives in a project-local `.env` file (see
`.env.example`) — nothing auto-loads it, source it explicitly:
`set -a && source .env && set +a && <command>`. See
`run-evaluation-suite` before touching `.env` or anything that reads a
secret from it; getting the sourcing wrong has twice printed a real
token into a terminal/conversation in this project's history.

`python3 src/build_embeddings.py` must be run once before the app can
answer anything live (retrieval reads `data/embeddings.npz`, which
doesn't exist until you build it) — see `bedrock-rag-retrieval` before
assuming it's already there or re-running it casually.

`tests/` covers the intent classifier, retriever machinery, routing/
verifier logic, and both LLM clients (Bedrock and Groq) + the embeddings
client — all mocked except `test_live_bedrock.py` and `test_live_rag.py`,
both skipped by default (both make real, billed Bedrock calls — LLM and
embeddings respectively). Keep new tests free of real network calls
unless they're explicitly gated the same way. Load
**`write-compliance-tests`** before adding tests — it has the coverage
pattern, mocking conventions, and where each kind of test belongs.

## Before changing specific areas, load the matching skill

This repo has project skills under `.claude/skills/` with the
detail that used to live in this file. Load the matching one before
editing rather than guessing from this summary:

- **`edit-compliance-graph`** — before touching `src/agents.py` or the
  LangGraph pipeline (nodes, edges, `ComplianceState`, routing).
- **`add-ecfr-regulation`** — before adding/expanding the regulatory
  knowledge base (`data/kb.json`, `data/raw/`, `src/parse_ecfr.py`).
- **`bedrock-llm-integration`** — before changing the final-answer LLM
  call (`src/bedrock_llm.py`, `src/groq_llm.py`, `src/llm_prompts.py`,
  provider/model choice, prompts, cost/fallback behavior — covers both
  Bedrock and Groq, which are interchangeable here).
- **`bedrock-rag-retrieval`** — before changing `src/retriever.py`,
  `src/bedrock_embeddings.py`, or `src/build_embeddings.py` (the
  dense-embedding retrieval pipeline, its one-time indexing job, cost/
  concurrency, and `MIN_SCORE` calibration).
- **`write-compliance-tests`** — before adding or updating any test,
  including the tests for a new feature (see Conventions below).
- **`edit-compliance-ui`** — before touching `app_ui.py`, the Streamlit
  web UI (layout, the Bedrock/Groq answer-generation toggle, chat history).
- **`run-evaluation-suite`** — before running, changing, or interpreting
  anything in `eval/` (the templated 1,416-case eval, the Groq LLM-judge
  pass, and the `.env` secrets convention they introduced).

## Conventions

- Keep modules dependency-light and runnable standalone (`if __name__ ==
  "__main__":` smoke tests exist in `retriever.py`, `classifier.py`,
  `bedrock_llm.py`, `groq_llm.py`, `agents.py` — keep that pattern for
  new modules). The `groq` package is intentionally **not** in
  `requirements.txt` — it's an opt-in extra, only needed if
  `LLM_PROVIDER=groq` is actually selected.
- No framework beyond LangGraph/langchain_aws for orchestration, and
  Streamlit for the UI — don't introduce CrewAI, additional orchestration
  layers, or a different web/UI framework unless asked.
- This is a portfolio/demo recreation, not production code: prefer small,
  readable modules over defensive engineering for cases that can't happen
  here (single-user CLI, one in-memory knowledge base, no concurrent
  requests).
- **Every new feature gets test coverage in the same change** — a new
  node, intent/domain, regulatory part, or CLI flag isn't done until
  `tests/` covers it (clear-cut + at least one edge case). Load
  `write-compliance-tests` first; don't hand this back to the user as a
  follow-up. The one standing exception is `app_ui.py`: Streamlit UI
  rendering has no automated test here (manual verification only —
  see `edit-compliance-ui`), which is itself a documented limitation,
  not license to skip testing anything else it calls into.
- Known limitations of the demo (unverified LLM citations, four U.S.-only
  regulatory domains with no GDPR/SOC2, retrieval degrading on generic
  vocabulary at larger synthetic scale — see why in README.md) are
  documented in README.md's "Honest limitations" section — don't silently
  "fix" or paper over these without updating that section.
- A real, measured one (**fixed and closed** — see
  [GitHub issue #1](https://github.com/Aniteja-Reddy-Mutyala/Financial-agents/issues/1)
  for the full history): citation-domain-correctness (99.9%
  structurally) does not mean answer correctness — a semantic LLM-judge
  pass found HIPAA/FDA answers substantively correct only 30%/46% of the
  time despite citing the right regulation. A first fix attempt (full-
  section-text evidence + a stricter anti-hallucination prompt) was
  tried and reverted because it didn't generalize. Four narrower,
  separately-targeted fixes landed instead — `NO_PAD_INTENTS` in
  `agents.py`, sentence-boundary chunking and bounded neighbor-chunk
  expansion in `retriever.py`, and a clause-awareness addition to
  `llm_prompts.py`'s `SYSTEM_PROMPT` — and were confirmed, via a
  controlled re-run (full FDA+HIPAA case set, deterministic generation,
  two independent judge providers), to raise FDA to 76.7% and HIPAA to
  78.9%. See the issue (closed, with the full writeup) and README.md's
  "Honest limitations" for the complete history, including the known
  caveats on that measurement (judge independence, residual ~21-23%
  error rate) — read both before touching this area again, since the
  residual error rate is a known *ongoing* limitation, not evidence this
  fix didn't work.
