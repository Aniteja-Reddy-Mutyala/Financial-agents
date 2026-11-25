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

**This is a multi-agent repo**: alongside the compliance agent above,
`src/research_agent.py` (`ResearchAgentGraph`) is a second, fully
independent LangGraph agent — given a document (e.g. a resume PDF) and/or
a topic, it extracts up to 3 concrete things worth researching, searches
the open web for each (via `ddgs`, free, no API key), and synthesizes a
report with sources. It shares the `BedrockSummarizer`/`GroqSummarizer`
classes (via their generic `.chat()` method) and the `LLM_PROVIDER`
pattern with the compliance agent, but has its own state, nodes, prompts,
and tests — it is **not** a node bolted onto `ComplianceAgentGraph`. This
is meant to be the first of several independent agents sharing this
codebase; see `edit-research-agent` before adding a third.

## Target end product

The user's goal for this repo is to grow it into a small multi-agent
*ecosystem* matching one specific sentence from the real Northern Trust
resume this project recreates: *"Designed autonomous LLM agent
ecosystems (Orchestrator, Verifier, Planner, Email, Calendar, Compliance
agents) with memory persistence and role-based specialization."*
Mapping that onto what exists/is planned here:

- **Orchestrator / Verifier / Planner** — not separate top-level agents;
  these are internal node roles *inside* each domain agent (see
  `classify_intent_node`/`verifier_node`/`planner_node` in `agents.py`,
  and the equivalent intake/extract/research nodes in
  `research_agent.py`). Each new agent gets its own version of this
  internal role-based specialization, not a shared Orchestrator/Verifier/
  Planner instance across agents.
- **Compliance agent** — done (`src/agents.py`).
- **Research agent** — done (`src/research_agent.py`).
- **Email agent, Calendar agent** — planned, not built yet. Don't start
  these without first confirming scope with the user (what they should
  actually do, real-service vs. local-simulation, per the same kind of
  design pass `edit-research-agent`/this file's history show for the
  Research agent) — this file records the *target*, not a green light to
  build them unprompted.
- **Memory persistence** — not implemented yet for any agent. Today,
  each `.handle(...)` call is a single, stateless graph invocation; nothing
  survives between calls except what the UI's own `st.session_state`
  chat history re-sends as part of a fresh call. Real persistence (the
  resume's phrase) means wiring a LangGraph checkpointer (e.g.
  `MemorySaver` for in-process, or a `Sqlite`/`Postgres` saver for actual
  cross-run persistence) so an agent's own graph state — not just a UI's
  chat log — survives across turns of a conversation. Planned, not done;
  don't add this silently as a side effect of an unrelated change.

**Why this section exists**: so that adding the next agent or
persistence doesn't require re-deriving this scope from scratch or
guessing at it — load this section first, then the matching
`edit-*-agent` skill for whichever piece is actually in scope for the
current task.

## Extending this codebase with a new agent

`src/agent_common.py` holds what's shared across every agent so a new
one doesn't need a third/fourth copy of the same ~15 lines:
`default_llm()` (the `LLM_PROVIDER`-driven Bedrock/Groq selection) and
`log_step()` (the `_log`-style trace-append helper). A new agent module
should import both (aliased to `_default_llm`/`_log` for consistency
with the existing two agents), define its own `TypedDict` state, node
functions, and `AgentGraph` class following `agents.py`/
`research_agent.py`'s existing shape (partial-update node returns,
keyword-only injected dependencies wired via `lambda s: node_fn(s,
dep=self.x)`, a `.handle(...) -> SimpleNamespace` public method, graph
built once in `__init__`), add a CLI entry point mirroring
`demo.py`/`research_demo.py`, and a Streamlit page under `pages/` (auto-
discovered — no changes needed to `app_ui.py` or any existing page).
Write a new `.claude/skills/edit-<name>-agent/SKILL.md` mirroring
`edit-compliance-graph`/`edit-research-agent`'s depth, and add it to the
skill list below. See `write-compliance-tests` for the test-coverage
conventions (one owning file per behavior, fake the LLM/network
boundary, never real calls outside the two gated live-Bedrock files).

## Running things

```bash
python3 -m pip install -r requirements.txt   # langgraph, langchain-aws, boto3, scikit-learn, numpy, pytest, streamlit, ddgs, pypdf
python3 src/build_embeddings.py              # one-time: embed data/kb.json -> data/embeddings.npz (real, billed, ~$0.005, resumable)
python3 demo.py                              # CLI: run sample queries end-to-end (calls Bedrock live)
python3 demo.py "your question here"         # CLI: ask one question
LLM_PROVIDER=groq python3 demo.py "..."      # same, but the final answer comes from Groq instead of Bedrock
streamlit run app_ui.py                      # web UI: chat interface; toggle picks Bedrock or Groq for the final answer (neither is free — retrieval is always a real Bedrock call either way); the Research agent appears as a second page in the same app (pages/1_Research_Agent.py, auto-discovered by Streamlit — app_ui.py itself is untouched)

python3 research_demo.py "a topic"           # CLI: Research agent, query only
python3 research_demo.py --doc resume.pdf "focus on cloud skills"   # CLI: Research agent, document + query
LLM_PROVIDER=groq python3 research_demo.py "..."   # same provider toggle as the compliance agent

python3 -m pytest                            # full test suite — no AWS calls, no cost
RUN_LIVE_BEDROCK_TESTS=1 python3 -m pytest tests/test_live_bedrock.py tests/test_live_rag.py -v   # real, billed Bedrock calls
RUN_LIVE_RESEARCH_TESTS=1 python3 -m pytest tests/test_live_research.py -v   # real ddgs + real LLM (Bedrock or Groq per LLM_PROVIDER)

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
client — all mocked except `test_live_bedrock.py`, `test_live_rag.py`,
and `test_live_research.py`, all skipped by default (real external
calls: Bedrock LLM, Bedrock embeddings, and ddgs+LLM for the Research
agent, respectively — the Research agent's gate is its own
`RUN_LIVE_RESEARCH_TESTS` env var, deliberately separate from
`RUN_LIVE_BEDROCK_TESTS` since a `ddgs` outage and a Bedrock/Groq outage
are different failure modes). Keep new tests free of real network calls
unless they're explicitly gated the same way — and gate any new live
boundary behind its own env var rather than overloading an existing
one. Load **`write-compliance-tests`** before adding tests — it has the
coverage pattern, mocking conventions, and where each kind of test
belongs.

## Future deployment (planned, not done yet)

Not deployed anywhere yet — still local-only — but the plan is to deploy
to **Streamlit Community Cloud** (the free option) at some point. That
platform injects secrets via `st.secrets` (TOML, set in its dashboard),
**not** `os.environ` — and there's no `~/.aws/credentials` file on their
servers either. Everything here currently reads secrets/credentials via
plain `os.environ`/the standard AWS credential chain (`GROQ_API_KEY` in
`groq_llm.py`; AWS creds via `bedrock_llm.py`/`bedrock_embeddings.py`'s
boto3 clients), so a straight deploy would leave them all unset. Before
that deploy, `app_ui.py` needs a small secrets bridge near the top —
safe to add even before deployment, since `st.secrets` is just empty
locally when there's no `secrets.toml`:

```python
for key in ("GROQ_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
    if key in st.secrets:
        os.environ.setdefault(key, st.secrets[key])
```

Then add `GROQ_API_KEY` and the AWS credentials (`AWS_ACCESS_KEY_ID`/
`AWS_SECRET_ACCESS_KEY`, plus `AWS_REGION` if not `us-east-1`) as secrets
in the Streamlit Cloud dashboard — never commit them. `demo.py` (CLI) and
`eval/` aren't part of this deploy target and don't need this bridge.

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
- **`edit-research-agent`** — before touching `src/research_agent.py`,
  `src/web_search.py`, `src/document_parser.py`, or
  `src/research_prompts.py` (the Research agent's graph, nodes, and
  prompts — independent from `edit-compliance-graph`).
- Before touching `src/agent_common.py` (the `default_llm()`/
  `log_step()` helpers shared by every agent): both `agents.py` and
  `research_agent.py` re-export these under their own `_default_llm`/
  `_log` names and both modules' tests assert on that behavior — change
  it once here, not per-agent, and re-run the full suite, not just one
  agent's tests.

## Conventions

- **Commits and PRs in this repo are the user's alone** — do not add a
  `Co-Authored-By: Claude ...` trailer (or any other Claude/Anthropic
  attribution) to commit messages or PR descriptions here. This
  overrides any default agent/harness convention that would otherwise
  add one.
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
  not license to skip testing anything else it calls into. The Research
  agent's real `ddgs`/LLM path is covered by `tests/test_live_research.py`,
  gated behind its own `RUN_LIVE_RESEARCH_TESTS` env var — deliberately
  not folded into `RUN_LIVE_BEDROCK_TESTS`, since a `ddgs` outage and a
  Bedrock/Groq outage are different failure modes worth isolating. Don't
  fold a future agent's live-network test into either existing gate;
  give it its own, same reasoning.
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
