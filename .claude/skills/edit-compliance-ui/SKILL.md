---
name: edit-compliance-ui
description: Use before changing app_ui.py, the Streamlit web UI for the compliance agent — layout, the Bedrock/Groq answer-generation toggle, chat history, or anything rendered in the browser. Covers the caching pattern, how the toggle picks the LLM provider, and how to manually test it since it has no automated test coverage.
---

## A second agent has its own page — don't fold it in here

`pages/1_Research_Agent.py` is the Research agent's Streamlit page,
auto-discovered by Streamlit's multipage convention (any `.py` under
`pages/` next to `app_ui.py`) — it required zero changes to this file,
and should stay that way. If you're tempted to add Research-agent UI
logic into `app_ui.py` "since it's already open," don't — see
`edit-research-agent` instead, and `CLAUDE.md`'s "Target end product"
for why this repo keeps each agent's UI as its own page rather than
growing one shared page.

## It's presentation only — no business logic here

`app_ui.py` is a thin Streamlit front-end over `ComplianceAgentGraph` from
`src/agents.py`. If you find yourself writing intent logic, retrieval
logic, or prompt text inside `app_ui.py`, that belongs in `src/` instead —
load `edit-compliance-graph` or `bedrock-llm-integration` and put it
there, then just call it from the UI.

## Caching — don't rebuild the agent per interaction

`_load_agent(use_bedrock)` is wrapped in `@st.cache_resource`, keyed on
the `use_bedrock` toggle. This matters because `ComplianceAgentGraph()`
re-trains the intent classifier and reloads the embedding index from
disk from scratch — without caching, every single chat message would
silently redo that work. If you add a new toggle/setting that changes how
the agent is built, either fold it into `_load_agent`'s cache key or the
app will rebuild needlessly; don't just build a new agent inline in the
response-handling code.

## The toggle picks a Summarizer provider — it is NOT an offline/free mode

The "Use live Bedrock" sidebar toggle controls which `Summarizer` answers
the question: **on** = `BedrockSummarizer` (Amazon Nova Micro, the
default); **off** = `GroqSummarizer` (Groq's open-weight model) — see
`_load_agent` in `app_ui.py`. This is the same interchangeable-provider
design as `LLM_PROVIDER` elsewhere in the project (`bedrock-llm-integration`
skill) — the UI just exposes it as a toggle instead of an env var.
`GroqSummarizer` needs `GROQ_API_KEY` in the environment (nothing
auto-loads `.env` — source it before `streamlit run`, see
`run-evaluation-suite`'s `.env` conventions); if it's missing, or the
Groq call fails for any other reason, `GroqSummarizer.summarize()`
raises `GroqUnavailable`, which `summarizer_node` already catches and
falls back to `_template_draft` — the same tested fallback path a real
provider outage uses, so a missing key degrades gracefully rather than
crashing. **There used to be a third `_OfflineSummarizer` stub here
(always raised `BedrockUnavailable`, i.e. "skip the LLM call entirely")
— it was removed** once the toggle's off position meant "use Groq," not
"use no LLM"; don't re-add a template-only mode without the user asking
for it specifically, since it would make the toggle a 3-state control
with no UI to express the third state.

**This toggle does NOT make the whole app free, in either position.**
Retrieval (`Retriever`, backed by Bedrock Titan embeddings — see
`bedrock-rag-retrieval`) has no offline/mocked path in the UI: every
question, toggle on or off, makes one real, billed Bedrock embedding
call to search the KB, and now the toggle's off position makes a real
Groq call instead of a free one. Don't describe either position as
"offline" or "free" anywhere in the UI text or docs without that
caveat — an earlier version of this UI claimed the off position made
"no AWS calls, no cost," which was wrong from the moment retrieval moved
off TF-IDF, and is doubly wrong now that off also makes a real Groq
call.

## Citation de-duplication

`summarizer_node`/`escalate_node` (in `src/agents.py`) append a trailing
`"\n\nCitations: ..."` line to `answer` — useful for `demo.py`'s plain-text
CLI output. The UI renders citations separately as a styled caption, so
`_answer_body()` strips that trailing line before `st.write()`-ing the
answer. If you change the CLI's citation-line format in `agents.py`,
update `_answer_body()`'s split string to match, or the duplicate will
come back.

## Known limitation this UI will visibly surface

A query that's *stylistically* plausible as a compliance question but
about a genuinely uncovered topic (e.g. "what are the legal requirements
for starting a business") can still show up green/"grounded" in the UI,
with citations from an unrelated domain — dense embeddings recognize
"this sounds like formal/legal register" even when the actual subject
matter isn't in this project's 4 domains at all (documented in
`edit-compliance-graph`'s "known rough edge" and README's "Honest
limitations" — not the same as, and not fixed by, the padding/chunking
fixes that closed GitHub issue #1). That's existing backend behavior
being shown faithfully, not a UI bug — don't try to patch it from the UI
side (e.g. hiding the badge, second-guessing `verification["grounded"]`);
if it's ever fixed, it should be fixed in `verifier_node`/`retriever.py`,
and would need real semantic judgment (e.g. an LLM-judge second call),
not a threshold tweak — flag the added cost to the user first (see
`bedrock-llm-integration`'s cost-awareness rules) before attempting it.
(A different off-topic case — "what's the best pizza topping in
Chicago" — used to show the same symptom but is fixed: the classifier's
`has_signal` check now catches queries sharing zero training vocabulary
before they ever get padded with a wrong intent's description.)

## Testing a change — manual only, no pytest coverage

This file has no automated tests (Streamlit UI rendering isn't practically
unit-tested in this project — that's a deliberate, documented gap, not an
oversight; see README's "Honest limitations"). To verify a change:

1. `python3 -m pip install -r requirements.txt` (installs `streamlit`).
2. `streamlit run app_ui.py` (or via the Claude Code browser pane's
   `preview_start` with a `.claude/launch.json` entry — see the
   `compliance-ui` config already set up for this project).
3. Every question costs at least one real, billed Bedrock embedding call
   regardless of the toggle (see above), plus one more real call either
   way for the summarize step (Bedrock if the toggle is on, Groq if
   off) — there is no zero-cost path through this UI. Don't loop through
   all 5 example questions in both toggle positions "to be thorough"
   (see cost-awareness rules in `bedrock-llm-integration` and
   `bedrock-rag-retrieval`); one or two real questions per toggle
   position is enough to confirm a change works.
4. Click at least one example button, submit at least one free-typed
   question via the chat input, expand "Show reasoning trace" once, and
   reload the page to confirm it still renders from a clean session.
5. If you changed anything citation- or answer-formatting related,
   specifically check there's no duplicate citations line (see above).
