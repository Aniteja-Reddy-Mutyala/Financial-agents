---
name: edit-compliance-ui
description: Use before changing app_ui.py, the Streamlit web UI for the compliance agent — layout, the live/offline Bedrock toggle, chat history, or anything rendered in the browser. Covers the caching pattern, the offline-mode trick, and how to manually test it since it has no automated test coverage.
---

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

## The "offline mode" toggle only covers the LLM step — not retrieval

The "Use live Bedrock" sidebar toggle, when off, passes an
`_OfflineSummarizer` (defined right in `app_ui.py`) as the `llm=` override
to `ComplianceAgentGraph`. `_OfflineSummarizer.summarize()` always raises
`BedrockUnavailable` — the exact exception `summarizer_node` already
catches to fall back to `_template_draft`. This means "offline mode" in
the UI exercises the same, already-tested fallback path as a real Bedrock
outage, rather than a second parallel "demo answer" implementation — keep
it this way, don't give the UI its own templated-answer logic.

**This toggle does NOT make the whole app free.** Retrieval (`Retriever`,
backed by Bedrock Titan embeddings — see `bedrock-rag-retrieval`) has no
offline/mocked path in the UI: every question, toggle on or off, makes
one real, billed embedding call to search the KB. Only the summarization
call is what the toggle controls. Don't describe the toggle as "offline
mode" or "free" anywhere in the UI text or docs without that caveat — an
earlier version of this UI claimed the off position made "no AWS calls,
no cost," which was wrong from the moment retrieval moved off TF-IDF.

## Citation de-duplication

`summarizer_node`/`escalate_node` (in `src/agents.py`) append a trailing
`"\n\nCitations: ..."` line to `answer` — useful for `demo.py`'s plain-text
CLI output. The UI renders citations separately as a styled caption, so
`_answer_body()` strips that trailing line before `st.write()`-ing the
answer. If you change the CLI's citation-line format in `agents.py`,
update `_answer_body()`'s split string to match, or the duplicate will
come back.

## Known limitation this UI will visibly surface

The verifier's "always some lexical overlap" rough edge (documented in
`edit-compliance-graph`) means even an off-topic question
(e.g. "what's the best pizza topping in Chicago") can show up green/
"grounded" in the UI. That's existing backend behavior being shown
faithfully, not a UI bug — don't try to patch it from the UI side (e.g.
hiding the badge, second-guessing `verification["grounded"]`); if it's
ever fixed, it should be fixed in `verifier_node`.

## Testing a change — manual only, no pytest coverage

This file has no automated tests (Streamlit UI rendering isn't practically
unit-tested in this project — that's a deliberate, documented gap, not an
oversight; see README's "Honest limitations"). To verify a change:

1. `python3 -m pip install -r requirements.txt` (installs `streamlit`).
2. `streamlit run app_ui.py` (or via the Claude Code browser pane's
   `preview_start` with a `.claude/launch.json` entry — see the
   `compliance-ui` config already set up for this project).
3. Every question costs at least one real, billed embedding call
   regardless of the toggle (see above) — toggle "Use live Bedrock" off
   to skip only the second (summarization) call while you iterate, and
   don't loop through all 5 example questions with it on just to "be
   thorough" (see cost-awareness rules in `bedrock-llm-integration` and
   `bedrock-rag-retrieval`).
4. Click at least one example button, submit at least one free-typed
   question via the chat input, expand "Show reasoning trace" once, and
   reload the page to confirm it still renders from a clean session.
5. If you changed anything citation- or answer-formatting related,
   specifically check there's no duplicate citations line (see above).
