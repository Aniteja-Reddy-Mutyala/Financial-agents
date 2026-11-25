---
name: edit-research-agent
description: Use before changing src/research_agent.py, src/web_search.py, src/document_parser.py, or src/research_prompts.py — the Research agent's LangGraph pipeline (nodes, edges, ResearchState, routing), independent from edit-compliance-graph. Covers the graph shape, the topic cap, why ddgs/pypdf over paid alternatives, and the .chat() method on BedrockSummarizer/GroqSummarizer.
---

## This is a second, independent agent — not a node on ComplianceAgentGraph

`src/research_agent.py`'s `ResearchAgentGraph` is a fully separate
`langgraph.graph.StateGraph`, state (`ResearchState`), and set of nodes
from `src/agents.py`'s `ComplianceAgentGraph` — deliberately, since this
repo is meant to host several independent agents over time (see
`CLAUDE.md`'s "Target end product"), not one graph growing a second
domain. It shares with the compliance agent: the
`BedrockSummarizer`/`GroqSummarizer` classes (via their generic
`.chat()` method — see below), and `src/agent_common.py`'s
`default_llm()`/`log_step()` helpers (re-exported here as `_default_llm`/
`_log`, same names each module used before this was extracted — see
`agent_common.py`'s docstring for why a second near-identical copy was
the trigger to extract, not the first).

## Graph shape

```
            --[document_path given]--> parse_document --+
intake ----<                                             >--> extract_topics -> web_research -> synthesize -> END
            --[query only]------------------------------+
```

`ResearchState` fields: `query`, `document_path` (`""` if none given —
can be a path/str or a file-like object, e.g. Streamlit's
`UploadedFile`), `document_text`, `topics` (hard-capped to `MAX_TOPICS =
3`), `search_results` (`dict[topic -> list[{title,url,snippet}]]`),
`answer`, `trace`.

- **`intake`** is a no-op routing node (`route_after_intake` reads
  `state.get("document_path")` to decide `parse_document` vs.
  `extract_topics`).
- **`parse_document_node`** calls the injected `document_parser`
  (`document_parser.parse_document` by default). On `DocumentParseError`
  it does **not** propagate — it logs the failure and sets
  `document_text=""`, falling through to topic extraction using `query`
  alone. Same graceful-degrade spirit as `summarizer_node`'s template
  fallback in the compliance agent: a bad document shouldn't crash a
  request that still has a usable query.
- **`extract_topics_node`** is the first LLM call (`.chat()`), only made
  when `document_text` is non-empty — a query-only request skips it
  entirely and uses `topics = [query]`, since there's nothing to extract
  from a topic that's already just one topic. **`MAX_TOPICS = 3` is
  enforced in code** (`topics[:3]`) regardless of what the LLM returns —
  this is the actual cost-control mechanism, not just a prompt
  instruction, same "structural guarantee, not a trust assumption"
  pattern this repo already uses for anti-hallucination in the
  compliance agent.
- **`web_research_node`** calls `WebSearcher.search(topic)` per topic.
  A single topic's `WebSearchUnavailable` is caught **per-topic**, not
  per-request — logged, stored as `[]` for that topic, the other topics
  still get researched.
- **`synthesize_node`** is the second LLM call. If **every** topic came
  back with zero results, it skips the LLM call entirely and returns a
  template "no web results found for: ..." answer — don't let the model
  guess with nothing to ground on, same principle as `verifier_node`'s
  grounded/escalate split, just implemented as a plain `if` here since
  there's only one next step (`END`) either way, not a separate graph
  edge. If the LLM call itself fails (`BedrockUnavailable`/
  `GroqUnavailable`), it falls back to a **different** template that
  renders the raw search results it does have — don't collapse this into
  the same "no results" template as the truly-empty case; a real set of
  shown-but-unsynthesized results is a different, more useful failure
  mode than genuinely finding nothing, and conflating them was an actual
  bug caught in testing (`_template_draft` branches on
  `any(search_results.get(t) for t in topics)`).

## Node function conventions — same as edit-compliance-graph

Partial-update return (never mutate `state`), every node calls `_log`,
long-lived dependencies are keyword-only args wired via `lambda s:
node_fn(s, dependency=self.x)` in `_build_graph` — including
`document_parser` itself, which is an injectable constructor arg on
`ResearchAgentGraph` (`document_parser=None`, defaulting to
`document_parser.parse_document`) specifically so tests can swap in a
fake without real file I/O. (A Python default-argument value is bound
once at function-definition time, so making `document_parser` a
keyword-only *node* argument wired from `self.document_parser` — rather
than a default value baked into `parse_document_node`'s own signature —
is what actually makes it overridable per-instance; this was a real bug
caught while writing `tests/test_research_agent.py`.)

## `.chat()` — the generic method both Summarizer classes share

`BedrockSummarizer`/`GroqSummarizer` (in `bedrock_llm.py`/`groq_llm.py`)
each expose a third method, `.chat(system_prompt, user_message) -> str`,
alongside their compliance-specific `.summarize()`/
`.respond_ungrounded()` — same lazy-client + typed-`*Unavailable`-
exception pattern, just with the caller supplying both prompts directly
instead of the method building them from `llm_prompts.py`. This is the
Research agent's **only** way of calling an LLM; it never touches
`.summarize()`/`.respond_ungrounded()`, and the compliance agent never
touches `.chat()`. Both Summarizer classes got `.chat()` added together
(same shared-interface discipline as the rest of this project — don't
add a capability to only one provider).

`src/research_prompts.py` supplies the two prompts used with `.chat()`
(`TOPIC_EXTRACTION_SYSTEM_PROMPT`/`build_topic_extraction_message` and
`RESEARCH_SYNTHESIS_SYSTEM_PROMPT`/`build_synthesis_message`), mirroring
`llm_prompts.py`'s one-constant-plus-one-builder shape. Unlike
`llm_prompts.py`, there's no risk of the two providers drifting apart
here — `.chat()`'s signature is identical for both, so there's nothing
provider-specific to diverge.

## `_default_llm()`/`_log` are shared via `src/agent_common.py`

Both `agents.py` and `research_agent.py` now import these from
`agent_common.py` (`default_llm`/`log_step`, re-exported under each
module's original `_default_llm`/`_log` names so neither module's own
public interface changed). This used to be two separate ~10-15-line
copies — deliberately *not* extracted after just the first duplicate
(see the git history / `CLAUDE.md`'s "three similar lines" rule), but
extracted once the planned multi-agent roster (`CLAUDE.md`'s "Target
end product": Email, Calendar agents still to come) made a third and
fourth copy a near-certainty rather than a hypothetical. **Both
existing modules' test suites (`test_agents_integration.py`,
`test_research_agent.py`) still import `_default_llm` from their own
module** (`from agents import _default_llm`, `from research_agent
import _default_llm`) and pass unchanged — the extraction only moved
the implementation, not the public name each module exposes. A new
agent should import `default_llm`/`log_step` from `agent_common`
directly (aliased to `_default_llm`/`_log` for consistency with the
existing two), not write a third copy.

## Why `ddgs` and `pypdf`, not a paid search API or a heavier parsing stack

Web search uses `ddgs` (`src/web_search.py::WebSearcher`) — the current
PyPI name for the renamed `duckduckgo_search` package — specifically
because it's free and needs no API key, matching this project's existing
bias toward free/cheap over paid (the same reasoning that added Groq as
a free LLM alternative to Bedrock). A purpose-built search API (e.g.
Tavily) was considered and explicitly rejected for v1: it would add a
new secret to manage for a feature this project didn't strictly need a
paid tier for. `ddgs`'s result dicts use `title`/`href`/`body` keys —
confirmed with a real call during implementation, not assumed from
memory, since a renamed/evolving package is exactly the kind of thing
that can silently shift its shape across releases; re-confirm this if
`ddgs` is ever upgraded and something downstream starts getting empty
titles/urls/snippets.

Document parsing uses `pypdf` (`src/document_parser.py::parse_document`)
for the same free/lightweight reasoning, and only supports `.pdf`/`.txt`/
`.md` — no OCR (a scanned/image-only PDF raises `DocumentParseError`,
it is not silently handled as empty), no `.docx`. `parse_document`
accepts a path/str **or a file-like object** (not just a path)
specifically so Streamlit's `st.file_uploader` result can be passed
straight through without ever writing a temp file — `filename_hint`
supplies the extension when the source has no `.suffix`/`.name` of its
own. Extracted text is truncated to `MAX_EXTRACTED_CHARS = 20_000`
(naive head-truncation, not head+tail) to bound prompt size/cost — a
real resume is typically well under this, so this mostly matters for
protecting against someone uploading something much larger.

## Entry points

`research_demo.py` (CLI, mirrors `demo.py`) and `pages/1_Research_Agent.py`
(Streamlit page, **auto-discovered** by Streamlit's multipage convention
— any `.py` file under `pages/` next to `app_ui.py` gets added to the
sidebar nav automatically, so `app_ui.py` itself needed zero changes for
this to appear). Both construct a bare `ResearchAgentGraph()` the same
way `demo.py`/`app_ui.py` construct `ComplianceAgentGraph()`.

## Adding a new node

Same procedure as `edit-compliance-graph`: write the node function
(partial-update return, `_log` call) → `graph.add_node(...)` in
`_build_graph` → wire with `add_edge`/`add_conditional_edges` → smoke-
test via `python3 src/research_agent.py` (one hardcoded query-only run)
→ confirm via `python3 research_demo.py "..."` end-to-end, ideally once
with a real `--doc` too since the document path exercises more of the
graph than the query-only shortcut does.

## Test coverage — and the one deliberate gap

`tests/test_web_search.py`, `tests/test_document_parser.py`, and
`tests/test_research_agent.py` cover this agent (see
`write-compliance-tests` for the general conventions); `.chat()` itself
is tested alongside `.summarize()`/`.respond_ungrounded()` in the
existing `tests/test_bedrock_llm.py`/`tests/test_groq_llm.py`, not a new
file, since it's a method on classes those files already own.

**No live/gated test exists for a real `ddgs` call or a real `.chat()`
call in this agent's end-to-end path** — a deliberate v1 scope cut, not
an oversight (see README's "Honest limitations"). This project's
existing rule is "never add a third ungated live-AWS test file"
(`test_live_bedrock.py`/`test_live_rag.py` are the only two); the same
reasoning extends to a new live-network boundary. If you want this
covered later, gate it behind its own env var (e.g.
`RUN_LIVE_RESEARCH_TESTS=1`) rather than overloading
`RUN_LIVE_BEDROCK_TESTS` — a `ddgs` outage and a Bedrock outage are
different failure modes worth being able to isolate.
