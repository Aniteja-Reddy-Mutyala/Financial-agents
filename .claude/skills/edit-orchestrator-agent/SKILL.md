---
name: edit-orchestrator-agent
description: Use before changing src/orchestrator_agent.py or src/orchestrator_prompts.py — the top-level Orchestrator -> Planner -> domain agents -> Orchestrator flow that sits in front of ComplianceAgentGraph/ResearchAgentGraph, distinct from edit-compliance-graph's and edit-research-agent's per-agent internal Orchestrator/Planner node roles.
---

## What this is, and what it is NOT

`src/orchestrator_agent.py`'s `OrchestratorGraph` is the top-level
Orchestrator/Planner pair described in `CLAUDE.md`'s "Target end
product" / "Cross-agent integration" section — a **cross-agent** layer
that receives a raw user query and the final composed answer, distinct
from each domain agent's own **internal** Orchestrator/Planner node
roles (`agents.py`'s `classify_intent_node`/`planner_node`,
`research_agent.py`'s equivalents). Those internal roles are untouched
by this agent — `OrchestratorGraph` never reaches into
`ComplianceAgentGraph`/`ResearchAgentGraph` internals, it only calls
their existing `.handle(query) -> SimpleNamespace(answer=..., trace=...)`
entry points, same as `demo.py`/`research_demo.py` do today. Don't
special-case a particular domain agent inside `orchestrator_agent.py` —
the whole point is that a new agent only needs to be added to the
`agents` dict and the planner prompt's agent list (see below), not
given its own branch here.

## Graph shape

```
plan -> dispatch -> compose -> END
```

`OrchestratorState` fields: `query`, `subtasks`
(`[{"agent": str, "task": str}, ...]`, from the planner), `subtask_results`
(same, plus `{"answer": str, "trace": list}` per subtask, from
dispatch), `answer`, `trace`.

- **`planner_node`** is the only LLM call that decides *routing* — it
  sends `orchestrator_prompts.PLANNER_SYSTEM_PROMPT` plus the raw query
  to `.chat()` and expects one `agent_name: subtask text` line **per
  agent the request actually needs, not one per available agent** (see
  `_parse_plan`). Confirmed with a real Groq call during manual
  testing: a purely compliance-shaped query produced a one-line plan
  and `dispatch_node` never touched the Research agent at all — this
  isn't a forced fan-out, don't "fix" it into one by always including
  every agent in `available_agents`. **Any line whose agent
  name isn't a key of `available_agents` is silently dropped** — this
  is what keeps a planner hallucinating an unknown agent name from
  dispatching anywhere. If parsing produces zero valid subtasks (LLM
  unavailable, or every line failed to parse), it falls back to
  `DEFAULT_AGENT` (`"compliance"`) with the **full, unmodified query**
  — per `CLAUDE.md`'s "Cross-agent integration," this project's
  population is assumed to mostly ask finance/compliance-shaped
  queries, so compliance is the safer default than dropping the
  request. Don't change `DEFAULT_AGENT` without re-reading that
  reasoning in `CLAUDE.md`.
- **`dispatch_node`** loops over `state["subtasks"]` and calls
  `self.agents[subtask["agent"]].handle(subtask["task"])` for each —
  this is the **only** place this module calls into a domain agent,
  and it does so through each agent's already-existing public
  `.handle(...)` method, never anything more specific. A subtask whose
  agent isn't in `self.agents` would `KeyError` here — this can't
  happen in practice because `planner_node` already filtered against
  `available_agents = self.agents.keys()`, but if you ever decouple
  those two (e.g. a static agent list instead of `self.agents.keys()`),
  re-add a guard.
- **`compose_node`** has two paths: **exactly one** subtask result
  forwards that agent's `answer` **verbatim, with no LLM call at all**
  — there's nothing to compose from one answer, and skipping the call
  saves a real cost. **More than one** subtask result calls `.chat()`
  with `orchestrator_prompts.COMPOSER_SYSTEM_PROMPT` to weave them into
  one answer; on `BedrockUnavailable`/`GroqUnavailable` it falls back to
  concatenating each agent's own answer under an `[agent_name]` header
  rather than losing content — same "never drop real content on an LLM
  failure" principle as `agents.py`'s `summarizer_node`/
  `research_agent.py`'s `synthesize_node` template fallbacks.

## Node function conventions — same as edit-compliance-graph/edit-research-agent

Partial-update return (never mutate `state`), every node calls `_log`
(imported from `agent_common.py` as `_log`, same as the other two
agents), long-lived dependencies are keyword-only args wired via
`lambda s: node_fn(s, dependency=self.x)` in `_build_graph`.

## `agents` — the injectable agent registry

`OrchestratorGraph.__init__(self, llm=None, agents=None)` defaults
`agents` to `{"compliance": ComplianceAgentGraph(llm=self.llm),
"research": ResearchAgentGraph(llm=self.llm)}` when not given —
**`self.llm` is passed down explicitly**, not left for each domain
agent to resolve its own `_default_llm()`. Omitting this was a real bug
caught by manually testing the Streamlit page: with the "Use live
Bedrock" toggle off (`llm=GroqSummarizer()`), the Orchestrator's own
plan/compose calls correctly used Groq, but `ComplianceAgentGraph()`/
`ResearchAgentGraph()` built with no `llm=` each called their own
`_default_llm()`, which reads `LLM_PROVIDER` from the process env —
defaulting to Bedrock regardless of the toggle, since nothing had set
`LLM_PROVIDER=groq` in that shell. Passing `self.llm` down makes the
toggle actually apply everywhere this orchestrator calls an LLM, which
is what `pages/2_Orchestrator_Agent.py`'s own caption already claimed.
**Tests must always pass
`agents=` explicitly** (with fake domain-agent stand-ins — see Test
coverage below) — constructing the real defaults builds a real
`Retriever` (`ComplianceAgentGraph()`'s constructor), which is a real,
billed Bedrock embedding client, not something a plumbing test should
ever trigger. This mirrors `write-compliance-tests`'s existing rule
about never calling `Retriever.build()` in a free test.

Adding a future agent (Email, Calendar — see `CLAUDE.md`'s "Target end
product"): construct it in the default `agents` dict under a short
lowercase key, add that same key + a one-line description to
`orchestrator_prompts.PLANNER_SYSTEM_PROMPT`'s agent list (keep the two
in sync — the planner can only ever route to names it's told about),
and that's the whole integration; `dispatch_node`/`compose_node` need
no changes.

## `_default_llm()`/`_log` are shared via `src/agent_common.py`

Same pattern as `agents.py`/`research_agent.py` — imported from
`agent_common.py` and re-exported as `_default_llm`/`_log` so this
module's own tests can `from orchestrator_agent import _default_llm`
the same way the other two agents' tests do.

## `src/orchestrator_prompts.py`

Mirrors `research_prompts.py`'s one-`*_SYSTEM_PROMPT`-constant-plus-one-
`build_*_message`-function shape, for the same two `.chat()` calls
(`PLANNER_SYSTEM_PROMPT`/`build_planner_message` and
`COMPOSER_SYSTEM_PROMPT`/`build_composer_message`). Both providers share
this prompt unchanged (same `.chat()` interface both already expose),
same as the Research agent — there's no provider-specific prompt
drift risk here either.

## Entry points

`orchestrator_demo.py` (CLI, mirrors `demo.py`/`research_demo.py`) and
`pages/2_Orchestrator_Agent.py` (Streamlit page, auto-discovered — no
changes needed to `app_ui.py` or any other page). Both construct a bare
`OrchestratorGraph()`, which in turn constructs real
`ComplianceAgentGraph()`/`ResearchAgentGraph()` instances — running
either entry point for real exercises the full real pipeline (real
Bedrock retrieval embedding call included), same cost profile as
running `demo.py` and `research_demo.py` back to back plus the planner/
compose LLM calls.

## Test coverage

`tests/test_orchestrator_agent.py` owns this agent — `_parse_plan`'s
agent-name filtering/case-insensitivity/malformed-line handling, the
single-vs-multi-subtask compose branch, the planner-LLM-failure and
composer-LLM-failure fallbacks, and its own copy of the
`_default_llm`/`LLM_PROVIDER` coverage (same one-owning-file-per-module
reasoning as `test_agents_integration.py`/`test_research_agent.py`).
Fakes both the LLM (`.chat()`-capable fake, same shape as
`test_research_agent.py::_FakeChatSummarizer`) and the domain agents
themselves (`_FakeDomainAgent` — a bare `.handle(query) ->
SimpleNamespace(answer=..., trace=...)` stand-in, passed in via
`agents={...}`) — never the real `ComplianceAgentGraph`/
`ResearchAgentGraph` defaults, for the `Retriever.build()` cost reason
above. `test_default_agents_are_built_with_the_orchestrators_llm`
specifically guards the llm-passthrough bug described in the `agents`
section above — it monkeypatches `orchestrator_agent.ComplianceAgentGraph`/
`ResearchAgentGraph` themselves (not `agents=`) to assert the
constructor actually receives `self.llm`, without building a real
`Retriever`. No live-network test file exists for this agent yet: if one is
added, it should call the real `OrchestratorGraph()` end-to-end and get
its own env-var gate, same reasoning as `test_live_research.py`'s
`RUN_LIVE_RESEARCH_TESTS` being separate from `RUN_LIVE_BEDROCK_TESTS`.
