---
name: edit-compliance-graph
description: Use before changing src/agents.py or the LangGraph pipeline in this project (nodes, edges, ComplianceState, routing logic, adding a new agent/node). Covers the graph shape, node-function conventions, the conditional escalation edge, and how ComplianceAgentGraph wires it all together.
---

## Graph shape

`src/agents.py` builds a `langgraph.graph.StateGraph` over a
`ComplianceState` TypedDict (`query`, `intent`, `plan`, `evidence`,
`verification`, `answer`, `trace`). The graph is linear except for one
conditional edge:

```
classify_intent -> plan -> retrieve -> verify --[grounded]--> summarize -> END
                                              \--[not grounded]--> escalate -> END
```

- `verify` is the compliance gate: it decides `grounded` from retrieval
  score against `MIN_SCORE`, and `route_after_verify` sends the state to
  either `summarize` (calls the LLM) or `escalate` (templated human-handoff
  message, no LLM call). Never let `summarize` run on ungrounded state —
  that's the whole point of the verifier agent. `MIN_SCORE` is tuned to
  whatever similarity metric `retriever.py` currently produces (dense
  Titan-embedding cosine similarity, not the TF-IDF cosine it used to be)
  — if you change how `retriever.py` scores things, `MIN_SCORE` almost
  certainly needs recalibrating too; see `bedrock-rag-retrieval`.

## Node function conventions

- Every node function takes the current state dict and returns a **partial
  update** dict — LangGraph merges it into the running state. Don't mutate
  `state` in place; build and return new values.
- Every node appends a line to `trace` via the `_log(state, agent, note)`
  helper so the CLI can print a readable audit trail. Any new node must do
  this too, so the trace stays a complete record of what happened.
- Node functions that need a long-lived object (the intent classifier, the
  retriever, the Bedrock client) take it as a keyword-only argument and are
  wired into the graph via a `lambda s: node_fn(s, dependency=self.x)` in
  `_build_graph` — see the existing nodes for the pattern. Don't reach for
  module-level globals instead.

## ComplianceAgentGraph

`ComplianceAgentGraph` (aliased as `OrchestratorAgent` for backward
compatibility with `demo.py`) owns the long-lived objects — intent
classifier, retriever, Bedrock client — and compiles the graph once in
`__init__`. Don't rebuild the graph per-query; `handle(query)` should only
call `self.app.invoke(...)`.

## Adding a new node or agent

1. Write the node function following the conventions above (partial-update
   return, `_log` call).
2. Add it with `graph.add_node(...)` in `_build_graph`.
3. Wire it in with `add_edge` or `add_conditional_edges` as appropriate —
   don't leave a node unreachable from `classify_intent`'s entry point.
4. Smoke-test directly: `python3 src/agents.py` runs one hardcoded query
   through the whole graph and prints the trace + answer. Check the new
   node's trace line appears in the right place.
5. Then run `python3 demo.py` to confirm the CLI still works end-to-end.

## Fixed: unconditional intent-description padding (the old "known rough edge")

`plan` used to always append the (possibly wrong) intent's description to
the retrieval query, which could guarantee lexical/semantic overlap with
the knowledge base even for an off-topic or misclassified query — see the
real incidents: "Tell me the format of SAR" misclassified as
`sox_internal_controls` pulling in a SOX section, and later "what's the
best pizza topping in Chicago" classified as `sox_internal_controls` at
13% confidence and still grounding, because the padding alone scored
above `MIN_SCORE` regardless of the query's real topic.

Two cheaper fixes were tried and empirically ruled out before this one —
worth knowing so you don't re-try them: (1) a **raw-query-only score
floor** — rejected because a real, legitimately on-topic but vague query
("what is the threshold") scored 0.176 raw-alone, inside the same range
as genuinely off-topic queries (0.13-0.16) — no clean separating
threshold exists. (2) **classifier confidence as a gate** — rejected
because this classifier legitimately produces low confidence (15-30%) on
many *correctly* classified queries too (tiny softmax over 8 classes with
a small training set), so a confidence floor would reject good answers
along with bad ones.

**The actual fix**: `classify_intent_node`'s `IntentClassifier.predict()`
now returns `has_signal` (`src/classifier.py`) — whether the query shared
*any* vocabulary with the training set (`vectorizer.transform(query).nnz
> 0`). This is a different, more reliable signal than confidence: a
genuinely off-topic query shares literally zero training vocabulary, so
its top "intent" is pure class-prior-intercept noise, not a real
classification, regardless of what confidence number comes out.
`planner_node` now only pads the retrieval query with the intent
description when `has_signal` is `True`; otherwise it searches the raw
query alone. `retriever_node` skips the now-redundant second search when
`plan["retrieval_query"] == state["query"]` (no padding happened), saving
a real Bedrock call in exactly the case that used to produce a false
positive.

**What this does NOT fix** — a different, deeper limitation found while
testing the fix, not yet closed: a query that is *stylistically*
plausible as a compliance question but about a genuinely uncovered topic
(e.g. "what are the legal requirements for starting a business," "what is
the penalty for violating a contract") can still score well above
`MIN_SCORE` on dense-embedding similarity alone, no padding involved —
dense embeddings correctly recognize "this sounds like formal/legal
register" even when the specific subject matter isn't in this project's
4 domains at all. This is a fundamentally different problem (retrieval
itself giving a plausible-looking score to the wrong subject matter, not
an artifact of the padding mechanism) and would need real semantic
judgment (e.g. an LLM-judge second call) to close — not something a
threshold or signal-detection trick on top of embedding scores can fix.
If you pick this up, flag the added cost/latency to the user first (every
grounded query would need a third Bedrock call) — see
`bedrock-llm-integration`'s cost-awareness rules — and update README's
"Honest limitations" to match, don't just quietly patch it in.
