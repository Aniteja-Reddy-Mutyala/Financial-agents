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

`verify` is the compliance gate: it decides `grounded` from retrieval
score against `MIN_SCORE`, and `route_after_verify` sends the state to
either `summarize` (an LLM call grounded in retrieved evidence) or
`escalate` (an LLM call with NO evidence — see "escalate_node: LLM-
powered, never given evidence" below; it is not the static message it
used to be). Never let `summarize` run on ungrounded state, and never
let `escalate_node` receive any retrieved evidence — those are the two
halves of the whole anti-hallucination design here: the grounded path
only ever cites what it was given, and the ungrounded path has nothing
to cite no matter what it says. `MIN_SCORE` is tuned to whatever
similarity metric `retriever.py` currently produces (dense Titan-
embedding cosine similarity, not the TF-IDF cosine it used to be) — if
you change how `retriever.py` scores things, `MIN_SCORE` almost
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

## Fixed: `NO_PAD_INTENTS` — padding hurts broad, multi-topic intents even when `has_signal` is true

A *different* padding problem from the one above (that one was about a
*wrong/noise* intent description; this one is about a *correct*
classification whose description is still too broad to help): a query
can be correctly, confidently classified into `fda_e_records` or
`hipaa_privacy_security` and still have its retrieval query degraded by
padding, because those two intents' single fixed description ("FDA
electronic records and electronic signature requirements" / "HIPAA
Privacy and Security Rule requirements...") can't represent every
sub-topic under it. Measured (GitHub issue #1): padding with the
FDA description skewed retrieval toward "signature" vocabulary even for
a direct, correctly-classified "audit trail" question, so `21 CFR 11.10`
(the section that actually covers audit trails) never made the top
results. HIPAA (102 sections, much broader) has the same shape of
problem.

`planner_node` now checks `intent in NO_PAD_INTENTS` *before* the
`has_signal` check — for `fda_e_records`/`hipaa_privacy_security`, it
never pads regardless of `has_signal`, and searches the raw query alone
(same `retriever_node` short-circuit as the `has_signal=False` path
above, so this also saves a Bedrock call). Don't add a third intent to
`NO_PAD_INTENTS` just because it has many sections — the two already in
it were added because padding was *measured* to hurt them, not because
of section count alone; measure before adding more.

This fix, plus a sentence-boundary chunking fix and a bounded neighbor-
chunk evidence expansion in `retriever.py` (see `bedrock-rag-retrieval`)
and a clause-awareness prompt addition (see `bedrock-llm-integration`),
closed GitHub issue #1 — see README's "Honest limitations" for the full,
measured before/after.

## escalate_node: LLM-powered, never given evidence — and why that's the whole point

A greeting ("Hi") used to either get force-fit into a random regulatory
domain (confusing) or, after the has_signal fix, correctly fail to
ground and get a static "route to a human compliance officer" message —
a strange reply to "hi." `escalate_node` now calls the LLM too (via a
new `Summarizer.respond_ungrounded(query)` method, separate from
`.summarize()` — see `bedrock-llm-integration`), using a prompt
(`llm_prompts.py::UNGROUNDED_SYSTEM_PROMPT`) that lets the model decide
the tone: warm and conversational for a greeting, or an honest "I don't
have grounded evidence for that" for a real-sounding but uncovered
compliance question.

**This is deliberately the *only* place a greeting is handled** — there
is no separate chitchat detector, classifier label, or phrase list
anywhere. A greeting is not special-cased at all; it just happens not to
ground (same as any other query that doesn't clear `MIN_SCORE`), and
`escalate_node` handles every ungrounded query the same way regardless
of why it's ungrounded.

**Two earlier, narrower attempts were tried and replaced by this one:**
(1) adding `"chitchat"` as a 9th trained label directly on
`IntentClassifier` — reverted after a measured regression: with only ~3
training examples per class already, the new class's intercept was
enough to flip several previously-correct real compliance queries (e.g.
"Tell me the format of SAR," sharing the token "tell" with a chitchat
example) purely from the fragile model's shifted decision boundary, not
genuine ambiguity; (2) a separate, deterministic `is_chitchat(query)`
phrase/substring check run before the real classifier — this avoided
regressing the classifier, but needed constant hand-expansion for every
new phrasing a real user typed ("Hi,there" with no space, "tell me
something about yourself") and could never cover all of them. Letting
the LLM itself judge tone — with no evidence to misuse, since
`escalate_node` never has any — handles arbitrary phrasing without a
maintained list and without touching the classifier at all. If you're
tempted to special-case greeting detection again, read this history
first; the LLM-judgment approach here is the one that actually stopped
needing patches.

**Why this doesn't reopen the hallucination risk** `summarize()`'s
prompt has needed real fixes for (see `bedrock-llm-integration`'s
"Prompting" section and GitHub issue #1): `escalate_node` is only ever
reached when retrieval did NOT ground the query, and it is never handed
`state["evidence"]` — not filtered, not summarized, not at all. The
model has nothing to cite from in this path no matter how it's
prompted, so there's no instruction for it to ignore or evidence for it
to misuse. That's a structural guarantee from the code, not a trust
assumption about the model's behavior.

`escalate_node`'s static fallback (used only when the LLM call itself
fails — `SummarizerUnavailable`) is unchanged from before: the templated
"could not confidently ground this answer... recommend routing to a
human compliance officer" message, so a provider outage degrades the
same way it always has rather than showing nothing.
