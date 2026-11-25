---
name: bedrock-llm-integration
description: Use when changing the final-answer LLM integration in this project — src/bedrock_llm.py, src/groq_llm.py, src/llm_prompts.py, the summarize node's model/provider, prompts, or anything that calls Bedrock or Groq for the final answer. Covers both providers (model-agnostic by design), cost awareness for each, the shared prompt contract, and the fallback contract. For the embeddings/retrieval half (src/bedrock_embeddings.py, src/build_embeddings.py, src/retriever.py — always Bedrock, Groq has no embeddings API), see bedrock-rag-retrieval instead.
---

## Two interchangeable providers, one shared prompt

The final-answer Summarizer step is **model-agnostic by design**:
`src/bedrock_llm.py::BedrockSummarizer` (Amazon Nova Micro, the original
default) and `src/groq_llm.py::GroqSummarizer` (`openai/gpt-oss-20b` on
Groq, an open-weight model, no AWS cost for this step) both implement
the exact same interface — `.model_id`, `.summarize(query,
intent_description, evidence)` for a grounded compliance answer (called
from `summarizer_node`), and `.respond_ungrounded(query)` for every
ungrounded query — greetings included — (called from `escalate_node`;
see `edit-compliance-graph`'s "escalate_node: LLM-powered, never given
evidence" section for why this single method replaced an earlier
separate chitchat-detection mechanism) — both raising a typed
`*Unavailable` exception on any failure, and both import their prompt
text from `src/llm_prompts.py` instead of defining their own
(`SYSTEM_PROMPT`/`build_user_message` for `.summarize()`,
`UNGROUNDED_SYSTEM_PROMPT`/`build_ungrounded_user_message` for
`.respond_ungrounded()` — two separate prompts, since `.summarize()`'s
prompt is specifically about refusing to answer from anything but
retrieved excerpts, which is wrong for a call that's never given any).
**`.respond_ungrounded()` must never be passed retrieved evidence** —
that it structurally has none to cite is what makes it safe from
hallucinated citations, not its prompt wording; don't "improve" it by
handing it evidence someday. **Never edit a prompt or message template
in just one of `bedrock_llm.py`/`groq_llm.py`** — they'd silently drift
into answering the same question differently depending on which
provider happens to be configured. Change `llm_prompts.py` once; both
providers pick it up. If you add a third method to this interface, add
it to both files and to the fakes in
`tests/test_bedrock_llm.py`/`test_groq_llm.py` and
`tests/test_agents_integration.py::_FakeSummarizer` — a fake missing a
method only breaks the one test that happens to exercise that path,
easy to miss.

`src/agents.py::_default_llm()` picks the provider from the
`LLM_PROVIDER` env var (`"bedrock"` or `"groq"`, case-insensitive,
defaults to `"bedrock"`), read fresh on every call — not cached at
import time, so it's safe to flip between `ComplianceAgentGraph()`
instances in the same process. An explicit `llm=` override to
`ComplianceAgentGraph` always wins over `LLM_PROVIDER`.

```bash
python3 demo.py "..."                    # Bedrock (default)
LLM_PROVIDER=groq python3 demo.py "..."  # Groq instead
```

**Retrieval is unaffected by this choice.** Groq has no embeddings API,
so dense retrieval always uses Bedrock Titan embeddings regardless of
which provider answers the question — a live query with `LLM_PROVIDER=groq`
still makes one real Bedrock call (embed) plus one real Groq call
(summarize), not zero Bedrock calls. See `bedrock-rag-retrieval`.

## Model and client — per provider

- **Bedrock**: `amazon.nova-micro-v1:0`, set in
  `src/bedrock_llm.py::DEFAULT_MODEL_ID`. Swap to `amazon.nova-lite-v1:0`
  or `amazon.nova-pro-v1:0` for quality over cost — same code path, just
  change the constant (or pass `model_id=` to `BedrockSummarizer`).
  Client: `langchain_aws.ChatBedrockConverse`. Region/credentials come
  from the environment (`AWS_REGION`, default `us-east-1`) and the
  standard AWS credential chain (`~/.aws/credentials`) — don't hardcode
  either. `BedrockSummarizer(temperature=...)` defaults to `0.2`; pass
  `0.0` for deterministic generation when you need a controlled
  before/after comparison rather than production use (`eval/run_eval.py
  --temperature 0.0` wires this through — see `run-evaluation-suite`).
- **Groq**: `openai/gpt-oss-20b`, set in `src/groq_llm.py::DEFAULT_MODEL_ID`
  — this exact model is already proven to work against this project's
  Groq account (it's also `eval/llm_judge.py`'s judge model); several
  other candidates (`llama-3.3-70b-versatile` and others) failed outright
  during that investigation — don't swap the default without re-verifying
  a candidate actually exists on this account's Groq lineup first (`client.models.list()`).
  Client: the `groq` package (`pip install groq` — intentionally **not**
  in `requirements.txt`, since this provider is opt-in). Needs
  `GROQ_API_KEY` (see `.env.example` and `run-evaluation-suite`'s `.env`
  conventions). **`gpt-oss-20b` is a reasoning model** — it spends a
  variable, sometimes large, share of its token budget on internal
  reasoning before the visible answer (measured: up to 271 of 310
  completion tokens on one call). `GroqSummarizer.summarize()` uses
  `max_tokens=800` for this reason and explicitly raises
  `GroqUnavailable` if the model returns empty content (budget exhausted
  on reasoning alone) rather than returning a blank answer — don't lower
  `max_tokens` without re-testing against a few real queries first.

## Cost awareness — both are real, not mocks

- **Bedrock**: every call is billed. Nova Micro is fractions of a cent
  per query at this scale.
- **Groq**: also a real API call, though typically free/cheap at light
  usage on their current tier.

Either way:
- Don't add retry loops, batch-over-many-queries scripts, or anything
  that multiplies call volume without telling the user first and giving
  a rough cost estimate.
- Don't switch the default provider or model to a larger/pricier one
  without flagging the cost difference — that's a cost decision, not a
  pure code change.

## Fallback contract — don't break this, for either provider

`BedrockSummarizer.summarize()` raises `BedrockUnavailable`;
`GroqSummarizer.summarize()` raises `GroqUnavailable`. `summarizer_node`
in `src/agents.py` catches both at once via the `SummarizerUnavailable`
tuple and falls back to a templated draft (`_template_draft`) so a
missing/expired credential or a provider outage never crashes the demo.
If you touch either `summarize()`:

- Keep it raising its own `*Unavailable` (not letting raw SDK exceptions
  propagate) so the fallback still catches it.
- If you add a third provider, add its exception type to
  `SummarizerUnavailable` in `agents.py` — don't just catch it locally
  and assume the tuple still covers it.
- Don't add a silent retry-with-backoff inside `summarize()` that could
  turn one user query into many billed calls — surface the failure and
  let the existing fallback handle it instead.

## Prompting

`src/llm_prompts.py::SYSTEM_PROMPT` instructs the model to answer only
from the provided excerpts, cite section numbers, and (added for
[GitHub issue #1](https://github.com/Aniteja-Reddy-Mutyala/Financial-agents/issues/1),
now closed) read every clause in every excerpt rather than just the
first or most prominent one — added after measuring the model ignoring
a later lettered clause (e.g. `(b)` after focusing on `(a)`/`(c)`) even
when that clause was present in its evidence. None of this is
**enforced mechanically** for either provider — Bedrock has been
observed once citing a section not present in the retrieved evidence
(likely drawn from pretraining), and the clause-awareness addition
improved the *content* of at least one checked answer but didn't move
the aggregate judge metric outside noise on a 50-case sample (see the
issue's full history and README's "Honest limitations" — a *much*
larger, deterministic-generation re-run was what actually confirmed
real improvement, not the prompt change judged in isolation). If you're
asked to make citations trustworthy, that means adding a verification
step (e.g. regex-extract cited "31 CFR ####.###" patterns from the
response and check each one is in `state["verification"]["citations"]`)
— don't just reword the prompt and assume it's fixed; test it against a
few queries **on both providers** and report what you found either way.
A fix was already tried and reverted once for not generalizing (full-
section-text evidence + a stricter anti-hallucination prompt, see the
issue) — if you try another prompt-only change here, test it narrowly
(one added instruction, not a broad rewrite) and re-run a controlled
eval (large sample, `BedrockSummarizer(temperature=0.0)` — see
`run-evaluation-suite`'s "Controlled re-measurement") before trusting
any before/after delta; `BedrockSummarizer`'s default `temperature=0.2`
means a single small-sample run can't distinguish a real effect from
run-to-run generation noise (measured: the same 5-case slice scored
60%, 80%, then 60% across three runs with no code change in between).

## Testing a change

```bash
python3 src/bedrock_llm.py                    # standalone smoke test, Bedrock
python3 src/groq_llm.py                       # standalone smoke test, Groq
python3 src/agents.py                         # full graph on one hardcoded query (Bedrock)
python3 demo.py                               # full CLI, sample queries, Bedrock
LLM_PROVIDER=groq python3 demo.py             # full CLI, sample queries, Groq
python3 -m pytest tests/test_bedrock_llm.py tests/test_groq_llm.py -v   # mocked, both providers
python3 -m pytest tests/test_agents_integration.py -k default_llm -v     # provider-selection logic
```
