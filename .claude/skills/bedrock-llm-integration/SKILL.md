---
name: bedrock-llm-integration
description: Use when changing the LLM/Bedrock integration in this project — src/bedrock_llm.py, the summarize node's model, prompts, or anything that calls AWS for the final answer. Covers which model is used, cost awareness, the langchain_aws client pattern, and the fallback contract. For the embeddings/retrieval half of the Bedrock integration (src/bedrock_embeddings.py, src/build_embeddings.py, src/retriever.py), see bedrock-rag-retrieval instead — a live query now makes one call of each kind.
---

## Model and client

- Model: `amazon.nova-micro-v1:0` (Amazon's cheapest on-demand Bedrock text
  model), set in `src/bedrock_llm.py::DEFAULT_MODEL_ID`. Swap to
  `amazon.nova-lite-v1:0` or `amazon.nova-pro-v1:0` for quality over cost —
  same code path, just change the constant (or pass `model_id=` when
  constructing `BedrockSummarizer`).
- Client: `langchain_aws.ChatBedrockConverse`, invoked with a
  `SystemMessage` + `HumanMessage` pair from `langchain_core.messages`.
  Keep using this package (not raw `boto3`) for any new LLM call in this
  project — it's the integration the rest of the codebase and docs assume.
- Region/credentials come from the environment (`AWS_REGION`, default
  `us-east-1`) and the standard AWS credential chain (`~/.aws/credentials`).
  Don't hardcode a region or read credentials directly.

## Cost awareness — this is real, not a mock

Every Bedrock call is a real, billed API call against whatever AWS account
is configured. It's small per call (Nova Micro is fractions of a cent per
query at this scale), but:

- Don't add retry loops, batch-over-many-queries scripts, or anything that
  multiplies call volume without telling the user first and giving a rough
  cost estimate.
- Don't switch the default model to a larger one (Nova Pro, Claude-on-
  Bedrock) without flagging the cost difference to the user — that's a
  cost decision, not a pure code change.

## Fallback contract — don't break this

`BedrockSummarizer.summarize()` raises `BedrockUnavailable` on any failure
(network, auth, throttling, access-not-granted). `summarizer_node` in
`src/agents.py` catches exactly that exception and falls back to a
templated draft (`_template_draft`) so a missing/expired AWS session never
crashes the demo. If you touch `summarize()`:

- Keep it raising `BedrockUnavailable` (not letting raw boto3/langchain
  exceptions propagate) so the fallback still catches it.
- Don't add a silent retry-with-backoff inside `summarize()` that could
  turn one user query into many billed calls — surface the failure and let
  the existing fallback handle it instead.

## Prompting

The system prompt in `src/bedrock_llm.py::SYSTEM_PROMPT` instructs the
model to answer only from the provided excerpts and cite section numbers.
This is **not enforced mechanically** — the model has been observed once
citing a section not present in the retrieved evidence (likely drawn from
pretraining). If you're asked to make citations trustworthy, that means
adding a verification step (e.g. regex-extract cited "31 CFR ####.###"
patterns from the response and check each one is in
`state["verification"]["citations"]`) — don't just reword the prompt and
assume it's fixed; test it against a few queries and report what you
found either way.

## Testing a change

```bash
python3 src/bedrock_llm.py   # standalone smoke test with one hardcoded excerpt
python3 src/agents.py        # full graph on one hardcoded query
python3 demo.py              # full CLI with sample queries
```
