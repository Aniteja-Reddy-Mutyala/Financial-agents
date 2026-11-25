---
name: run-evaluation-suite
description: Use before running, changing, or interpreting anything in eval/ — the templated 1,416-case evaluation suite (eval/generate_dataset.py, eval/run_eval.py, eval/generate_report.py) and the LLM-judge pass (eval/llm_judge.py, eval/run_judge.py, now with both a Groq and a Bedrock judge provider). Covers ground-truth design, real cost/time, the .env secrets convention it introduced, a known gpt-oss judge quirk, and how to run a controlled (deterministic, intent-filtered) re-measurement.
---

## What this is, and what it isn't

`eval/` is a one-off/occasional quality-measurement harness, separate from
`tests/` — it answers "how good are the real answers at scale," not
"does the code work" (that's what `tests/` + `write-compliance-tests`
cover). It is **not** part of the default `pytest` run and has no CI
wiring; run it manually when you want a quality snapshot, not on every
change.

## The three-stage pipeline

1. **`eval/generate_dataset.py`** — free, deterministic (fixed seed), no
   AWS calls. Templated scenario generation across all 8 intents,
   producing `eval/dataset.json` (~1,416 cases). Each case carries
   auto-derived ground truth: `expected_intent`,
   `expected_citation_prefix` (a CFR title/part, e.g. `"31 CFR"` or
   `"21 CFR 11"` — **deliberately not one exact section**, see below),
   and `expected_keywords` (informational, not a pass/fail gate).
2. **`eval/run_eval.py`** — real cost (~$0.03 for the full set: one
   embed + one summarize call per case), real time (ran ~23 min once,
   but has hit much worse throttling before — see
   `bedrock-rag-retrieval`). Same resilience pattern as
   `build_embeddings.py`: modest concurrency, backoff, checkpointing
   (`eval/.eval_checkpoint.json`) so an interruption resumes instead of
   restarting. Produces `eval/results.json`. Also takes `--intents
   a,b,c` (restrict to specific `expected_intent` values — e.g.
   `fda_e_records,hipaa_privacy_security`, so a focused re-check isn't
   diluted across all 8 intents) and `--temperature` (override
   `BedrockSummarizer`'s default 0.2; pass `0.0` for deterministic
   generation in a controlled before/after comparison — see "Controlled
   re-measurement" below).
   **`eval/results.json` is meant to hold the full 1,416-case dataset's
   results** — running with `--intents` or `--sample` overwrites it with
   a partial set. Don't commit that partial file over the full one (see
   git history: this happened once and was caught before commit by
   diffing case counts) — either restore the full-dataset version before
   committing, or treat a filtered run's `results.json`/`judge_results.json`
   as a local scratch artifact you don't stage.
3. **`eval/generate_report.py`** — free (just reads `results.json`).
   Computes precision/recall/F1 per intent from a real confusion matrix,
   plus grounding and citation-domain-correctness rates. Writes
   `eval/REPORT.md`.

## Why ground truth is a CFR prefix, not one exact section

A first version of `generate_dataset.py` hand-picked one "correct"
citation per intent. A pilot run showed this was wrong: the real
retriever often (correctly!) cites a *different*, equally-valid section
in the same regulatory family — e.g. FDA citing `21 CFR 11.10` instead
of a hand-picked `11.100`/`11.200`/`11.70` set. Don't re-narrow this back
to exact sections without re-running a pilot first; you'll just
reintroduce false "failures" that are actually correct answers.

## The LLM-judge pass — a different, deeper kind of check

`run_eval.py`'s scorer is **purely structural**: intent label match,
citation-prefix match, keyword substring search. It never reads the
answer for meaning. `eval/llm_judge.py` + `eval/run_judge.py` add a real
semantic check — an LLM reads the actual question+answer and gives a
correct/incorrect verdict — **against answers already in
`eval/results.json`, with zero new retrieval/generation calls** (it
judges existing output, it doesn't re-run the agent).

Two interchangeable judge providers now (`run_judge.py --provider
groq|bedrock`, default `groq`) — same shared prompt
(`JUDGE_SYSTEM_PROMPT`/`_build_user_prompt` in `llm_judge.py`; never edit
one provider's prompt without the other, same rule as
`bedrock-llm-integration`'s shared-prompt convention):

- **Groq (`GroqJudge`, default): `openai/gpt-oss-20b`**. Needs
  `GROQ_API_KEY`. Free/cheap, but has a **real daily token quota per
  model** that a run of a few hundred cases can exhaust mid-run (measured:
  a 360-case run died with `429` rate-limit errors after ~148-160 cases).
  `--model` switches to a different Groq model's separate quota
  (`openai/gpt-oss-120b` has worked as a fallback) rather than waiting for
  the daily reset.
  - **`gpt-oss-*` is a reasoning model — give it real `max_tokens`
    headroom.** Measured: a single judge call burned 271 of 310
    completion tokens on hidden reasoning before the visible JSON
    verdict. At `max_tokens=400`, 49 of 120 calls (41%) came back with
    empty content. Raising it to 800 dropped that to 9/120 (7.5%).
  - **Don't assume every `judge_correct: null` is this quirk, though** —
    check `judge_reasoning` first. `"unparseable judge output: ''"` is
    the empty-content/budget quirk; `"judge call failed: ... 429 ..."`
    is the daily-quota wall, a different problem with a different fix
    (switch model/provider, not raise `max_tokens`).
- **Bedrock (`BedrockJudge`): `amazon.nova-lite-v1:0`**. Added
  specifically to work around Groq's daily quota (a separate, billed
  resource with its own limits, not Groq's) — use `--provider bedrock`
  to judge via it instead, or `--provider bedrock --retry-unparsed` to
  fill in just the cases a prior (any-provider) run left as
  `judge_correct: null`, without re-spending calls on cases already
  judged.
  - **Deliberately not the same model as `BedrockSummarizer`'s Nova
    Micro** — a model judging its own (or a sibling model's) output
    risks leniency toward its own mistakes (self-evaluation bias), which
    defeats the point of an external judge. Nova Lite is a different
    *tier*, which is weaker independence than a different *provider*
    would give.
  - **Claude models were tried first for better independence and are
    NOT usable on this account**: `AccessDeniedException` /
    `INVALID_PAYMENT_INSTRUMENT` — an AWS Marketplace subscription gap,
    not a code bug. Don't re-attempt a Claude judge model without first
    confirming (e.g. via `boto3.client("bedrock").list_foundation_models()`
    plus one real invoke) that this account's access actually works now.
  - A Hugging Face Inference API version was tried even earlier and is
    kept as unused reference (same file, git history) — blocked on
    prepaid-credit requirements. Don't re-attempt HF unless the user
    explicitly wants to pay for it.

## A 120-case judged sample found real answer-correctness gaps — now fixed, see the issue for the full history

The structural report says 99.9% citation-domain-correct. A 120-case
judged sample found only 64.9% substantively correct — and HIPAA/FDA,
which looked like the *strongest* domains structurally (100%/98.9% F1),
were the *weakest* under semantic review (30%/46% correct). Citation-
correct does not mean answer-correct. Tracked as
[GitHub issue #1](https://github.com/Aniteja-Reddy-Mutyala/Financial-agents/issues/1)
(now **closed**) and `eval/JUDGE_REPORT.md`. Root causes: chunking
truncation, `planner_node`'s fixed intent-description padding skewing
FDA retrieval toward "signature" vocabulary regardless of the actual
sub-topic, and (found later) the retriever returning only a section's
single best-scoring chunk even when the decisive clause lived in a
different chunk of the same section. A first fix attempt (full-section-
text evidence + a stricter prompt) was tried and reverted for not
generalizing; four narrower fixes landed instead (see `edit-compliance-
graph` and `bedrock-rag-retrieval` for where each one lives) and were
confirmed, via the controlled re-measurement below, to raise FDA to
76.7% and HIPAA to 78.9%. If you touch retrieval, chunking, or the
judged-answer prompt again, re-run the controlled check below
afterward — don't trust the structural report alone, that's exactly
what this finding is about, and a residual ~21-23% error rate on
FDA/HIPAA is a known *ongoing* limitation, not proof the fix didn't
work.

## Controlled re-measurement — removing noise from a before/after comparison

A single small-sample judge run is **not a reliable way to tell whether a
retrieval/prompt change helped**, for two compounding reasons found the
hard way in issue #1's history:

1. **Small sample, cross-domain dilution.** A random `--sample 50` across
   all 8 intents gives only ~6-7 cases for any one domain — one flipped
   verdict is a ~15-20 point swing. Use `run_eval.py --intents
   fda_e_records,hipaa_privacy_security` (no `--sample`, or a large one)
   to get the full ~180-per-intent set instead of a diluted slice.
2. **Generation isn't deterministic.** `BedrockSummarizer` defaults to
   `temperature=0.2`, so re-running `run_eval.py` against *identical*
   retrieved evidence can produce different answer wording each time —
   measured: the same 5-case HIPAA slice scored 60%, then 80%, then 60%
   across three runs with no code change in between, purely from
   sampling noise. Use `run_eval.py --temperature 0.0` to remove this
   variable before trusting a before/after delta.

```bash
python3 eval/run_eval.py --intents fda_e_records,hipaa_privacy_security --temperature 0.0
set -a && source .env && set +a && python3 eval/run_judge.py --sample <full count> --model openai/gpt-oss-120b
python3 eval/run_judge.py --provider bedrock --retry-unparsed   # fill in any quota-cut cases
```

Even with both of these controlled, one judge's verdict on one answer
can still differ run-to-run on a borderline case (the judge call itself
isn't guaranteed deterministic at `temperature=0.0` the same way
re-running it twice can disagree) — a large sample size is what makes
the aggregate number trustworthy, not any single case's verdict.

## Secrets: project-local `.env`, not the shell profile

This is the one part of the project that uses a `.env` file
(`HF_TOKEN`, `GROQ_API_KEY`, `GITHUB_TOKEN` — see `.env.example`) instead
of `~/.zshenv`, by deliberate choice (project-scoped secrets). **Nothing
here auto-loads `.env`** — no `python-dotenv` dependency — so source it
explicitly before a command that needs one of these:

```bash
set -a && source .env && set +a && python3 eval/run_judge.py
```

**Never let a sourcing or parsing error print a secret value.** This
happened twice in this project's real history from careless commands
(a bare `grep -n` matching the line, and a shell syntax error echoing a
token as an unrecognized command after a stray space following `=`) —
both required rotating the exposed token. Verify a `.env` value is set
by checking length only (`${#VARNAME}` in a shell, or reading the file
in Python and printing `len(value)`), never by `source`-ing it in a way
whose error path could echo the raw value, and never by `cat`/`grep`-ing
the file without a flag that suppresses matched content.

`GITHUB_TOKEN` needs **write** access to actually post a comment or close
an issue, not just read — a token scoped read-only (or a fine-grained
token missing "Issues: Read and write") fails with `403: Resource not
accessible by personal access token` on `POST .../issues/{n}/comments`,
not a clearer permissions error. If you hit that, the fix is the user
regenerating the token with write scope, not a code workaround.

## Running a subset / iterating cheaply

```bash
python3 eval/generate_dataset.py                 # free, rebuilds eval/dataset.json
python3 eval/run_eval.py --estimate-only         # see cost/time before committing
python3 eval/run_eval.py --sample 50             # pilot on a random subset first
python3 eval/generate_report.py                  # free, from whatever results.json exists
set -a && source .env && set +a && python3 eval/run_judge.py --sample 50
```

Always pilot a change on `--sample 50`-ish before re-running the full
1,416 cases — re-running the whole set costs real time (checkpointed,
but still) and, for the judge pass, real Groq calls.
