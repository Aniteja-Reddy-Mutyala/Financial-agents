---
name: run-evaluation-suite
description: Use before running, changing, or interpreting anything in eval/ — the templated 1,416-case evaluation suite (eval/generate_dataset.py, eval/run_eval.py, eval/generate_report.py) and the open-weight LLM-judge pass (eval/llm_judge.py, eval/run_judge.py). Covers ground-truth design, real cost/time, the .env secrets convention it introduced, and a known gpt-oss judge quirk.
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
   restarting. Produces `eval/results.json`.
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
semantic check — an open-weight model reads the actual question+answer
and gives a correct/incorrect verdict — **against answers already in
`eval/results.json`, with zero new Bedrock calls** (it judges existing
output, it doesn't re-run the agent).

- **Current judge: `openai/gpt-oss-20b` via Groq** (`GroqJudge` in
  `llm_judge.py`). Needs `GROQ_API_KEY`.
- **A Hugging Face Inference API version was tried first and is kept as
  unused reference** (same file, git history) — that route hit a real
  wall: this account's HF Inference Providers needed prepaid credits for
  every model tried, even after enabling the free "HF Inference API"
  provider. Don't re-attempt HF unless the user explicitly wants to pay
  for it.
- **`gpt-oss-20b` is a reasoning model — give it real `max_tokens`
  headroom.** Measured: a single judge call burned 271 of 310 completion
  tokens on hidden reasoning before the visible JSON verdict. At
  `max_tokens=400`, 49 of 120 calls (41%) came back with empty content
  (reasoning alone exhausted the budget). Raising it to 800 dropped that
  to 9/120 (7.5%). If you see a wave of `judge_correct: null` /
  "unparseable judge output: ''" results, this is almost certainly why —
  raise `max_tokens` further, don't assume the model or prompt is broken.

## The real finding this eval surfaced — read before trusting citation-correctness alone

The structural report says 99.9% citation-domain-correct. A 120-case
judged sample found only 64.9% substantively correct — and HIPAA/FDA,
which look like the *strongest* domains structurally (100%/98.9% F1),
are the *weakest* under semantic review (30%/46%). Citation-correct does
not mean answer-correct. Full writeup, root causes found so far
(chunking truncation; `planner_node`'s fixed intent-description padding
skewing FDA retrieval toward "signature" vocabulary regardless of the
actual sub-topic), and a fix that was tried and reverted (didn't
generalize — see git history) are tracked in
[GitHub issue #1](https://github.com/Aniteja-Reddy-Mutyala/Financial-agents/issues/1)
and `eval/JUDGE_REPORT.md`. If you pick this up, re-run the judge pass
afterward to confirm it actually moved the needle — don't trust the
structural report alone, that's exactly what this finding is about.

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
