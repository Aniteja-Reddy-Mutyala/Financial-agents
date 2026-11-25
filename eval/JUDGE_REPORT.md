# LLM-Judge Report

**Judge model:** `openai/gpt-oss-20b` via Groq (open-weight, no new Bedrock cost)
**Sample:** 120 of 1,415 cases from the full eval run, judged against the
*already-generated* answers (no new agent/Bedrock calls — see `run_judge.py`).
**Parsed:** 111/120 (9 returned empty/truncated judge output even after
raising `max_tokens` to 800 — `gpt-oss-20b` spends a variable, sometimes
large, share of its token budget on internal reasoning before the visible
JSON verdict).

## Why this exists

`run_eval.py`'s own scorer never reads the generated answer for meaning —
it only checks structural properties (intent label match, citation
CFR-prefix match, keyword presence). That gave misleadingly reassuring
numbers: 100% grounded, 99.9% citation-domain-correct. This judge reads
the actual question + answer and gives a real semantic verdict: is the
answer substantively correct and actually responsive, not just well-cited.

## Headline finding: citation-correct ≠ answer-correct

| Metric | Value |
|---|---|
| Citation domain-correct (structural, full 1,415) | 99.9% |
| Judge says substantively correct (this 120-case sample) | 64.9% |

**That ~35-point gap is the real finding.** The system almost always cites
the right regulation, but a meaningful fraction of its answers misstate,
omit, or fail to actually address what that regulation requires.

## Per-intent judge-correct rate (111 parsed cases)

| Intent | N | Judge-correct |
|---|---|---|
| hipaa_privacy_security | 10 | 30% |
| fda_e_records | 24 | 46% |
| info_sharing | 16 | 50% |
| sox_internal_controls | 10 | 60% |
| ctr_filing | 16 | 62% |
| aml_program | 16 | 94% |
| recordkeeping | 11 | 100% |
| sar_filing | 8 | 100% |

This is a striking reversal of the structural report: HIPAA and FDA
scored a perfect 100% precision/recall/F1 and 100% citation-domain-correct
in the full structural eval — they were the project's apparent strongest
domains. Under semantic judging, they're the **weakest** — the system
reliably finds and cites the right section but frequently gets the
*content* of what it requires wrong (e.g. stating a safeguard or
notification isn't required when the regulation actually mandates it).

## Concrete examples of real-looking errors (not judge nitpicking)

- *"Does HIPAA require notification when PHI is accessed without
  authorization?"* — system said no notification required; HIPAA's
  breach notification rule says otherwise. This is a potentially
  consequential wrong answer for a compliance tool to give.
- *"Does our system need signature validation for device history
  records?"* — system said electronic signatures don't need validation;
  21 CFR 11.100(2) says they do.
- *"What must a health plan do when PHI is disclosed in response to a
  subpoena?"* — cited 45 CFR 164.514 instead of 164.512(b), and omitted
  the actual subpoena-specific obligations.

## Caveats — read before treating 64.9% as gospel

- **Sample, not the full set.** 120 of 1,415 — a real signal, but not
  exhaustive; a specific intent's rate (e.g. `sar_filing`'s 8-case 100%)
  has wide uncertainty at this N.
- **The judge is itself an LLM and can be wrong.** A couple of its
  "wrong" verdicts look more like a stricter completeness bar than a
  factual error (e.g. flagging an appropriately cautious "it's unclear
  without more context" answer as not "directly addressing" the question).
  Spot-check `eval/judge_results.json`'s `judge_reasoning` field before
  treating any individual verdict as ground truth.
- **9 cases (7.5%) produced no parseable verdict at all** and are
  excluded from the rates above, not counted as failures or successes.
- This judges `src/bedrock_llm.py`'s Nova Micro output quality — it does
  not re-test retrieval or intent classification, both already covered
  by the structural eval.

## What this suggests, not prescribes

The structural eval (citation domain, grounding) is necessary but not
sufficient to call an answer "correct" — this project's own README should
not claim high accuracy based on citation-correctness alone without this
caveat. Closing the ~35-point gap would mean improving `bedrock_llm.py`'s
prompt/grounding discipline (e.g. requiring the model to quote the
specific requirement, not just name the section) — a real follow-up,
not something this eval run itself fixes.
