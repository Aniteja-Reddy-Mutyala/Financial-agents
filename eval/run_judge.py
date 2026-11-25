"""Runs the Groq-hosted open-weight LLM judge over a sample of
eval/results.json's already-generated answers, producing
eval/judge_results.json and a short comparison report: does the judge's
semantic verdict agree with run_eval.py's purely structural
citation_domain_correct check?

Does NOT call the compliance agent again — zero new Bedrock calls. Each
case here is one Groq API call.

Usage:
    python3 eval/run_judge.py --sample 120
"""
import argparse
import json
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.json"
DATASET_PATH = HERE / "dataset.json"
JUDGE_RESULTS_PATH = HERE / "judge_results.json"

SEED = 20261124


def _build_judge(provider, model_id):
    from llm_judge import (
        DEFAULT_BEDROCK_MODEL,
        DEFAULT_MODEL,
        BedrockJudge,
        BedrockJudgeUnavailable,
        GroqJudge,
        GroqJudgeUnavailable,
    )

    if provider == "bedrock":
        return BedrockJudge(model_id=model_id or DEFAULT_BEDROCK_MODEL), BedrockJudgeUnavailable
    return GroqJudge(model_id=model_id or DEFAULT_MODEL), GroqJudgeUnavailable


def run(sample_size, model_id=None, provider="groq", retry_unparsed=False):
    results = {r["case_id"]: r for r in json.loads(RESULTS_PATH.read_text()) if "error" not in r}
    dataset = {c["case_id"]: c for c in json.loads(DATASET_PATH.read_text())}

    existing_by_id = {}
    if retry_unparsed:
        # Only (re-)judge the cases a prior run left as judge_correct=None
        # (e.g. Groq's daily quota cutting a run off mid-way) — keep every
        # already-parsed verdict as-is instead of re-spending calls on it.
        existing = json.loads(JUDGE_RESULTS_PATH.read_text()) if JUDGE_RESULTS_PATH.exists() else []
        existing_by_id = {j["case_id"]: j for j in existing}
        case_ids = [cid for cid, j in existing_by_id.items() if j["judge_correct"] is None]
        print(f"Retrying {len(case_ids)} previously-unparsed cases out of {len(existing_by_id)} total.")
    else:
        random.seed(SEED)
        case_ids = random.sample(list(results.keys()), min(sample_size, len(results)))

    judge, judge_unavailable = _build_judge(provider, model_id)
    judged_by_id = dict(existing_by_id)
    t0 = time.time()
    for i, case_id in enumerate(case_ids, 1):
        r = results[case_id]
        c = dataset[case_id]
        try:
            verdict = judge.judge(
                query=r["query"],
                answer=r["answer"],
                expected_citation_prefix=c["expected_citation_prefix"],
                expected_keywords=c["expected_keywords"],
            )
        except judge_unavailable as exc:
            verdict = {"correct": None, "reasoning": f"judge call failed: {exc}"}
        judged_by_id[case_id] = {
            "case_id": case_id,
            "query": r["query"],
            "expected_intent": r["expected_intent"],
            "actual_intent": r["actual_intent"],
            "citation_domain_correct": r["citation_domain_correct"],
            "judge_correct": verdict["correct"],
            "judge_reasoning": verdict["reasoning"],
        }
        if i % 20 == 0 or i == len(case_ids):
            print(f"  {i}/{len(case_ids)} judged ({time.time() - t0:.0f}s elapsed)")

    judged = list(judged_by_id.values())
    JUDGE_RESULTS_PATH.write_text(json.dumps(judged, indent=2))
    print(f"\nWrote {len(judged)} judge verdicts to {JUDGE_RESULTS_PATH}")
    print_comparison(judged)


def print_comparison(judged):
    scored = [j for j in judged if j["judge_correct"] is not None]
    unparseable = len(judged) - len(scored)
    n = len(scored)
    if n == 0:
        print("No parseable judge verdicts.")
        return

    judge_correct_rate = sum(j["judge_correct"] for j in scored) / n
    both_correct = sum(j["judge_correct"] and j["citation_domain_correct"] for j in scored)
    judge_says_wrong_but_domain_right = [
        j for j in scored if not j["judge_correct"] and j["citation_domain_correct"]
    ]
    judge_says_right_but_domain_wrong = [
        j for j in scored if j["judge_correct"] and not j["citation_domain_correct"]
    ]

    print(f"\n=== Judge vs. structural scorer, {n} parsed (of {len(judged)}, {unparseable} unparseable) ===")
    print(f"Judge says correct: {judge_correct_rate:.1%}")
    print(f"Agree (both say correct): {both_correct / n:.1%}")
    print(f"Judge says WRONG despite correct citation domain: {len(judge_says_wrong_but_domain_right)} "
          f"({len(judge_says_wrong_but_domain_right) / n:.1%}) — these are answers that cited the right "
          f"regulation but the judge found substantively unconvincing")
    print(f"Judge says correct despite wrong citation domain: {len(judge_says_right_but_domain_wrong)} "
          f"({len(judge_says_right_but_domain_wrong) / n:.1%})")

    if judge_says_wrong_but_domain_right:
        print("\nExamples the judge flagged as substantively wrong (first 5):")
        for j in judge_says_wrong_but_domain_right[:5]:
            print(f"  [{j['expected_intent']}] {j['query']!r}")
            print(f"    judge reasoning: {j['judge_reasoning']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=120)
    parser.add_argument(
        "--model",
        default=None,
        help="Model id to use as judge (default depends on --provider). For "
        "Groq, switching models works around that model's own daily token "
        "quota; for Bedrock, pick any text model this account has access to.",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "bedrock"],
        default="groq",
        help="Which judge to use. 'bedrock' avoids Groq's daily quota "
        "entirely (a separate, billed resource) — see llm_judge.py's "
        "module docstring for why its default model differs from the "
        "Bedrock model that generates the answers being judged.",
    )
    parser.add_argument(
        "--retry-unparsed",
        action="store_true",
        help="Only (re-)judge cases eval/judge_results.json left as "
        "judge_correct=None (e.g. a prior Groq run cut off by its daily "
        "quota), keeping every already-parsed verdict unchanged. Ignores "
        "--sample.",
    )
    args = parser.parse_args()
    run(args.sample, model_id=args.model, provider=args.provider, retry_unparsed=args.retry_unparsed)
