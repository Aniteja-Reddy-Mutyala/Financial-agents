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


def run(sample_size):
    from llm_judge import GroqJudge, GroqJudgeUnavailable

    results = {r["case_id"]: r for r in json.loads(RESULTS_PATH.read_text()) if "error" not in r}
    dataset = {c["case_id"]: c for c in json.loads(DATASET_PATH.read_text())}

    random.seed(SEED)
    case_ids = random.sample(list(results.keys()), min(sample_size, len(results)))

    judge = GroqJudge()
    judged = []
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
        except GroqJudgeUnavailable as exc:
            verdict = {"correct": None, "reasoning": f"judge call failed: {exc}"}
        judged.append(
            {
                "case_id": case_id,
                "query": r["query"],
                "expected_intent": r["expected_intent"],
                "actual_intent": r["actual_intent"],
                "citation_domain_correct": r["citation_domain_correct"],
                "judge_correct": verdict["correct"],
                "judge_reasoning": verdict["reasoning"],
            }
        )
        if i % 20 == 0 or i == len(case_ids):
            print(f"  {i}/{len(case_ids)} judged ({time.time() - t0:.0f}s elapsed)")

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
    args = parser.parse_args()
    run(args.sample)
