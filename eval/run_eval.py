"""Runs eval/dataset.json through the REAL ComplianceAgentGraph (real,
billed Bedrock embedding + summarization calls) and scores each case
against its auto-derived ground truth, producing eval/results.json and a
printed accuracy report.

This is NOT part of the default pytest suite and is NOT meant to be run
casually or on every change — see the cost/time estimate this script
prints before it does anything, and confirm before letting the full run
proceed. Mirrors build_embeddings.py's resilience pattern (modest
concurrency, backoff on throttling, checkpointing every N cases) since
this account's measured Bedrock throughput is the same limited resource
either way.

Usage:
    python3 eval/run_eval.py --estimate-only        # just print the cost/time estimate
    python3 eval/run_eval.py --sample 50             # pilot run on a random subset
    python3 eval/run_eval.py                         # full run (resumes from checkpoint if interrupted)
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))

DATASET_PATH = HERE / "dataset.json"
RESULTS_PATH = HERE / "results.json"
CHECKPOINT_PATH = HERE / ".eval_checkpoint.json"

MAX_WORKERS = 3
CHECKPOINT_EVERY = 25
MAX_RETRIES_PER_CASE = 4

# Rough, measured-not-guessed cost basis (see bedrock-rag-retrieval /
# bedrock-llm-integration skills): ~$0.02/1M tokens for Titan embeddings,
# ~$0.035/1M tokens for Nova Micro, short queries + short-ish answers.
EST_COST_PER_CASE = 0.00002  # ~$0.02 per 1,000 cases, i.e. a couple cents for the whole set
# Measured: build_embeddings.py's 1,420 EMBED-ONLY calls took ~1,369s
# (~1s/call) at this account's throttling rate, 3-way concurrency. Each
# eval case makes one embed call AND one (typically slower) text-
# generation call, so budget generously, not optimistically.
EST_SECONDS_PER_CASE = 2.5


def estimate(n_cases):
    print(f"Estimated for {n_cases} cases:")
    print(f"  Real Bedrock calls: ~{n_cases * 2} (1 embed + 1 summarize per case)")
    print(f"  Estimated cost:     ~${n_cases * EST_COST_PER_CASE:.3f}")
    print(f"  Estimated time:     ~{n_cases * EST_SECONDS_PER_CASE / 60:.0f} minutes "
          f"(this account's Bedrock throughput is measurably limited — see "
          f"bedrock-rag-retrieval skill; could run longer if throttling is worse than measured)")


def _score_case(case, result):
    answer_lower = result.answer.lower()
    citations = sorted(result.verification.get("citations", []))
    prefix = case["expected_citation_prefix"]
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "expected_intent": case["expected_intent"],
        "actual_intent": result.intent["intent"],
        "intent_correct": result.intent["intent"] == case["expected_intent"],
        "grounded": result.verification["grounded"],
        # Primary correctness signal: did it cite something from the
        # right regulatory family (CFR title, or title+part for the
        # narrower non-BSA/AML domains)? See generate_dataset.py's
        # GROUND_TRUTH comment for why this is prefix-based, not an exact
        # section match.
        "citation_domain_correct": any(c.startswith(prefix) for c in citations),
        # Informational only, not a pass/fail gate — phrasing varies even
        # when the answer is correctly grounded (see generate_dataset.py).
        "keyword_hit": any(kw.lower() in answer_lower for kw in case["expected_keywords"]),
        "citations": citations,
        "answer": result.answer,
    }


def _run_one_with_backoff(agent, case):
    delay = 1.0
    last_exc = None
    for _ in range(MAX_RETRIES_PER_CASE):
        try:
            result = agent.handle(case["query"])
            return _score_case(case, result)
        except Exception as exc:  # noqa: BLE001 — want to retry almost anything transient
            last_exc = exc
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 15)
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "expected_intent": case["expected_intent"],
        "error": str(last_exc),
    }


def _load_checkpoint():
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        return {int(k): v for k, v in data.items()}
    return {}


def run(sample=None, intents=None, temperature=None):
    from agents import ComplianceAgentGraph

    dataset = json.loads(DATASET_PATH.read_text())
    if intents:
        dataset = [c for c in dataset if c["expected_intent"] in intents]
    if sample:
        random.seed(20261124)
        dataset = random.sample(dataset, min(sample, len(dataset)))

    estimate(len(dataset))

    done = _load_checkpoint()
    if done:
        print(f"Resuming from checkpoint: {len(done)}/{len(dataset)} cases already run.")
    remaining = [c for c in dataset if c["case_id"] not in done]

    if temperature is not None:
        # Forces deterministic generation for a controlled before/after
        # comparison — BedrockSummarizer's default (0.2) means re-running
        # this script against identical retrieved evidence can still
        # produce different answer text each time, which makes a single
        # run's accuracy numbers unreliable for judging whether a retrieval
        # or prompt change actually helped. See GitHub issue #1 history.
        from bedrock_llm import BedrockSummarizer

        agent = ComplianceAgentGraph(llm=BedrockSummarizer(temperature=temperature))
    else:
        agent = ComplianceAgentGraph()  # real retriever + real Bedrock LLM, built once, shared
    t0 = time.time()
    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one_with_backoff, agent, c): c["case_id"] for c in remaining}
        for future in as_completed(futures):
            case_id = futures[future]
            done[case_id] = future.result()
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= CHECKPOINT_EVERY:
                CHECKPOINT_PATH.write_text(json.dumps(done))
                completed_since_checkpoint = 0
            if len(done) % 50 == 0 or len(done) == len(dataset):
                print(f"  {len(done)}/{len(dataset)} cases run ({time.time() - t0:.0f}s elapsed this run)")

    CHECKPOINT_PATH.write_text(json.dumps(done))
    results = [done[c["case_id"]] for c in dataset]
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    if sample is None:
        CHECKPOINT_PATH.unlink(missing_ok=True)

    print(f"\nWrote {len(results)} scored results to {RESULTS_PATH}")
    print_report(results)


def print_report(results):
    from collections import defaultdict

    errored = [r for r in results if "error" in r]
    scored = [r for r in results if "error" not in r]
    print(f"\n=== Report: {len(scored)} scored, {len(errored)} errored (after retries) ===")

    by_intent = defaultdict(list)
    for r in scored:
        by_intent[r["expected_intent"]].append(r)

    print(f"\n{'Intent':24s} {'N':>5s} {'Intent✓':>8s} {'Grounded':>9s} {'Domain✓':>8s} {'Fact-kw':>8s}")
    for intent, rows in sorted(by_intent.items()):
        n = len(rows)
        print(
            f"{intent:24s} {n:5d} "
            f"{sum(r['intent_correct'] for r in rows) / n:8.0%} "
            f"{sum(r['grounded'] for r in rows) / n:9.0%} "
            f"{sum(r['citation_domain_correct'] for r in rows) / n:8.0%} "
            f"{sum(r['keyword_hit'] for r in rows) / n:8.0%}"
        )

    n = len(scored)
    if n:
        print(f"\n{'OVERALL':24s} {n:5d} "
              f"{sum(r['intent_correct'] for r in scored) / n:8.0%} "
              f"{sum(r['grounded'] for r in scored) / n:9.0%} "
              f"{sum(r['citation_domain_correct'] for r in scored) / n:8.0%} "
              f"{sum(r['keyword_hit'] for r in scored) / n:8.0%}")
    print("\n(Domain✓ = cited the right CFR title/part, the primary correctness check.")
    print(" Fact-kw is informational only — see generate_dataset.py's GROUND_TRUTH comment"
          " for why it's not a pass/fail gate.)")

    # The real failure signal: wrong intent, not grounded, or grounded but
    # citing the wrong regulatory family entirely.
    failures = [r for r in scored if not (r["intent_correct"] and r["grounded"] and r["citation_domain_correct"])]
    print(f"\n{len(failures)} cases failed (wrong intent, not grounded, or wrong-domain citation). First 5:")
    for r in failures[:5]:
        print(f"  [{r['expected_intent']} -> {r['actual_intent']}] {r['query']!r}")
        print(f"    grounded={r['grounded']}, citations={r['citations']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Run only N random cases (pilot run)")
    parser.add_argument("--estimate-only", action="store_true", help="Print cost/time estimate and exit")
    parser.add_argument(
        "--intents",
        default=None,
        help="Comma-separated expected_intent values to restrict the run to "
        "(e.g. fda_e_records,hipaa_privacy_security), so a controlled "
        "comparison isn't diluted by the other 6 intents.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override BedrockSummarizer's temperature (default 0.2) for this "
        "run — pass 0.0 for deterministic generation in a controlled "
        "before/after comparison.",
    )
    args = parser.parse_args()
    intents = set(args.intents.split(",")) if args.intents else None

    if args.estimate_only:
        dataset = json.loads(DATASET_PATH.read_text())
        if intents:
            dataset = [c for c in dataset if c["expected_intent"] in intents]
        n = args.sample or len(dataset)
        estimate(n)
    else:
        run(sample=args.sample, intents=intents, temperature=args.temperature)
