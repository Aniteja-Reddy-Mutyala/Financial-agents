"""Builds a full metrics report (accuracy, precision, recall, F1 per
intent class, plus grounding/citation/fact-keyword rates) from
eval/results.json and writes it to eval/REPORT.md.

Precision/recall/F1 are standard multi-class classification metrics
computed from the intent confusion matrix (expected_intent vs.
actual_intent) — not just raw accuracy, per the metrics this project's
eval suite was asked to report.

Run after eval/run_eval.py has produced eval/results.json:
    python3 eval/generate_report.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.json"
REPORT_PATH = HERE / "REPORT.md"


def _prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def build_report(results):
    scored = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    all_intents = sorted({r["expected_intent"] for r in scored} | {r["actual_intent"] for r in scored})

    # Confusion matrix: confusion[expected][actual] = count
    confusion = defaultdict(Counter)
    for r in scored:
        confusion[r["expected_intent"]][r["actual_intent"]] += 1

    # Per-class precision/recall/F1 from the confusion matrix.
    per_class = {}
    for intent in all_intents:
        tp = confusion[intent][intent]
        fn = sum(confusion[intent][other] for other in all_intents if other != intent)
        fp = sum(confusion[other][intent] for other in all_intents if other != intent)
        precision, recall, f1 = _prf1(tp, fp, fn)
        support = sum(confusion[intent].values())
        rows = [r for r in scored if r["expected_intent"] == intent]
        per_class[intent] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "grounded_rate": sum(r["grounded"] for r in rows) / support if support else 0.0,
            "domain_correct_rate": sum(r["citation_domain_correct"] for r in rows) / support if support else 0.0,
            "fact_keyword_rate": sum(r["keyword_hit"] for r in rows) / support if support else 0.0,
        }

    n = len(scored)
    accuracy = sum(r["intent_correct"] for r in scored) / n if n else 0.0
    macro_precision = sum(c["precision"] for c in per_class.values()) / len(per_class) if per_class else 0.0
    macro_recall = sum(c["recall"] for c in per_class.values()) / len(per_class) if per_class else 0.0
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0.0
    overall_grounded = sum(r["grounded"] for r in scored) / n if n else 0.0
    overall_domain_correct = sum(r["citation_domain_correct"] for r in scored) / n if n else 0.0
    overall_keyword = sum(r["keyword_hit"] for r in scored) / n if n else 0.0

    lines = []
    lines.append("# Evaluation Report\n")
    lines.append(f"**Cases scored:** {n}  **Errored (after retries):** {len(errored)}\n")
    lines.append(
        "Metrics are computed on **intent classification** (precision/recall/F1 from the\n"
        "confusion matrix below) plus three pipeline-quality rates per class: whether the\n"
        "answer was marked grounded, whether its citations came from the correct CFR\n"
        "title/part (the primary correctness check — see generate_dataset.py's\n"
        "GROUND_TRUTH comment for why this is prefix-based, not an exact-section match),\n"
        "and an informational fact-keyword mention rate (phrasing varies even in correct\n"
        "answers, so this is not a pass/fail signal).\n"
    )

    lines.append("## Overall\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy (intent) | {accuracy:.1%} |")
    lines.append(f"| Macro-avg precision | {macro_precision:.1%} |")
    lines.append(f"| Macro-avg recall | {macro_recall:.1%} |")
    lines.append(f"| Macro-avg F1 | {macro_f1:.1%} |")
    lines.append(f"| Grounded rate | {overall_grounded:.1%} |")
    lines.append(f"| Citation domain-correct rate | {overall_domain_correct:.1%} |")
    lines.append(f"| Fact-keyword mention rate (informational) | {overall_keyword:.1%} |\n")

    lines.append("## Per-intent class\n")
    lines.append("| Intent | N | Precision | Recall | F1 | Grounded | Domain✓ | Fact-kw |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for intent in all_intents:
        c = per_class[intent]
        lines.append(
            f"| {intent} | {c['support']} | {c['precision']:.1%} | {c['recall']:.1%} | {c['f1']:.1%} "
            f"| {c['grounded_rate']:.1%} | {c['domain_correct_rate']:.1%} | {c['fact_keyword_rate']:.1%} |"
        )

    lines.append("\n## Confusion matrix (rows = expected, columns = actual predicted intent)\n")
    header = "| expected \\\\ actual | " + " | ".join(all_intents) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(all_intents) + 1))
    for expected in all_intents:
        row = [str(confusion[expected][actual]) for actual in all_intents]
        lines.append(f"| **{expected}** | " + " | ".join(row) + " |")

    failures = [
        r for r in scored
        if not (r["intent_correct"] and r["grounded"] and r["citation_domain_correct"])
    ]
    lines.append(f"\n## Failures ({len(failures)} of {n})\n")
    lines.append(
        "A \"failure\" here means wrong intent label, not grounded, or grounded in the\n"
        "wrong regulatory family entirely — not a missing fact-keyword, which is\n"
        "informational only (see above).\n"
    )
    for r in failures[:30]:
        lines.append(
            f"- **[{r['expected_intent']} → {r['actual_intent']}]** {r['query']!r}  \n"
            f"  grounded={r['grounded']}, citations={r['citations']}"
        )
    if len(failures) > 30:
        lines.append(f"\n...and {len(failures) - 30} more (see results.json for the full list).")

    if errored:
        lines.append(f"\n## Errored cases ({len(errored)})\n")
        for r in errored[:10]:
            lines.append(f"- {r['query']!r}: {r['error']}")

    return "\n".join(lines)


if __name__ == "__main__":
    results = json.loads(RESULTS_PATH.read_text())
    report = build_report(results)
    REPORT_PATH.write_text(report)
    print(f"Wrote report to {REPORT_PATH}")
    print()
    print(report[:3000])
