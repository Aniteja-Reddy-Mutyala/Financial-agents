#!/usr/bin/env python3
"""CLI demo for the top-level Orchestrator (orchestrator_agent.py) — splits
a query across the Compliance and Research agents and composes one answer.

Usage:
    python3 orchestrator_demo.py "do we need to file a SAR for a wire transfer, and look up Kubernetes security best practices"
    python3 orchestrator_demo.py            # runs a sample query
    LLM_PROVIDER=groq python3 orchestrator_demo.py "..."   # Groq instead of Bedrock
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from orchestrator_agent import OrchestratorGraph  # noqa: E402

SAMPLE_QUERY = (
    "do we need to file a SAR for a structured cash deposit pattern, "
    "and also look up recent guidance on Kubernetes security"
)


def run(orchestrator, query):
    print("=" * 70)
    print(f"QUERY: {query}")
    print("-" * 70)
    result = orchestrator.handle(query)
    print("\n".join(result.trace))
    print()
    print(result.answer)
    print()


def main():
    query = " ".join(sys.argv[1:]) or SAMPLE_QUERY
    orchestrator = OrchestratorGraph()
    run(orchestrator, query)


if __name__ == "__main__":
    main()
