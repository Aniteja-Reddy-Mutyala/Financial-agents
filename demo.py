#!/usr/bin/env python3
"""CLI demo for the mini BSA/AML compliance agent.

Usage:
    python3 demo.py "do we need to file a SAR for this wire transfer?"
    python3 demo.py            # runs a few sample queries
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agents import OrchestratorAgent  # noqa: E402

SAMPLE_QUERIES = [
    "do we need to file a SAR for a structured cash deposit pattern",
    "what is the CTR threshold for a cash deposit",
    "how long must we retain records of wire transfers",
    "can we tell another bank about a SAR we filed",
]


def run_query(orchestrator, query):
    print("=" * 70)
    print(f"QUERY: {query}")
    print("-" * 70)
    result = orchestrator.handle(query)
    print("\n".join(result.trace))
    print()
    print(result.answer)
    print()


def main():
    orchestrator = OrchestratorAgent()
    queries = sys.argv[1:] or SAMPLE_QUERIES
    for q in queries:
        run_query(orchestrator, q)


if __name__ == "__main__":
    main()
