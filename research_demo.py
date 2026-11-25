#!/usr/bin/env python3
"""CLI demo for the Research agent.

Usage:
    python3 research_demo.py "Kubernetes"
    python3 research_demo.py --doc resume.pdf
    python3 research_demo.py --doc resume.pdf "focus on cloud skills"
    python3 research_demo.py            # runs a sample query
    LLM_PROVIDER=groq python3 research_demo.py "..."   # Groq instead of Bedrock
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from research_agent import ResearchAgentGraph  # noqa: E402

SAMPLE_QUERY = "Kubernetes"


def run(agent, query, document_path):
    print("=" * 70)
    print(f"QUERY: {query!r}  DOCUMENT: {document_path!r}")
    print("-" * 70)
    result = agent.handle(query=query, document_path=document_path)
    print("\n".join(result.trace))
    print()
    print(result.answer)
    print()


def main():
    args = sys.argv[1:]
    document_path = None
    if "--doc" in args:
        i = args.index("--doc")
        document_path = args[i + 1]
        args = args[:i] + args[i + 2:]
    query = " ".join(args) if args else ("" if document_path else SAMPLE_QUERY)

    agent = ResearchAgentGraph()
    run(agent, query, document_path)


if __name__ == "__main__":
    main()
