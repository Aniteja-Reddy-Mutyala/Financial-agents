"""Opt-in end-to-end tests against REAL external services for the
Research agent: a real ddgs web search call, and a real LLM .chat() call
(Bedrock or Groq, whichever LLM_PROVIDER selects — same default-to-
bedrock-unless-set behavior as ResearchAgentGraph() everywhere else).

Gated behind its OWN env var, deliberately separate from
RUN_LIVE_BEDROCK_TESTS — a ddgs outage and a Bedrock/Groq outage are
different failure modes worth being able to isolate (see
edit-research-agent skill, "Test coverage — and the one deliberate gap",
and write-compliance-tests' file-ownership table). This was a
deliberate v1 scope cut when the Research agent first landed; this file
closes it. Run explicitly with:

    RUN_LIVE_RESEARCH_TESTS=1 python3 -m pytest tests/test_live_research.py -v
    LLM_PROVIDER=groq RUN_LIVE_RESEARCH_TESTS=1 python3 -m pytest tests/test_live_research.py -v

Real cost: one free ddgs call for the search-only test; one real,
billed Bedrock call (or free Groq call, if LLM_PROVIDER=groq) plus one
free ddgs call for the end-to-end test. Needs GROQ_API_KEY sourced
first if testing the Groq path (nothing auto-loads .env).
"""
import os

import pytest

from research_agent import ResearchAgentGraph
from web_search import WebSearcher

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_RESEARCH_TESTS") != "1",
    reason="live Research-agent test disabled by default (real external calls); "
    "set RUN_LIVE_RESEARCH_TESTS=1 to run it",
)


def test_live_web_search_returns_real_shaped_results():
    searcher = WebSearcher()
    results = searcher.search("Python programming language", max_results=3)

    assert results
    for r in results:
        assert r["title"]
        assert r["url"].startswith("http")
        assert r["snippet"]


def test_live_query_only_produces_a_real_synthesized_answer():
    agent = ResearchAgentGraph()  # real WebSearcher, real LLM (Bedrock or Groq per LLM_PROVIDER)
    result = agent.handle(query="Kubernetes")

    assert result.topics == ["Kubernetes"]
    assert result.search_results["Kubernetes"]
    assert result.answer
    assert "unverified" in result.answer.lower() or "not been fact" in result.answer.lower()
    assert any("ResearchSummarizer" in line and "generated via" in line for line in result.trace)
