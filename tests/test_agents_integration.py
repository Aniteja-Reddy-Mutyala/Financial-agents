"""Full-graph integration tests with a fake LLM and a fake retriever/embedder
(no AWS calls, no cost) — these test the graph's PLUMBING (routing, citation
passthrough, Bedrock-failure fallback, crash-resistance), not real domain
relevance. Real relevance (e.g. "does this query actually find 31 CFR
1010.311 out of the real KB") requires a real Bedrock embedding call, so
those checks live in tests/test_live_rag.py instead, gated the same way as
test_live_bedrock.py.
"""
import pytest

from agents import ComplianceAgentGraph, _default_llm
from bedrock_llm import BedrockSummarizer
from groq_llm import GroqSummarizer
from retriever import Chunk, Retriever


class _FakeSummarizer:
    """Stands in for BedrockSummarizer — same interface, no network."""

    model_id = "fake-test-model"

    def __init__(self, reply="This is a fake generated answer."):
        self.reply = reply
        self.calls = []

    def summarize(self, query, intent_description, evidence):
        self.calls.append({"query": query, "intent_description": intent_description, "evidence": evidence})
        return self.reply


class _EmptyRetriever:
    """Always returns zero hits, forcing the verifier to deny grounding."""

    def search(self, query, top_k=3):
        return []


class _FixedVectorEmbedder:
    """Returns the same vector for every query, no matter the text — used
    only to deterministically drive a synthetic Retriever's scoring, never
    to assert anything about real semantic relevance."""

    def __init__(self, vector):
        self.vector = vector

    def embed_query(self, text):
        return self.vector


@pytest.fixture(scope="module")
def grounded_retriever():
    """A tiny synthetic 2-section corpus whose single chunk always scores
    well above MIN_SCORE, so every non-empty query deterministically takes
    the grounded/summarize branch — for testing that branch's plumbing
    without a real (billed) embedding call or depending on real relevance."""
    chunks = [
        Chunk(chunk_id="31 CFR 1020.320#0", source_id="31 CFR 1020.320",
              title="Reports by banks of suspicious transactions.",
              text="A suspicious transaction shall be reported by completing a SAR."),
        Chunk(chunk_id="31 CFR 1010.311#0", source_id="31 CFR 1010.311",
              title="Filing obligations.",
              text="A currency transaction report shall be filed for transactions over $10,000."),
    ]
    import numpy as np

    matrix = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return Retriever(chunks, matrix, _FixedVectorEmbedder([1.0, 0.0]))


# --- Grounded (summarize) branch --------------------------------------------


def test_grounded_query_calls_llm_and_includes_citations(grounded_retriever):
    fake_llm = _FakeSummarizer(reply="Yes, a SAR is required here.")
    agent = ComplianceAgentGraph(llm=fake_llm, retriever=grounded_retriever)

    result = agent.handle("do we need to file a SAR for a structured cash deposit pattern")

    assert result.intent["intent"] == "sar_filing"
    assert result.verification["grounded"] is True
    assert len(fake_llm.calls) == 1  # LLM was actually invoked
    assert "Yes, a SAR is required here." in result.answer
    assert "Citations:" in result.answer
    assert any(c in result.answer for c in result.verification["citations"])
    assert any("[Orchestrator]" in line for line in result.trace)
    assert any("[Planner]" in line for line in result.trace)
    assert any("[Retriever]" in line for line in result.trace)
    assert any("[ComplianceVerifier]" in line for line in result.trace)
    assert any("generated via Bedrock" in line or "Summarizer" in line for line in result.trace)


def test_summarizer_receives_only_cited_evidence(grounded_retriever):
    fake_llm = _FakeSummarizer()
    agent = ComplianceAgentGraph(llm=fake_llm, retriever=grounded_retriever)
    result = agent.handle("what is the CTR threshold for a cash deposit")

    passed_ids = {e["source_id"] for e in fake_llm.calls[0]["evidence"]}
    assert passed_ids == set(result.verification["citations"])


# --- Not-grounded (escalate) branch -----------------------------------------


def test_ungrounded_query_skips_llm_and_escalates():
    fake_llm = _FakeSummarizer()
    agent = ComplianceAgentGraph(llm=fake_llm, retriever=_EmptyRetriever())

    result = agent.handle("do we need to file a SAR for this wire transfer")

    assert result.verification["grounded"] is False
    assert fake_llm.calls == []  # LLM must NOT be called when ungrounded
    assert "⚠️" in result.answer
    assert "human compliance officer" in result.answer
    assert "Citations:" not in result.answer


def test_bedrock_failure_falls_back_to_template_not_crash(grounded_retriever):
    class _FailingSummarizer:
        model_id = "fake-failing-model"

        def summarize(self, **kwargs):
            from bedrock_llm import BedrockUnavailable

            raise BedrockUnavailable("simulated outage")

    agent = ComplianceAgentGraph(llm=_FailingSummarizer(), retriever=grounded_retriever)
    result = agent.handle("do we need to file a SAR for a structured cash deposit pattern")

    # Still grounded (retrieval succeeded) but LLM failed -> template fallback,
    # not a crash and not a silent empty answer.
    assert result.verification["grounded"] is True
    assert "Relevant regulatory text" in result.answer
    assert "Citations:" in result.answer
    assert any("LLM unavailable" in line for line in result.trace)


# --- Edge-case queries through the whole pipeline ---------------------------


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "???",
        "a" * 2000,
        "¿Necesitamos presentar un SAR?",
        "what's the best pizza topping in Chicago",  # off-topic
    ],
)
def test_edge_case_queries_do_not_crash(grounded_retriever, query):
    fake_llm = _FakeSummarizer()
    agent = ComplianceAgentGraph(llm=fake_llm, retriever=grounded_retriever)
    result = agent.handle(query)
    assert isinstance(result.answer, str) and result.answer != ""
    assert isinstance(result.trace, list) and len(result.trace) >= 1


# --- _default_llm: LLM_PROVIDER selection (model-agnostic summarizer) ------
# Constructing BedrockSummarizer()/GroqSummarizer() makes no network call
# (both lazy-load their client on first real use), so these are free and
# safe to run without credentials.


def test_default_llm_is_bedrock_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert isinstance(_default_llm(), BedrockSummarizer)


def test_default_llm_is_bedrock_explicitly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    assert isinstance(_default_llm(), BedrockSummarizer)


def test_default_llm_is_groq_when_selected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert isinstance(_default_llm(), GroqSummarizer)


def test_default_llm_provider_selection_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "GROQ")
    assert isinstance(_default_llm(), GroqSummarizer)


def test_default_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="openai"):
        _default_llm()


def test_default_llm_is_read_fresh_not_cached_at_import(monkeypatch):
    # Regression guard: an earlier version of this selection logic read
    # LLM_PROVIDER into a module-level constant at import time, so
    # switching it mid-process (e.g. between two ComplianceAgentGraph()
    # instances, or across tests) silently had no effect.
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    assert isinstance(_default_llm(), BedrockSummarizer)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert isinstance(_default_llm(), GroqSummarizer)


def test_compliance_agent_graph_uses_default_llm_when_none_passed(monkeypatch, grounded_retriever):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    agent = ComplianceAgentGraph(retriever=grounded_retriever)
    assert isinstance(agent.llm, GroqSummarizer)


def test_compliance_agent_graph_explicit_llm_overrides_provider_env(monkeypatch, grounded_retriever):
    # An explicit llm= always wins, regardless of LLM_PROVIDER.
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    fake_llm = _FakeSummarizer()
    agent = ComplianceAgentGraph(llm=fake_llm, retriever=grounded_retriever)
    assert agent.llm is fake_llm
