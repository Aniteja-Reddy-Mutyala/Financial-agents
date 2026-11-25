"""Full-graph integration tests for ResearchAgentGraph with a fake LLM and
a fake web searcher (no network calls, no cost) — tests the graph's
PLUMBING (intake routing, topic capping, per-topic search-failure
isolation, all-empty fallback, LLM-failure fallback), not real web
relevance. Mirrors test_agents_integration.py's pattern.
"""
import pytest

from research_agent import ResearchAgentGraph, _default_llm
from web_search import WebSearchUnavailable


class _FakeChatSummarizer:
    """Stands in for BedrockSummarizer/GroqSummarizer — same .chat()
    interface, no network."""

    model_id = "fake-test-model"

    def __init__(self, replies=None):
        # replies: list of strings returned in call order, or a single
        # string reused for every call.
        self.replies = replies
        self.calls = []
        self._call_index = 0

    def chat(self, system_prompt, user_message):
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        if isinstance(self.replies, list):
            reply = self.replies[self._call_index]
            self._call_index += 1
            return reply
        return self.replies or "fake synthesized answer"


class _FailingChatSummarizer:
    model_id = "fake-test-model"

    def __init__(self, exc):
        self.exc = exc
        self.calls = []

    def chat(self, system_prompt, user_message):
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        raise self.exc


class _FakeWebSearcher:
    """Returns scripted results per topic; records every call."""

    def __init__(self, results_by_topic=None, fail_for=None):
        self.results_by_topic = results_by_topic or {}
        self.fail_for = fail_for or set()
        self.calls = []

    def search(self, query, max_results=None):
        self.calls.append(query)
        if query in self.fail_for:
            raise WebSearchUnavailable(f"simulated failure for {query}")
        return self.results_by_topic.get(query, [])


class _EmptySearcher:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results=None):
        self.calls.append(query)
        return []


SAMPLE_RESULTS = [{"title": "Kubernetes docs", "url": "https://kubernetes.io/docs", "snippet": "What is K8s"}]


# --- Query-only -------------------------------------------------------------


def test_query_only_skips_extraction_llm_call_and_uses_query_as_the_one_topic():
    fake_llm = _FakeChatSummarizer(replies="A synthesized answer about Kubernetes.")
    fake_searcher = _FakeWebSearcher(results_by_topic={"Kubernetes": SAMPLE_RESULTS})
    agent = ResearchAgentGraph(llm=fake_llm, web_searcher=fake_searcher)

    result = agent.handle(query="Kubernetes")

    assert result.topics == ["Kubernetes"]
    # Only one .chat() call — the synthesis one; extraction was skipped.
    assert len(fake_llm.calls) == 1
    assert fake_searcher.calls == ["Kubernetes"]
    assert result.answer == "A synthesized answer about Kubernetes."
    assert any("ResearchPlanner" in line for line in result.trace)


def test_handle_with_neither_query_nor_document_raises_value_error():
    agent = ResearchAgentGraph(llm=_FakeChatSummarizer(), web_searcher=_EmptySearcher())
    with pytest.raises(ValueError):
        agent.handle()


# --- Document (+ query) chained case -----------------------------------------


def test_document_and_query_chained_extracts_and_caps_topics():
    # Extraction LLM call returns 4 topics; the graph must cap to 3.
    fake_llm = _FakeChatSummarizer(replies=[
        "Kubernetes\nTerraform\nAcme Corp\nExtra Topic",
        "A synthesized research report.",
    ])
    fake_searcher = _FakeWebSearcher(results_by_topic={
        "Kubernetes": SAMPLE_RESULTS, "Terraform": SAMPLE_RESULTS, "Acme Corp": SAMPLE_RESULTS,
    })

    def fake_parser(path):
        return "Senior engineer skilled in Kubernetes, Terraform. Previously at Acme Corp."

    agent = ResearchAgentGraph(llm=fake_llm, web_searcher=fake_searcher, document_parser=fake_parser)

    result = agent.handle(query="focus on cloud skills", document_path="resume.pdf")

    assert result.topics == ["Kubernetes", "Terraform", "Acme Corp"]
    assert len(fake_searcher.calls) == 3
    assert result.answer == "A synthesized research report."


def test_document_parse_failure_degrades_to_query_only():
    fake_llm = _FakeChatSummarizer(replies="An answer from query alone.")
    fake_searcher = _FakeWebSearcher(results_by_topic={"Kubernetes": SAMPLE_RESULTS})
    agent = ResearchAgentGraph(llm=fake_llm, web_searcher=fake_searcher)

    # document_path points at a real file that will fail to parse
    # (unsupported extension) — parse_document_node must catch this and
    # fall through using the query alone, not crash.
    result = agent.handle(query="Kubernetes", document_path="notes.docx")

    assert result.document_text == ""
    assert result.topics == ["Kubernetes"]
    assert any("document parse failed" in line for line in result.trace)


# --- Web research resilience -------------------------------------------------


def test_one_topics_search_failure_does_not_kill_the_others(tmp_path):
    doc = tmp_path / "resume.txt"
    doc.write_text("Experienced in TopicA and TopicB.")
    fake_llm = _FakeChatSummarizer(replies=[
        "TopicA\nTopicB",
        "A synthesized answer using only TopicB's results.",
    ])
    fake_searcher = _FakeWebSearcher(
        results_by_topic={"TopicB": SAMPLE_RESULTS}, fail_for={"TopicA"}
    )
    agent = ResearchAgentGraph(llm=fake_llm, web_searcher=fake_searcher)

    result = agent.handle(document_path=str(doc))

    assert set(fake_searcher.calls) == {"TopicA", "TopicB"}
    assert result.search_results["TopicA"] == []
    assert result.search_results["TopicB"] == SAMPLE_RESULTS
    assert result.answer == "A synthesized answer using only TopicB's results."
    assert any("web search failed for 'TopicA'" in line for line in result.trace)


def test_all_topics_empty_results_uses_template_fallback_without_llm_synthesis_call():
    fake_llm = _FakeChatSummarizer(replies=["Kubernetes"])
    fake_searcher = _EmptySearcher()
    agent = ResearchAgentGraph(llm=fake_llm, web_searcher=fake_searcher)

    result = agent.handle(query="Kubernetes")

    # Only the (skipped, query-only) extraction path ran no LLM call at
    # all; synthesize_node must also skip its LLM call when every topic
    # came back empty.
    assert len(fake_llm.calls) == 0
    assert "No web search results were found for: Kubernetes" in result.answer


def test_synthesis_llm_failure_falls_back_to_template_draft_with_raw_results():
    from bedrock_llm import BedrockUnavailable

    fake_searcher = _FakeWebSearcher(results_by_topic={"Kubernetes": SAMPLE_RESULTS})
    fake_llm = _FailingChatSummarizer(exc=BedrockUnavailable("simulated provider outage"))
    agent = ResearchAgentGraph(llm=fake_llm, web_searcher=fake_searcher)

    result = agent.handle(query="Kubernetes")

    # Results existed (search succeeded) but synthesis failed — the raw
    # results must still show up, not get silently dropped.
    assert "Kubernetes docs" in result.answer
    assert "https://kubernetes.io/docs" in result.answer
    assert "No web search results were found for" not in result.answer
    assert any("LLM unavailable" in line for line in result.trace)


# --- Provider selection (mirrors test_agents_integration.py's _default_llm coverage) ---


def test_default_llm_is_bedrock_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from bedrock_llm import BedrockSummarizer

    assert isinstance(_default_llm(), BedrockSummarizer)


def test_default_llm_is_groq_when_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    from groq_llm import GroqSummarizer

    assert isinstance(_default_llm(), GroqSummarizer)


def test_default_llm_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "GROQ")
    from groq_llm import GroqSummarizer

    assert isinstance(_default_llm(), GroqSummarizer)


def test_default_llm_raises_on_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "something-else")
    with pytest.raises(ValueError):
        _default_llm()
