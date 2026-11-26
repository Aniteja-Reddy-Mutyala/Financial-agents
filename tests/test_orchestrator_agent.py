"""Full-graph integration tests for OrchestratorGraph (the top-level
Orchestrator -> Planner -> domain agents -> Orchestrator flow in
src/orchestrator_agent.py) with a fake LLM and fake domain agents — no
real AWS/LLM/network calls. Tests the graph's PLUMBING (plan parsing,
the single-agent-dispatch-no-extra-LLM-call shortcut, multi-agent
composition, fallback on planner/composer LLM failure), not real
planning quality. Mirrors test_agents_integration.py/test_research_agent.py's
pattern.
"""
import pytest

from bedrock_llm import BedrockUnavailable
from orchestrator_agent import DEFAULT_AGENT, OrchestratorGraph, _default_llm, _parse_plan


class _FakeChatSummarizer:
    """Stands in for BedrockSummarizer/GroqSummarizer — same .chat()
    interface, no network."""

    model_id = "fake-test-model"

    def __init__(self, replies=None):
        self.replies = replies
        self.calls = []
        self._call_index = 0

    def chat(self, system_prompt, user_message):
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        if isinstance(self.replies, list):
            reply = self.replies[self._call_index]
            self._call_index += 1
            return reply
        return self.replies or "fake plan or composition"


class _FailingChatSummarizer:
    model_id = "fake-test-model"

    def __init__(self, exc):
        self.exc = exc
        self.calls = []

    def chat(self, system_prompt, user_message):
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        raise self.exc


class _FakeDomainAgent:
    """Stands in for ComplianceAgentGraph/ResearchAgentGraph — same
    .handle(query) -> SimpleNamespace(answer=..., trace=...) interface."""

    def __init__(self, answer="fake agent answer"):
        self.answer = answer
        self.calls = []

    def handle(self, query):
        from types import SimpleNamespace

        self.calls.append(query)
        return SimpleNamespace(answer=self.answer, trace=[f"handled '{query}'"])


# --- _parse_plan --------------------------------------------------------


def test_parse_plan_extracts_known_agents_only():
    raw = "compliance: do we need a SAR\nresearch: Kubernetes security\nbogus: ignored"
    subtasks = _parse_plan(raw, {"compliance", "research"})
    assert subtasks == [
        {"agent": "compliance", "task": "do we need a SAR"},
        {"agent": "research", "task": "Kubernetes security"},
    ]


def test_parse_plan_is_case_insensitive_on_agent_name():
    subtasks = _parse_plan("COMPLIANCE: do we need a SAR", {"compliance"})
    assert subtasks == [{"agent": "compliance", "task": "do we need a SAR"}]


def test_parse_plan_skips_lines_with_no_colon_or_empty_task():
    subtasks = _parse_plan("not a valid line\ncompliance: ", {"compliance"})
    assert subtasks == []


# --- Single-agent dispatch ------------------------------------------------


def test_single_agent_plan_forwards_its_answer_with_no_composer_llm_call():
    fake_llm = _FakeChatSummarizer(replies="compliance: do we need a SAR")
    fake_compliance = _FakeDomainAgent(answer="You need to file a SAR because...")
    fake_research = _FakeDomainAgent(answer="should not be called")
    orch = OrchestratorGraph(llm=fake_llm, agents={"compliance": fake_compliance, "research": fake_research})

    result = orch.handle("do we need a SAR")

    assert fake_compliance.calls == ["do we need a SAR"]
    assert fake_research.calls == []
    # Only the planner call — compose_node must skip the LLM entirely
    # when there's exactly one subtask result.
    assert len(fake_llm.calls) == 1
    assert result.answer == "You need to file a SAR because..."
    assert any("single agent (compliance)" in line for line in result.trace)


# --- Multi-agent dispatch + composition -----------------------------------


def test_multi_agent_plan_dispatches_to_each_and_composes_via_llm():
    fake_llm = _FakeChatSummarizer(replies=[
        "compliance: do we need a SAR\nresearch: Kubernetes security",
        "A single composed answer covering both.",
    ])
    fake_compliance = _FakeDomainAgent(answer="SAR answer")
    fake_research = _FakeDomainAgent(answer="Kubernetes answer")
    orch = OrchestratorGraph(llm=fake_llm, agents={"compliance": fake_compliance, "research": fake_research})

    result = orch.handle("do we need a SAR, and look up Kubernetes security")

    assert fake_compliance.calls == ["do we need a SAR"]
    assert fake_research.calls == ["Kubernetes security"]
    assert len(fake_llm.calls) == 2
    assert result.answer == "A single composed answer covering both."
    assert any("dispatched to compliance" in line for line in result.trace)
    assert any("dispatched to research" in line for line in result.trace)


def test_composer_llm_failure_falls_back_to_concatenating_agent_answers():
    fake_llm = _FakeChatSummarizer(replies="compliance: a\nresearch: b")
    # Swap the second .chat() call (compose) to fail — simplest is a
    # dedicated fake whose first call succeeds then raises.
    class _PlanThenFailSummarizer:
        model_id = "fake-test-model"
        calls = []

        def chat(self, system_prompt, user_message):
            self.calls.append(user_message)
            if len(self.calls) == 1:
                return "compliance: a\nresearch: b"
            raise BedrockUnavailable("simulated outage")

    fake_llm = _PlanThenFailSummarizer()
    fake_compliance = _FakeDomainAgent(answer="Compliance answer")
    fake_research = _FakeDomainAgent(answer="Research answer")
    orch = OrchestratorGraph(llm=fake_llm, agents={"compliance": fake_compliance, "research": fake_research})

    result = orch.handle("a and b")

    assert "Compliance answer" in result.answer
    assert "Research answer" in result.answer
    assert any("LLM unavailable" in line for line in result.trace)


# --- Planner fallback ------------------------------------------------------


def test_planner_llm_failure_falls_back_to_default_agent_with_full_query():
    fake_llm = _FailingChatSummarizer(exc=BedrockUnavailable("simulated outage"))
    fake_compliance = _FakeDomainAgent(answer="Compliance answer")
    fake_research = _FakeDomainAgent(answer="should not be called")
    orch = OrchestratorGraph(llm=fake_llm, agents={"compliance": fake_compliance, "research": fake_research})

    result = orch.handle("some ambiguous query")

    assert fake_compliance.calls == ["some ambiguous query"]
    assert fake_research.calls == []
    assert result.answer == "Compliance answer"


def test_planner_output_that_parses_to_no_known_agent_falls_back_to_default_agent():
    fake_llm = _FakeChatSummarizer(replies="not a valid plan line at all")
    fake_compliance = _FakeDomainAgent(answer="Compliance answer")
    orch = OrchestratorGraph(llm=fake_llm, agents={"compliance": fake_compliance, "research": _FakeDomainAgent()})

    result = orch.handle("some query")

    assert fake_compliance.calls == ["some query"]
    assert result.answer == "Compliance answer"


def test_default_agent_is_compliance():
    assert DEFAULT_AGENT == "compliance"


# --- Default agents get the orchestrator's own llm, not their own _default_llm() ---


def test_default_agents_are_built_with_the_orchestrators_llm(monkeypatch):
    # Regression test for a real bug caught by manually testing the
    # Streamlit page: the default compliance/research agents must be
    # constructed with this orchestrator's own `llm`, not left to call
    # their own _default_llm() (which reads LLM_PROVIDER from the
    # process env and would silently ignore a `llm=` override like the
    # UI toggle). Monkeypatches the two classes orchestrator_agent.py
    # imports them as, so no real Retriever/AWS client gets built.
    import orchestrator_agent

    captured = {}

    class _FakeComplianceGraph:
        def __init__(self, llm=None):
            captured["compliance_llm"] = llm

    class _FakeResearchGraph:
        def __init__(self, llm=None):
            captured["research_llm"] = llm

    monkeypatch.setattr(orchestrator_agent, "ComplianceAgentGraph", _FakeComplianceGraph)
    monkeypatch.setattr(orchestrator_agent, "ResearchAgentGraph", _FakeResearchGraph)

    fake_llm = _FakeChatSummarizer()
    orchestrator_agent.OrchestratorGraph(llm=fake_llm)

    assert captured["compliance_llm"] is fake_llm
    assert captured["research_llm"] is fake_llm


# --- Provider selection (mirrors test_research_agent.py's _default_llm coverage) ---


def test_default_llm_is_bedrock_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from bedrock_llm import BedrockSummarizer

    assert isinstance(_default_llm(), BedrockSummarizer)


def test_default_llm_is_groq_when_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    from groq_llm import GroqSummarizer

    assert isinstance(_default_llm(), GroqSummarizer)


def test_default_llm_raises_on_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "something-else")
    with pytest.raises(ValueError):
        _default_llm()
