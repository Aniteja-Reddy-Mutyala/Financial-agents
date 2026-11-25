"""Tests for GroqSummarizer — mocked, no real Groq API calls (no cost).

We bypass the lazy `client` property by setting `_client` directly to a
fake object, mirroring test_bedrock_llm.py's pattern for BedrockSummarizer.
"""
import pytest

from groq_llm import GroqSummarizer, GroqUnavailable


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, reply="A fake grounded answer citing 31 CFR 1020.320."):
        self.reply = reply
        self.received_kwargs = None

    def create(self, **kwargs):
        self.received_kwargs = kwargs
        return _FakeResponse(self.reply)


class _FakeChat:
    def __init__(self, reply):
        self.completions = _FakeCompletions(reply)


class _FakeGroqClient:
    def __init__(self, reply="A fake grounded answer citing 31 CFR 1020.320."):
        self.chat = _FakeChat(reply)


class _FailingCompletions:
    def create(self, **kwargs):
        raise RuntimeError("simulated Groq throttling / access-denied error")


class _FailingGroqClient:
    def __init__(self):
        self.chat = type("_C", (), {"completions": _FailingCompletions()})()


def _summarizer_with_fake_client(fake_client):
    s = GroqSummarizer(api_key="fake-key-not-used")
    s._client = fake_client  # bypass lazy Groq() construction
    return s


EVIDENCE = [
    {
        "source_id": "31 CFR 1020.320",
        "title": "Reports by banks of suspicious transactions.",
        "text": "A suspicious transaction shall be reported by completing a SAR.",
    }
]


def test_summarize_returns_stripped_model_content():
    fake = _FakeGroqClient(reply="  Yes, file a SAR.  \n")
    summarizer = _summarizer_with_fake_client(fake)
    result = summarizer.summarize(
        query="do we need to file a SAR?",
        intent_description="SAR filing obligations",
        evidence=EVIDENCE,
    )
    assert result == "Yes, file a SAR."


def test_summarize_passes_system_and_user_messages():
    fake = _FakeGroqClient()
    summarizer = _summarizer_with_fake_client(fake)
    summarizer.summarize(
        query="do we need to file a SAR?",
        intent_description="SAR filing obligations",
        evidence=EVIDENCE,
    )
    kwargs = fake.chat.completions.received_kwargs
    assert kwargs is not None
    messages = kwargs["messages"]
    assert len(messages) == 2
    system_msg, user_msg = messages
    assert system_msg["role"] == "system"
    assert "regulatory compliance assistant" in system_msg["content"]
    assert user_msg["role"] == "user"
    assert "do we need to file a SAR?" in user_msg["content"]
    assert "31 CFR 1020.320" in user_msg["content"]


def test_summarize_raises_groq_unavailable_on_failure():
    summarizer = _summarizer_with_fake_client(_FailingGroqClient())
    with pytest.raises(GroqUnavailable):
        summarizer.summarize(
            query="do we need to file a SAR?",
            intent_description="SAR filing obligations",
            evidence=EVIDENCE,
        )


def test_summarize_raises_groq_unavailable_on_empty_content():
    # Regression guard for the real gpt-oss failure mode documented in
    # eval/llm_judge.py: a reasoning model can burn its whole max_tokens
    # budget on hidden reasoning and return empty visible content. That
    # must be treated as a provider failure (triggering the template
    # fallback in agents.py), not returned as a blank "answer".
    fake = _FakeGroqClient(reply="")
    summarizer = _summarizer_with_fake_client(fake)
    with pytest.raises(GroqUnavailable):
        summarizer.summarize(
            query="do we need to file a SAR?",
            intent_description="SAR filing obligations",
            evidence=EVIDENCE,
        )


def test_summarize_with_empty_evidence_does_not_crash():
    fake = _FakeGroqClient(reply="No relevant regulation found.")
    summarizer = _summarizer_with_fake_client(fake)
    result = summarizer.summarize(
        query="unrelated question", intent_description="unknown", evidence=[]
    )
    assert result == "No relevant regulation found."


def test_chitchat_returns_stripped_model_content():
    fake = _FakeGroqClient(reply="  Hi! I'm a compliance assistant.  \n")
    summarizer = _summarizer_with_fake_client(fake)
    result = summarizer.chitchat("hi")
    assert result == "Hi! I'm a compliance assistant."


def test_chitchat_passes_chitchat_system_prompt_not_the_compliance_one():
    fake = _FakeGroqClient()
    summarizer = _summarizer_with_fake_client(fake)
    summarizer.chitchat("tell me about yourself")
    messages = fake.chat.completions.received_kwargs["messages"]
    system_msg, user_msg = messages
    assert "conversational front end" in system_msg["content"]
    assert "Answer ONLY using the provided regulatory excerpts" not in system_msg["content"]
    assert "tell me about yourself" in user_msg["content"]


def test_chitchat_raises_groq_unavailable_on_failure():
    summarizer = _summarizer_with_fake_client(_FailingGroqClient())
    with pytest.raises(GroqUnavailable):
        summarizer.chitchat("hi")


def test_chitchat_raises_groq_unavailable_on_empty_content():
    fake = _FakeGroqClient(reply="")
    summarizer = _summarizer_with_fake_client(fake)
    with pytest.raises(GroqUnavailable):
        summarizer.chitchat("hi")


def test_client_property_raises_groq_unavailable_with_no_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    summarizer = GroqSummarizer(api_key=None)
    with pytest.raises(GroqUnavailable, match="GROQ_API_KEY"):
        summarizer.client


def test_default_model_id_matches_the_proven_judge_model():
    # Regression guard: openai/gpt-oss-20b is the model already verified
    # to work against this project's Groq account (see eval/llm_judge.py
    # and run-evaluation-suite skill) — don't drift to an unverified model
    # as the default without re-testing, the same way several other
    # candidate models failed outright during that investigation.
    assert GroqSummarizer().model_id == "openai/gpt-oss-20b"
