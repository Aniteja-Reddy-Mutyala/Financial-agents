"""Tests for BedrockSummarizer — mocked, no real AWS/Bedrock calls (no cost).

We bypass the lazy `llm` property by setting `_llm` directly to a fake
object, so these tests never import boto3/langchain_aws's network path.
"""
import pytest

from bedrock_llm import BedrockSummarizer, BedrockUnavailable


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeChatModel:
    def __init__(self, reply="A fake grounded answer citing 31 CFR 1020.320."):
        self.reply = reply
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return _FakeResponse(self.reply)


class _FailingChatModel:
    def invoke(self, messages):
        raise RuntimeError("simulated Bedrock throttling / access-denied error")


def _summarizer_with_fake_llm(fake_llm):
    s = BedrockSummarizer()
    s._llm = fake_llm  # bypass lazy ChatBedrockConverse construction
    return s


EVIDENCE = [
    {
        "source_id": "31 CFR 1020.320",
        "title": "Reports by banks of suspicious transactions.",
        "text": "A suspicious transaction shall be reported by completing a SAR.",
    }
]


def test_summarize_returns_stripped_model_content():
    fake = _FakeChatModel(reply="  Yes, file a SAR.  \n")
    summarizer = _summarizer_with_fake_llm(fake)
    result = summarizer.summarize(
        query="do we need to file a SAR?",
        intent_description="SAR filing obligations",
        evidence=EVIDENCE,
    )
    assert result == "Yes, file a SAR."


def test_summarize_passes_system_and_human_messages():
    fake = _FakeChatModel()
    summarizer = _summarizer_with_fake_llm(fake)
    summarizer.summarize(
        query="do we need to file a SAR?",
        intent_description="SAR filing obligations",
        evidence=EVIDENCE,
    )
    assert fake.received_messages is not None
    assert len(fake.received_messages) == 2
    system_msg, human_msg = fake.received_messages
    assert "regulatory compliance assistant" in system_msg.content
    assert "do we need to file a SAR?" in human_msg.content
    assert "31 CFR 1020.320" in human_msg.content


def test_summarize_raises_bedrock_unavailable_on_failure():
    summarizer = _summarizer_with_fake_llm(_FailingChatModel())
    with pytest.raises(BedrockUnavailable):
        summarizer.summarize(
            query="do we need to file a SAR?",
            intent_description="SAR filing obligations",
            evidence=EVIDENCE,
        )


def test_summarize_with_empty_evidence_does_not_crash():
    fake = _FakeChatModel(reply="No relevant regulation found.")
    summarizer = _summarizer_with_fake_llm(fake)
    result = summarizer.summarize(
        query="unrelated question", intent_description="unknown", evidence=[]
    )
    assert result == "No relevant regulation found."


def test_respond_ungrounded_returns_stripped_model_content():
    fake = _FakeChatModel(reply="  Hi! I'm a compliance assistant.  \n")
    summarizer = _summarizer_with_fake_llm(fake)
    result = summarizer.respond_ungrounded("hi")
    assert result == "Hi! I'm a compliance assistant."


def test_respond_ungrounded_passes_ungrounded_system_prompt_not_the_compliance_one():
    fake = _FakeChatModel()
    summarizer = _summarizer_with_fake_llm(fake)
    summarizer.respond_ungrounded("tell me about yourself")
    assert fake.received_messages is not None
    system_msg, human_msg = fake.received_messages
    # Must use the ungrounded prompt, not SYSTEM_PROMPT (which demands
    # answering only from regulatory excerpts — wrong when there are none).
    assert "conversational front end" in system_msg.content
    assert "Answer ONLY using the provided regulatory excerpts" not in system_msg.content
    assert "tell me about yourself" in human_msg.content


def test_respond_ungrounded_raises_bedrock_unavailable_on_failure():
    summarizer = _summarizer_with_fake_llm(_FailingChatModel())
    with pytest.raises(BedrockUnavailable):
        summarizer.respond_ungrounded("hi")


def test_default_model_id_is_cheapest_nova_model():
    # Regression guard: this project is specifically meant to use the
    # cheapest Amazon model. If this ever changes it should be a deliberate
    # decision (see bedrock-llm-integration skill), not an accident.
    assert BedrockSummarizer().model_id == "amazon.nova-micro-v1:0"
