"""langchain_aws wrapper around Amazon Bedrock, using a cheap Nova model.

Stands in for the "hybrid LLM inference" (Claude 3.5/4, Claude Instant v1)
part of the original project, routed through AWS Bedrock via langchain_aws
(the same package named on the resume) since that's what's available here.
Defaults to Amazon Nova Micro — the cheapest Amazon text model on Bedrock —
and raises BedrockUnavailable on any failure so callers can fall back
gracefully instead of hard-failing.

One of two interchangeable Summarizer providers — see groq_llm.py for the
other, and agents.py's ComplianceAgentGraph for how a provider is chosen.
Both share the exact same prompt text (llm_prompts.py) so switching
providers changes cost/latency/model quality, not what's being asked.

Also used (via the generic .chat() method below) by the separate
Research agent (research_agent.py) — see research_prompts.py. .chat()
is deliberately generic (caller supplies both prompts) rather than
compliance-specific like .summarize()/.respond_ungrounded(), so it can
be shared across agents without either one depending on the other's
prompt file.
"""
import os

from langchain_core.messages import HumanMessage, SystemMessage

from llm_prompts import (
    SYSTEM_PROMPT,
    UNGROUNDED_SYSTEM_PROMPT,
    build_ungrounded_user_message,
    build_user_message,
)

DEFAULT_MODEL_ID = "amazon.nova-micro-v1:0"
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")


class BedrockUnavailable(Exception):
    pass


class BedrockSummarizer:
    """Thin facade over langchain_aws.ChatBedrockConverse."""

    def __init__(self, model_id=DEFAULT_MODEL_ID, region=DEFAULT_REGION, temperature=0.2):
        self.model_id = model_id
        self.region = region
        self.temperature = temperature
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from langchain_aws import ChatBedrockConverse  # lazy import

            self._llm = ChatBedrockConverse(
                model=self.model_id,
                region_name=self.region,
                temperature=self.temperature,
                max_tokens=400,
            )
        return self._llm

    def summarize(self, query, intent_description, evidence):
        """evidence: list of {source_id, title, text} dicts already filtered to citations."""
        user_message = build_user_message(query, intent_description, evidence)
        try:
            response = self.llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
            )
            return response.content.strip()
        except Exception as exc:  # noqa: BLE001 - want to degrade gracefully either way
            raise BedrockUnavailable(str(exc)) from exc

    def respond_ungrounded(self, query):
        """A reply for a query retrieval did NOT ground — see escalate_node
        in agents.py and llm_prompts.py's UNGROUNDED_SYSTEM_PROMPT. Never
        passed any retrieved evidence, so it cannot cite a hallucinated
        regulation no matter what it says — that's enforced by this method
        simply having none to hand it, not by the prompt's wording alone."""
        user_message = build_ungrounded_user_message(query)
        try:
            response = self.llm.invoke(
                [SystemMessage(content=UNGROUNDED_SYSTEM_PROMPT), HumanMessage(content=user_message)]
            )
            return response.content.strip()
        except Exception as exc:  # noqa: BLE001 - want to degrade gracefully either way
            raise BedrockUnavailable(str(exc)) from exc

    def chat(self, system_prompt, user_message):
        """Generic LLM call for callers outside the compliance prompt
        contract (currently: the Research agent) — caller supplies both
        prompts directly instead of this method building them from
        llm_prompts.py."""
        try:
            response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            )
            return response.content.strip()
        except Exception as exc:  # noqa: BLE001 - want to degrade gracefully either way
            raise BedrockUnavailable(str(exc)) from exc


if __name__ == "__main__":
    llm = BedrockSummarizer()
    print(
        llm.summarize(
            query="do we need to file a SAR for this wire transfer?",
            intent_description="Suspicious Activity Report (SAR) filing obligations",
            evidence=[
                {
                    "source_id": "31 CFR 1020.320",
                    "title": "Reports by banks of suspicious transactions.",
                    "text": "A suspicious transaction shall be reported by completing a "
                    "Suspicious Activity Report (SAR)...",
                }
            ],
        )
    )
