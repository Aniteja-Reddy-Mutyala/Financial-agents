"""langchain_aws wrapper around Amazon Bedrock, using a cheap Nova model.

Stands in for the "hybrid LLM inference" (Claude 3.5/4, Claude Instant v1)
part of the original project, routed through AWS Bedrock via langchain_aws
(the same package named on the resume) since that's what's available here.
Defaults to Amazon Nova Micro — the cheapest Amazon text model on Bedrock —
and raises BedrockUnavailable on any failure so callers can fall back
gracefully instead of hard-failing.
"""
import os

from langchain_core.messages import HumanMessage, SystemMessage

DEFAULT_MODEL_ID = "amazon.nova-micro-v1:0"
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")

SYSTEM_PROMPT = (
    "You are a regulatory compliance assistant covering BSA/AML (31 CFR), "
    "FDA electronic records (21 CFR Part 11), HIPAA (45 CFR), and SOX-related "
    "SEC rules (17 CFR). Answer ONLY using the provided regulatory excerpts. "
    "Be concise and precise, cite the exact section numbers given in the "
    "excerpts (e.g. 31 CFR 1020.320, 45 CFR 164.502), and never state an "
    "obligation that isn't supported by the excerpts. If the excerpts don't "
    "fully answer the question, say what's missing."
)


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
        excerpt_block = "\n\n".join(
            f"{e['source_id']} — {e['title']}\n{e['text']}" for e in evidence
        )
        user_message = (
            f"Compliance question: {query}\n"
            f"Detected intent: {intent_description}\n\n"
            f"Regulatory excerpts:\n{excerpt_block}\n\n"
            "Write a short answer (3-6 sentences) for a compliance analyst, "
            "citing the specific regulatory sections used (as given above)."
        )
        try:
            response = self.llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
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
