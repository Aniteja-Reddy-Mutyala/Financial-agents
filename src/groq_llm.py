"""Groq wrapper around an open-weight model, as an alternative Summarizer
provider to bedrock_llm.py's BedrockSummarizer — same interface
(`.model_id`, `.summarize(query, intent_description, evidence)`, raises a
typed *Unavailable exception on any failure), so agents.py's
ComplianceAgentGraph can use either one interchangeably. See agents.py
for how a provider is selected (LLM_PROVIDER env var or an explicit
`llm=` override) and the run-evaluation-suite skill for why
`openai/gpt-oss-20b` was already proven to work against this project's
Groq account (it's also eval/llm_judge.py's judge model).

Needs the `groq` package (`pip install groq` — not in requirements.txt;
this provider is opt-in, only needed if you actually select it) and a
Groq API key: set GROQ_API_KEY (see .env.example). Both providers share
the exact same prompt text (llm_prompts.py) — switching providers trades
cost/latency/model quality, not what's being asked.

Cost/latency note: unlike Bedrock Nova Micro (a small, fast, dedicated
text model), `gpt-oss-20b` is a reasoning model that spends a variable,
sometimes large, share of its token budget on internal reasoning before
the visible answer (measured in eval/llm_judge.py: up to 271 of 310
completion tokens on a single call). max_tokens here is set generously
for the same reason max_tokens=800 was needed for the judge pass — too
low and you get an empty response, not a truncated one.
"""
import os

from llm_prompts import SYSTEM_PROMPT, build_user_message

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class GroqUnavailable(Exception):
    pass


class GroqSummarizer:
    """Thin facade over the groq SDK's chat completions endpoint."""

    def __init__(self, model_id=DEFAULT_MODEL_ID, api_key=None, temperature=0.2):
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from groq import Groq  # lazy import

            if not self.api_key:
                raise GroqUnavailable(
                    "No Groq API key found — set the GROQ_API_KEY environment variable."
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    def summarize(self, query, intent_description, evidence):
        """evidence: list of {source_id, title, text} dicts already filtered to citations."""
        user_message = build_user_message(query, intent_description, evidence)
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=800,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                # gpt-oss burned its whole budget on hidden reasoning —
                # same failure mode documented in eval/llm_judge.py.
                raise GroqUnavailable(
                    f"{self.model_id} returned empty content (likely exhausted "
                    "max_tokens on internal reasoning before any visible output)"
                )
            return content.strip()
        except GroqUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - want to degrade gracefully either way
            raise GroqUnavailable(str(exc)) from exc


if __name__ == "__main__":
    llm = GroqSummarizer()
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
