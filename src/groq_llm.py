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
completion tokens on a single call). `reasoning_effort="low"` is passed
on every call here to cut down on that hidden token spend directly
(Groq's API exposes "none"/"low"/"medium"/"high" for this model) rather
than just buying more max_tokens headroom and hoping — this is the root-
cause fix; max_tokens=1200 and the finish_reason check below are the
safety net for whatever hidden-reasoning spend still happens even at low
effort. A visible-but-truncated answer (non-empty content,
`finish_reason="length"`) is the same underlying failure as an empty one
and is treated the same way: raised as GroqUnavailable rather than
silently returned mid-sentence.
"""
import os

from llm_prompts import (
    SYSTEM_PROMPT,
    UNGROUNDED_SYSTEM_PROMPT,
    build_ungrounded_user_message,
    build_user_message,
)

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
                max_tokens=1200,
                temperature=self.temperature,
                reasoning_effort="low",
            )
            choice = response.choices[0]
            content = choice.message.content
            if not content or not content.strip():
                # gpt-oss burned its whole budget on hidden reasoning —
                # same failure mode documented in eval/llm_judge.py.
                raise GroqUnavailable(
                    f"{self.model_id} returned empty content (likely exhausted "
                    "max_tokens on internal reasoning before any visible output)"
                )
            if choice.finish_reason == "length":
                # Same failure mode, just with some visible content before
                # the budget ran out — a sentence cut off mid-word is worse
                # than a clear fallback, so treat it as unavailable too.
                raise GroqUnavailable(
                    f"{self.model_id} hit max_tokens before finishing the answer "
                    "(finish_reason='length') — treating as unavailable rather "
                    "than returning a truncated answer"
                )
            return content.strip()
        except GroqUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - want to degrade gracefully either way
            raise GroqUnavailable(str(exc)) from exc

    def respond_ungrounded(self, query):
        """A reply for a query retrieval did NOT ground — see escalate_node
        in agents.py and llm_prompts.py's UNGROUNDED_SYSTEM_PROMPT. Never
        passed any retrieved evidence, so it cannot cite a hallucinated
        regulation no matter what it says — that's enforced by this method
        simply having none to hand it, not by the prompt's wording alone."""
        user_message = build_ungrounded_user_message(query)
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": UNGROUNDED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1200,
                temperature=self.temperature,
                reasoning_effort="low",
            )
            choice = response.choices[0]
            content = choice.message.content
            if not content or not content.strip():
                raise GroqUnavailable(
                    f"{self.model_id} returned empty content (likely exhausted "
                    "max_tokens on internal reasoning before any visible output)"
                )
            if choice.finish_reason == "length":
                raise GroqUnavailable(
                    f"{self.model_id} hit max_tokens before finishing the answer "
                    "(finish_reason='length') — treating as unavailable rather "
                    "than returning a truncated answer"
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
