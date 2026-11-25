"""Shared, framework-level helpers reused by every agent in this repo —
currently `agents.py`'s ComplianceAgentGraph and `research_agent.py`'s
ResearchAgentGraph, and meant to be reused by whatever agent comes next
(Email, Calendar — see CLAUDE.md's "Target end product" for the planned
multi-agent roster). Extracted here once a second agent needed the exact
same ~15-line provider-selection logic and trace-logging helper, rather
than copying a third time — see the "three similar lines" rule in
CLAUDE.md's Conventions for why this wasn't done after just the first
duplicate.

Each agent module re-exports what it needs under its own existing name
(e.g. `agents.py` does `from agent_common import default_llm as
_default_llm`) so nothing calling into an agent module's own public
interface has to change because of this extraction.
"""
import os
from typing import List, Union

from bedrock_llm import BedrockSummarizer
from groq_llm import GroqSummarizer

LLMProvider = Union[BedrockSummarizer, GroqSummarizer]


def default_llm(default="bedrock") -> LLMProvider:
    """Which LLM provider an agent builds by default when no explicit
    `llm=` override is passed — "bedrock" (Amazon Nova Micro) or "groq"
    (an open-weight model, no AWS cost), chosen via the LLM_PROVIDER env
    var. Read fresh on every call, not cached at import time, so
    switching it between agent instances in the same process (tests
    included) actually takes effect. Shared by every agent — see
    bedrock-llm-integration and edit-research-agent skills for the two
    providers' per-method contracts (.summarize()/.respond_ungrounded()
    for the compliance agent, .chat() for the research agent)."""
    provider = os.environ.get("LLM_PROVIDER", default).lower()
    if provider == "groq":
        return GroqSummarizer()
    if provider != "bedrock":
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r} — expected 'bedrock' or 'groq'")
    return BedrockSummarizer()


def log_step(state: dict, agent: str, note: str) -> List[str]:
    """Appends a `"[{agent}] {note}"` line to a *copy* of state["trace"]
    (never mutates state in place — every node function returns this as
    part of its partial-update dict). Shared across agents so every
    agent's reasoning trace reads the same way in the CLI/UI."""
    trace = list(state.get("trace", []))
    trace.append(f"[{agent}] {note}")
    return trace
