#!/usr/bin/env python3
"""Streamlit page for the top-level Orchestrator (orchestrator_agent.py) —
splits a query across the Compliance and Research agents and composes
one answer. Auto-discovered by Streamlit's multipage convention, same as
pages/1_Research_Agent.py.

Presentation layer only — all business logic stays in
src/orchestrator_agent.py.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st  # noqa: E402

from groq_llm import GroqSummarizer  # noqa: E402
from orchestrator_agent import OrchestratorGraph  # noqa: E402

# Same st.secrets -> os.environ bridge as app_ui.py / pages/1_Research_Agent.py
# — each Streamlit page script runs independently, so this needs to be
# here too. See CLAUDE.md's "Future deployment".
try:
    for _key in ("GROQ_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
        if _key in st.secrets:
            os.environ.setdefault(_key, st.secrets[_key])
except st.errors.StreamlitSecretNotFoundError:
    pass

st.set_page_config(page_title="Orchestrator Agent", page_icon="🧭", layout="centered")


@st.cache_resource(show_spinner="Loading Orchestrator agent…")
def _load_agent(use_bedrock: bool) -> OrchestratorGraph:
    # Same interchangeable-provider design as the other two pages — off =
    # GroqSummarizer, falls back to concatenating each agent's own answer
    # (not a crash) if that call fails. Builds its own Compliance and
    # Research agent instances, each with the same LLM provider choice.
    llm = None if use_bedrock else GroqSummarizer()
    return OrchestratorGraph(llm=llm)


st.title("🧭 Orchestrator Agent")
st.caption(
    "Give it a request that may span the Compliance agent, the Research "
    "agent, or both — it plans which agent(s) are needed, runs each via "
    "its existing pipeline, and composes one final answer."
)

with st.sidebar:
    st.header("Settings")
    use_bedrock = st.toggle(
        "Use live Bedrock for the final answer",
        value=False,
        help=(
            "On = Amazon Nova Micro plans and (if more than one agent "
            "runs) composes the final answer. Off = Groq's open-weight "
            "model instead (needs GROQ_API_KEY). Either provider choice "
            "is also used by the Compliance and Research agents this "
            "page builds. The Compliance agent's retrieval step is "
            "always a real, billed Bedrock embedding call regardless of "
            "this toggle."
        ),
    )
    st.caption(
        "⚠️ Up to 2 extra LLM calls (plan + compose) on top of whichever "
        "agent(s) the plan dispatches to — the Compliance agent's own "
        "LLM call and/or the Research agent's up to 2 LLM calls, plus "
        "any real Bedrock retrieval/web-search calls those agents make."
    )
    st.divider()
    st.caption(
        "Known limitation: the plan step's agent choice isn't verified "
        "against the query — a wrong plan dispatches to the wrong "
        "agent(s), the same structural risk as the Compliance agent's "
        "own intent classifier (see README's \"Honest limitations\")."
    )

agent = _load_agent(use_bedrock)

query = st.text_area("Request")

if st.button("Run", type="primary"):
    if not query.strip():
        st.warning("Enter a request.")
    else:
        with st.spinner("Planning, dispatching to agent(s), composing…"):
            result = agent.handle(query)

        st.write(result.answer)
        with st.expander("Show reasoning trace"):
            st.code("\n".join(result.trace), language=None)
