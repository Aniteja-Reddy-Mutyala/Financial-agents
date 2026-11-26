#!/usr/bin/env python3
"""Streamlit page for the Research agent — a second, independent agent
from the compliance one (app_ui.py). Auto-discovered by Streamlit's
multipage convention (any .py under pages/ next to the main app_ui.py
entry script) — app_ui.py itself needs no changes for this to appear.

Presentation layer only — all business logic stays in
src/research_agent.py. See the edit-research-agent skill before
changing this file.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st  # noqa: E402

from groq_llm import GroqSummarizer  # noqa: E402
from research_agent import ResearchAgentGraph  # noqa: E402

# Streamlit re-runs each page script independently (app_ui.py's own top-level
# code doesn't execute for this page), so the st.secrets -> os.environ bridge
# needs to be here too — see app_ui.py and CLAUDE.md's "Future deployment".
# Locally, with no secrets.toml, st.secrets raises StreamlitSecretNotFoundError
# on access (not just an empty mapping), so this is wrapped to stay a no-op.
try:
    for _key in ("GROQ_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
        if _key in st.secrets:
            os.environ.setdefault(_key, st.secrets[_key])
except st.errors.StreamlitSecretNotFoundError:
    pass

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading Research agent…")
def _load_agent(use_bedrock: bool) -> ResearchAgentGraph:
    # Same interchangeable-provider design as app_ui.py's compliance
    # agent toggle — off = GroqSummarizer, falls back to a templated
    # draft (not a crash) if GROQ_API_KEY is missing or the call fails.
    llm = None if use_bedrock else GroqSummarizer()
    return ResearchAgentGraph(llm=llm)


st.title("🔎 Research Agent")
st.caption(
    "Give it a topic/question and/or upload a document (PDF or text, "
    "e.g. a resume) — it extracts up to 3 concrete things worth "
    "researching, looks each one up on the open web, and synthesizes "
    "one report with sources. Web results are unverified; see the "
    "report's own closing caveat."
)

with st.sidebar:
    st.header("Settings")
    use_bedrock = st.toggle(
        "Use live Bedrock for the final answer",
        value=False,
        help=(
            "On = Amazon Nova Micro generates the topic extraction and "
            "synthesis (2 real, billed AWS calls). Off = Groq's "
            "open-weight model instead (needs GROQ_API_KEY — nothing "
            "auto-loads .env, source it before `streamlit run`); falls "
            "back to a templated draft if that key is missing or the "
            "call fails, so this never crashes. Either way, web search "
            "itself (ddgs) is free and makes no AWS/Groq call."
        ),
    )
    st.caption(
        "⚠️ Up to 2 LLM calls (topic extraction + synthesis) per request, "
        "plus up to 3 free web searches (no API key) — one per topic."
    )
    st.divider()
    st.caption(
        "Known limitation: web search results are not fact-checked — "
        "the synthesis is instructed to flag disagreement and thin "
        "coverage, but a citation existing doesn't guarantee the "
        "summarized claim is accurate."
    )

agent = _load_agent(use_bedrock)

query = st.text_input("Topic or question (optional if you upload a document)")
uploaded = st.file_uploader("Document (optional — PDF or .txt)", type=["pdf", "txt", "md"])

if st.button("Research", type="primary"):
    if not query and not uploaded:
        st.warning("Enter a topic/question, upload a document, or both.")
    else:
        with st.spinner("Extracting topics, searching the web, synthesizing…"):
            # A document that fails to parse degrades gracefully inside
            # parse_document_node (falls through to the query alone) —
            # it never raises out of .handle(), so no try/except needed
            # here; a failed parse just shows up in the trace below.
            result = agent.handle(query=query, document_path=uploaded)

        st.write(result.answer)
        with st.expander("Show reasoning trace"):
            st.code("\n".join(result.trace), language=None)
