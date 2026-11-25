#!/usr/bin/env python3
"""Streamlit chat UI for the mini compliance agent.

Presentation layer only — all business logic stays in `src/agents.py`.
Run with:

    streamlit run app_ui.py

See the `edit-compliance-ui` skill before changing this file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from agents import ComplianceAgentGraph  # noqa: E402
from bedrock_llm import BedrockUnavailable  # noqa: E402
from classifier import LABEL_DESCRIPTIONS  # noqa: E402

st.set_page_config(page_title="Compliance Agent", page_icon="⚖️", layout="centered")


class _OfflineSummarizer:
    """Drop-in for BedrockSummarizer that always raises BedrockUnavailable.

    Lets the UI's "skip Bedrock" toggle reuse summarizer_node's existing,
    tested template fallback instead of adding a second answer-generation
    code path just for the UI.
    """

    model_id = "offline-template-only"

    def summarize(self, **kwargs):
        raise BedrockUnavailable("offline mode selected in UI")


@st.cache_resource(show_spinner="Loading knowledge base and models…")
def _load_agent(use_bedrock: bool) -> ComplianceAgentGraph:
    llm = None if use_bedrock else _OfflineSummarizer()
    return ComplianceAgentGraph(llm=llm)


def _answer_body(answer: str) -> str:
    """Drop the trailing "Citations: ..." line that summarizer_node/
    escalate_node append to `answer` — the CLI (demo.py) wants that line
    inline, but this UI renders citations as its own styled caption, so
    showing both would duplicate the same list."""
    return answer.rsplit("\n\nCitations:", 1)[0]


EXAMPLE_QUERIES = [
    "Do we need to file a SAR for a structured cash deposit pattern?",
    "What is the CTR threshold for a cash deposit?",
    "What safeguards does HIPAA require for protected health information?",
    "What must the CEO certify about internal controls each quarter?",
    "What audit trail does the FDA require for electronic signatures?",
]

st.title("⚖️ Mini Regulatory Compliance Agent")
st.caption(
    "LangGraph multi-agent pipeline over real regulatory text — "
    "BSA/AML (31 CFR), FDA e-records (21 CFR 11), HIPAA (45 CFR), "
    "SOX-implementing SEC rules (17 CFR) — grounded via Amazon Bedrock."
)

with st.sidebar:
    st.header("Settings")
    use_bedrock = st.toggle(
        "Use live Bedrock for the final answer",
        value=True,
        help=(
            "Retrieval always makes one real, billed Bedrock embedding call "
            "per question, on or off (that's how dense-embedding search "
            "works — see bedrock-rag-retrieval skill). This toggle only "
            "controls the summarization step: On = also makes a real Nova "
            "Micro call to generate the answer. Off = skip that one and use "
            "the same template fallback the pipeline uses on a real Bedrock "
            "outage, saving the (larger of the two, still small) LLM call."
        ),
    )
    if use_bedrock:
        st.caption("⚠️ Each question makes 2 real, billed AWS calls (embed + summarize).")
    else:
        st.caption("⚠️ Each question still makes 1 real, billed AWS call (embed only).")

    st.divider()
    st.header("Regulatory domains")
    for label, desc in LABEL_DESCRIPTIONS.items():
        st.caption(f"**{label}**  \n{desc}")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption(
        "Known limitations: unverified LLM citations, four U.S.-only "
        "domains (no GDPR/SOC2), retrieval needs a real Bedrock call per "
        "question (no free mode — see README's \"Honest limitations\")."
    )

agent = _load_agent(use_bedrock)

if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    st.markdown("**Try an example question:**")
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"example::{q}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        intent = turn["intent"]
        grounded = turn["grounded"]
        badge = "🟢" if grounded else "🟠"
        st.caption(
            f"{badge} Intent: **{intent['intent']}** "
            f"({intent['confidence']:.0%} confidence) · "
            f"{'grounded' if grounded else 'escalated — not grounded'}"
        )
        st.write(_answer_body(turn["answer"]))
        if turn["citations"]:
            st.caption("Citations: " + ", ".join(turn["citations"]))
        with st.expander("Show reasoning trace"):
            st.code("\n".join(turn["trace"]), language=None)

pending = st.session_state.pop("pending_query", None)
typed = st.chat_input("Ask a compliance question…")
query = pending or typed

if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Classifying intent, retrieving evidence, generating answer…"):
            result = agent.handle(query)
        intent = result.intent
        grounded = result.verification["grounded"]
        citations = result.verification["citations"]
        badge = "🟢" if grounded else "🟠"
        st.caption(
            f"{badge} Intent: **{intent['intent']}** "
            f"({intent['confidence']:.0%} confidence) · "
            f"{'grounded' if grounded else 'escalated — not grounded'}"
        )
        st.write(_answer_body(result.answer))
        if citations:
            st.caption("Citations: " + ", ".join(citations))
        with st.expander("Show reasoning trace"):
            st.code("\n".join(result.trace), language=None)

    st.session_state.history.append(
        {
            "query": query,
            "answer": result.answer,
            "intent": intent,
            "grounded": grounded,
            "citations": citations,
            "trace": result.trace,
        }
    )
    st.rerun()
