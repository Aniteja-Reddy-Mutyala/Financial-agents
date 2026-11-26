#!/usr/bin/env python3
"""Streamlit chat UI for the mini compliance agent.

Presentation layer only — all business logic stays in `src/agents.py`.
Run with:

    streamlit run app_ui.py

See the `edit-compliance-ui` skill before changing this file.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from agents import ComplianceAgentGraph  # noqa: E402
from classifier import LABEL_DESCRIPTIONS  # noqa: E402
from groq_llm import GroqSummarizer  # noqa: E402

# Streamlit Community Cloud injects secrets via st.secrets, not os.environ
# (and has no ~/.aws/credentials) — bridge them so every module here that
# reads via os.environ keeps working unchanged. Locally, with no
# secrets.toml, st.secrets raises StreamlitSecretNotFoundError on access
# (not just an empty mapping), so this is wrapped to stay a no-op there.
try:
    for _key in ("GROQ_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
        if _key in st.secrets:
            os.environ.setdefault(_key, st.secrets[_key])
except st.errors.StreamlitSecretNotFoundError:
    pass

st.set_page_config(page_title="Compliance Agent", page_icon="⚖️", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base and models…")
def _load_agent(use_bedrock: bool) -> ComplianceAgentGraph:
    # Off = GroqSummarizer, not a fully offline/template-only stub — same
    # interchangeable-provider design as LLM_PROVIDER elsewhere in this
    # project (see bedrock-llm-integration skill). If GROQ_API_KEY isn't
    # set (nothing auto-loads .env — source it before `streamlit run`),
    # GroqSummarizer.summarize() raises GroqUnavailable on first use,
    # which summarizer_node already catches and falls back to the
    # templated draft, so a missing key degrades gracefully rather than
    # crashing the app.
    llm = None if use_bedrock else GroqSummarizer()
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
        value=False,
        help=(
            "Retrieval always makes one real, billed Bedrock embedding call "
            "per question, on or off (that's how dense-embedding search "
            "works — see bedrock-rag-retrieval skill). This toggle only "
            "controls the summarization step: On = Amazon Nova Micro "
            "generates the answer (real, billed AWS call). Off = Groq's "
            "open-weight model generates it instead (needs GROQ_API_KEY — "
            "see .env.example; nothing auto-loads it, source it before "
            "running `streamlit run`). If that key is missing or the Groq "
            "call fails, it falls back to the same templated draft the "
            "pipeline uses on any provider outage, so this never crashes. "
            "Groq's model (gpt-oss-20b) is a reasoning model that spends a "
            "variable, sometimes large, share of its token budget on hidden "
            "reasoning before the visible answer — on a longer or multi-part "
            "question it can occasionally exhaust that budget mid-answer, "
            "which now also falls back to the templated draft rather than "
            "showing a cut-off answer. Bedrock has no equivalent limit."
        ),
    )
    if use_bedrock:
        st.caption("⚠️ Each question makes 2 real, billed AWS calls (embed + summarize).")
    else:
        st.caption(
            "⚠️ Each question makes 1 real, billed AWS call (embed) plus 1 "
            "Groq call (summarize) — see the help tooltip if GROQ_API_KEY "
            "isn't set. Groq may fall back to a generic answer on longer or "
            "multi-part questions (reasoning-token budget); Bedrock doesn't "
            "have this limit."
        )

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
