"""Multi-agent compliance pipeline built on LangGraph.

Mirrors the Orchestrator / Planner / Verifier / Summarizer agent roles and
the LangGraph-based multi-agent orchestration described in the original
Northern Trust project, at a scale that runs locally against a small
regulatory knowledge base and a cheap Amazon Bedrock model.

Graph shape:

    classify_intent --[chitchat]--> chitchat -> END
                    \\--[else]-----> plan -> retrieve -> verify --[grounded]--> summarize -> END
                                                                \\--[not grounded]--> escalate -> END

The conditional edge out of `verify` is the "error recovery" behavior from
the resume: when retrieval doesn't clear the confidence bar, the graph
routes to a human-escalation node instead of letting an LLM guess.

The conditional edge out of `classify_intent` is separate: a greeting or
"tell me about yourself"-style message (detected by classifier.py's
is_chitchat() — a deterministic phrase check, NOT a label on
IntentClassifier's model; see CHITCHAT_LABEL's comment for why) skips
retrieval entirely and gets a short conversational reply instead of
either a fabricated regulatory answer or the "route to a human"
escalation message, which would be a strange response to "hi."
"""
import os
from types import SimpleNamespace
from typing import List, Optional, TypedDict, Union

from langgraph.graph import END, StateGraph

from bedrock_llm import BedrockSummarizer, BedrockUnavailable
from classifier import CHITCHAT_DESCRIPTION, CHITCHAT_LABEL, IntentClassifier, is_chitchat
from groq_llm import GroqSummarizer, GroqUnavailable
from retriever import Retriever

Summarizer = Union[BedrockSummarizer, GroqSummarizer]
SummarizerUnavailable = (BedrockUnavailable, GroqUnavailable)

def _default_llm() -> Summarizer:
    """Which Summarizer provider ComplianceAgentGraph builds by default when
    no `llm=` override is passed — "bedrock" (Amazon Nova Micro, the
    original default) or "groq" (an open-weight model, no AWS cost for
    this step), chosen via the LLM_PROVIDER env var. Read fresh on every
    call, not cached at import time, so switching it between
    ComplianceAgentGraph() instances in the same process (tests included)
    actually takes effect. Retrieval/embeddings are unaffected either way:
    Groq has no embeddings API, so Bedrock Titan embeddings are used
    regardless of this setting — see bedrock-rag-retrieval and
    run-evaluation-suite skills.
    """
    provider = os.environ.get("LLM_PROVIDER", "bedrock").lower()
    if provider == "groq":
        return GroqSummarizer()
    if provider != "bedrock":
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r} — expected 'bedrock' or 'groq'")
    return BedrockSummarizer()

# Intents broad enough to cover several distinct sub-topics under one
# fixed description — padding the retrieval query with that description
# skews retrieval toward whichever sub-topic the description's wording
# happens to lexically/semantically favor, regardless of what the user
# actually asked. Measured (GitHub issue #1): fda_e_records's description
# ("FDA electronic records and electronic signature requirements") kept
# pulling in signature-focused sections even for a direct "audit trail"
# question, so 21 CFR 11.10 (the section that actually covers audit
# trails) never made the top 5 — the LLM then answered confidently from
# the wrong evidence. hipaa_privacy_security (102 sections) has the same
# shape of problem, likely worse. See planner_node.
NO_PAD_INTENTS = {"fda_e_records", "hipaa_privacy_security"}

# Tuned for dense Bedrock Titan-embedding cosine similarity, NOT the old
# TF-IDF cosine scale (was 0.08). Measured against the real built index:
# genuinely relevant queries scored 0.37-0.70 top-hit; genuinely off-topic
# queries ("what's the best pizza topping in Chicago", "capital of
# France", etc.) scored 0.13-0.16 — dense embeddings have a much higher
# noise floor than TF-IDF, where near-zero meant near-zero. 0.25 sits with
# real margin on both sides of that measured gap. If you change
# retriever.py's similarity metric again, re-measure and recalibrate this
# the same way — see bedrock-rag-retrieval skill.
MIN_SCORE = 0.25


class ComplianceState(TypedDict, total=False):
    query: str
    intent: dict
    plan: dict
    evidence: list
    verification: dict
    answer: str
    trace: List[str]


def _log(state: ComplianceState, agent: str, note: str) -> List[str]:
    trace = list(state.get("trace", []))
    trace.append(f"[{agent}] {note}")
    return trace


def _unique_evidence(state: ComplianceState):
    citations = state["verification"]["citations"]
    seen = set()
    out = []
    for hit in state["evidence"]:
        if hit["source_id"] not in citations or hit["source_id"] in seen:
            continue
        seen.add(hit["source_id"])
        out.append(hit)
    return out


# --- Node functions -------------------------------------------------------


def classify_intent_node(state: ComplianceState, *, intent_clf: IntentClassifier) -> dict:
    # is_chitchat() runs first and is fully independent of IntentClassifier
    # (see classifier.py's CHITCHAT_LABEL comment for why it's not a 9th
    # trained label) — when it matches, skip the real classifier call
    # entirely rather than discarding its guess, so the trace/UI badge
    # never shows a mismatched real-domain label next to a chitchat reply.
    if is_chitchat(state["query"]):
        intent = {
            "intent": CHITCHAT_LABEL,
            "description": CHITCHAT_DESCRIPTION,
            "confidence": 1.0,
            "has_signal": True,
        }
    else:
        intent = intent_clf.predict(state["query"])
    trace = _log(state, "Orchestrator", f"classified intent -> {intent['intent']}")
    return {"intent": intent, "trace": trace}


def route_after_classify(state: ComplianceState) -> str:
    return "chitchat" if state["intent"]["intent"] == CHITCHAT_LABEL else "plan"


def planner_node(state: ComplianceState) -> dict:
    intent = state["intent"]["intent"]
    top_k = 4 if intent == "aml_program" else 3
    if intent in NO_PAD_INTENTS:
        # This intent's one fixed description can't represent all of its
        # sub-topics well (see NO_PAD_INTENTS above) — pad with nothing
        # and let the raw query alone drive retrieval, same as the
        # has_signal=False path below.
        query = state["query"]
    elif state["intent"]["has_signal"]:
        query = f"{state['query']} {state['intent']['description']}"
    else:
        # The classifier shared zero training vocabulary with this query —
        # its intent guess is statistical noise (pure class-prior
        # intercepts), not a real classification. Padding the retrieval
        # query with a noise-driven description would inflate the score
        # for whatever domain the model happened to guess, regardless of
        # the query's actual topic — this was the real, measured mechanism
        # behind the "known rough edge" in edit-compliance-graph. Search
        # the user's own words only instead; dense embeddings are often
        # enough to ground a genuinely on-topic query without any padding
        # crutch (measured: 5 of 6 deliberately vague-but-legitimate test
        # queries already cleared MIN_SCORE on the raw query alone).
        query = state["query"]
    plan = {"retrieval_query": query, "top_k": top_k}
    trace = _log(
        state,
        "Planner",
        f"retrieval_query='{query}', top_k={top_k}, "
        f"has_signal={state['intent']['has_signal']}",
    )
    return {"plan": plan, "trace": trace}


def retriever_node(state: ComplianceState, *, retriever: Retriever) -> dict:
    top_k = state["plan"]["top_k"]
    retrieval_query = state["plan"]["retrieval_query"]
    raw_hits = retriever.search(state["query"], top_k=top_k)
    if retrieval_query == state["query"]:
        # planner_node already decided not to pad (has_signal=False) —
        # searching the identical string again would just repeat the same
        # real, billed embedding call for no new information.
        hits = raw_hits
    else:
        # Search both the raw user query and the Planner's intent-augmented
        # query, then merge by best score per section. Searching the
        # augmented query alone let a wrong/low-confidence intent
        # classification pad the evidence with a section from the wrong
        # regulatory domain purely because the (possibly wrong) intent's
        # description text matches it — see the edit-compliance-graph
        # skill's "known rough edge." Merging in the raw query means a
        # section that genuinely matches what the user actually asked
        # still competes for and usually wins the top slots, even when the
        # intent classifier guessed wrong (but still had real signal).
        planned_hits = retriever.search(retrieval_query, top_k=top_k)
        best_by_source = {}
        for hit in raw_hits + planned_hits:
            existing = best_by_source.get(hit["source_id"])
            if existing is None or hit["score"] > existing["score"]:
                best_by_source[hit["source_id"]] = hit
        hits = sorted(best_by_source.values(), key=lambda h: h["score"], reverse=True)[:top_k]
    trace = _log(state, "Retriever", f"found {len(hits)} relevant regulatory sections")
    return {"evidence": hits, "trace": trace}


def verifier_node(state: ComplianceState) -> dict:
    strong = [h for h in state["evidence"] if h["score"] >= MIN_SCORE]
    grounded = len(strong) > 0
    seen, citations = set(), []
    for h in strong:
        if h["source_id"] not in seen:
            seen.add(h["source_id"])
            citations.append(h["source_id"])
    verification = {
        "grounded": grounded,
        "citations": citations,
        "reason": (
            "sufficient regulatory evidence retrieved"
            if grounded
            else "no evidence cleared the confidence threshold — escalate to human compliance officer"
        ),
    }
    trace = _log(state, "ComplianceVerifier", f"grounded={grounded}, citations={citations}")
    return {"verification": verification, "trace": trace}


def route_after_verify(state: ComplianceState) -> str:
    return "summarize" if state["verification"]["grounded"] else "escalate"


def _template_draft(state: ComplianceState, evidence) -> str:
    lines = [
        f"Query intent: {state['intent']['description']} "
        f"(confidence {state['intent']['confidence']:.0%})",
        "",
        "Relevant regulatory text:",
    ]
    for hit in evidence:
        lines.append(f"\n• {hit['source_id']} — {hit['title']}")
        lines.append(f"  \"{hit['text'][:300].rstrip()}...\"")
    lines.append(f"\nCitations: {', '.join(state['verification']['citations'])}")
    return "\n".join(lines)


def summarizer_node(state: ComplianceState, *, llm: Summarizer) -> dict:
    evidence = _unique_evidence(state)
    try:
        llm_answer = llm.summarize(
            query=state["query"],
            intent_description=state["intent"]["description"],
            evidence=evidence,
        )
        trace = _log(state, "Summarizer", f"generated via {llm.model_id}")
        citations = ", ".join(state["verification"]["citations"])
        answer = f"{llm_answer}\n\nCitations: {citations}"
    except SummarizerUnavailable as exc:
        trace = _log(state, "Summarizer", f"LLM unavailable ({exc}); using template fallback")
        answer = _template_draft(state, evidence)
    return {"answer": answer, "trace": trace}


def chitchat_node(state: ComplianceState, *, llm: Summarizer) -> dict:
    """Greeting/small-talk reply, bypassing plan/retrieve/verify entirely —
    see route_after_classify. Sets a `verification` dict (grounded=True,
    no citations) purely so downstream consumers (app_ui.py, demo.py) that
    read `result.verification["grounded"]`/`["citations"]` unconditionally
    don't need a special case for this path.
    """
    try:
        llm_answer = llm.chitchat(state["query"])
        trace = _log(state, "Summarizer", f"generated chitchat reply via {llm.model_id}")
        answer = llm_answer
    except SummarizerUnavailable as exc:
        trace = _log(state, "Summarizer", f"LLM unavailable ({exc}); using static chitchat fallback")
        answer = (
            "Hi! I'm a compliance assistant that answers BSA/AML, FDA "
            "electronic records, HIPAA, and SOX questions by retrieving and "
            "citing the actual regulatory text — try asking something like "
            "\"what is the CTR threshold for a cash deposit?\""
        )
    verification = {"grounded": True, "citations": [], "reason": "chitchat — not a regulatory question"}
    return {"answer": answer, "verification": verification, "trace": trace}


def escalate_node(state: ComplianceState) -> dict:
    trace = _log(state, "Summarizer", "not grounded; routing to human compliance officer")
    answer = (
        f"Query intent: {state['intent']['description']} "
        f"(confidence {state['intent']['confidence']:.0%})\n\n"
        "⚠️ Could not confidently ground this answer in the regulatory "
        "knowledge base. Recommend routing to a human compliance officer."
    )
    return {"answer": answer, "trace": trace}


# --- Graph assembly ---------------------------------------------------------


class ComplianceAgentGraph:
    """Builds and runs the LangGraph compliance pipeline."""

    def __init__(self, llm: Optional[Summarizer] = None, retriever: Optional[Retriever] = None):
        self.intent_clf = IntentClassifier()
        self.retriever = retriever or Retriever.build()
        self.llm = llm or _default_llm()
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ComplianceState)
        graph.add_node("classify_intent", lambda s: classify_intent_node(s, intent_clf=self.intent_clf))
        graph.add_node("chitchat", lambda s: chitchat_node(s, llm=self.llm))
        graph.add_node("plan", planner_node)
        graph.add_node("retrieve", lambda s: retriever_node(s, retriever=self.retriever))
        graph.add_node("verify", verifier_node)
        graph.add_node("summarize", lambda s: summarizer_node(s, llm=self.llm))
        graph.add_node("escalate", escalate_node)

        graph.set_entry_point("classify_intent")
        graph.add_conditional_edges(
            "classify_intent", route_after_classify, {"chitchat": "chitchat", "plan": "plan"}
        )
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "verify")
        graph.add_conditional_edges(
            "verify", route_after_verify, {"summarize": "summarize", "escalate": "escalate"}
        )
        graph.add_edge("chitchat", END)
        graph.add_edge("summarize", END)
        graph.add_edge("escalate", END)
        return graph.compile()

    def handle(self, query: str) -> SimpleNamespace:
        final_state = self.app.invoke({"query": query, "trace": []})
        return SimpleNamespace(**final_state)


# Backward-compatible alias used by demo.py
OrchestratorAgent = ComplianceAgentGraph


if __name__ == "__main__":
    orch = ComplianceAgentGraph()
    result = orch.handle("do we need to file a SAR for a structured cash deposit pattern")
    print("\n".join(result.trace))
    print("\n--- ANSWER ---\n")
    print(result.answer)
