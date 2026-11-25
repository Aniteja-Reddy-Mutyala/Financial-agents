"""Research agent — a second, fully independent LangGraph agent alongside
agents.py's ComplianceAgentGraph (the first of what will become several
agents sharing this codebase — see .claude/skills/edit-research-agent).

Given a document (PDF/text — a resume, for example) and/or a topic/
question, it extracts up to 3 concrete things worth researching, looks
each one up on the open web (via web_search.py's WebSearcher, free, no
API key), and synthesizes one report with sources. Gracefully degrades
to query-only (the query itself is the one topic — no extraction LLM
call needed) or document-only (topics extracted from the document alone)
when only one input is given.

Graph shape:

    intake --[document given]--> parse_document --> extract_topics
           --[query only]------------------------------^
    extract_topics -> web_research -> synthesize -> END

Deliberately does NOT share agents.py's ComplianceState or node
functions, but DOES share agent_common.py's default_llm()/log_step()
helpers (as `_default_llm`/`_log` here, same names as before this was
extracted — see agent_common.py's docstring and CLAUDE.md's "Target end
product" for why this is now shared rather than duplicated a third
time). Also shares the Summarizer classes themselves
(BedrockSummarizer/GroqSummarizer) via their generic .chat() method,
so this agent is model-agnostic between Bedrock and Groq exactly like
the compliance agent, via the same LLM_PROVIDER env var.
"""
from types import SimpleNamespace
from typing import List, TypedDict, Union

from langgraph.graph import END, StateGraph

from agent_common import default_llm as _default_llm
from agent_common import log_step as _log
from bedrock_llm import BedrockSummarizer, BedrockUnavailable
from document_parser import DocumentParseError, parse_document
from groq_llm import GroqSummarizer, GroqUnavailable
from research_prompts import (
    RESEARCH_SYNTHESIS_SYSTEM_PROMPT,
    TOPIC_EXTRACTION_SYSTEM_PROMPT,
    build_synthesis_message,
    build_topic_extraction_message,
)
from web_search import WebSearcher, WebSearchUnavailable

Chatter = Union[BedrockSummarizer, GroqSummarizer]
ChatterUnavailable = (BedrockUnavailable, GroqUnavailable)

MAX_TOPICS = 3  # hard cap enforced in code (not just the prompt) — bounds
                # both the LLM's extraction output and the number of
                # billed web-search calls per request.


class ResearchState(TypedDict, total=False):
    query: str
    document_path: str  # "" if none given; can also be a file-like object
    document_text: str
    topics: list
    search_results: dict
    answer: str
    trace: List[str]


# --- Node functions --------------------------------------------------------


def route_after_intake(state: ResearchState) -> str:
    return "parse_document" if state.get("document_path") else "extract_topics"


def parse_document_node(state: ResearchState, *, document_parser) -> dict:
    try:
        text = document_parser(state["document_path"])
        trace = _log(state, "ResearchOrchestrator", f"parsed document ({len(text)} chars)")
        return {"document_text": text, "trace": trace}
    except DocumentParseError as exc:
        # Graceful degrade, same spirit as summarizer_node's template
        # fallback: a bad document shouldn't crash the request if there's
        # still a query to work from.
        trace = _log(state, "ResearchOrchestrator", f"document parse failed ({exc}) — continuing with query alone")
        return {"document_text": "", "trace": trace}


def extract_topics_node(state: ResearchState, *, llm: Chatter) -> dict:
    query = state.get("query", "")
    document_text = state.get("document_text", "")

    if document_text:
        user_message = build_topic_extraction_message(query, document_text)
        try:
            raw = llm.chat(TOPIC_EXTRACTION_SYSTEM_PROMPT, user_message)
            topics = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
        except ChatterUnavailable:
            # No topics extracted — fall back to the query alone if there
            # is one, otherwise there's nothing left to research.
            topics = [query] if query else []
    else:
        # Query-only: the query itself is the one topic. Skips the
        # extraction LLM call entirely — nothing to extract from.
        topics = [query] if query else []

    topics = [t for t in topics if t][:MAX_TOPICS]
    trace = _log(state, "ResearchPlanner", f"topics={topics}")
    return {"topics": topics, "trace": trace}


def web_research_node(state: ResearchState, *, web_searcher: WebSearcher) -> dict:
    results = {}
    trace = list(state.get("trace", []))
    for topic in state.get("topics", []):
        try:
            results[topic] = web_searcher.search(topic)
            trace.append(f"[ResearchRetriever] found {len(results[topic])} results for '{topic}'")
        except WebSearchUnavailable as exc:
            # One topic's search failing shouldn't kill the others.
            results[topic] = []
            trace.append(f"[ResearchRetriever] web search failed for '{topic}': {exc}")
    return {"search_results": results, "trace": trace}


def _template_draft(state: ResearchState) -> str:
    """Non-LLM fallback. Two distinct cases: no topic found any results at
    all (nothing to show but say so), vs. results exist but synthesis
    failed (show the raw results instead of losing them)."""
    document_text = state.get("document_text", "")
    topics = state.get("topics", [])
    search_results = state.get("search_results", {})
    lines = []
    if document_text:
        lines.append(f"Document summary (first 300 chars): {document_text[:300].rstrip()}...")
        lines.append("")

    if not any(search_results.get(t) for t in topics):
        lines.append("No web search results were found for: " + ", ".join(topics) + ".")
        return "\n".join(lines)

    lines.append("Raw web search results (unverified; synthesis unavailable):")
    for topic in topics:
        results = search_results.get(topic, [])
        lines.append(f"\n{topic}:")
        if not results:
            lines.append("  (no results)")
            continue
        for r in results:
            lines.append(f"  • {r['title']} ({r['url']}): {r['snippet'][:200].rstrip()}...")
    return "\n".join(lines)


def synthesize_node(state: ResearchState, *, llm: Chatter) -> dict:
    topics = state.get("topics", [])
    search_results = state.get("search_results", {})

    if not any(search_results.get(t) for t in topics):
        # Nothing to ground a synthesis in — don't let the model guess
        # with no sources, same principle as verifier_node's
        # grounded/escalate split in the compliance agent.
        answer = _template_draft(state)
        trace = _log(state, "ResearchSummarizer", "no web results for any topic — using template fallback")
        return {"answer": answer, "trace": trace}

    user_message = build_synthesis_message(state.get("document_text", ""), topics, search_results)
    try:
        answer = llm.chat(RESEARCH_SYNTHESIS_SYSTEM_PROMPT, user_message)
        trace = _log(state, "ResearchSummarizer", f"generated via {llm.model_id}")
    except ChatterUnavailable as exc:
        answer = _template_draft(state)
        trace = _log(state, "ResearchSummarizer", f"LLM unavailable ({exc}); using template fallback")
    return {"answer": answer, "trace": trace}


class ResearchAgentGraph:
    def __init__(self, llm: Chatter = None, web_searcher: WebSearcher = None, document_parser=None):
        self.llm = llm or _default_llm()
        self.web_searcher = web_searcher or WebSearcher()
        self.document_parser = document_parser or parse_document
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("intake", lambda s: {})
        graph.add_node("parse_document", lambda s: parse_document_node(s, document_parser=self.document_parser))
        graph.add_node("extract_topics", lambda s: extract_topics_node(s, llm=self.llm))
        graph.add_node("web_research", lambda s: web_research_node(s, web_searcher=self.web_searcher))
        graph.add_node("synthesize", lambda s: synthesize_node(s, llm=self.llm))

        graph.set_entry_point("intake")
        graph.add_conditional_edges(
            "intake", route_after_intake,
            {"parse_document": "parse_document", "extract_topics": "extract_topics"},
        )
        graph.add_edge("parse_document", "extract_topics")
        graph.add_edge("extract_topics", "web_research")
        graph.add_edge("web_research", "synthesize")
        graph.add_edge("synthesize", END)
        return graph.compile()

    def handle(self, query: str = "", document_path=None) -> SimpleNamespace:
        if not query and not document_path:
            raise ValueError("ResearchAgentGraph.handle() needs a query, a document, or both")
        final_state = self.app.invoke({"query": query, "document_path": document_path or "", "trace": []})
        return SimpleNamespace(**final_state)


if __name__ == "__main__":
    agent = ResearchAgentGraph()
    result = agent.handle(query="Kubernetes")
    print("\n".join(result.trace))
    print("\n--- ANSWER ---\n")
    print(result.answer)
