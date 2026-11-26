"""Top-level Orchestrator -> Planner -> domain agents -> Orchestrator
flow, as scoped in CLAUDE.md's "Target end product" / "Cross-agent
integration" section — a *different*, cross-agent pair from each domain
agent's own internal Orchestrator/Planner node roles (agents.py's
classify_intent_node/planner_node, research_agent.py's equivalents).

This is the one call site meant to receive a raw user query, decide
which existing domain agent(s) (ComplianceAgentGraph, ResearchAgentGraph,
...) it needs, run each one via its existing, unchanged `.handle(...)`
entry point, and compose their answers into one final response.

Graph shape:

    plan -> dispatch -> compose -> END

`plan` asks the LLM to split the query into subtasks, one per agent
that's actually needed (see orchestrator_prompts.PLANNER_SYSTEM_PROMPT).
`dispatch` calls each assigned agent's `.handle(...)` in turn — no
per-agent special-casing, just a lookup in self.agents keyed by the
planner's own agent names. `compose` returns a single agent's answer
untouched (no extra LLM call needed) or, when more than one agent ran,
asks the LLM to weave them into one coherent answer.
"""
from types import SimpleNamespace
from typing import Dict, List, Optional, TypedDict, Union

from langgraph.graph import END, StateGraph

from agent_common import default_llm as _default_llm
from agent_common import log_step as _log
from agents import ComplianceAgentGraph
from bedrock_llm import BedrockSummarizer, BedrockUnavailable
from groq_llm import GroqSummarizer, GroqUnavailable
from orchestrator_prompts import (
    COMPOSER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_composer_message,
    build_planner_message,
)
from research_agent import ResearchAgentGraph

Chatter = Union[BedrockSummarizer, GroqSummarizer]
ChatterUnavailable = (BedrockUnavailable, GroqUnavailable)

# The population this project targets is assumed to mostly ask finance/
# compliance-shaped queries (see CLAUDE.md's "Cross-agent integration"),
# so when planning fails outright (LLM unavailable, or the LLM's output
# doesn't parse into any known agent), route the whole query to
# compliance rather than silently answering nothing.
DEFAULT_AGENT = "compliance"


class OrchestratorState(TypedDict, total=False):
    query: str
    subtasks: list  # [{"agent": str, "task": str}, ...]
    subtask_results: list  # same, plus {"answer": str, "trace": list}
    answer: str
    trace: List[str]


# --- Node functions ---------------------------------------------------------


def _parse_plan(raw: str, available_agents) -> list:
    subtasks = []
    for line in raw.splitlines():
        if ":" not in line:
            continue
        name, task = line.split(":", 1)
        name = name.strip().lower()
        task = task.strip()
        if name in available_agents and task:
            subtasks.append({"agent": name, "task": task})
    return subtasks


def planner_node(state: OrchestratorState, *, llm: Chatter, available_agents) -> dict:
    query = state["query"]
    try:
        raw = llm.chat(PLANNER_SYSTEM_PROMPT, build_planner_message(query))
        subtasks = _parse_plan(raw, available_agents)
    except ChatterUnavailable:
        subtasks = []

    if not subtasks:
        # Either the LLM call failed or its output didn't parse into any
        # known agent — fall back to the single default agent rather
        # than dropping the request (see DEFAULT_AGENT above).
        subtasks = [{"agent": DEFAULT_AGENT, "task": query}]

    trace = _log(state, "Orchestrator", f"plan -> {subtasks}")
    return {"subtasks": subtasks, "trace": trace}


def dispatch_node(state: OrchestratorState, *, agents: Dict[str, object]) -> dict:
    results = []
    trace = list(state.get("trace", []))
    for subtask in state["subtasks"]:
        agent = agents[subtask["agent"]]
        result = agent.handle(subtask["task"])
        results.append({**subtask, "answer": result.answer, "trace": list(result.trace)})
        trace.append(f"[Planner] dispatched to {subtask['agent']}: '{subtask['task']}'")
        trace.extend(f"  [{subtask['agent']}] {line}" for line in result.trace)
    return {"subtask_results": results, "trace": trace}


def compose_node(state: OrchestratorState, *, llm: Chatter) -> dict:
    results = state["subtask_results"]
    if len(results) == 1:
        # Nothing to compose — forward the one agent's answer verbatim,
        # with no extra LLM call needed (and nothing for one to add).
        trace = _log(state, "Orchestrator", f"single agent ({results[0]['agent']}) — forwarding its answer")
        return {"answer": results[0]["answer"], "trace": trace}

    try:
        answer = llm.chat(COMPOSER_SYSTEM_PROMPT, build_composer_message(state["query"], results))
        trace = _log(state, "Orchestrator", f"composed via {llm.model_id}")
    except ChatterUnavailable as exc:
        # Non-LLM fallback: show each agent's own answer rather than
        # losing the content, same principle as the domain agents'
        # template-draft fallbacks.
        parts = [f"[{r['agent']}] {r['answer']}" for r in results]
        answer = "\n\n".join(parts)
        trace = _log(state, "Orchestrator", f"LLM unavailable ({exc}); concatenating agent answers")
    return {"answer": answer, "trace": trace}


class OrchestratorGraph:
    """Builds and runs the top-level Orchestrator -> Planner -> domain
    agents -> Orchestrator graph. `agents` defaults to one
    ComplianceAgentGraph and one ResearchAgentGraph instance, keyed by
    the same names the planner prompt is told about — override for
    tests, or to add a future agent (Email, Calendar, ...) without
    touching this class."""

    def __init__(self, llm: Optional[Chatter] = None, agents: Optional[Dict[str, object]] = None):
        self.llm = llm or _default_llm()
        # Pass self.llm down to the default agents too — otherwise each
        # one would call its own _default_llm() internally and always
        # pick Bedrock unless LLM_PROVIDER=groq is set in the process
        # env, silently ignoring whatever provider this orchestrator was
        # actually built with (e.g. a UI toggle). Only applies to these
        # defaults — an explicit `agents=` override (tests, a future
        # agent) is used exactly as given.
        self.agents = agents if agents is not None else {
            "compliance": ComplianceAgentGraph(llm=self.llm),
            "research": ResearchAgentGraph(llm=self.llm),
        }
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)
        graph.add_node("plan", lambda s: planner_node(s, llm=self.llm, available_agents=self.agents.keys()))
        graph.add_node("dispatch", lambda s: dispatch_node(s, agents=self.agents))
        graph.add_node("compose", lambda s: compose_node(s, llm=self.llm))

        graph.set_entry_point("plan")
        graph.add_edge("plan", "dispatch")
        graph.add_edge("dispatch", "compose")
        graph.add_edge("compose", END)
        return graph.compile()

    def handle(self, query: str) -> SimpleNamespace:
        final_state = self.app.invoke({"query": query, "trace": []})
        return SimpleNamespace(**final_state)


if __name__ == "__main__":
    orch = OrchestratorGraph()
    result = orch.handle(
        "do we need to file a SAR for a structured cash deposit pattern, "
        "and also look up recent guidance on Kubernetes security"
    )
    print("\n".join(result.trace))
    print("\n--- ANSWER ---\n")
    print(result.answer)
