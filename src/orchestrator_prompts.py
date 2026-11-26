"""Shared prompt text for the top-level Orchestrator/Planner
(orchestrator_agent.py). Mirrors research_prompts.py's shape (one
*_SYSTEM_PROMPT constant + one build_*_message function per LLM call).
Both calls use BedrockSummarizer/GroqSummarizer's generic .chat() method,
same as research_prompts.py.
"""

# Keep this in sync with orchestrator_agent.py's AVAILABLE_AGENTS keys/
# descriptions — the planner can only ever route to names it's told
# about here.
PLANNER_SYSTEM_PROMPT = (
    "You are a planner that splits a user's request into subtasks for "
    "specialized agents. Available agents:\n"
    "- compliance: answers BSA/AML, FDA e-records, HIPAA, or SOX "
    "regulatory compliance questions, grounded in regulatory text.\n"
    "- research: researches a topic on the open web and synthesizes a "
    "report with sources.\n"
    "Decide which agent(s) the request actually needs and write one "
    "line per agent you choose, in the exact format "
    "'agent_name: subtask text', where subtask text is the piece of "
    "the original request that agent should handle, rephrased to "
    "stand alone if needed. Use ONLY the agent names listed above, "
    "lowercase, exactly as spelled. If the request needs only one "
    "agent, output only one line. Output nothing else — no numbering, "
    "no preamble, no commentary."
)


def build_planner_message(query: str) -> str:
    return f"User request: {query}\n\nSplit this into subtasks for the agents above."


COMPOSER_SYSTEM_PROMPT = (
    "You compose a single, coherent final answer for a user from one or "
    "more specialized agents' answers to pieces of their original "
    "request. Address the user's original request directly and weave "
    "the agents' content into flowing prose — don't just concatenate "
    "or label it by agent name. Preserve any citations or source URLs "
    "each agent included, verbatim. Do not add claims beyond what the "
    "agents provided."
)


def build_composer_message(query: str, subtask_results: list) -> str:
    parts = [f"User's original request: {query}"]
    for item in subtask_results:
        parts.append(f"\nSubtask ({item['agent']}): {item['task']}\nAgent's answer:\n{item['answer']}")
    parts.append("\nCompose the single final answer described above.")
    return "\n".join(parts)
