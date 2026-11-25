"""Shared prompt text for the Summarizer step, used by both
bedrock_llm.py and groq_llm.py so the two providers can't silently drift
apart into answering the same question differently depending on which
one happens to be configured. If you change the prompt, both providers
get the change — don't copy-paste a tweak into just one.
"""

SYSTEM_PROMPT = (
    "You are a regulatory compliance assistant covering BSA/AML (31 CFR), "
    "FDA electronic records (21 CFR Part 11), HIPAA (45 CFR), and SOX-related "
    "SEC rules (17 CFR). Answer ONLY using the provided regulatory excerpts. "
    "Be concise and precise, cite the exact section numbers given in the "
    "excerpts (e.g. 31 CFR 1020.320, 45 CFR 164.502), and never state an "
    "obligation that isn't supported by the excerpts. If the excerpts don't "
    "fully answer the question, say what's missing. Each excerpt may contain "
    "more than one numbered or lettered clause (e.g. (a), (b), (c)) — read "
    "every clause in every excerpt before answering; a requirement in a "
    "later clause (e.g. identity verification in clause (b)) is just as "
    "binding as one in an earlier or more prominent clause, and skipping it "
    "produces a wrong answer even when the citation is correct."
)


def build_user_message(query, intent_description, evidence):
    """evidence: list of {source_id, title, text} dicts already filtered to citations."""
    excerpt_block = "\n\n".join(
        f"{e['source_id']} — {e['title']}\n{e['text']}" for e in evidence
    )
    return (
        f"Compliance question: {query}\n"
        f"Detected intent: {intent_description}\n\n"
        f"Regulatory excerpts:\n{excerpt_block}\n\n"
        "Write a short answer (3-6 sentences) for a compliance analyst, "
        "citing the specific regulatory sections used (as given above)."
    )
