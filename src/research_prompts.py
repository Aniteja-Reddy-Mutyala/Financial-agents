"""Shared prompt text for the Research agent (research_agent.py). Mirrors
llm_prompts.py's shape (one *_SYSTEM_PROMPT constant + one build_*_message
function per LLM call), but there's only one call site per prompt here —
both BedrockSummarizer and GroqSummarizer expose the same generic
.chat(system_prompt, user_message) method (see bedrock_llm.py/groq_llm.py),
so there's no risk of the two providers drifting apart in tone the way
llm_prompts.py was originally written to prevent for the compliance agent.
"""

TOPIC_EXTRACTION_SYSTEM_PROMPT = (
    "You extract concrete, researchable topics from a document and/or a "
    "user's question — things like specific skills, technologies, "
    "companies, tools, or named claims that are worth looking up on the "
    "web. You are NOT answering anything yourself here. Return at most 3 "
    "topics, one per line, each a short noun phrase (2-6 words) specific "
    "enough to search the web for — prefer 'Kubernetes' or 'Northern "
    "Trust compliance platform' over vague phrases like 'technical "
    "skills' or 'work history.' If fewer than 3 genuinely distinct, "
    "concrete topics are present, return fewer rather than padding the "
    "list with vague ones. Return nothing else — no numbering, no "
    "preamble, just the topic lines."
)


def build_topic_extraction_message(query, document_text):
    parts = []
    if document_text:
        parts.append(f"Document text:\n{document_text}")
    if query:
        parts.append(f"User's question/focus: {query}")
    return "\n\n".join(parts) + "\n\nList up to 3 topics to research, one per line."


RESEARCH_SYNTHESIS_SYSTEM_PROMPT = (
    "You write a short research report combining (optionally) a parsed "
    "document and web search results on up to 3 topics. These web "
    "results are UNVERIFIED, user-generated or publisher content from "
    "an open web search — not fact-checked, not regulatory or "
    "authoritative text, and may be outdated, promotional, or simply "
    "wrong. Treat every web source as a lead to weigh, not a fact to "
    "repeat uncritically. For each topic, summarize what the sources "
    "say, cite the specific URL(s) you drew from inline, and explicitly "
    "flag when sources disagree or when you could only find thin/low-"
    "quality coverage. If a topic returned no search results, say so "
    "plainly rather than inventing information to fill the gap. Close "
    "with one short line reminding the reader these are unverified web "
    "sources, not vetted research. Write in flowing prose per topic, not "
    "a bare link dump."
)


def build_synthesis_message(document_text, topics, search_results):
    parts = []
    if document_text:
        parts.append(f"Document text:\n{document_text}")
    for topic in topics:
        results = search_results.get(topic, [])
        if not results:
            parts.append(f"Topic: {topic}\n(no web search results found)")
            continue
        block = "\n".join(f"- {r['title']} ({r['url']}): {r['snippet']}" for r in results)
        parts.append(f"Topic: {topic}\n{block}")
    return "\n\n".join(parts) + "\n\nWrite the research report described above."
