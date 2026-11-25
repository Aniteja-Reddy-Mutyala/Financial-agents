"""Free, no-API-key web search for the Research agent (src/research_agent.py),
via the `ddgs` package (the current PyPI name for the renamed
`duckduckgo_search` — see .claude/skills/edit-research-agent/SKILL.md for
why this was picked over a paid search API). Same lazy-client + typed-
exception pattern as BedrockSummarizer/GroqSummarizer/BedrockEmbedder.
"""


class WebSearchUnavailable(Exception):
    pass


class WebSearcher:
    """Thin facade over ddgs.DDGS."""

    def __init__(self, max_results=5):
        self.max_results = max_results
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from ddgs import DDGS  # lazy import

            self._client = DDGS()
        return self._client

    def search(self, query, max_results=None):
        """Returns [{"title", "url", "snippet"}, ...]. Empty/whitespace
        query short-circuits to [] without calling the client, same
        convention as Retriever.search(). Raises WebSearchUnavailable on
        any client failure."""
        if not query or not query.strip():
            return []
        try:
            raw = self.client.text(query, max_results=max_results or self.max_results)
        except Exception as exc:
            raise WebSearchUnavailable(str(exc)) from exc
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in raw
        ]


if __name__ == "__main__":
    searcher = WebSearcher()
    for result in searcher.search("Python programming language", max_results=2):
        print(result["title"], "-", result["url"])
