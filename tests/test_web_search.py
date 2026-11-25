"""Tests for WebSearcher — mocked, no real ddgs network calls."""
import pytest

from web_search import WebSearcher, WebSearchUnavailable


class _FakeDDGSClient:
    def __init__(self, results=None, exc=None):
        self.results = results if results is not None else []
        self.exc = exc
        self.received_kwargs = None

    def text(self, query, **kwargs):
        self.received_kwargs = {"query": query, **kwargs}
        if self.exc is not None:
            raise self.exc
        return self.results


def _searcher_with_fake_client(fake_client):
    s = WebSearcher()
    s._client = fake_client  # bypass lazy DDGS() construction
    return s


def test_clear_cut_search_shapes_results():
    fake = _FakeDDGSClient(results=[
        {"title": "Kubernetes docs", "href": "https://kubernetes.io/docs", "body": "What is Kubernetes"},
    ])
    searcher = _searcher_with_fake_client(fake)
    results = searcher.search("Kubernetes")
    assert results == [
        {"title": "Kubernetes docs", "url": "https://kubernetes.io/docs", "snippet": "What is Kubernetes"}
    ]


def test_clear_cut_passes_max_results():
    fake = _FakeDDGSClient(results=[])
    searcher = _searcher_with_fake_client(fake)
    searcher.search("Kubernetes", max_results=7)
    assert fake.received_kwargs["max_results"] == 7


def test_edge_empty_query_short_circuits_without_calling_client():
    fake = _FakeDDGSClient(results=[{"title": "x", "href": "y", "body": "z"}])
    searcher = _searcher_with_fake_client(fake)
    assert searcher.search("") == []
    assert searcher.search("   ") == []
    assert fake.received_kwargs is None


def test_edge_client_failure_raises_web_search_unavailable():
    fake = _FakeDDGSClient(exc=RuntimeError("simulated DDGS rate limit"))
    searcher = _searcher_with_fake_client(fake)
    with pytest.raises(WebSearchUnavailable):
        searcher.search("Kubernetes")


def test_edge_missing_fields_default_to_empty_strings():
    fake = _FakeDDGSClient(results=[{}])
    searcher = _searcher_with_fake_client(fake)
    assert searcher.search("Kubernetes") == [{"title": "", "url": "", "snippet": ""}]
