"""Tests for BedrockEmbedder — mocked, no real AWS/Bedrock calls (no cost).

We bypass the lazy `client` property by setting `_client` directly to a
fake object, mirroring test_bedrock_llm.py's pattern for BedrockSummarizer.
"""
import pytest

from bedrock_embeddings import BedrockEmbedder, BedrockEmbeddingUnavailable


class _FakeEmbeddingsClient:
    def __init__(self, vector=None):
        self.vector = vector or [0.1, 0.2, 0.3]
        self.queried_texts = []
        self.documents_batches = []

    def embed_query(self, text):
        self.queried_texts.append(text)
        return self.vector

    def embed_documents(self, texts):
        self.documents_batches.append(texts)
        return [self.vector for _ in texts]


class _FailingEmbeddingsClient:
    def embed_query(self, text):
        raise RuntimeError("simulated Bedrock throttling / access-denied error")

    def embed_documents(self, texts):
        raise RuntimeError("simulated Bedrock throttling / access-denied error")


def _embedder_with_fake_client(fake_client):
    e = BedrockEmbedder()
    e._client = fake_client  # bypass lazy BedrockEmbeddings construction
    return e


def test_embed_query_returns_vector_from_client():
    fake = _FakeEmbeddingsClient(vector=[1.0, 2.0, 3.0])
    embedder = _embedder_with_fake_client(fake)
    result = embedder.embed_query("what is the CTR threshold for a cash deposit")
    assert result == [1.0, 2.0, 3.0]
    assert fake.queried_texts == ["what is the CTR threshold for a cash deposit"]


def test_embed_query_raises_bedrock_embedding_unavailable_on_failure():
    embedder = _embedder_with_fake_client(_FailingEmbeddingsClient())
    with pytest.raises(BedrockEmbeddingUnavailable):
        embedder.embed_query("do we need to file a SAR?")


def test_embed_documents_batches_through_to_client():
    fake = _FakeEmbeddingsClient()
    embedder = _embedder_with_fake_client(fake)
    result = embedder.embed_documents(["chunk one", "chunk two"])
    assert len(result) == 2
    assert fake.documents_batches == [["chunk one", "chunk two"]]


def test_embed_documents_raises_bedrock_embedding_unavailable_on_failure():
    embedder = _embedder_with_fake_client(_FailingEmbeddingsClient())
    with pytest.raises(BedrockEmbeddingUnavailable):
        embedder.embed_documents(["a chunk"])


def test_default_model_is_titan_embed_v2_at_512_dims():
    # Regression guard, same spirit as bedrock_llm's cheapest-model test:
    # if this ever changes it should be a deliberate decision (cost/quality
    # tradeoff, and every cached vector in data/embeddings.npz would need
    # re-embedding to match), not an accident.
    embedder = BedrockEmbedder()
    assert embedder.model_id == "amazon.titan-embed-text-v2:0"
    assert embedder.dimensions == 512
