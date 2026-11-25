"""langchain_aws wrapper around Amazon Bedrock Titan embeddings.

Stands in for the PGVector/Pinecone/ChromaDB dense-embedding retrieval
described in the original project — this is the real embedding half of
it, routed through AWS Bedrock via langchain_aws (same package family as
bedrock_llm.py) since that's what's available here. Defaults to Titan
Embed Text v2 at 512 dimensions with L2 normalization enabled, which lets
cosine similarity be computed as a plain dot product (see retriever.py).

Mirrors bedrock_llm.py's shape on purpose: lazy client, raises a typed
exception (BedrockEmbeddingUnavailable) on any failure instead of letting
raw boto3/langchain exceptions propagate, same rationale as
BedrockSummarizer/BedrockUnavailable.
"""
import os

DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_DIMENSIONS = 512


class BedrockEmbeddingUnavailable(Exception):
    pass


class BedrockEmbedder:
    """Thin facade over langchain_aws.BedrockEmbeddings."""

    def __init__(
        self,
        model_id=DEFAULT_MODEL_ID,
        region=DEFAULT_REGION,
        dimensions=DEFAULT_DIMENSIONS,
    ):
        self.model_id = model_id
        self.region = region
        self.dimensions = dimensions
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from langchain_aws import BedrockEmbeddings  # lazy import

            self._client = BedrockEmbeddings(
                model_id=self.model_id,
                region_name=self.region,
                dimensions=self.dimensions,
                normalize=True,  # lets callers use a dot product as cosine similarity
            )
        return self._client

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.client.embed_query(text)
        except Exception as exc:  # noqa: BLE001 - want to degrade gracefully either way
            raise BedrockEmbeddingUnavailable(str(exc)) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embeds many texts in one call where the API allows it.

        Used only by build_embeddings.py (the one-time indexing script) —
        query-time embedding always goes through embed_query, one call per
        user query, since there's only ever one query to embed at a time.
        """
        try:
            return self.client.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            raise BedrockEmbeddingUnavailable(str(exc)) from exc


if __name__ == "__main__":
    embedder = BedrockEmbedder()
    vec = embedder.embed_query("what is the CTR threshold for a cash deposit")
    print(f"model={embedder.model_id} dims={len(vec)} first5={vec[:5]}")
