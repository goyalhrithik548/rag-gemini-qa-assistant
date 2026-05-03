"""Embedding model factory."""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings


class SafeGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """Embedding wrapper that uses one-text requests for better model compatibility."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            embeddings.extend(
                super().embed_documents(
                    [text],
                    batch_size=1,
                    task_type="RETRIEVAL_DOCUMENT",
                )
            )
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(text, task_type="RETRIEVAL_QUERY")


def build_embeddings_model(api_key: str, model_name: str) -> SafeGoogleGenerativeAIEmbeddings:
    """Create the Gemini embedding model used for FAISS indexing and retrieval."""
    return SafeGoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
    )
