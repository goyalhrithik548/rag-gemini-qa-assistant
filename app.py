"""CLI entry point for the Gemini-powered RAG assistant."""

from __future__ import annotations

import sys
from pathlib import Path

from config import AppConfig, ConfigurationError, load_config
from src.embeddings import build_embeddings_model
from src.loader import load_text_documents
from src.memory import build_conversation_memory
from src.qa_chain import build_qa_chain
from src.splitter import split_documents
from src.utils import file_sha256, format_source_details
from src.vector_store import (
    build_and_save_vector_store,
    load_index_manifest,
    load_vector_store,
    save_index_manifest,
)


def _describe_exception(exc: Exception) -> str:
    """Convert common runtime failures into clearer user-facing messages."""
    message = str(exc)
    if "PERMISSION_DENIED" in message and (
        "generativelanguage.googleapis.com" in message or "denied access" in message.lower()
    ):
        return (
            "Gemini API access was denied for the manual key in config.py. Use a Google API key "
            "that has Gemini access enabled, then run the app again."
        )
    return message


def _build_index_manifest(config: AppConfig) -> dict[str, str | int]:
    """Describe the current source/config used to build the FAISS index."""
    return {
        "document_path": str(config.document_path),
        "document_hash": file_sha256(config.document_path),
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }


def _build_or_load_vector_store(config: AppConfig):
    """Load an existing FAISS index or create a new one from the text document."""
    embeddings = build_embeddings_model(
        api_key=config.google_api_key,
        model_name=config.embedding_model,
    )
    expected_manifest = _build_index_manifest(config)

    if config.faiss_index_dir.exists():
        saved_manifest = load_index_manifest(config.faiss_index_dir)
        if saved_manifest == expected_manifest:
            print(f"[startup] Loading existing FAISS index from: {config.faiss_index_dir}")
            try:
                return load_vector_store(config.faiss_index_dir, embeddings)
            except Exception as exc:
                print(
                    "[startup] Existing index could not be loaded. Rebuilding... "
                    f"({_describe_exception(exc)})"
                )
        else:
            print("[startup] Source document or chunk settings changed. Rebuilding FAISS index...")

    print(f"[startup] Building FAISS index from: {config.document_path}")
    documents = load_text_documents(config.document_path)
    chunks = split_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("No document chunks were created from the text file.")

    print(f"[startup] Created {len(chunks)} chunks. Saving FAISS index locally...")
    vector_store = build_and_save_vector_store(
        documents=chunks,
        embeddings=embeddings,
        index_path=config.faiss_index_dir,
    )
    save_index_manifest(config.faiss_index_dir, expected_manifest)
    return vector_store


def _print_welcome(document_path: Path) -> None:
    print("\nRAG-based AI Q&A Assistant")
    print(f"Document: {document_path.name}")
    print("Type your question and press Enter.")
    print("Type 'exit' to quit.\n")


def main() -> int:
    """Run the interactive CLI application."""
    try:
        config = load_config()
        vector_store = _build_or_load_vector_store(config)
        memory = build_conversation_memory()
        qa_chain = build_qa_chain(
            vector_store=vector_store,
            memory=memory,
            api_key=config.google_api_key,
            chat_model_name=config.chat_model,
            retrieval_k=config.retrieval_k,
        )
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        print(f"[error] {_describe_exception(exc)}")
        return 1

    _print_welcome(config.document_path)

    while True:
        try:
            question = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting assistant.")
            return 0

        if question.lower() == "exit":
            print("Exiting assistant.")
            return 0

        if not question:
            print("Please enter a non-empty question.")
            continue

        try:
            response = qa_chain.invoke({"question": question})
            answer = str(response.get("answer", "")).strip()
            sources = format_source_details(response.get("source_documents", []))

            print("\nAnswer:")
            print(answer or "I could not generate an answer.")
            if sources:
                print(f"\nSources: {sources}")
            print()
        except Exception as exc:  # pragma: no cover - runtime API/network guard
            print(f"[error] Failed to answer the question: {_describe_exception(exc)}\n")


if __name__ == "__main__":
    sys.exit(main())
