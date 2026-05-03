"""FAISS vector store helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def build_and_save_vector_store(
    documents: List[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
    index_path: Path,
) -> FAISS:
    """Create a FAISS index from documents and save it locally."""
    if index_path.exists():
        shutil.rmtree(index_path)

    vector_store = FAISS.from_documents(documents, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))
    return vector_store


def load_vector_store(index_path: Path, embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_index_manifest(index_path: Path) -> dict[str, Any] | None:
    """Load the saved index manifest if it exists."""
    manifest_path = index_path / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_index_manifest(index_path: Path, manifest: dict[str, Any]) -> None:
    """Persist metadata about the FAISS index inputs."""
    manifest_path = index_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
