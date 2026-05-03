"""Text splitting helpers."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 250,      # 🔥 Reduced for better granularity
    chunk_overlap: int = 50,    # 🔥 Keeps context between chunks
) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]   # removed "" to force splits earlier
    )

    chunks = splitter.split_documents(documents)

    # Add metadata for better traceability
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_number"] = index
        chunk.metadata.setdefault(
            "source_name",
            chunk.metadata.get("source", "document")
        )

    # 🔍 Debug: Print chunk info (optional but recommended for now)
    print(f"\n[DEBUG] Created {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks[:5], start=1):  # print first 5 only
        print(f"--- Chunk {i} ---")
        print(chunk.page_content[:200])
        print()

    return chunks