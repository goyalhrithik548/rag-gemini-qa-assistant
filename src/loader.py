"""Document loading utilities for the RAG assistant."""

from __future__ import annotations

from pathlib import Path
import re

from langchain_core.documents import Document


def load_text_documents(text_path: Path) -> list[Document]:
    """Load and split text file into logical sections."""

    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    raw_text = text_path.read_text(encoding="utf-8")

    # 🔥 Split by section headings (## ...)
    sections = re.split(r"\n## ", raw_text)

    documents = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        documents.append(
            Document(
                page_content=section,
                metadata={
                    "source": str(text_path),
                    "source_name": text_path.name,
                    "document_type": "text",
                },
            )
        )

    print(f"\n[DEBUG] Created {len(documents)} base documents (before chunking)\n")

    return documents