"""Small reusable utility helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


def normalise_text(text: str) -> str:
    """Collapse noisy whitespace into cleaner retrieval-friendly text."""
    text = text or ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_meaningful_text(text: str, minimum_characters: int = 20) -> bool:
    """Check whether a text block contains enough content to be useful."""
    candidate = re.sub(r"\s+", " ", text or "").strip()
    return len(candidate) >= minimum_characters


def file_sha256(path: Path) -> str:
    """Create a stable SHA-256 fingerprint for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_source_details(documents: Iterable[Document]) -> str:
    """Format source chunk metadata for CLI display."""
    chunk_numbers: set[int] = set()
    source_names: set[str] = set()

    for document in documents or []:
        metadata = document.metadata or {}
        chunk_number = metadata.get("chunk_number")
        source_name = metadata.get("source_name")

        if chunk_number is not None:
            chunk_numbers.add(int(chunk_number))
        if source_name:
            source_names.add(str(source_name))

    if not source_names and not chunk_numbers:
        return ""

    source_label = ", ".join(sorted(source_names)) if source_names else "document"
    if not chunk_numbers:
        return source_label

    joined_chunks = ", ".join(str(number) for number in sorted(chunk_numbers))
    return f"{source_label} | chunks {joined_chunks}"
