"""Application configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Manual project configuration
GOOGLE_API_KEY = "your_api_key"
GEMINI_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "models/gemini-embedding-2"
SOURCE_FILE_NAME = "sample_file.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4


class ConfigurationError(ValueError):
    """Raised when a required configuration value is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    """Strongly-typed runtime configuration."""

    base_dir: Path
    data_dir: Path
    document_path: Path
    faiss_index_dir: Path
    google_api_key: str
    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int


def load_config() -> AppConfig:
    """Load and validate project settings from in-code configuration."""
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    document_path = data_dir / SOURCE_FILE_NAME
    faiss_index_dir = base_dir / "faiss_index"

    if not GOOGLE_API_KEY.strip():
        raise ConfigurationError(
            "Missing GOOGLE_API_KEY in config.py. Add your Gemini API key to the "
            "GOOGLE_API_KEY constant before running the app."
        )
    if not document_path.exists():
        raise FileNotFoundError(
            f"Missing input text file: {document_path}. Place {SOURCE_FILE_NAME} inside the data folder."
        )
    if CHUNK_SIZE <= 0:
        raise ConfigurationError("CHUNK_SIZE must be a positive integer.")
    if CHUNK_OVERLAP < 0:
        raise ConfigurationError("CHUNK_OVERLAP cannot be negative.")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ConfigurationError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    if RETRIEVAL_K <= 0:
        raise ConfigurationError("RETRIEVAL_K must be a positive integer.")

    return AppConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        document_path=document_path,
        faiss_index_dir=faiss_index_dir,
        google_api_key=GOOGLE_API_KEY.strip(),
        embedding_model=EMBEDDING_MODEL,
        chat_model=GEMINI_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        retrieval_k=RETRIEVAL_K,
    )
