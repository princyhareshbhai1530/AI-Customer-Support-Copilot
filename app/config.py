"""Central configuration. All settings can be overridden in a `.env` file."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.2

    # --- Paths ---
    knowledge_base_dir: Path = BASE_DIR / "data" / "knowledge_base"
    vector_store_dir: Path = BASE_DIR / "data" / "vector_store"
    db_path: Path = BASE_DIR / "data" / "tickets.db"

    # --- RAG tuning ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retriever_k: int = 4

    # --- Branding ---
    company_name: str = "Acme SaaS Inc."

    @property
    def ready(self) -> bool:
        """True when an API key is configured (AI features available)."""
        return bool(self.openai_api_key and self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Make the key visible to the OpenAI / LangChain SDKs as well.
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    return settings
