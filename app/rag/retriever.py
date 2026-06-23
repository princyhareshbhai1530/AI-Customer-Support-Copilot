"""Load the persisted Chroma vector store and expose a retriever."""
from __future__ import annotations

from pathlib import Path

from ..config import get_settings


def vector_store_exists() -> bool:
    settings = get_settings()
    vs = Path(settings.vector_store_dir)
    return vs.exists() and any(vs.iterdir())


def get_vector_store():
    settings = get_settings()
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    return Chroma(
        persist_directory=str(settings.vector_store_dir),
        embedding_function=embeddings,
    )


def get_retriever(k: int | None = None):
    settings = get_settings()
    return get_vector_store().as_retriever(
        search_kwargs={"k": k or settings.retriever_k}
    )
