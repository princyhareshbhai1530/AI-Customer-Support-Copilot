"""Retrieval helper that returns [] instead of raising when the vector
store is missing — so email drafting still works before ingestion."""
from __future__ import annotations

from langchain_core.documents import Document

from ..rag.retriever import get_retriever, vector_store_exists


def safe_retrieve(query: str) -> list[Document]:
    if not vector_store_exists():
        return []
    try:
        return get_retriever().invoke(query)
    except Exception:
        return []
