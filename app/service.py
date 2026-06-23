"""High-level service layer used by BOTH the FastAPI backend and the
Streamlit dashboard. Every customer interaction is logged for analytics."""
from __future__ import annotations

from . import db
from .config import get_settings
from .rag.chain import answer_query
from .workflow.classifier import classify_ticket
from .workflow.email_drafter import draft_email


def handle_query(message: str, channel: str = "chat", classify: bool = True) -> dict:
    """Answer a customer query with RAG, optionally classify it, and log it."""
    rag = answer_query(message)

    classification = None
    if classify:
        try:
            classification = classify_ticket(message).model_dump()
        except Exception:
            classification = None

    db.log_ticket(
        customer_message=message,
        channel=channel,
        category=(classification or {}).get("category"),
        priority=(classification or {}).get("priority"),
        sentiment=(classification or {}).get("sentiment"),
        suggested_team=(classification or {}).get("suggested_team"),
        answer=rag["answer"],
        sources=rag["sources"],
    )

    return {
        "answer": rag["answer"],
        "sources": rag["sources"],
        "num_chunks": rag["num_chunks"],
        "classification": classification,
    }


def triage(message: str, channel: str = "triage") -> dict:
    """Classify and route a ticket without generating an answer."""
    result = classify_ticket(message).model_dump()
    db.log_ticket(
        customer_message=message,
        channel=channel,
        category=result["category"],
        priority=result["priority"],
        sentiment=result["sentiment"],
        suggested_team=result["suggested_team"],
    )
    return result


def make_email(
    message: str, tone: str = "professional and warm", use_kb: bool = True
) -> dict:
    return draft_email(message, tone=tone, use_knowledge_base=use_kb)


def get_analytics() -> dict:
    return db.analytics_summary()


def is_ready() -> bool:
    return get_settings().ready
