"""Workflow automation: classify an incoming message into a category,
priority, and sentiment, and suggest a team to route it to.

Uses ChatOpenAI structured output so the result is a validated object.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..config import get_settings

# Default routing table: category -> team.
ROUTING_TABLE = {
    "Billing": "Finance / Billing",
    "Technical": "Engineering Support (Tier 2)",
    "Shipping": "Logistics",
    "Returns": "Returns & Refunds",
    "Account": "Account Management",
    "General": "Front-line Support (Tier 1)",
}


class TicketClassification(BaseModel):
    """Structured result of classifying a support message."""

    category: Literal[
        "Billing", "Technical", "Shipping", "Returns", "Account", "General"
    ] = Field(description="Best-fitting support category.")
    priority: Literal["Low", "Medium", "High", "Urgent"] = Field(
        description="Urgency: Urgent = outage/security/legal/very angry customer."
    )
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(
        description="Customer's emotional tone."
    )
    summary: str = Field(description="One-sentence summary of the issue.")
    suggested_team: str = Field(
        default="", description="Team that should own this ticket."
    )


def classify_ticket(message: str) -> TicketClassification:
    settings = get_settings()
    if not settings.ready:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.chat_model, temperature=0, api_key=settings.openai_api_key
    ).with_structured_output(TicketClassification)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You triage customer support messages for {company}. "
                "Classify the message precisely and concisely.",
            ),
            ("human", "{message}"),
        ]
    )
    result: TicketClassification = (prompt | llm).invoke(
        {"company": settings.company_name, "message": message}
    )

    # Fill the routing destination from our table if the model left it blank.
    if not result.suggested_team:
        result.suggested_team = ROUTING_TABLE.get(result.category, "Front-line Support")
    return result
