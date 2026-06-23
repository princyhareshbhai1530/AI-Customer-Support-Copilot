"""Draft a customer-facing email reply, optionally grounded in the
knowledge base via RAG."""
from __future__ import annotations

from ..config import get_settings

_PROMPT = """You are a customer support agent for {company}.
Write a complete email reply to the customer below.

Tone: {tone}
Guidelines:
- Greet the customer and acknowledge their issue.
- Use ONLY the reference context for factual claims (policies, prices, steps).
  If the context lacks a needed detail, write a placeholder in [square
  brackets] rather than inventing facts.
- Keep it clear and well structured. End with a professional sign-off from
  "The {company} Support Team".

Reference context from the knowledge base:
{context}

Customer message:
{message}
"""


def draft_email(
    customer_message: str,
    tone: str = "professional and warm",
    use_knowledge_base: bool = True,
) -> dict:
    settings = get_settings()
    if not settings.ready:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    context = "(knowledge base not used)"
    sources: list[str] = []
    if use_knowledge_base:
        from .retriever_bridge import safe_retrieve

        docs = safe_retrieve(customer_message)
        if docs:
            from ..rag.chain import format_docs

            context = format_docs(docs)
            sources = sorted({d.metadata.get("source", "?") for d in docs})

    prompt = ChatPromptTemplate.from_template(_PROMPT)
    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=0.4,
        api_key=settings.openai_api_key,
    )
    email = (prompt | llm | StrOutputParser()).invoke(
        {
            "company": settings.company_name,
            "tone": tone,
            "context": context,
            "message": customer_message,
        }
    )
    return {"email": email, "sources": sources}
