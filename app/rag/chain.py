"""RAG question-answering chain (LangChain Expression Language)."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..config import get_settings
from .retriever import get_retriever, vector_store_exists

_SYSTEM_PROMPT = """You are the customer-support copilot for {company}.
Answer the customer's question using ONLY the context below, which is drawn
from the company knowledge base.

Rules:
- Be concise, friendly, and accurate.
- If the context does not contain the answer, say you don't have that
  information on file and offer to escalate to a human agent. Do not invent
  policies, prices, or facts.
- When you use a fact, you may refer to it naturally (e.g. "per our return
  policy").

Context:
{context}
"""

_HUMAN_PROMPT = "Customer question: {question}"


def format_docs(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "knowledge base")
        blocks.append(f"[{i}] (source: {src})\n{d.page_content}")
    return "\n\n".join(blocks)


def answer_query(question: str) -> dict:
    """Retrieve context and generate an answer. Returns answer + sources."""
    settings = get_settings()
    if not settings.ready:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not vector_store_exists():
        raise RuntimeError(
            "Vector store not found. Run `python -m scripts.ingest` first."
        )

    from langchain_openai import ChatOpenAI

    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = format_docs(docs) if docs else "(no relevant documents found)"

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
    )
    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {"company": settings.company_name, "context": context, "question": question}
    )

    sources = sorted({d.metadata.get("source", "knowledge base") for d in docs})
    return {"answer": answer, "sources": sources, "num_chunks": len(docs)}
