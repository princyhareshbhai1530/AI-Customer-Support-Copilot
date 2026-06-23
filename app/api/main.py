"""FastAPI backend for the AI Customer Support Copilot.

Run with:  uvicorn app.api.main:app --reload
Interactive docs at http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .. import service
from ..config import get_settings
from ..db import fetch_tickets, init_db

app = FastAPI(
    title="AI Customer Support Copilot API",
    description="RAG-powered support copilot: answer queries, draft emails, "
    "and triage tickets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


# --------------------------- request models ---------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., examples=["How long do I have to return an item?"])
    channel: str = "api"
    classify: bool = True


class TriageRequest(BaseModel):
    message: str
    channel: str = "api"


class EmailRequest(BaseModel):
    message: str
    tone: str = "professional and warm"
    use_knowledge_base: bool = True


# ------------------------------ routes -------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ai_ready": get_settings().ready}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    _require_ready()
    try:
        return service.handle_query(req.message, req.channel, classify=req.classify)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/classify")
def classify(req: TriageRequest) -> dict:
    _require_ready()
    try:
        return service.triage(req.message, req.channel)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/draft-email")
def draft_email(req: EmailRequest) -> dict:
    _require_ready()
    try:
        return service.make_email(req.message, req.tone, req.use_knowledge_base)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/tickets")
def tickets(limit: int = 50) -> dict:
    return {"tickets": fetch_tickets(limit=limit)}


@app.get("/analytics")
def analytics() -> dict:
    return service.get_analytics()


def _require_ready() -> None:
    if not get_settings().ready:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured. Add it to your .env file.",
        )
