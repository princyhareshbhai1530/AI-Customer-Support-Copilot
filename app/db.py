"""Lightweight SQLite store that logs every handled ticket.

This powers the analytics dashboard and demonstrates a persistence layer
without requiring an external database service.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    channel         TEXT,
    customer_message TEXT NOT NULL,
    category        TEXT,
    priority        TEXT,
    sentiment       TEXT,
    suggested_team  TEXT,
    answer          TEXT,
    sources         TEXT
);
"""


@contextmanager
def _conn():
    settings = get_settings()
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def log_ticket(
    *,
    customer_message: str,
    channel: str = "chat",
    category: str | None = None,
    priority: str | None = None,
    sentiment: str | None = None,
    suggested_team: str | None = None,
    answer: str | None = None,
    sources: Iterable[str] | None = None,
    created_at: str | None = None,
) -> int:
    init_db()
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO tickets
                (created_at, channel, customer_message, category, priority,
                 sentiment, suggested_team, answer, sources)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at or datetime.now(timezone.utc).isoformat(),
                channel,
                customer_message,
                category,
                priority,
                sentiment,
                suggested_team,
                answer,
                json.dumps(list(sources) if sources else []),
            ),
        )
        return int(cur.lastrowid)


def fetch_tickets(limit: int | None = None) -> list[dict]:
    init_db()
    sql = "SELECT * FROM tickets ORDER BY created_at DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with _conn() as conn:
        rows = conn.execute(sql).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["sources"] = json.loads(d.get("sources") or "[]")
        except json.JSONDecodeError:
            d["sources"] = []
        out.append(d)
    return out


def analytics_summary() -> dict:
    """Aggregate counts used by the dashboard and the /analytics endpoint."""
    tickets = fetch_tickets()

    def _counts(field: str) -> dict:
        counts: dict[str, int] = {}
        for t in tickets:
            key = t.get(field) or "Unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    return {
        "total": len(tickets),
        "by_category": _counts("category"),
        "by_priority": _counts("priority"),
        "by_sentiment": _counts("sentiment"),
    }
