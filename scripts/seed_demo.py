"""Seed the database with realistic demo tickets so the analytics dashboard
has data to show — without spending any OpenAI credits.

Usage (from project root):
    python -m scripts.seed_demo
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import init_db, log_ticket
from app.workflow.classifier import ROUTING_TABLE

SAMPLES = {
    "Billing": [
        "I was charged twice for my subscription this month.",
        "How do I get a refund on my annual plan?",
        "My invoice doesn't match the price I signed up for.",
    ],
    "Technical": [
        "The dashboard keeps crashing when I open analytics.",
        "I can't log in, it says my password is invalid.",
        "The API is returning 500 errors since this morning.",
    ],
    "Shipping": [
        "Where is my order? It was supposed to arrive yesterday.",
        "Can I upgrade to express shipping after ordering?",
        "Do you ship to Canada?",
    ],
    "Returns": [
        "My webcam arrived damaged, I want a replacement.",
        "How do I return a laptop stand I bought last week?",
        "I received the wrong item in my order.",
    ],
    "Account": [
        "How do I add another team member to my account?",
        "I need to change the email on my account.",
        "Can I downgrade from Growth to Starter?",
    ],
    "General": [
        "Do you offer a free trial?",
        "What are your support hours?",
        "Is my data secure on your platform?",
    ],
}

PRIORITIES = ["Low", "Medium", "High", "Urgent"]
PRIORITY_WEIGHTS = [0.35, 0.4, 0.18, 0.07]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
SENTIMENT_WEIGHTS = [0.2, 0.5, 0.3]
CHANNELS = ["chat", "email", "api", "dashboard"]


def main(n: int = 60) -> None:
    init_db()
    random.seed(7)
    now = datetime.now(timezone.utc)
    for _ in range(n):
        category = random.choice(list(SAMPLES.keys()))
        message = random.choice(SAMPLES[category])
        priority = random.choices(PRIORITIES, PRIORITY_WEIGHTS)[0]
        sentiment = random.choices(SENTIMENTS, SENTIMENT_WEIGHTS)[0]
        created = now - timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        log_ticket(
            customer_message=message,
            channel=random.choice(CHANNELS),
            category=category,
            priority=priority,
            sentiment=sentiment,
            suggested_team=ROUTING_TABLE[category],
            created_at=created.isoformat(),
        )
    print(f"Seeded {n} demo tickets. Open the Analytics tab to view them.")


if __name__ == "__main__":
    main()
