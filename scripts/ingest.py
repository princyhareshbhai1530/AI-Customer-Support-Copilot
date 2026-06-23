"""Build the vector store from the knowledge base.

Usage (from project root):
    python -m scripts.ingest
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag.ingest import build_vector_store


def main() -> None:
    settings = get_settings()
    if not settings.ready:
        print("ERROR: OPENAI_API_KEY is not set. Add it to your .env file.")
        raise SystemExit(1)

    print(f"Reading knowledge base from: {settings.knowledge_base_dir}")
    stats = build_vector_store(reset=True)
    print("Ingestion complete.")
    print(f"  Files indexed : {', '.join(stats['files'])}")
    print(f"  Documents     : {stats['documents']}")
    print(f"  Chunks        : {stats['chunks']}")
    print(f"  Vector store  : {stats['vector_store']}")


if __name__ == "__main__":
    main()
