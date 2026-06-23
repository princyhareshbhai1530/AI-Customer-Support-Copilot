"""Ingestion pipeline: load knowledge-base files, chunk them, embed with
OpenAI, and persist into a Chroma vector store on disk."""
from __future__ import annotations

import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings


def load_documents(kb_dir: Path) -> list[Document]:
    """Load .md / .txt / .pdf files from the knowledge base directory."""
    kb_dir = Path(kb_dir)
    docs: list[Document] = []
    if not kb_dir.exists():
        return docs

    for path in sorted(kb_dir.glob("**/*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": path.name}))
        elif suffix == ".pdf":
            # Imported lazily so the project still runs without pypdf installed.
            from langchain_community.document_loaders import PyPDFLoader

            for d in PyPDFLoader(str(path)).load():
                d.metadata["source"] = path.name
                docs.append(d)
    return docs


def build_vector_store(reset: bool = True) -> dict:
    """(Re)build the vector store from the knowledge base. Returns stats."""
    settings = get_settings()
    if not settings.ready:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file before ingesting."
        )

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    documents = load_documents(settings.knowledge_base_dir)
    if not documents:
        raise RuntimeError(
            f"No documents found in {settings.knowledge_base_dir}. "
            "Add .md/.txt/.pdf files first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    if reset and Path(settings.vector_store_dir).exists():
        shutil.rmtree(settings.vector_store_dir)
    Path(settings.vector_store_dir).mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.vector_store_dir),
    )

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "files": sorted({d.metadata.get("source", "?") for d in documents}),
        "vector_store": str(settings.vector_store_dir),
    }
