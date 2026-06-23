# AI Customer Support Copilot

A Generative-AI customer support copilot for small and medium-sized businesses.
It answers customer queries from a company knowledge base (RAG), drafts email
replies, and automatically triages incoming tickets — exposed through both a
**FastAPI** backend and an interactive **Streamlit** dashboard.

> Built with **Python, OpenAI API, LangChain, ChromaDB, FastAPI, Streamlit**.

---

## Features

| Capability | Where | How it works |
|---|---|---|
| **Answer customer queries** | Copilot tab / `POST /chat` | RAG over the knowledge base with source attribution |
| **Draft emails** | Email tab / `POST /draft-email` | LLM reply grounded in retrieved policies |
| **Knowledge-base retrieval** | All AI features | LangChain + ChromaDB vector store, OpenAI embeddings |
| **Workflow automation (triage)** | Triage tab / `POST /classify` | Structured classification → category, priority, sentiment, team routing |
| **Interactive dashboard** | Streamlit | Live analytics on every handled ticket |
| **Scalable API** | FastAPI | Documented endpoints at `/docs` for integration |

## Architecture

```
                    ┌────────────────────┐
                    │  Knowledge base     │  data/knowledge_base/*.md|txt|pdf
                    └─────────┬──────────┘
                              │  scripts/ingest.py
                              ▼
        OpenAI embeddings → ChromaDB vector store (data/vector_store/)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   app/rag (retrieve + answer)     app/workflow (classify + draft email)
              └───────────────┬───────────────┘
                              ▼
                    app/service.py  ──►  SQLite log (data/tickets.db)
              ┌───────────────┴───────────────┐
              ▼                               ▼
   FastAPI backend (app/api)        Streamlit dashboard (dashboard/)
   uvicorn app.api.main:app         streamlit run dashboard/streamlit_app.py
```

Both the API and the dashboard call the **same** `app/service.py` layer, so
business logic lives in one place. Every interaction is logged to SQLite, which
powers the analytics tab.

## Project structure

```
ai-support-copilot/
├── app/
│   ├── config.py            # settings from .env
│   ├── db.py                # SQLite ticket log + analytics
│   ├── service.py           # orchestration used by API + dashboard
│   ├── rag/
│   │   ├── ingest.py        # load → chunk → embed → persist
│   │   ├── retriever.py     # load Chroma store
│   │   └── chain.py         # RAG answer chain (LCEL)
│   ├── workflow/
│   │   ├── classifier.py    # structured triage + routing
│   │   └── email_drafter.py # grounded email drafting
│   └── api/main.py          # FastAPI endpoints
├── dashboard/streamlit_app.py
├── data/knowledge_base/     # sample company docs (edit these!)
├── scripts/
│   ├── ingest.py            # build the vector store
│   └── seed_demo.py         # seed demo analytics (no API key needed)
├── requirements.txt
└── .env.example
```

---

## Run it on Windows — step by step

### 0. Prerequisites
- **Python 3.10–3.12** — install from <https://www.python.org/downloads/> and
  **tick "Add python.exe to PATH"** during setup.
- An **OpenAI API key** — <https://platform.openai.com/api-keys>.

Verify Python in **PowerShell**:
```powershell
python --version
```

### 1. Open the project folder
Unzip the project, then in PowerShell:
```powershell
cd path\to\ai-support-copilot
```

### 2. Create and activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script ("running scripts is disabled"),
run this once, then activate again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
> Tip: using **Command Prompt (cmd)** instead? Activate with `venv\Scripts\activate.bat`.

You should see `(venv)` at the start of the prompt.

### 3. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(First install pulls in LangChain + ChromaDB and takes a few minutes.)

### 4. Add your API key
```powershell
copy .env.example .env
notepad .env
```
Replace `sk-your-key-here` with your real key, save, and close Notepad.

### 5. Build the knowledge-base index
```powershell
python -m scripts.ingest
```
This reads `data/knowledge_base/`, embeds it, and writes the vector store to
`data/vector_store/`. Re-run it whenever you change the knowledge base.

### 6. (Optional) Seed demo analytics
So the Analytics tab has data immediately — costs **no** API credits:
```powershell
python -m scripts.seed_demo
```

### 7. Run the dashboard
```powershell
streamlit run dashboard/streamlit_app.py
```
It opens at <http://localhost:8501>. Use the **Copilot**, **Email drafter**,
**Ticket triage**, and **Analytics** tabs.

### 8. (Optional) Run the API
Open a **second** PowerShell window, `cd` to the project, activate the venv
(`venv\Scripts\Activate.ps1`), then:
```powershell
uvicorn app.api.main:app --reload
```
Interactive API docs: <http://localhost:8000/docs>.

Quick test from a third terminal:
```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"How long do I have to return an item?\"}"
```

---

## Using your own knowledge base
Drop your own `.md`, `.txt`, or `.pdf` files into `data/knowledge_base/`
(delete the samples if you like), then re-run `python -m scripts.ingest`.
Change the company name and models in `.env`.

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Service + key status |
| `POST` | `/chat` | RAG answer (`{"message": "..."}`) |
| `POST` | `/classify` | Triage a ticket |
| `POST` | `/draft-email` | Draft a reply (`{"message": "...", "tone": "..."}`) |
| `GET`  | `/tickets?limit=50` | Recent logged tickets |
| `GET`  | `/analytics` | Aggregate counts |

## Cost & models
Defaults are the low-cost `gpt-4o-mini` (chat) and `text-embedding-3-small`
(embeddings); a typical demo costs a few cents. Change them in `.env`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | Make sure `.env` exists (not `.env.example`) and holds your key. |
| `Vector store not found` | Run `python -m scripts.ingest`. |
| PowerShell won't activate venv | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `streamlit` / `uvicorn` not recognized | The venv isn't active — re-run `venv\Scripts\Activate.ps1`. |
| ChromaDB build error on install | Upgrade pip (`python -m pip install --upgrade pip`) and retry; ensure Python 3.10–3.12. |
| Port already in use | Use `streamlit run ... --server.port 8502` or `uvicorn ... --port 8001`. |

## Notes for portfolio use
- Business logic is fully separated from the UI/API for clarity and testability.
- The classifier uses LLM **structured output** (validated Pydantic schema), so
  routing is reliable rather than free-text parsing.
- The RAG prompt is constrained to the retrieved context and instructed not to
  fabricate policies — answers cite their source files.
