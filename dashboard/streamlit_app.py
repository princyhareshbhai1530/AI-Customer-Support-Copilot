"""Interactive Streamlit dashboard for the AI Customer Support Copilot.

Run from the project root:
    streamlit run dashboard/streamlit_app.py

The dashboard imports the core service layer directly, so it works
standalone. The FastAPI backend (app/api/main.py) exposes the same logic
for programmatic / integration use.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run via `streamlit run`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import service
from app.config import get_settings
from app.db import fetch_tickets, init_db
from app.rag.retriever import vector_store_exists

st.set_page_config(
    page_title="AI Support Copilot",
    page_icon="🎧",
    layout="wide",
)

# ------------------------------- styling -------------------------------
st.markdown(
    """
    <style>
      .stApp { background: #0e1117; }
      .badge {
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:0.78rem; font-weight:600; margin-right:6px;
      }
      .b-urgent { background:#5b1a1a; color:#ff8a8a; }
      .b-high   { background:#5b3a1a; color:#ffc48a; }
      .b-medium { background:#1a3a5b; color:#8ac4ff; }
      .b-low    { background:#1a5b32; color:#8affb0; }
      .b-team   { background:#2a2a3a; color:#c9c9ff; }
      .src-pill {
        display:inline-block; padding:2px 8px; border-radius:6px;
        background:#1c2230; color:#9fb4d8; font-size:0.75rem; margin:2px 4px 2px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()
init_db()


def _priority_badge(priority: str) -> str:
    cls = {"Urgent": "b-urgent", "High": "b-high", "Medium": "b-medium", "Low": "b-low"}
    return f'<span class="badge {cls.get(priority, "b-medium")}">{priority}</span>'


# ------------------------------- sidebar -------------------------------
with st.sidebar:
    st.title("🎧 Support Copilot")
    st.caption(f"Knowledge base for **{settings.company_name}**")

    if settings.ready:
        st.success("OpenAI API key detected")
    else:
        st.error("No OPENAI_API_KEY found. AI features are disabled.")
        st.caption("Add it to your `.env` file, then restart.")

    if vector_store_exists():
        st.info("Vector store: ready ✅")
    else:
        st.warning("Vector store not built. Run `python -m scripts.ingest`.")

    st.divider()
    st.caption(
        f"Chat model: `{settings.chat_model}`\n\n"
        f"Embeddings: `{settings.embedding_model}`"
    )


def _guard() -> bool:
    """Return True if AI features can run; otherwise show guidance."""
    if not settings.ready:
        st.error("Set `OPENAI_API_KEY` in your `.env` file to use this feature.")
        return False
    if not vector_store_exists():
        st.warning(
            "Knowledge base not indexed yet. Run `python -m scripts.ingest` "
            "in your terminal, then refresh."
        )
        return False
    return True


tab_chat, tab_email, tab_triage, tab_analytics = st.tabs(
    ["💬 Copilot", "✉️ Email drafter", "🧭 Ticket triage", "📊 Analytics"]
)

# ------------------------------- chat tab -------------------------------
with tab_chat:
    st.subheader("Ask the knowledge base")
    st.caption("Grounded answers with source attribution (RAG).")

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                pills = "".join(
                    f'<span class="src-pill">📄 {s}</span>' for s in turn["sources"]
                )
                st.markdown(pills, unsafe_allow_html=True)

    question = st.chat_input("e.g. How long do I have to return an item?")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        if _guard():
            with st.chat_message("assistant"):
                with st.spinner("Searching the knowledge base..."):
                    try:
                        result = service.handle_query(question, channel="dashboard")
                        st.markdown(result["answer"])
                        if result["sources"]:
                            pills = "".join(
                                f'<span class="src-pill">📄 {s}</span>'
                                for s in result["sources"]
                            )
                            st.markdown(pills, unsafe_allow_html=True)
                        if result.get("classification"):
                            c = result["classification"]
                            st.markdown(
                                _priority_badge(c["priority"])
                                + f'<span class="badge b-team">{c["category"]}'
                                f' → {c["suggested_team"]}</span>',
                                unsafe_allow_html=True,
                            )
                        st.session_state.history.append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Error: {exc}")

# ------------------------------- email tab -------------------------------
with tab_email:
    st.subheader("Draft a reply")
    st.caption("Generates a full email grounded in your policies.")

    col1, col2 = st.columns([3, 1])
    with col1:
        msg = st.text_area(
            "Customer message",
            height=160,
            placeholder="My order #12345 arrived damaged and I'd like a refund...",
        )
    with col2:
        tone = st.selectbox(
            "Tone",
            ["professional and warm", "concise and formal", "friendly and casual",
             "apologetic and reassuring"],
        )
        use_kb = st.toggle("Use knowledge base", value=True)

    if st.button("Draft email", type="primary"):
        if not msg.strip():
            st.warning("Enter a customer message first.")
        elif _guard():
            with st.spinner("Drafting..."):
                try:
                    out = service.make_email(msg, tone=tone, use_kb=use_kb)
                    st.text_area("Draft", out["email"], height=320)
                    if out["sources"]:
                        st.caption("Grounded in: " + ", ".join(out["sources"]))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error: {exc}")

# ------------------------------- triage tab -------------------------------
with tab_triage:
    st.subheader("Classify & route a ticket")
    st.caption("Workflow automation: category, priority, sentiment, routing.")

    t_msg = st.text_area(
        "Incoming message",
        height=140,
        placeholder="The app keeps crashing every time I try to log in!",
    )
    if st.button("Triage ticket", type="primary"):
        if not t_msg.strip():
            st.warning("Enter a message first.")
        elif settings.ready:
            with st.spinner("Classifying..."):
                try:
                    r = service.triage(t_msg)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Category", r["category"])
                    c2.metric("Priority", r["priority"])
                    c3.metric("Sentiment", r["sentiment"])
                    st.markdown(
                        f'**Route to:** <span class="badge b-team">'
                        f'{r["suggested_team"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.info(r["summary"])
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error: {exc}")
        else:
            st.error("Set `OPENAI_API_KEY` to use triage.")

# ------------------------------- analytics tab -------------------------------
with tab_analytics:
    st.subheader("Support analytics")
    st.caption("Live from every ticket handled by the copilot.")

    tickets = fetch_tickets()
    if not tickets:
        st.info(
            "No tickets logged yet. Use the other tabs, or seed demo data with "
            "`python -m scripts.seed_demo`."
        )
    else:
        df = pd.DataFrame(tickets)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total tickets", len(df))
        urgent = int((df["priority"].isin(["Urgent", "High"])).sum())
        m2.metric("High / urgent", urgent)
        neg = int((df["sentiment"] == "Negative").sum())
        m3.metric("Negative sentiment", neg)
        m4.metric("Categories", df["category"].nunique())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**By category**")
            st.bar_chart(df["category"].value_counts())
        with c2:
            st.markdown("**By priority**")
            st.bar_chart(df["priority"].value_counts())

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**By sentiment**")
            st.bar_chart(df["sentiment"].value_counts())
        with c4:
            st.markdown("**Volume over time**")
            by_day = (
                df.dropna(subset=["created_at"])
                .set_index("created_at")
                .resample("D")
                .size()
            )
            st.line_chart(by_day)

        st.markdown("**Recent tickets**")
        show = df[
            ["created_at", "channel", "category", "priority", "sentiment",
             "customer_message"]
        ].head(25)
        st.dataframe(show, hide_index=True)
