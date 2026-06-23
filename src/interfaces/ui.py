import json
import requests
import streamlit as st

from src.config import settings

API_URL = settings.api_url

st.set_page_config(
    page_title="MyNotebookLM",
    page_icon="📚",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------

def post(endpoint: str, payload=None, files=None):
    url = f"{API_URL}{endpoint}"

    if files:
        r = requests.post(url, files=files)
    else:
        r = requests.post(url, json=payload)

    r.raise_for_status()
    return r.json()


def render_citations(citations):
    if not citations:
        return

    st.markdown("### Sources")

    for c in citations:
        st.markdown(
            f"""
            **{c['source_marker']}**

            - File: `{c['filename']}`
            - Page: `{c['page']}`
            - Chunk: `{c.get('chunk_id','')}`
            """
        )


def render_chunks(chunks):
    if not chunks:
        return

    with st.expander("Retrieved Chunks"):
        for idx, chunk in enumerate(chunks, start=1):
            st.markdown(
                f"### Chunk {idx} "
                f"(score={chunk['score']:.4f})"
            )

            st.code(
                chunk["text"],
                language="text",
            )


# -----------------------------
# Session state
# -----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("📚 MyNotebookLM")

    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
    )

    if uploaded:

        if st.button("Index Document"):

            files = {
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    "application/pdf",
                )
            }

            with st.spinner("Indexing PDF..."):

                result = post(
                    "/upload",
                    files=files,
                )

            st.success("Indexed successfully")

            st.json(result)

    st.markdown("---")

    st.subheader("Retrieval")

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.markdown("---")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# -----------------------------
# Main
# -----------------------------

st.title("📚 MyNotebookLM")

tab_chat, tab_summary, tab_quiz, tab_flash = st.tabs(
    [
        "💬 Chat",
        "📝 Summary",
        "❓ Quiz",
        "🧠 Flashcards",
    ]
)

# =====================================================
# CHAT
# =====================================================

with tab_chat:

    st.subheader("Ask Questions")

    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Thinking..."):

            result = post(
                "/ask",
                {
                    "question": question,
                    "k": top_k,
                },
            )

        answer = result["answer"]

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        with st.chat_message("assistant"):

            st.markdown(answer)

            render_citations(
                result.get("citations", [])
            )

            render_chunks(
                result.get("chunks", [])
            )

# =====================================================
# SUMMARY
# =====================================================

with tab_summary:

    st.subheader("Generate Summary")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Topic / Query",
            key="summary_query",
        )

    with col2:
        summary_k = st.number_input(
            "K",
            min_value=1,
            max_value=64,
            value=12,
        )

    if st.button("Generate Summary"):

        with st.spinner("Summarizing..."):

            result = post(
                "/summarize",
                {
                    "query": query or None,
                    "k": summary_k,
                },
            )

        st.markdown("## Summary")

        st.write(result["summary"])

        st.markdown("## Key Points")

        for item in result["key_points"]:
            st.markdown(f"- {item}")

        render_citations(
            result.get("citations", [])
        )

        render_chunks(
            result.get("chunks", [])
        )

        st.download_button(
            "Download JSON",
            data=json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="summary.json",
            mime="application/json",
        )

# =====================================================
# QUIZ
# =====================================================

with tab_quiz:

    st.subheader("Generate Quiz")

    quiz_count = st.slider(
        "Number of questions",
        1,
        20,
        8,
    )

    quiz_query = st.text_input(
        "Optional topic",
        key="quiz_query",
    )

    if st.button("Generate Quiz"):

        with st.spinner("Creating quiz..."):

            result = post(
                "/quiz",
                {
                    "query": quiz_query or None,
                    "count": quiz_count,
                },
            )

        for idx, item in enumerate(
            result["items"],
            start=1,
        ):

            st.markdown(
                f"### Q{idx}. {item['question']}"
            )

            answer = st.radio(
                "",
                item["options"],
                key=f"quiz_{idx}",
            )

            with st.expander("Show Answer"):

                st.success(
                    item["options"][
                        item["correct_index"]
                    ]
                )

                st.write(
                    item["explanation"]
                )

                if item.get("topic"):
                    st.caption(
                        f"Topic: {item['topic']}"
                    )

        render_citations(
            result.get("citations", [])
        )

        st.download_button(
            "Download Quiz JSON",
            data=json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="quiz.json",
            mime="application/json",
        )

# =====================================================
# FLASHCARDS
# =====================================================

with tab_flash:

    st.subheader("Generate Flashcards")

    flash_count = st.slider(
        "Number of cards",
        1,
        50,
        15,
    )

    flash_query = st.text_input(
        "Optional topic",
        key="flash_query",
    )

    if st.button("Generate Flashcards"):

        with st.spinner("Creating flashcards..."):

            result = post(
                "/flashcard",
                {
                    "query": flash_query or None,
                    "count": flash_count,
                },
            )

        cards = result["cards"]

        for idx, card in enumerate(
            cards,
            start=1,
        ):

            with st.expander(
                f"{idx}. {card['front']}"
            ):

                st.markdown(
                    f"**Answer**\n\n{card['back']}"
                )

                if card.get("hint"):
                    st.info(
                        f"Hint: {card['hint']}"
                    )

                if card.get("topic"):
                    st.caption(
                        f"Topic: {card['topic']}"
                    )

        render_citations(
            result.get("citations", [])
        )

        st.download_button(
            "Download Flashcards JSON",
            data=json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            file_name="flashcards.json",
            mime="application/json",
        )
