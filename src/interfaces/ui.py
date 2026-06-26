"""
MyNotebookLM — Gradio UI
Connects to the RAG Learning API (src/api.py).
"""

import json
import requests
import gradio as gr
from src.config import settings

API_URL = settings.api_url


# ─────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────

def _post(endpoint: str, payload: dict | None = None, files=None):
    url = f"{API_URL}{endpoint}"
    try:
        if files:
            r = requests.post(url, files=files, timeout=120)
        else:
            r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"❌ Cannot connect to API at {API_URL}. Is the server running?"
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        return None, f"❌ API error {e.response.status_code}: {detail or str(e)}"
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"


def _fmt_citations(citations: list) -> str:
    if not citations:
        return ""
    lines = ["\n---\n**Sources**\n"]
    for c in citations:
        marker = c.get("source_marker", "?")
        filename = c.get("filename", "")
        page = c.get("page", "")
        lines.append(f"- **[{marker}]** `{filename}` — page {page}")
    return "\n".join(lines)


def _fmt_chunks(chunks: list) -> str:
    if not chunks:
        return ""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", 0)
        text = chunk.get("text", "").strip()
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "")
        page = meta.get("page", "")
        lines.append(
            f"### Chunk {i}  `score={score:.4f}`  —  `{filename}` p.{page}\n\n"
            f"```\n{text}\n```\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────
# Feature handlers
# ─────────────────────────────────────────

def upload_pdf(file):
    if file is None:
        return "No file selected."
    with open(file.name, "rb") as f:
        data, err = _post("/upload", files={"file": (file.name, f, "application/pdf")})
    if err:
        return err
    return (
        f"✅ **Indexed successfully**\n\n"
        f"- File: `{data['filename']}`\n"
        f"- Chunks indexed: `{data['chunks_indexed']}`"
    )


def reindex_images():
    data, err = _post("/reindex-images")
    if err:
        return err
    return f"✅ Reindex complete — {data.get('images_indexed', 0)} images indexed across {data.get('pdfs', 0)} PDFs."


def ask_question(question: str, k: int, document_filter: str, show_chunks: bool):
    if not question.strip():
        return "Please enter a question.", ""

    filters = {}
    if document_filter.strip():
        filters["filename"] = document_filter.strip()

    payload = {
        "question": question,
        "k": k,
        "filters": filters or None,
    }
    data, err = _post("/ask", payload)
    if err:
        return err, ""

    answer_md = data.get("answer", "")
    answer_md += _fmt_citations(data.get("citations", []))

    chunks_md = _fmt_chunks(data.get("chunks", [])) if show_chunks else ""
    return answer_md, chunks_md


def generate_summary(query: str, document: str, k: int):
    payload = {
        "query": query.strip() or None,
        "document": document.strip() or None,
        "k": k,
    }
    data, err = _post("/summarize", payload)
    if err:
        return err, "", ""

    summary_md = data.get("summary", "")
    key_points_md = "\n".join(f"- {p}" for p in data.get("key_points", []))
    citations_md = _fmt_citations(data.get("citations", []))

    return summary_md, key_points_md, citations_md


def generate_quiz(query: str, document: str, count: int):
    payload = {
        "query": query.strip() or None,
        "document": document.strip() or None,
        "count": count,
    }
    data, err = _post("/quiz", payload)
    if err:
        return err, None

    items = data.get("items", [])
    if not items:
        return "No quiz items generated.", None

    # Build readable markdown
    lines = []
    for i, item in enumerate(items, 1):
        options = item.get("options", [])
        correct_idx = item.get("correct_index", 0)
        explanation = item.get("explanation", "")
        topic = item.get("topic", "")
        difficulty = item.get("difficulty", "")

        lines.append(f"### Q{i}. {item['question']}\n")
        for j, opt in enumerate(options):
            prefix = "✅" if j == correct_idx else "○"
            lines.append(f"{prefix} **{chr(65+j)}.** {opt}")

        lines.append(f"\n> **Explanation:** {explanation}")
        if topic:
            lines.append(f"> *Topic: {topic}*")
        if difficulty:
            lines.append(f"> *Difficulty: {difficulty}*")
        lines.append("\n---")

    quiz_md = "\n".join(lines)
    raw_json = json.dumps(data, indent=2, ensure_ascii=False)
    return quiz_md, raw_json


def generate_flashcards(query: str, document: str, count: int):
    payload = {
        "query": query.strip() or None,
        "document": document.strip() or None,
        "count": count,
    }
    data, err = _post("/flashcard", payload)
    if err:
        return err, None

    cards = data.get("cards", [])
    if not cards:
        return "No flashcards generated.", None

    lines = []
    for i, card in enumerate(cards, 1):
        front = card.get("front", "")
        back = card.get("back", "")
        hint = card.get("hint", "")
        topic = card.get("topic", "")

        lines.append(f"### Card {i}")
        lines.append(f"**Q:** {front}\n")
        lines.append(f"**A:** {back}")
        if hint:
            lines.append(f"\n💡 *Hint: {hint}*")
        if topic:
            lines.append(f"🏷️ *Topic: {topic}*")
        lines.append("\n---")

    cards_md = "\n".join(lines)
    raw_json = json.dumps(data, indent=2, ensure_ascii=False)
    return cards_md, raw_json


# ─────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────

CSS = """
:root {
    --radius: 10px;
}

/* Tab bar */
.tab-nav button {
    font-weight: 600;
    font-size: 0.95rem;
}

/* Output panels */
.output-panel {
    background: var(--background-fill-secondary);
    border-radius: var(--radius);
    padding: 1rem;
    min-height: 120px;
}

/* Accordion / expander headers */
.gr-accordion > .label-wrap {
    font-weight: 600;
}

/* Download button */
.dl-btn {
    max-width: 200px;
}
"""

with gr.Blocks(
    title="📚 MyNotebookLM",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_md,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    ),
    css=CSS,
) as demo:

    gr.Markdown("# 📚 MyNotebookLM\nAsk questions, summarize, quiz, and study your PDF documents.")

    # ── Documents tab ──────────────────────────────────────
    with gr.Tab("📁 Documents"):
        gr.Markdown("### Upload & Index PDFs")

        with gr.Row():
            with gr.Column(scale=2):
                pdf_file = gr.File(
                    label="Select PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                upload_btn = gr.Button("⬆ Upload & Index", variant="primary")
                upload_status = gr.Markdown()

            with gr.Column(scale=1):
                gr.Markdown("### Vision Index")
                gr.Markdown(
                    "Re-embed all page images for multimodal retrieval "
                    "(`text_vl` / `hybrid` mode only)."
                )
                reindex_btn = gr.Button("🔄 Re-index Images", variant="secondary")
                reindex_status = gr.Markdown()

        upload_btn.click(upload_pdf, inputs=pdf_file, outputs=upload_status)
        reindex_btn.click(reindex_images, outputs=reindex_status)

    # ── Chat tab ───────────────────────────────────────────
    with gr.Tab("💬 Chat"):
        gr.Markdown("### Ask Questions")

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What does the document say about…?",
                    lines=2,
                )
            with gr.Column(scale=1):
                k_slider = gr.Slider(1, 20, value=5, step=1, label="Top K chunks")
                doc_filter = gr.Textbox(
                    label="Filter by filename (optional)",
                    placeholder="e.g. lecture_notes.pdf",
                )
                show_chunks = gr.Checkbox(label="Show retrieved chunks", value=False)

        ask_btn = gr.Button("Ask", variant="primary")

        answer_out = gr.Markdown(label="Answer", elem_classes="output-panel")

        with gr.Accordion("Retrieved chunks", open=False):
            chunks_out = gr.Markdown()

        ask_btn.click(
            ask_question,
            inputs=[question_input, k_slider, doc_filter, show_chunks],
            outputs=[answer_out, chunks_out],
        )
        question_input.submit(
            ask_question,
            inputs=[question_input, k_slider, doc_filter, show_chunks],
            outputs=[answer_out, chunks_out],
        )

    # ── Summary tab ────────────────────────────────────────
    with gr.Tab("📝 Summary"):
        gr.Markdown("### Generate Summary")
        gr.Markdown(
            "Leave both fields empty to summarize the entire corpus. "
            "Fill in **Query** for a topic-focused summary, or **Document** to scope to one file."
        )

        with gr.Row():
            sum_query = gr.Textbox(
                label="Topic / Query (optional)",
                placeholder="e.g. neural network training methods",
            )
            sum_doc = gr.Textbox(
                label="Filename (optional)",
                placeholder="e.g. paper.pdf",
            )
            sum_k = gr.Number(value=12, minimum=1, maximum=64, label="K", precision=0)

        sum_btn = gr.Button("Generate Summary", variant="primary")

        with gr.Row():
            with gr.Column(scale=2):
                sum_out = gr.Markdown(label="Summary", elem_classes="output-panel")
            with gr.Column(scale=1):
                keypoints_out = gr.Markdown(label="Key Points", elem_classes="output-panel")

        sum_citations_out = gr.Markdown()

        sum_btn.click(
            generate_summary,
            inputs=[sum_query, sum_doc, sum_k],
            outputs=[sum_out, keypoints_out, sum_citations_out],
        )

    # ── Quiz tab ───────────────────────────────────────────
    with gr.Tab("❓ Quiz"):
        gr.Markdown("### Generate Quiz")

        with gr.Row():
            quiz_query = gr.Textbox(
                label="Topic / Query (optional)",
                placeholder="e.g. transformer architecture",
            )
            quiz_doc = gr.Textbox(
                label="Filename (optional)",
                placeholder="e.g. textbook.pdf",
            )
            quiz_count = gr.Slider(1, 20, value=8, step=1, label="Number of questions")

        quiz_btn = gr.Button("Generate Quiz", variant="primary")

        quiz_out = gr.Markdown(label="Quiz", elem_classes="output-panel")

        with gr.Accordion("Download raw JSON", open=False):
            quiz_json = gr.Code(language="json", label="quiz.json")

        quiz_btn.click(
            generate_quiz,
            inputs=[quiz_query, quiz_doc, quiz_count],
            outputs=[quiz_out, quiz_json],
        )

    # ── Flashcards tab ─────────────────────────────────────
    with gr.Tab("🧠 Flashcards"):
        gr.Markdown("### Generate Flashcards")

        with gr.Row():
            flash_query = gr.Textbox(
                label="Topic / Query (optional)",
                placeholder="e.g. backpropagation",
            )
            flash_doc = gr.Textbox(
                label="Filename (optional)",
                placeholder="e.g. notes.pdf",
            )
            flash_count = gr.Slider(1, 50, value=15, step=1, label="Number of cards")

        flash_btn = gr.Button("Generate Flashcards", variant="primary")

        flash_out = gr.Markdown(label="Flashcards", elem_classes="output-panel")

        with gr.Accordion("Download raw JSON", open=False):
            flash_json = gr.Code(language="json", label="flashcards.json")

        flash_btn.click(
            generate_flashcards,
            inputs=[flash_query, flash_doc, flash_count],
            outputs=[flash_out, flash_json],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)