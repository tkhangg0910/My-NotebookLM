from src.rag import retrieve, fetch_all_chunks
from src.schemas import Summary, QuizItem, QuizSet, FlashcardSet, Flashcard
from src.config import get_settings
from src.rag import render_prompt, format_citations
from src.llm import invoke_llm
import json
import re
from pydantic import ValidationError
from loguru import logger
settings=get_settings()
SUMMARY_SINGLE_TEMPLATE = "summary_single.j2"
SUMMARY_MAP_TEMPLATE = "summary_map.j2"
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.j2"
QUIZ_TEMPLATE = "quiz.j2"
FLASHCARDS_TEMPLATE = "flashcard.j2"
def _validate_summary_payload(payload):
    """
    Validate and normalize summary JSON returned by the LLM.

    Expected schema:
    {
        "summary": str,
        "key_points": list[str]
    }

    Returns:
        tuple[str, list[str]]
    """

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected summary payload to be a dict, got {type(payload).__name__}"
        )

    summary = payload.get("summary", "")
    key_points = payload.get("key_points", [])

    # normalize summary
    if summary is None:
        summary = ""
    elif not isinstance(summary, str):
        summary = str(summary)

    # normalize key_points
    if key_points is None:
        key_points = []
    elif isinstance(key_points, str):
        key_points = [key_points]
    elif not isinstance(key_points, list):
        raise ValueError(
            f"Expected key_points to be a list, got {type(key_points).__name__}"
        )

    normalized_key_points = []

    for item in key_points:
        if item is None:
            continue

        text = str(item).strip()

        if text:
            normalized_key_points.append(text)

    return summary.strip(), normalized_key_points

def _parse_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Internal JSON parsing error: {e}")
            logger.error(f"Faulty string: {json_str}")
            return {}
    else:
        logger.error(f"No JSON structure found in LLM output: {text}")
        return {}


def _resolve_target(document, query, filters, k, retrieval_k):
    effective_filters = dict(filters or {})
    
    if document:
        effective_filters["filename"] = document
    
    if query:
        chunks = retrieve(query, k=k or retrieval_k, filters=effective_filters)
        return chunks, "query", query
    if effective_filters:
        chunks = fetch_all_chunks(filters=effective_filters)
        scope = "document" if document else "filter"
        target = ", ".join(f"{k}={v}" for k, v in effective_filters.items())
        return chunks, scope, target
    return fetch_all_chunks(filters=None), "corpus", None

def summarize(document=None, query=None, filters=None, k=None):
    chunks, scope, target = _resolve_target(
        document,
        query,
        filters,
        k,
        settings.summarize_retrieval_k,
    )

    if not chunks:
        return Summary(
            scope=scope,
            target=target,
            summary="",
            key_points=[],
            citations=[],
            chunks=[],
        )

    if scope == "query":
        payload = _parse_json(
            invoke_llm(
                render_prompt(
                    SUMMARY_SINGLE_TEMPLATE,
                    chunks=chunks,
                )
            )
        )

        summary_text, key_points = _validate_summary_payload(payload)

    # Document / Filter / Corpus summary
    elif len(chunks) <= settings.summarize_batch_size:
        payload = _parse_json(
            invoke_llm(
                render_prompt(
                    SUMMARY_SINGLE_TEMPLATE,
                    chunks=chunks,
                )
            )
        )

        summary_text, key_points = _validate_summary_payload(payload)

    else:
        partials = []

        for start in range(
            0,
            len(chunks),
            settings.summarize_batch_size,
        ):
            batch = chunks[
                start : start + settings.summarize_batch_size
            ]

            payload = _parse_json(
                invoke_llm(
                    render_prompt(
                        SUMMARY_MAP_TEMPLATE,
                        chunks=batch
                    )
                )
            )

            summary_text, key_points = \
                _validate_summary_payload(payload)

            partials.append(
                {
                    "summary": summary_text,
                    "key_points": key_points,
                }
            )

        payload = _parse_json(
            invoke_llm(
                render_prompt(
                    SUMMARY_REDUCE_TEMPLATE,
                    partials=partials,
                )
            )
        )

        summary_text, key_points = \
            _validate_summary_payload(payload)

    return Summary(
        scope=scope,
        target=target,
        summary=summary_text,
        key_points=key_points,
        citations=format_citations(chunks),
        chunks=chunks,
    )
    
def _validate_items(payload, key, model_class, dedup_field, label, valid_markers):
    raw_items = payload.get(key)
    items, seen = [], set()
    for raw in raw_items:
        try:
            item = model_class.model_validate(raw)
        except ValidationError:
            continue
        
        norm = str(getattr(item, dedup_field, "")).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        markers = [m for m in item.source_markers if m in valid_markers]
        items.append(item.model_copy(update={"source_markers": markers}))
    
    if not items:
        raise RuntimeError(f"No valid {label} produced.")
    return items

def generate_quiz(document=None, query=None, filters=None, count=None, k=None):
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )
    
    n = count or settings.quiz_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(QUIZ_TEMPLATE, chunks=chunks, count=n)
    payload = _parse_json(invoke_llm(prompt))
    items = _validate_items(payload, "items", QuizItem, "question", "quiz items",valid_markers)

    return QuizSet(scope=scope, target=target, items=items, chunks=chunks,citations=format_citations(chunks))

def generate_flashcards(document=None, query=None, filters=None, count=None, k=None):
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )
    n = count or settings.quiz_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(FLASHCARDS_TEMPLATE, chunks=chunks, count=n)
    payload = _parse_json(invoke_llm(prompt))

    cards = _validate_items(payload, "cards", Flashcard, "front", "flashcards", valid_markers)

    return FlashcardSet(scope=scope,target=target, cards=cards, chunks=chunks,citations=format_citations(chunks))
