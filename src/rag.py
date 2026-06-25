from src.schemas import ChunkMetadata, RetrievedChunk, Citation, RagAnswer
from src.indexing import get_vector_store
from src.config import get_settings
from src.filters import filters_to_qdrant
from src.llm import invoke_llm
from src.store import get_client
from functools import lru_cache
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from src.reranker import get_reranker

PROMPTS_DIR = "src/prompts"
ANSWER_TEMPLATE = "answer.j2"

def scroll_all(collection_name, scroll_filter=None, batch_size=1000):
    client = get_client()

    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False, 
        )

        yield points

        if next_offset is None:
            break

        offset = next_offset


settings=get_settings()

def retrieve(query, k=None, filters=None, collection_name=None):
    hits = get_vector_store(collection_name).similarity_search_with_score(
        query=query,
        k=settings.top_c,
        filter=filters_to_qdrant(filters),
    )
    
    pairs = [
    (query, doc.page_content)
    for doc, _ in hits
    ]

    scores = get_reranker().predict(pairs)

    reranked = sorted(
        zip(hits, scores),
        key=lambda x: x[1],
        reverse=True
    )
    k = k if k else settings.top_k
    reranked = reranked[:k]
    return [
        RetrievedChunk(
            text=doc.page_content,
            score=float(score),
            metadata=ChunkMetadata(**doc.metadata),
        )
        for (doc, _), score in reranked
    ]
    
def fetch_all_chunks(filters=None, collection_name=None):
    name = collection_name or settings.qdrant_collection
    results = []

    for page in scroll_all(name, scroll_filter=filters_to_qdrant(filters)):
        for point in page:
            payload = point.payload or {}
            meta, text = payload.get("metadata") or {}, payload.get("page_content") or ""
            
            if meta and text:
                results.append(RetrievedChunk(text=text, score=0.0, metadata=ChunkMetadata(**meta)))

        return sorted(results, key=lambda r: (
            r.metadata.filename,
            r.metadata.page,
            int(r.metadata.chunk_id.rsplit(":", 1)[-1]),
        ))

@lru_cache(maxsize=1)
def _jinja_env():
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False, undefined=StrictUndefined,
        trim_blocks=True, lstrip_blocks=True,
    )

def render_prompt(template_name, **context):
    return _jinja_env().get_template(template_name).render(**context)

def format_citations(chunks):
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
            )
        for i, c in enumerate(chunks, start=1)
    ]
    

def answer(question, k=None, filters=None, collection_name=None):
    chunks = retrieve(question, k=k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."
        )

    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=chunks)
    text = invoke_llm(prompt)
    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(chunks),
        chunks=chunks,
    )
