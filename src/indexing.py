import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_settings
from collections import defaultdict
from src.schemas import ChunkMetadata
import uuid
from pathlib import Path
from src.store import get_vector_store, ensure_collection, ensure_image_collection
from langchain_experimental.text_splitter import SemanticChunker
from src.store import get_embeddings, get_client
from src.vision import embed_images
from qdrant_client.http import models as qmodels

import pymupdf

settings = get_settings()
def vision_enabled():
    return settings.rag_mode in {
        "text_vl",
        "hybrid",
    }
def discover_pdfs():
    return list(settings.data_dir.rglob("*.pdf"))

def _document_id(path):
    raw = f"{path.name}:{path.stat().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def _chunk_id(doc_id, page, index):
    return f"{doc_id}:{page}:{index}"

def _load_pdf_text(path):
    pages = PyPDFLoader(str(path)).load()
    doc_id = _document_id(path)

    for doc in pages:
        page_number = int(doc.metadata.get("page", 0)) + 1

        doc.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section"),
            "page_image": str(
                settings.image_dir /
                f"{doc_id}_{page_number}.png"
            ),
        }

    return pages

def _extract_page_images(path):
    doc_id = _document_id(path)

    settings.image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = []

    with pymupdf.open(str(path)) as pdf:
        for page_idx in range(len(pdf)):
            page_number = page_idx + 1

            image_path = (
                settings.image_dir /
                f"{doc_id}_{page_number}.png"
            )

            if not image_path.exists():
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False,
                )
                pix.save(image_path)

            images.append(
                {
                    "document_id": doc_id,
                    "filename": path.name,
                    "page": page_number,
                    "image_path": str(image_path),
                }
            )

    return images

def _load_pdf(path):
    pages = _load_pdf_text(path)
    page_images = []
    if vision_enabled():
        page_images = _extract_page_images(path)

    return pages, page_images

def _splitter(chunk_size=None, chunk_overlap=None):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )

def semantic_chunker():
    return SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type="interquartile",
    )
    
def build_chunks(pdf_paths, chunk_size=None, chunk_overlap=None, chunker=None):
    page_docs = []
    all_page_images = []

    for path in pdf_paths:
        pages, page_images = _load_pdf(path)
        page_docs.extend(pages)
        all_page_images.extend(page_images)

    default_chunker = semantic_chunker() if settings.chunker_type=="semantic" else _splitter(chunk_size, chunk_overlap) 
    splitter = chunker or default_chunker
    chunks = splitter.split_documents(page_docs)
    per_doc_counter = defaultdict(int)
    
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc_counter[doc_id]
        per_doc_counter[doc_id] += 1
        meta = ChunkMetadata(
            document_id=doc_id,
            filename=chunk.metadata["filename"],
            source=chunk.metadata["source"],
            page=chunk.metadata["page"],
            chunk_id=_chunk_id(doc_id, chunk.metadata["page"], idx),
            section=chunk.metadata.get("section"),
        )
        chunk.metadata = meta.model_dump()
    return chunks, all_page_images

def index_chunks(chunks, collection_name=None):
    if not chunks:
        return 0

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    get_vector_store(collection_name=collection_name).add_documents(chunks, ids=ids)
    return len(chunks)

def index_images(page_images):
    if not page_images:
        return 0

    image_paths = [
        item["image_path"]
        for item in page_images
    ]

    embeddings = embed_images(image_paths)

    client = get_client()

    points = []

    for idx, (meta, emb) in enumerate(
        zip(page_images, embeddings)
    ):
        image_id = (
            f"{meta['document_id']}"
            f":{meta['page']}"
        )
        points.append(
            qmodels.PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        image_id,
                    )
                ),
                vector={
                    "original": emb.tolist()
                },
                payload=meta,
            )
        )

    client.upsert(
        collection_name=settings.image_collection,
        points=points,
        wait=True,
    )

    return len(points)

def reindex_images(recreate=False):
    pdfs = discover_pdfs()

    ensure_image_collection(
        recreate=recreate,
    )

    all_page_images = []

    for pdf in pdfs:
        page_images = _extract_page_images(pdf)
        all_page_images.extend(page_images)

    count = index_images(
        all_page_images
    )

    return {
        "pdfs": len(pdfs),
        "images_indexed": count,
    }

def ingest(recreate=False, collection_name=None, chunker=None, chunk_size=None, chunk_overlap=None):
    pdfs = discover_pdfs()
    ensure_collection(recreate=recreate, collection_name=collection_name)
    chunks, page_images = build_chunks(pdfs, chunker=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    text_count = index_chunks(chunks, collection_name=collection_name)
    
    image_count = 0
    if vision_enabled():
        ensure_image_collection(recreate=recreate, collection_name=collection_name)
        image_count = index_images(page_images)

    return {
        "text_chunks": text_count,
        "page_images": image_count,
    }

def save_and_ingest_pdf(file_bytes, filename):
    safe_name = Path(filename).name
    dest = settings.data_dir / safe_name
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file_bytes)

    ensure_collection(recreate=False)
    chunks, page_images = build_chunks([dest])

    text_count = index_chunks(chunks)
    image_count = 0

    if vision_enabled():
        ensure_image_collection(recreate=False)
        image_count = index_images(page_images)

    return {
        "filename": safe_name,
        "chunks_indexed": text_count,
        "images_indexed": image_count,
    }
