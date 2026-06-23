from fastapi import FastAPI, UploadFile, File

from src.rag import answer
from src.learning import (
    summarize as summarize_learning,
    generate_quiz,
    generate_flashcards,
)

from src.filters import MetadataFilter, filters_to_dict
from src.schemas import (   
    RagAnswer,
    Summary,
    QuizSet,
    FlashcardSet,
)

from src.indexing import save_and_ingest_pdf

from pydantic import BaseModel, Field

from pydantic import BaseModel

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    message: str

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=64)
    filters: MetadataFilter | None = None


class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)


class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None

    count: int | None = Field(
        default=None,
        ge=1,
        le=50,
    )

    k: int | None = Field(
        default=None,
        ge=1,
        le=64,
    )


class FlashcardsRequest(QuizRequest):
    pass


app = FastAPI(
    title="RAG Learning API",
    description="Grounded Q&A, summaries, quizzes, and flashcards over indexed PDFs.",
    version="0.1.0",
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    content = await file.read()

    return save_and_ingest_pdf(
        content,
        file.filename or "",
    )


@app.post("/ask", response_model=RagAnswer)
def ask(req: AskRequest):
    return answer(
        req.question,
        k=req.k,
        filters=filters_to_dict(req.filters),
    )



@app.post("/summarize", response_model=Summary)
def summarize(req: SummarizeRequest):
    return summarize_learning(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        k=req.k,
    )


@app.post("/quiz", response_model=QuizSet)
def quiz(req: QuizRequest):
    return generate_quiz(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        count=req.count,
        k=req.k,
    )

@app.post("/flashcard", response_model=FlashcardSet)
def flashcard(req: FlashcardsRequest):
    return generate_flashcards(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        count=req.count,
        k=req.k,
    )