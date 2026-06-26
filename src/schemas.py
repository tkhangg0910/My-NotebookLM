from pydantic import BaseModel, Field, model_validator
from typing import Literal

class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    source: str
    page: int
    chunk_id: str
    section: str | None = None
    page_image: str | None = None
    
class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: ChunkMetadata
    
class Citation(BaseModel):
    source_index: int
    source_marker: str
    filename: str
    page: int
    section: str | None = None
    chunk_id: str | None = None

class RagAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    
class Summary(BaseModel):
    scope: Literal["query","document", "filter", "corpus"]
    target: str | None = None
    summary: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)

class QuizItem(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int
    explanation: str
    source_markers: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    topic: str | None = None

    @model_validator(mode="after")
    def _validate_correct_index(self) -> "QuizItem":
        if not 0 <= self.correct_index < len(self.options):
            raise ValueError("correct_index out of range")
        return self

class Flashcard(BaseModel):
    front: str
    back: str
    hint: str | None = None
    topic: str | None = None
    source_markers: list[str] = Field(default_factory=list)

class FlashcardSet(BaseModel):
    scope: Literal["query", "document", "filter", "corpus"]
    target: str | None = None
    cards: list[Flashcard] = Field(default_factory=list)  
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    
class QuizSet(BaseModel):
    scope: Literal["query", "document", "filter", "corpus"]
    target: str | None = None
    items: list[QuizItem] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)