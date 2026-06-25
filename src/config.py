from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import Field, model_validator
from typing import Literal
from functools import lru_cache
QuantizationType = Literal[
    "none",
    "IQ2_M",
    "IQ3_M",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q3_K_XL",
    "Q4_0",
    "Q4_K_M",
    "Q5_0",
    "Q5_K_M",
    "Q8_0",
]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")
    rag_mode: Literal["text_only", "text_vl", "hybrid"] = "text_only"
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage/qdrant")
    image_dir: Path = Path("storage/images")

    qdrant_collection: str = "rag_chunks"
    image_collection: str = "rag_images"
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    chunker_type: Literal["recursive", "semantic"] = "recursive"
    top_k: int = Field(default=5, ge=1, le=64)
    top_c: int = Field(default=20, ge=1, le=64)

    embedding_model: str = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"
    reranker_model: str = "BAAI/bge-reranker-base"

    llm_provider: Literal["hf_local", "gemini", "vllm","llamacpp"] = "hf_local"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    
    hf_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    vision_model: str = "vidore/colqwen2-v1.0"
    hf_device: int = 0
    hf_max_new_tokens: int = Field(default=2048, ge=1)
    hf_quantization: QuantizationType = "none"
    gguf_model_path: Path | None = None

    gemini_model: str = "gemini-2.5-flash"
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    vllm_api_base: str = "http://localhost:8001/v1"
    vllm_api_key: str = "EMPTY"
    
    summarize_batch_size: int = Field(default=10, ge=1)
    summarize_retrieval_k: int = Field(default=12, ge=1, le=128)
    generation_retrieval_k: int = Field(default=16, ge=1, le=128)
    
    quiz_default_count: int = Field(default=8, ge=1, le=50)
    flashcards_default_count: int = Field(default=15, ge=1, le=100)
    api_url: str = "http://localhost:8000"
        
    @model_validator(mode="after")
    def validate_config(self)-> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.hf_device <-1:
            raise ValueError("hf_device must be-1 for CPU or >= 0 for CUDA.")
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when llm_provider=’gemini’.")
        return self
    
@lru_cache(maxsize=1)
def get_settings()-> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    return settings

settings = get_settings()