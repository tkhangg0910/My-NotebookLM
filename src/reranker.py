from sentence_transformers import CrossEncoder
from functools import lru_cache
from src.config import get_settings
settings = get_settings()

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        settings.reranker_model
    )