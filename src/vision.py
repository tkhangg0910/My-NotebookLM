from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from functools import lru_cache
from src.config import get_settings
import torch
from transformers.utils.import_utils import is_flash_attn_2_available
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

settings = get_settings()

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return Image.open(
            self.image_paths[idx]
        ).convert("RGB")

@lru_cache(maxsize=1)
def get_vision_model():
    model = ColQwen2.from_pretrained(
        settings.vision_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(settings.vision_model)
    return model, processor


def embed_images(image_paths, batch_size=8):
    model, processor = get_vision_model()
    dataset = ImageDataset(image_paths)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(model.device)
                for k, v in batch.items()
            }

            batch_emb = model(**batch)

            embeddings.extend(
                torch.unbind(
                    batch_emb.cpu()
                )
            )

    return embeddings

def embed_query(query: str):
    model, processor = get_vision_model()

    batch = processor.process_queries(
        [query]
    )

    batch = {
        k: v.to(model.device)
        for k, v in batch.items()
    }

    with torch.no_grad():
        emb = model(**batch)

    return emb[0]