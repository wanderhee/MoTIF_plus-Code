import re
from pathlib import Path

from datasets import Dataset, DatasetDict


def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_texts(text_dir, min_length: int = 50) -> list:
    texts = []
    for path in sorted(Path(text_dir).rglob("*.txt")):
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if len(text) >= min_length:
            texts.append(text)
    return texts


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 128) -> list:
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if len(chunk) >= overlap:
            chunks.append(chunk)
    return chunks


def build_pretrain_dataset(
    text_dir: str,
    chunk_size: int = 1024,
    overlap: int = 128,
    min_length: int = 50,
    train_split: float = 0.9,
    seed: int = 42,
) -> DatasetDict:
    texts = load_texts(text_dir, min_length=min_length)
    chunks = [c for t in texts for c in chunk_text(t, chunk_size, overlap)]
    if not chunks:
        raise FileNotFoundError(f"No usable text found under {text_dir}")

    dataset = Dataset.from_dict({"text": chunks})
    split = dataset.train_test_split(test_size=1 - train_split, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})
