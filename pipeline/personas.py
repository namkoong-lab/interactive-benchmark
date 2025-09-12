from datasets import load_dataset
import sys
from typing import Optional

_dataset_cache = None  # lazy-loaded HF dataset


def _get_dataset():
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = load_dataset("Tianyi-Lab/Personas", split="train")
    return _dataset_cache


def get_persona_description(index: int) -> str:
    """
    Return the descriptive persona text for a given index from the
    HuggingFace dataset "Tianyi-Lab/Personas".
    """
    dataset = _get_dataset()
    idx = int(index)
    if idx < 0 or idx >= len(dataset):
        raise IndexError(f"persona index {idx} out of range (0..{len(dataset)-1})")
    entry = dataset[idx]
    return entry["Llama-3.1-70B-Instruct_descriptive_persona"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python personas.py <persona_index>")
        sys.exit(1)
    try:
        idx = int(sys.argv[1])
    except ValueError:
        print("Error: persona_index must be an integer.")
        sys.exit(1)
    print(get_persona_description(idx))