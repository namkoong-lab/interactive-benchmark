from datasets import load_dataset
import sys


def get_persona_description(index: int) -> str:
    """
    Return the descriptive persona text for a given index from the
    HuggingFace dataset "Tianyi-Lab/Personas".
    """
    dataset = load_dataset("Tianyi-Lab/Personas", split="train")
    entry = dataset[int(index)]
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