import json
from datasets import load_dataset
import sys
import os

OUTPUT_DIR = "benchmark_metadata"

def load_meta(meta_indices_list, output_directory):
    if not meta_indices_list:
        print("No target indices provided.")
        return

    os.makedirs(output_directory, exist_ok=True)

    indices_to_process = set()
    for idx in meta_indices_list:
        file_path = os.path.join(output_directory, f"meta_{idx}.json")
        if os.path.exists(file_path):
            print(f"Metadata file for index {idx} already exists at {file_path}. Skipping generation for this index.")
        else:
            indices_to_process.add(idx)

    if not indices_to_process:
        print("All requested metadata files already exist. Nothing to do.")
        return

    max_index_needed = -1
    if indices_to_process:
        max_index_needed = max(indices_to_process)
    try:
        streaming_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_All_Beauty",
            split="full",
            streaming=True,
            trust_remote_code=True
        )
        # streaming_dataset = streaming_dataset.remove_columns("images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    metadata_found_count = 0
    for i, item in enumerate(streaming_dataset):
        if i in indices_to_process:
            file_path = os.path.join(output_directory, f"meta_{i}.json")
            if not os.path.exists(file_path):
                print(f"Found metadata for index {i}. Storing it to {file_path}")
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(item, f, indent=4, ensure_ascii=False)
                    metadata_found_count += 1
                    indices_to_process.remove(i)  
                except IOError as e:
                    print(f"Error writing file for metadata index {i}: {e}")
            else: 
                indices_to_process.remove(i)


        if not indices_to_process:
            print("All specified (and not pre-existing) metadata items have been found and saved.")
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python meta.py <comma_separated_indices>")
        sys.exit(1)

    try:
        meta_indices = list(map(int, sys.argv[1].split(',')))
        print(f"Requesting metadata for indices: {meta_indices}, output will be in '{OUTPUT_DIR}' directory.")
    except ValueError:
        print("Invalid input. Provide comma-separated integer indices like 4,5,6.")
        sys.exit(1)

    try:
        load_meta(meta_indices, OUTPUT_DIR)
        print(f"Metadata processing complete.")
    except Exception as e:
        print(f"An unexpected error occurred during metadata processing: {e}")
        sys.exit(1)
