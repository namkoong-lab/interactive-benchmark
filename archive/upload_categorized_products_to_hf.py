import os
import json
import subprocess
from huggingface_hub import HfApi

DATASET_DIR = os.path.join(os.path.dirname(__file__), "categorized_products")
HF_REPO = "gilberty005/categorized_6k"
COMBINED_FILE = os.path.join(DATASET_DIR, "all_products_combined.jsonl")
CATEGORIES_FILE = os.path.join(DATASET_DIR, "product_categories.json")


def delete_remote_files():
    api = HfApi()
    print(f"Fetching file list from {HF_REPO}...")
    repo_info = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    files_to_delete = [f for f in repo_info if not f.startswith('.')]
    if files_to_delete:
        print(f"Deleting {len(files_to_delete)} files from {HF_REPO}...")
        for f in files_to_delete:
            print(f"Deleting {f} ...")
            api.delete_file(path_in_repo=f, repo_id=HF_REPO, repo_type="dataset")
        print("All files deleted.")
    else:
        print("No files to delete.")

def combine_files():
    with open(COMBINED_FILE, "w", encoding="utf-8") as outfile:
        for fname in os.listdir(DATASET_DIR):
            if fname == "product_categories.json" or not fname.endswith(".json"):
                continue
            file_path = os.path.join(DATASET_DIR, fname)
            with open(file_path, "r", encoding="utf-8") as infile:
                try:
                    data = json.load(infile)
                    if isinstance(data, list):
                        for item in data:
                            outfile.write(json.dumps(item) + "\n")
                    elif isinstance(data, dict):
                        outfile.write(json.dumps(data) + "\n")
                except Exception as e:
                    print(f"Error reading {fname}: {e}")

def upload_files():
    print(f"Uploading {COMBINED_FILE} ...")
    subprocess.run([
        "huggingface-cli", "upload", HF_REPO, COMBINED_FILE, "--repo-type=dataset"
    ], check=True)
    print(f"Uploading {CATEGORIES_FILE} ...")
    subprocess.run([
        "huggingface-cli", "upload", HF_REPO, CATEGORIES_FILE, "--repo-type=dataset"
    ], check=True)
    print("Upload complete.")

if __name__ == "__main__":
    delete_remote_files()
    combine_files()
    upload_files() 