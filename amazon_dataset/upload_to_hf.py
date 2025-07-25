import os
import json
from glob import glob
from huggingface_hub import HfApi

repo_id = "gilberty005/amazon_20k"
base_dir = os.path.dirname(__file__)
folder_path = os.path.join(base_dir, "benchmark_metadata")
jsonl_path = os.path.join(base_dir, "benchmark_metadata.jsonl")

# Step 0: Delete all files currently in the Hugging Face dataset repo
api = HfApi()
print(f"Fetching file list from {repo_id}...")
repo_info = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
files_to_delete = [f for f in repo_info if not f.startswith('.')]

if files_to_delete:
    print(f"Deleting {len(files_to_delete)} files from {repo_id}...")
    for f in files_to_delete:
        print(f"Deleting {f} ...")
        api.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="dataset")
    print("All files deleted.")
else:
    print("No files to delete.")

# Step 1: Combine all JSON files into a single .jsonl file
json_files = sorted(glob(os.path.join(folder_path, "meta_*.json")))
total_files = len(json_files)
print(f"Combining {total_files} JSON files into {jsonl_path} ...")

with open(jsonl_path, "w", encoding="utf-8") as outfile:
    for fname in json_files:
        with open(fname, "r", encoding="utf-8") as infile:
            data = json.load(infile)
            outfile.write(json.dumps(data) + "\n")
print(f"Combined all JSONs into {jsonl_path}")

# Step 2: Upload the .jsonl file to Hugging Face
print(f"Uploading {jsonl_path} to {repo_id} on Hugging Face...")
api.upload_file(
    path_or_fileobj=jsonl_path,
    path_in_repo=os.path.basename(jsonl_path),
    repo_id=repo_id,
    repo_type="dataset"
)
print(f"Upload of {jsonl_path} to {repo_id} complete.")

# Step 3: Clean up the .jsonl file after upload
os.remove(jsonl_path)
print(f"Cleaned up temporary file {jsonl_path}.") 