from huggingface_hub import HfApi
import os

# Set your repo ID
repo_id = "gilberty005/amazon_20k"
# Always resolve the folder path relative to this script's location
folder_path = os.path.join(os.path.dirname(__file__), "benchmark_metadata")

# Create an instance of the API
api = HfApi()

# Upload the folder using the large folder upload method
api.upload_large_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="dataset"
)

print(f"Upload of {folder_path} to {repo_id} complete.") 