import os
import sys
import subprocess
import time
from typing import List
import json
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "categorized_products"
METADATA_DIR = "benchmark_metadata"
MAX_PRODUCTS = 200

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [OUTPUT_DIR, METADATA_DIR]:
        os.makedirs(directory, exist_ok=True)

def download_metadata(indices: List[int]):
    """Download metadata for the specified indices."""
    print("\n=== Downloading Metadata ===")
    indices_str = ','.join(map(str, indices))
    result = subprocess.run(['python', 'meta.py', indices_str], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error downloading metadata:")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)

def verify_data_files():
    """Verify that we have metadata files."""
    meta_files = [f for f in os.listdir(METADATA_DIR) if f.startswith('meta_')]
    if not meta_files:
        print(f"Error: No metadata files found in {METADATA_DIR}")
        return False
    print(f"\nFound {len(meta_files)} metadata files")
    return True

def run_categorization():
    """Run the categorization script."""
    print("\n=== Running Categorization ===")
    result = subprocess.run(['python', 'categorize.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error during categorization:")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)

def main():
    """Main pipeline execution."""
    print("Starting Amazon product categorization pipeline...")
    
    ensure_directories()
    indices = list(range(MAX_PRODUCTS))
    download_metadata(indices)
    
    if not verify_data_files():
        print("Error: Missing required data files. Exiting.")
        sys.exit(1)
    
    run_categorization()
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 