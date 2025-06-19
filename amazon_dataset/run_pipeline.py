import os
import sys
import subprocess
import time
import argparse
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

def get_last_processed_product() -> int:
    """Get the index of the last processed product by checking existing files."""
    if not os.path.exists(OUTPUT_DIR):
        return 0
    
    # Look for item_X.json files and find the highest number
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('item_') and f.endswith('.json')]
    if not existing_files:
        return 0
    
    # Extract numbers from filenames and find the maximum
    numbers = []
    for filename in existing_files:
        try:
            # Extract number from "item_X.json"
            number = int(filename.replace('item_', '').replace('.json', ''))
            numbers.append(number)
        except ValueError:
            continue
    
    return max(numbers) if numbers else 0

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

def run_categorization(start_from: int = 0, num_products: int = None):
    """Run the categorization script with optional start position and product count."""
    print("\n=== Running Categorization ===")
    
    # Build command with arguments
    cmd = ['python', 'categorize.py']
    
    if start_from > 0:
        cmd.extend(['--start-from', str(start_from)])
    
    if num_products:
        cmd.extend(['--num-products', str(num_products)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error during categorization:")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Amazon product categorization pipeline')
    parser.add_argument('--continue-from-last', action='store_true',
                       help='Continue from the last processed product')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start processing from this product index (0-based)')
    parser.add_argument('--num-products', type=int, default=None,
                       help='Number of products to process (default: all remaining)')
    parser.add_argument('--max-products', type=int, default=MAX_PRODUCTS,
                       help=f'Maximum total products to process (default: {MAX_PRODUCTS})')
    
    return parser.parse_args()

def main():
    """Main pipeline execution."""
    args = parse_arguments()
    
    print("Starting Amazon product categorization pipeline...")
    
    # Determine starting point
    if args.continue_from_last:
        start_from = get_last_processed_product()
        print(f"Continuing from last processed product: {start_from}")
    else:
        start_from = args.start_from
        print(f"Starting from product index: {start_from}")
    
    # Determine how many products to process
    if args.num_products:
        num_products = args.num_products
        print(f"Processing {num_products} additional products")
    else:
        # When continuing from last, allow processing beyond current max_products
        if args.continue_from_last:
            num_products = args.max_products - start_from
            if num_products <= 0:
                # If we've already reached max_products, allow processing more
                num_products = args.max_products
        else:
            num_products = args.max_products - start_from
        print(f"Processing all remaining products up to {args.max_products}")
    
    # Validate parameters
    if not args.continue_from_last and start_from >= args.max_products:
        print(f"Error: Start index {start_from} is >= max products {args.max_products}")
        sys.exit(1)
    
    if num_products <= 0:
        print(f"Error: Number of products to process must be positive, got {num_products}")
        sys.exit(1)
    
    # Only warn about exceeding max_products if not continuing from last
    if not args.continue_from_last and start_from + num_products > args.max_products:
        print(f"Warning: Requested {num_products} products starting from {start_from} would exceed max {args.max_products}")
        num_products = args.max_products - start_from
        print(f"Adjusted to process {num_products} products")
    
    ensure_directories()
    
    # Only download metadata if we need new products
    if start_from + num_products > get_last_processed_product():
        indices = list(range(start_from, start_from + num_products))
        download_metadata(indices)
    
    if not verify_data_files():
        print("Error: Missing required data files. Exiting.")
        sys.exit(1)
    
    run_categorization(start_from, num_products)
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 