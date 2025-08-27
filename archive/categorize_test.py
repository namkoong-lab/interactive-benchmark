import json
import os
from datasets import load_dataset
import openai
from typing import List, Dict, Any, Set
import time
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict
import sys
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Configuration
OUTPUT_DIR = "categorized_products10"
METADATA_DIR = "benchmark_metadata"
CATEGORIES_FILE = "product_categories.json"
INITIAL_SAMPLE_SIZE = 15  # Number of products to use for initial category generation
def load_products_from_local_repo(data_path: str,limit: int = None) -> List[Dict[str, Any]]:
    """
    Loads all product JSONL files from a local directory or a single file.
    """
    print(f"Loading files from local path: {data_path}...")
    
    all_data = []
    if os.path.isdir(data_path):
        # Find all .jsonl files in the directory
        file_paths = glob.glob(os.path.join(data_path, '*.jsonl'))
        if not file_paths:
             print(f"Warning: No .jsonl files found in directory: {data_path}")
    elif os.path.isfile(data_path):
        # It's a single file path
        file_paths = [data_path]
    else:
        print(f"Error: Path not found or is not a file/directory: {data_path}")
        return []

    for file_path in file_paths:
        print(f"Reading {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {i+1} in {os.path.basename(file_path)}. Skipping.")
                      # Stop processing more files if the limit has been reached
        if limit and len(all_data) >= limit:
            break
    
    print(f"Successfully loaded {len(all_data)} products.")
    return all_data

def load_product_data(meta_file: str) -> Dict[str, Any]:
    """Load metadata for a product."""
    product_data = {}
    meta_path = os.path.join(METADATA_DIR, meta_file)
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            product_data['metadata'] = json.load(f)
    else:
        print(f"Warning: No metadata file found: {meta_file}")
        return None
    return product_data

def load_category_hierarchy() -> Dict[str, Any]:
    """Load existing category hierarchy or return None if it doesn't exist."""
    hierarchy_path = os.path.join(OUTPUT_DIR, CATEGORIES_FILE)
    if os.path.exists(hierarchy_path):
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_category_hierarchy(hierarchy: Dict[str, Any]):
    """Save the category hierarchy to file."""
    hierarchy_path = os.path.join(OUTPUT_DIR, CATEGORIES_FILE)
    with open(hierarchy_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=4, ensure_ascii=False)

def print_category_hierarchy(hierarchy: Dict[str, Any], indent: int = 0):
    """Print the category hierarchy in a tree-like structure."""
    for category in hierarchy.get('categories', []):
        print(' ' * indent + f"- {category['name']}")
        if 'subcategories' in category:
            print_category_hierarchy({'categories': category['subcategories']}, indent + 4)

def generate_initial_category_hierarchy(products: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    print("\n=== Generating Initial Category Hierarchy ===")
    sample_products = products[:INITIAL_SAMPLE_SIZE]
    products_text = "\n\n".join([
        f"Product {i+1}:\nTitle: {p['metadata'].get('title', 'N/A')}\n"
        f"Description: {p['metadata'].get('description', 'N/A')}\n"
        f"Brand: {p['metadata'].get('brand', 'N/A')}\n"
        f"Main Category: {p['metadata'].get('main_category', 'N/A')}\n"
        f"Features: {p['metadata'].get('features', 'N/A')}\n"
        f"Price: {p['metadata'].get('price', 'N/A')}"
        for i, p in enumerate(sample_products)
    ])
    system_prompt = "You are a product categorization expert. Your task is to create a comprehensive, hierarchical category system that can be used to categorize a large number of products. Respond with ONLY the JSON object and nothing else."
    user_prompt = f"""Based on these {len(sample_products)} sample products and considering there are {len(products)} total products to categorize, create a comprehensive, hierarchical category system that can accommodate all products. The categories should be specific enough to be meaningful but not overly specific.\nCATEGORY HIERARCHY FORMAT INSTRUCTIONS:\nWhen generating or updating product_categories.json, you must follow these rules:\n1. The category hierarchy must be a single nested tree, starting from the root "categories" list.\n2. Each category is an object with:\n- "name": the category name (string)\n- "subcategories": a list of subcategory objects (may be omitted or empty if none)\n3. The root object must include:\n- "categories": the top-level list of category objects\n- "version": a string version number (e.g., "1.0")\n- "last_updated": a string timestamp (e.g., "2024-06-19 01:59:54")\nSample Products:{products_text}\nCreate a hierarchical category system following the format rules above. The categories should be consistent and reusable across all products."""
    
    # This now correctly uses the dispatcher function
    response_content = get_llm_response(
        model_name=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=2048
    )
    
    # Clean up the response in case the model adds markdown
    if '```json' in response_content:
        response_content = response_content.split('```json')[1].split('```')[0]
    
    category_hierarchy = json.loads(response_content)
    
    if 'version' not in category_hierarchy:
        category_hierarchy['version'] = "1.0"
    if 'last_updated' not in category_hierarchy:
        category_hierarchy['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S")

    save_category_hierarchy(category_hierarchy)
    print("\nInitial category hierarchy:")
    print_category_hierarchy(category_hierarchy)
    return category_hierarchy
def find_category_path(category_hierarchy: Dict[str, Any], category_name: str, current_path: List[str] = None) -> List[str]:
    """Find the path to a category in the hierarchy."""
    if current_path is None:
        current_path = []
    
    for category in category_hierarchy.get('categories', []):
        if category['name'] == category_name:
            return current_path + [category_name]
        if 'subcategories' in category:
            path = find_category_path({'categories': category['subcategories']}, category_name, current_path + [category['name']])
            if path:
                return path
    return None

def add_category_to_hierarchy(hierarchy, new_category, category_path):
    # Traverse the hierarchy to the parent node
    node = hierarchy['categories']
    for cat in category_path:
        found = False
        for sub in node:
            if sub['name'] == cat:
                if 'subcategories' not in sub:
                    sub['subcategories'] = []
                node = sub['subcategories']
                found = True
                break
        if not found:
            # If the path does not exist, create it
            new_node = {'name': cat, 'subcategories': []}
            node.append(new_node)
            node = new_node['subcategories']
    node.append(new_category)

def update_category_hierarchy(category_hierarchy, new_category, category_path):
    version_parts = category_hierarchy['version'].split('.')
    version_parts[-1] = str(int(version_parts[-1]) + 1)
    category_hierarchy['version'] = '.'.join(version_parts)
    category_hierarchy['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S")
    # Add new category as subcategory at the correct place
    add_category_to_hierarchy(category_hierarchy, new_category, category_path)
    hierarchy_path = os.path.join(OUTPUT_DIR, CATEGORIES_FILE)
    with open(hierarchy_path, 'w', encoding='utf-8') as f:
        json.dump(category_hierarchy, f, indent=4, ensure_ascii=False)
    return category_hierarchy

def categorize_product(product_data: Dict[str, Any], model: str = "gpt-4o", max_retries: int = 5, lock=None) -> Dict[str, Any]:
    import os
    OUTPUT_DIR = "categorized_products10"
    CATEGORIES_FILE = "product_categories.json"
    def load_category_hierarchy():
        hierarchy_path = os.path.join(OUTPUT_DIR, CATEGORIES_FILE)
        if os.path.exists(hierarchy_path):
            with open(hierarchy_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    metadata = product_data.get('metadata', {})
    prompt = f"""You are an expert product categorization AI. Your task is to categorize the given product into the most suitable category from the provided hierarchy.\n## 1. Context\n**Product Information:**\n- **Title:** {metadata.get('title', 'N/A')}\n- **Description:** {metadata.get('description', 'N/A')}\n- **Brand:** {metadata.get('brand', 'N/A')}\n**Existing Categories:**{{CATEGORIES_PLACEHOLDER}}\n## 2. Categorization Rules\n1.  **Prioritize Existing Categories:** Always try to find the best fit within the existing category tree first.\n2.  **Follow the Hierarchy:** Place specific items under their correct broader parent category.\n- ✅ **Correct:** A "Poster" belongs under `Home & Kitchen > Wall Art > Posters`.\n- ❌ **Incorrect:** Do not create `Home & Kitchen > Posters` if `Wall Art` already exists.\n3.  **Ensure Correct Placement:** The category must make logical sense.\n- ✅ **Correct:** An "Apron" belongs in `Home & Kitchen > Kitchen Accessories`.\n- ❌ **Incorrect:** An "Apron" does not belong in `Fashion > Accessories`.\n4.  **Propose a New Category (Only If Necessary):** Only propose a new category if absolutely no suitable category exists.\n## 3. Required Output Format\nRespond with a single JSON object. Do not add any explanations outside of the JSON.\n2**JSON Structure:**\n```json
{
  "category_path": ["Home & Kitchen", "Kitchen & Dining", "Cookware"],
  "confidence": 0.9,
  "new_category": {
    "name": "New Category Name"
  }
}"""
    client = openai.OpenAI()
    last_error = None
    for attempt in range(max_retries):
        try:
            # Always reload the latest hierarchy
            with lock:
                category_hierarchy = load_category_hierarchy()
                prompt_with_cats = prompt.replace("{CATEGORIES_PLACEHOLDER}", json.dumps(category_hierarchy, indent=2))
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a product categorization expert. Your task is to categorize products using the provided category hierarchy, and propose new categories when necessary."},
                    {"role": "user", "content": prompt_with_cats}
                ],
                temperature=0.3,
                max_tokens=150
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from OpenAI API")
            # Strip triple backticks and optional 'json' from start/end
            content_stripped = content.strip()
            if content_stripped.startswith('```'):
                content_stripped = content_stripped.lstrip('`').strip()
                if content_stripped.lower().startswith('json'):
                    content_stripped = content_stripped[4:].strip()
                if content_stripped.endswith('```'):
                    content_stripped = content_stripped[:-3].strip()
            try:
                cat = json.loads(content_stripped)
                # If new_category, update hierarchy and remove from result
                if 'new_category' in cat:
                    with lock:
                        category_hierarchy = load_category_hierarchy()
                        # Insert new_category as subcategory at the correct place
                        update_category_hierarchy(category_hierarchy, cat['new_category'], cat['category_path'])
                    del cat['new_category']
                return cat
            except Exception as e:
                last_error = e
                print(f"Invalid JSON from OpenAI: {content[:200]}... (truncated)")
        except Exception as e:
            last_error = e
            print(f"OpenAI API error (attempt {attempt+1}/{max_retries}): {e}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to categorize product after {max_retries} attempts. Last error: {last_error}")

# def categorize_products(start_from: int = 0, num_products: int = None, random_sample: bool = False, seed: int = None, num_workers: int = 8, model: str = "gpt-4o") -> None:
def categorize_products(data_path: str, start_from: int = 0, num_products: int = None, random_sample: bool = False, seed: int = None, num_workers: int = 8, model: str = "gpt-4o") -> None:

    """Categorize products using a consistent category hierarchy that can evolve."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lock = threading.Lock()
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    # Load all product metadata
    # meta_files = [f for f in os.listdir(METADATA_DIR) if f.startswith('meta_')]
    # if random_sample:
    #     if num_products and num_products < len(meta_files):
    #         meta_files = random.sample(meta_files, num_products)
    #         print(f"Randomly sampled {len(meta_files)} products from {len([f for f in os.listdir(METADATA_DIR) if f.startswith('meta_')])} total products")
    #     else:
    #         print(f"Processing all {len(meta_files)} products in random order")
    #         random.shuffle(meta_files)
    # else:
    #     meta_files.sort()
    #     if start_from > 0:
    #         meta_files = meta_files[start_from:]
    #         print(f"Starting from product index {start_from} (skipping first {start_from} products)")
    #     if num_products:
    #         meta_files = meta_files[:num_products]
    #         print(f"Processing {num_products} products")
    # print(f"\nFound {len(meta_files)} metadata files to process")
    # # Skip already categorized products
    # already_categorized = set()
    # if os.path.exists(OUTPUT_DIR):
    #     for f in os.listdir(OUTPUT_DIR):
    #         if f.startswith('item_') and f.endswith('.json'):
    #             try:
    #                 idx = int(f.split('_')[1].split('.')[0])
    #                 already_categorized.add(idx)
    #             except Exception:
    #                 continue
    # products = []
    # for i, meta_file in enumerate(meta_files):
    #     idx = i + 1
    #     if idx in already_categorized:
    #         continue
    #     product_data = load_product_data(meta_file)
    #     if product_data:
    #         products.append((idx, product_data, meta_file))
    # print(f"{len(products)} products to categorize (skipping {len(already_categorized)} already categorized)")
    
    dataset = load_products_from_local_repo(data_path,limit=num_products)
    if not dataset:
        print("No products loaded. Exiting.")
        return

    print("Preparing products for categorization...")
    products_to_process = []
    for i, item in enumerate(dataset):
        product_data = {'metadata': item['metadata']}
        item_id = item.get('item_id', i + 1)
        products_to_process.append((item_id, product_data))

    print("Preparing products for categorization...")
    products_to_process = []
    for i, item in enumerate(dataset):
        product_data = {'metadata': item['metadata']}
        
        item_id = item.get('item_id', i + 1)
        
        products_to_process.append((item_id, product_data))

    if random_sample:
        random.shuffle(products_to_process)
    
    if num_products:
        products_to_process = products_to_process[:num_products]

    already_categorized_ids = set()
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith('item_') and f.endswith('.json'):
                try:
                    item_id = int(f.split('_')[1].split('.')[0])
                    already_categorized_ids.add(item_id)
                except (IndexError, ValueError):
                    continue

    products = [p for p in products_to_process if p[0] not in already_categorized_ids]

    
    
    # Load or generate category hierarchy
    category_hierarchy = load_category_hierarchy()
    if not category_hierarchy:
        category_hierarchy = generate_initial_category_hierarchy([p[1] for p in products], model=model)
        print(f"model = {model}")
        print("\nInitial category hierarchy generated and saved to", os.path.join(OUTPUT_DIR, CATEGORIES_FILE))
    else:
        print("\nLoaded existing category hierarchy from", os.path.join(OUTPUT_DIR, CATEGORIES_FILE))
    # Parallel categorization
    successful_categorizations = 0
    errors = 0
    def process_one(args):
        idx, product = args
        try:
            cat = categorize_product(product, model=model, lock=lock)
            with lock:
                output_path = os.path.join(OUTPUT_DIR, f"item_{idx}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "item_id": idx,
                        "metadata": product.get('metadata', {}),
                        "categorization": cat
                    }, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"\nError categorizing item {idx}: {e}")
            return False
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one, args) for args in products]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Categorizing products (parallel)"):
            try:
                if f.result():
                    successful_categorizations += 1
                else:
                    errors += 1
            except Exception as e:
                print(f"\nException in thread: {e}")
                errors += 1
    print(f"\nSuccessfully categorized {successful_categorizations} out of {len(products)} products ({errors} errors)")

if __name__ == "__main__":
    legacy_num_products = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        legacy_num_products = int(sys.argv[1])
        sys.argv.pop(1)
    parser = argparse.ArgumentParser(description='Categorize Amazon products')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start processing from this product index (0-based) - only used when not using random sampling')
    parser.add_argument('--num-products', type=int, default=None,
                       help='Number of products to process (default: all remaining)')
    parser.add_argument('--random-sample', action='store_true',
                       help='Randomly sample products instead of processing sequentially')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible sampling (default: None)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of parallel workers for categorization (default: 8)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='OpenAI model to use for categorization (default: gpt-4o)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the local data file (.jsonl) or directory containing .jsonl files.')

    args = parser.parse_args()
    if legacy_num_products is not None:
        args.num_products = legacy_num_products
    if args.start_from < 0:
        print("Start index must be non-negative")
        sys.exit(1)
    if args.num_products is not None and args.num_products <= 0:
        print("Number of products must be positive")
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("Please ensure your OPENAI_API_KEY is set in the .env file")
        exit(1)
    if not os.path.exists(METADATA_DIR):
        print(f"Error: {METADATA_DIR} directory not found")
        exit(1)
    # categorize_products(args.start_from, args.num_products, args.random_sample, args.seed, args.num_workers, args.model) 
    categorize_products(args.data_path, args.start_from, args.num_products, args.random_sample, args.seed, args.num_workers, args.model)