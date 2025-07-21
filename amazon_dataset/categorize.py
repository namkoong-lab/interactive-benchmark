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
OUTPUT_DIR = "categorized_products"
METADATA_DIR = "benchmark_metadata"
CATEGORIES_FILE = "product_categories.json"
INITIAL_SAMPLE_SIZE = 20  # Number of products to use for initial category generation

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

def generate_initial_category_hierarchy(products: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    prompt = f"""Based on these {len(sample_products)} sample products and considering there are {len(products)} total products to categorize, \
create a comprehensive, hierarchical category system that can accommodate all products. The categories should be specific \
enough to be meaningful but not overly specific.\n\nIMPORTANT:\n- Always try to fit products into an existing category or subcategory, even if it is not a perfect match.\n- Only create a new category if there is truly no suitable existing category at any level.\n- If you must propose a new category, it should be a subcategory of the most relevant existing category, not a new top-level category.\n- Avoid creating categories that are too specific or that could be grouped under an existing broader category.\n\nCATEGORY HIERARCHY FORMAT INSTRUCTIONS:\nWhen generating or updating product_categories.json, you must follow these rules:\n\n1. The category hierarchy must be a single nested tree, starting from the root \"categories\" list.\n2. Each category is an object with:\n   - \"name\": the category name (string)\n   - \"subcategories\": a list of subcategory objects (may be omitted or empty if none)\n3. Do NOT use \"parent_category\" fields anywhere in the hierarchy.\n4. All categories and subcategories must be reachable from the root via nested \"subcategories\".\n5. The root object must include:\n   - \"categories\": the top-level list of category objects\n   - \"version\": a string version number (e.g., \"1.0\")\n   - \"last_updated\": a string timestamp (e.g., \"2024-06-19 01:59:54\")\n6. Example:\n{{\n  \"categories\": [\n    {{\n      \"name\": \"Main Category\",\n      \"subcategories\": [\n        {{\n          \"name\": \"Subcategory\",\n          \"subcategories\": [\n            {{ \"name\": \"Leaf Category\" }}\n          ]\n        }}\n      ]\n    }}\n  ],\n  \"version\": \"1.0\",\n  \"last_updated\": \"2024-06-19 01:59:54\"\n}}\n7. All category additions or updates must preserve this structure.\n\nSample Products:\n{products_text}\n\nCreate a hierarchical category system following the format rules above. The categories should be consistent and reusable across all products. For example:\n- \"Tech Products\" -> \"Wearable Technology\" -> \"Smart Watches\"\n- \"Home & Kitchen\" -> \"Kitchen Tools\" -> \"Utensils\"\n- \"Electronics\" -> \"Audio\" -> \"Headphones\"\n\nPlease ensure the categories are specific enough to be useful but not so specific that each product needs its own category."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a product categorization expert. Your task is to create a comprehensive, hierarchical category system that can be used to categorize a large number of products."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    category_hierarchy = json.loads(response.choices[0].message.content)
    category_hierarchy['version'] = "1.0"
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
    OUTPUT_DIR = "categorized_products"
    CATEGORIES_FILE = "product_categories.json"
    def load_category_hierarchy():
        hierarchy_path = os.path.join(OUTPUT_DIR, CATEGORIES_FILE)
        if os.path.exists(hierarchy_path):
            with open(hierarchy_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    metadata = product_data.get('metadata', {})
    prompt = f"""Product Information:\nTitle: {metadata.get('title', 'N/A')}\nDescription: {metadata.get('description', 'N/A')}\nBrand: {metadata.get('brand', 'N/A')}\nMain Category: {metadata.get('main_category', 'N/A')}\nFeatures: {metadata.get('features', 'N/A')}\nPrice: {metadata.get('price', 'N/A')}\n\nAvailable Categories:\n{{CATEGORIES_PLACEHOLDER}}\n\nIMPORTANT:\n- Always try to fit the product into an existing category or subcategory, even if it is not a perfect match.\n- Only propose a new category if there is truly no suitable existing category at any level.\n- If you must propose a new category, it should be a subcategory of the most relevant existing category, not a new top-level category.\n- Avoid creating categories that are too specific or that could be grouped under an existing broader category.\n\nCATEGORY HIERARCHY FORMAT INSTRUCTIONS:\nWhen generating or updating product_categories.json, you must follow these rules:\n\n1. The category hierarchy must be a single nested tree, starting from the root \"categories\" list.\n2. Each category is an object with:\n   - \"name\": the category name (string)\n   - \"subcategories\": a list of subcategory objects (may be omitted or empty if none)\n3. Do NOT use \"parent_category\" fields anywhere in the hierarchy.\n4. All categories and subcategories must be reachable from the root via nested \"subcategories\".\n5. The root object must include:\n   - \"categories\": the top-level list of category objects\n   - \"version\": a string version number (e.g., \"1.0\")\n   - \"last_updated\": a string timestamp (e.g., \"2024-06-19 01:59:54\")\n6. Example:\n{{\n  \"categories\": [\n    {{\n      \"name\": \"Main Category\",\n      \"subcategories\": [\n        {{\n          \"name\": \"Subcategory\",\n          \"subcategories\": [\n            {{ \"name\": \"Leaf Category\" }}\n          ]\n        }}\n      ]\n    }}\n  ],\n  \"version\": \"1.0\",\n  \"last_updated\": \"2024-06-19 01:59:54\"\n}}\n7. All category additions or updates must preserve this structure.\n\nCategorize this product using the categories provided above. If no existing category fits well, propose a new category.\n\nFormat your response as JSON:\n{{\n    \"category_path\": [\"Main Category\", \"Subcategory\", ...],  # List of categories from broadest to most specific\n    \"confidence\": 0.95,\n    \"new_category\": {{\n        \"name\": \"New Category Name\"\n        # Do NOT include 'parent_category'. Instead, the new category will be nested under the last element of the category_path as a subcategory.\n    }}  # Only include if a new category is needed\n}}"""
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

def categorize_products(start_from: int = 0, num_products: int = None, random_sample: bool = False, seed: int = None, num_workers: int = 8, model: str = "gpt-4o") -> None:
    """Categorize products using a consistent category hierarchy that can evolve."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lock = threading.Lock()
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    # Load all product metadata
    meta_files = [f for f in os.listdir(METADATA_DIR) if f.startswith('meta_')]
    if random_sample:
        if num_products and num_products < len(meta_files):
            meta_files = random.sample(meta_files, num_products)
            print(f"Randomly sampled {len(meta_files)} products from {len([f for f in os.listdir(METADATA_DIR) if f.startswith('meta_')])} total products")
        else:
            print(f"Processing all {len(meta_files)} products in random order")
            random.shuffle(meta_files)
    else:
        meta_files.sort()
        if start_from > 0:
            meta_files = meta_files[start_from:]
            print(f"Starting from product index {start_from} (skipping first {start_from} products)")
        if num_products:
            meta_files = meta_files[:num_products]
            print(f"Processing {num_products} products")
    print(f"\nFound {len(meta_files)} metadata files to process")
    # Skip already categorized products
    already_categorized = set()
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith('item_') and f.endswith('.json'):
                try:
                    idx = int(f.split('_')[1].split('.')[0])
                    already_categorized.add(idx)
                except Exception:
                    continue
    products = []
    for i, meta_file in enumerate(meta_files):
        idx = i + 1
        if idx in already_categorized:
            continue
        product_data = load_product_data(meta_file)
        if product_data:
            products.append((idx, product_data, meta_file))
    print(f"{len(products)} products to categorize (skipping {len(already_categorized)} already categorized)")
    # Load or generate category hierarchy
    category_hierarchy = load_category_hierarchy()
    if not category_hierarchy:
        category_hierarchy = generate_initial_category_hierarchy([p[1] for p in products])
        print("\nInitial category hierarchy generated and saved to", os.path.join(OUTPUT_DIR, CATEGORIES_FILE))
    else:
        print("\nLoaded existing category hierarchy from", os.path.join(OUTPUT_DIR, CATEGORIES_FILE))
    # Parallel categorization
    successful_categorizations = 0
    errors = 0
    def process_one(args):
        idx, product, meta_file = args
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
    # Check for legacy numeric argument first
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
    categorize_products(args.start_from, args.num_products, args.random_sample, args.seed, args.num_workers, args.model) 