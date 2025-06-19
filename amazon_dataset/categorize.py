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
    """Generate initial category hierarchy from a larger sample of products."""
    print("\n=== Generating Initial Category Hierarchy ===")
    
    # Use more products for initial categorization
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
    
    prompt = f"""Based on these {len(sample_products)} sample products and considering there are {len(products)} total products to categorize, 
create a comprehensive, hierarchical category system that can accommodate all products. The categories should be specific 
enough to be meaningful but not overly specific.

Sample Products:
{products_text}

Create a hierarchical category system with the following structure:
{{
    "categories": [
        {{
            "name": "Category Name",
            "subcategories": [
                {{
                    "name": "Subcategory Name",
                    "subcategories": [
                        {{
                            "name": "Specific Category Name"
                        }}
                    ]
                }}
            ]
        }}
    ],
    "version": "1.0",
    "last_updated": "timestamp"
}}

The categories should be consistent and reusable across all products. For example:
- "Tech Products" -> "Wearable Technology" -> "Smart Watches"
- "Home & Kitchen" -> "Kitchen Tools" -> "Utensils"
- "Electronics" -> "Audio" -> "Headphones"

Please ensure the categories are specific enough to be useful but not so specific that each product needs its own category."""

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

def add_category_to_hierarchy(category_hierarchy: Dict[str, Any], new_category: Dict[str, Any], parent_path: List[str]) -> Dict[str, Any]:
    """Add a new category to the hierarchy at the specified parent path."""
    if not parent_path:
        category_hierarchy['categories'].append(new_category)
        return category_hierarchy
    
    def add_to_path(categories, path, new_cat, depth=0):
        for category in categories:
            if category['name'] == path[depth]:
                if depth == len(path) - 1:
                    if 'subcategories' not in category:
                        category['subcategories'] = []
                    category['subcategories'].append(new_cat)
                else:
                    add_to_path(category['subcategories'], path, new_cat, depth + 1)
    
    add_to_path(category_hierarchy['categories'], parent_path, new_category)
    return category_hierarchy

def update_category_hierarchy(category_hierarchy: Dict[str, Any], new_category: Dict[str, Any]) -> Dict[str, Any]:
    """Update the category hierarchy with a new category."""
    # Increment version number
    version_parts = category_hierarchy['version'].split('.')
    version_parts[-1] = str(int(version_parts[-1]) + 1)
    category_hierarchy['version'] = '.'.join(version_parts)
    category_hierarchy['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add new category
    parent_path = find_category_path(category_hierarchy, new_category['parent_category'])
    if parent_path:
        category_hierarchy = add_category_to_hierarchy(category_hierarchy, new_category, parent_path)
    else:
        # If parent not found, add as top-level category
        category_hierarchy['categories'].append(new_category)
    
    # Save updated hierarchy
    save_category_hierarchy(category_hierarchy)
    return category_hierarchy

def categorize_product(product_data: Dict[str, Any], category_hierarchy: Dict[str, Any]) -> Dict[str, Any]:
    """Categorize a single product using the category hierarchy, adding new categories if needed."""
    metadata = product_data.get('metadata', {})
    
    prompt = f"""Product Information:
Title: {metadata.get('title', 'N/A')}
Description: {metadata.get('description', 'N/A')}
Brand: {metadata.get('brand', 'N/A')}
Main Category: {metadata.get('main_category', 'N/A')}
Features: {metadata.get('features', 'N/A')}
Price: {metadata.get('price', 'N/A')}

Available Categories:
{json.dumps(category_hierarchy, indent=2)}

Categorize this product using the categories provided above. If no existing category fits well, propose a new category.

Format your response as JSON:
{{
    "category_path": ["Main Category", "Subcategory", ...],  # List of categories from broadest to most specific
    "confidence": 0.95,
    "new_category": {{
        "name": "New Category Name",
        "parent_category": "Parent Category Name"
    }}  # Only include if a new category is needed
}}"""

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a product categorization expert. Your task is to categorize products using the provided category hierarchy, and propose new categories when necessary."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150  # Reduced token limit since we don't need explanations
    )
    
    categorization = json.loads(response.choices[0].message.content)
    
    # If a new category was proposed, update the hierarchy
    if 'new_category' in categorization:
        category_hierarchy = update_category_hierarchy(category_hierarchy, categorization['new_category'])
        # Remove the new_category from the response
        del categorization['new_category']
    
    return categorization

def categorize_products(start_from: int = 0, num_products: int = None) -> None:
    """Categorize products using a consistent category hierarchy that can evolve."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all product metadata
    meta_files = [f for f in os.listdir(METADATA_DIR) if f.startswith('meta_')]
    meta_files.sort()  # Ensure consistent ordering
    
    # Apply start_from filter
    if start_from > 0:
        meta_files = meta_files[start_from:]
        print(f"Starting from product index {start_from} (skipping first {start_from} products)")
    
    # Apply num_products filter
    if num_products:
        meta_files = meta_files[:num_products]
        print(f"Processing {num_products} products")
    
    print(f"\nFound {len(meta_files)} metadata files to process")
    
    products = []
    for meta_file in tqdm(meta_files, desc="Loading products"):
        product_data = load_product_data(meta_file)
        if product_data:
            products.append(product_data)
    
    # Load or generate category hierarchy
    category_hierarchy = load_category_hierarchy()
    if not category_hierarchy:
        category_hierarchy = generate_initial_category_hierarchy(products)
        print("\nInitial category hierarchy generated and saved to", os.path.join(OUTPUT_DIR, CATEGORIES_FILE))
    else:
        print("\nLoaded existing category hierarchy from", os.path.join(OUTPUT_DIR, CATEGORIES_FILE))
    
    # Categorize each product
    successful_categorizations = 0
    
    for idx, product in enumerate(tqdm(products, desc="Categorizing products"), start_from + 1):
        try:
            categorization = categorize_product(product, category_hierarchy)
            
            # Save individual product file with sequential numbering
            output_path = os.path.join(OUTPUT_DIR, f"item_{idx}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "item_id": idx,
                    "metadata": product.get('metadata', {}),
                    "categorization": categorization
                }, f, indent=2, ensure_ascii=False)
            
            successful_categorizations += 1
        except Exception as e:
            print(f"\nError categorizing item {idx}: {e}")
            continue
    
    print(f"\nSuccessfully categorized {successful_categorizations} out of {len(products)} products")

if __name__ == "__main__":
    # Check for legacy numeric argument first
    legacy_num_products = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        legacy_num_products = int(sys.argv[1])
        # Remove the legacy argument from sys.argv to avoid conflicts
        sys.argv.pop(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Categorize Amazon products')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start processing from this product index (0-based)')
    parser.add_argument('--num-products', type=int, default=None,
                       help='Number of products to process (default: all remaining)')
    
    args = parser.parse_args()
    
    # Handle legacy argument
    if legacy_num_products is not None:
        args.num_products = legacy_num_products
    
    # Validate arguments
    if args.start_from < 0:
        print("Start index must be non-negative")
        sys.exit(1)
    
    if args.num_products is not None and args.num_products <= 0:
        print("Number of products must be positive")
        sys.exit(1)
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Please ensure your OPENAI_API_KEY is set in the .env file")
        exit(1)
    
    # Check if data directories exist
    if not os.path.exists(METADATA_DIR):
        print(f"Error: {METADATA_DIR} directory not found")
        exit(1)
        
    categorize_products(args.start_from, args.num_products) 