import os
import json
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
CATEGORIES_FILE = os.path.join(BASE_DIR, "categorized_products", "product_categories.json")
PRODUCTS_DIR = os.path.join(BASE_DIR, "categorized_products")
OUTPUT_FILE = os.path.join(BASE_DIR, "product_categories_with_counts.json")

# Helper to recursively add counts to categories
def add_counts_to_categories(categories, count_map):
    for cat in categories:
        cat_name = cat["name"]
        cat["count"] = count_map.get(cat_name, 0)
        if "subcategories" in cat:
            add_counts_to_categories(cat["subcategories"], count_map)

# Helper to recursively find all category paths
def get_all_category_paths(categories, parent_path=None):
    if parent_path is None:
        parent_path = []
    paths = []
    for cat in categories:
        current_path = parent_path + [cat["name"]]
        paths.append(tuple(current_path))
        if "subcategories" in cat:
            paths.extend(get_all_category_paths(cat["subcategories"], current_path))
    return paths

# Helper to find the deepest matching category path for a product
def find_category_path(product, all_paths):
    # Looks for category_path in product['categorization']['category_path']
    if "categorization" in product and "category_path" in product["categorization"]:
        path = tuple(product["categorization"]["category_path"])
        if path in all_paths:
            return path
        # Try to match the longest possible prefix
        for i in range(len(path), 0, -1):
            if tuple(path[:i]) in all_paths:
                return tuple(path[:i])
    return None

# Load categories
with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
    categories_data = json.load(f)

# Get all category paths
all_paths = set(get_all_category_paths(categories_data["categories"]))

# Count items per category path
count_map = defaultdict(int)
for fname in os.listdir(PRODUCTS_DIR):
    if fname == "product_categories.json" or not fname.endswith(".json"):
        continue
    with open(os.path.join(PRODUCTS_DIR, fname), "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for product in data:
                path = find_category_path(product, all_paths)
                if path:
                    # Increment count for every node in the path
                    for i in range(1, len(path)+1):
                        count_map[path[:i][-1]] += 1
        except Exception as e:
            print(f"Error reading {fname}: {e}")

# Add counts to categories
add_counts_to_categories(categories_data["categories"], count_map)

# Write output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(categories_data, f, indent=2, ensure_ascii=False)

print(f"Wrote category counts to {OUTPUT_FILE}") 