import json
import os
from datasets import load_dataset
import sys
from tqdm import tqdm
import random

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "benchmark_metadata")
NUM_PRODUCTS = 20000
# Always resolve CATEGORY_SIZ ES_FILE relative to this script's directory
CATEGORY_SIZES_FILE = os.path.join(os.path.dirname(__file__), "category_sizes.json")


def read_category_sizes():
    if os.path.exists(CATEGORY_SIZES_FILE):
        try:
            with open(CATEGORY_SIZES_FILE, 'r') as f:
                sizes = json.load(f)
            return sizes
        except Exception:
            pass
    return {}

def write_category_sizes(sizes):
    with open(CATEGORY_SIZES_FILE, 'w') as f:
        json.dump(sizes, f, indent=2)

def get_category_sizes(force_update=False):
    sizes = read_category_sizes()
    categories = list(sizes.keys())

    if force_update or not sizes or any(not isinstance(sizes.get(cat, 0), int) or sizes[cat] <= 0 for cat in categories):
        print("Calculating category sizes from scratch...")
        categories = [
            "All_Beauty", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Baby_Products", "Books", "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty", "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_DVD", "Video_Games"
        ]
        for cat in categories:
            try:
                ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{cat}", split="full")
                n = ds.info.splits['full'].num_examples
                sizes[cat] = n
                print(f"{cat}: {n} products (scanned)")
            except Exception as e:
                print(f"Error loading category {cat} for size: {e}")
                sizes[cat] = 0
        write_category_sizes(sizes)
    else:
        print("All category sizes loaded from file.")
    return sizes, list(sizes.keys())


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
        streaming_dataset = streaming_dataset.remove_columns("images")
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

def load_products_with_description_filter(num_products=NUM_PRODUCTS, output_directory=OUTPUT_DIR, seed=None):
    print(f"Loading {num_products} products from all Amazon categories (weighted split, streaming, random jump)...")
    print("Filtering for products with non-empty descriptions and non-None prices...")

    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")

    os.makedirs(output_directory, exist_ok=True)

    # Step 1: Get the size of each category
    print("Getting category sizes...")
    category_sizes, categories = get_category_sizes()
    total_products = sum(category_sizes[cat] for cat in categories if category_sizes[cat] > 0)
    print(f"Total products across all categories: {total_products}")
    if total_products == 0:
        print("No products found in any category!")
        return False

    # Step 2: Calculate how many products to sample from each category
    print(f"Calculating weighted split for {num_products} products...")
    per_category_quota = {}
    remaining = num_products
    for i, cat in enumerate(categories):
        if category_sizes[cat] <= 0:
            per_category_quota[cat] = 0
            continue
        if i == len(categories) - 1:
            per_category_quota[cat] = remaining
        else:
            quota = int(round(num_products * (category_sizes[cat] / total_products)))
            per_category_quota[cat] = quota
            remaining -= quota
    print("Per-category quotas:")
    for cat in categories:
        print(f"  {cat}: {per_category_quota[cat]}")

    # Step 3: For each category, stream from the start, check validity, jump by random amount
    used_asins = set()
    products_saved = 0
    product_idx = 0
    for cat in categories:
        quota = per_category_quota[cat]
        if quota <= 0:
            continue
        print(f"Sampling {quota} products from {cat} using streaming mode and random jumps...")
        try:
            n = category_sizes[cat]
            ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{cat}", split="full", streaming=True)
            collected = 0
            i = 0
            stream = iter(ds)
            with tqdm(total=quota, desc=f"{cat} (sampling)") as cat_pbar:
                while collected < quota:
                    try:
                        item = next(stream)
                        i += 1
                    except StopIteration:
                        print(f"End of stream for {cat}. Collected {collected}/{quota}.")
                        break
                    # Filter for valid description
                    description = item.get('description', [])
                    if not description or (isinstance(description, list) and len(description) == 0):
                        continue
                    # Filter for valid price
                    price = item.get('price', None)
                    if price is None or price == "None" or price == "":
                        continue
                    # Deduplicate by ASIN
                    asin = item.get('asin', f"{cat}_{i}")
                    if asin in used_asins:
                        continue
                    used_asins.add(asin)
                    # Write product to disk immediately
                    file_path = os.path.join(output_directory, f"meta_{product_idx}.json")
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(item, f, indent=4, ensure_ascii=False)
                    except IOError as e:
                        print(f"Error writing file for product {product_idx}: {e}")
                        continue
                    collected += 1
                    products_saved += 1
                    product_idx += 1
                    cat_pbar.update(1)
                    # Jump by a random amount (at least 1)
                    jump = random.randint(1, max(1, n // quota))
                    for _ in range(jump - 1):
                        try:
                            next(stream)
                            i += 1
                        except StopIteration:
                            break
            if collected < quota:
                print(f"Warning: Only found {collected} valid products in {cat} (quota was {quota})")
        except Exception as e:
            print(f"Error loading category {cat}: {e}")
            continue

    print(f"\nSuccessfully saved {products_saved} products to {output_directory}/")
    print(f"Unique products used: {len(used_asins)}")
    return products_saved == num_products

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == "--load-20k" or sys.argv[1].startswith("--load-")):
        print("Amazon Product Loader - N Products Mode")
        print("=" * 50)
        # Parse number of products
        if sys.argv[1].startswith("--load-"):
            try:
                num_products = int(sys.argv[1].split("--load-")[1])
            except Exception:
                num_products = NUM_PRODUCTS
        else:
            num_products = NUM_PRODUCTS
        # Check for seed argument
        seed = None
        for arg in sys.argv[2:]:
            if arg.startswith("--seed="):
                try:
                    seed = int(arg.split("=")[1])
                except ValueError:
                    print("Invalid seed value. Using default random seed.")
        success = load_products_with_description_filter(num_products=num_products, seed=seed)
        if success:
            print("\n✅ Successfully loaded Amazon products!")
            print(f"📁 Products saved in: {OUTPUT_DIR}/")
            print(f"🔢 Total products: {num_products}")
        else:
            print("\n❌ Failed to load Amazon products.")
            print("Please check your internet connection and try again.")
    else:
        # Original mode: load specific indices
        if len(sys.argv) < 2:
            print("Usage: python meta.py <comma_separated_indices>")
            print("   or: python meta.py --load-N [--seed=123]")
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
