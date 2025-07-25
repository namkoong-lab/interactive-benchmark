import json
import argparse
import os

def prune_zero_count_categories(categories: list) -> list:
    """
    Recursively traverses the category list and removes any category
    or subcategory that has a count of 0.

    Args:
        categories: A list of category dictionaries.

    Returns:
        A new list of categories with zero-count items removed.
    """
    pruned_list = []
    for category in categories:
        if category.get("count", 1) > 0:
            if "subcategories" in category:
                pruned_subs = prune_zero_count_categories(category["subcategories"])
                if pruned_subs:
                    category["subcategories"] = pruned_subs
                else:
                    del category["subcategories"]
            
            pruned_list.append(category)
            
    return pruned_list

def main():
    """
    Main function to parse arguments and run the pruning script.
    """
    parser = argparse.ArgumentParser(
        description="A script to prune categories with a count of 0 from a JSON file."
    )
    parser.add_argument(
        "file_path", 
        type=str, 
        help="The path to the product_categories.json file."
    )
    args = parser.parse_args()

    file_path = args.file_path

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded '{file_path}'.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Please check the file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    if 'categories' not in data:
        print("Error: The JSON file must have a root key named 'categories'.")
        return

    # Prune the categories
    original_count = sum(1 for _ in json.dumps(data).split('"name":'))
    data['categories'] = prune_zero_count_categories(data['categories'])
    pruned_count = sum(1 for _ in json.dumps(data).split('"name":'))
    
    print(f"Pruning complete. Removed {original_count - pruned_count} zero-count categories.")

    base_name, extension = os.path.splitext(file_path)
    output_file_path = f"{base_name}_pruned{extension}"

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved the pruned hierarchy to '{output_file_path}'.")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

if __name__ == "__main__":
    main()
