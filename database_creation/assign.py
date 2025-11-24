import json
import argparse
from collections import deque
import tqdm
import os
import sqlite3 # MODIFIED: Import the sqlite3 library

class CategoryNode:
    """A node in the category hierarchy tree."""
    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.count = 0
        self.children = {}
        self.parent = None

    def __repr__(self):
        return f"CategoryNode(name='{self.name}', count={self.count}, children={len(self.children)})"

def parse_category_tree(filepath):
    # This function remains unchanged
    root = CategoryNode("Root", -1)
    parent_stack = {-4: root}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                leading_spaces = len(line) - len(line.lstrip(' '))
                level = leading_spaces // 4
                name = line.strip().lstrip('- ').strip()
                node = CategoryNode(name, level)
                parent_level = level - 1
                while parent_level * 4 not in parent_stack and parent_level >= -1:
                    parent_level -= 1
                parent = parent_stack[parent_level * 4]
                node.parent = parent
                parent.children[name] = node
                parent_stack[leading_spaces] = node
    except FileNotFoundError:
        print(f"Error: Category file not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"An error occurred while parsing the category tree: {e}")
        return None
    return root

def find_best_match(product_category, category_nodes):
    # This function remains unchanged
    norm_product_cat = product_category.lower().replace("'s", "").rstrip('s')
    for node_name, node in category_nodes.items():
        if node_name.lower() == norm_product_cat:
            return node
    product_cat_words = set(norm_product_cat.split())
    for node_name, node in category_nodes.items():
        norm_node_name_words = set(node_name.lower().replace("'s", "").split())
        if product_cat_words.issubset(norm_node_name_words):
            return node
    for node_name, node in category_nodes.items():
        norm_node_name = node_name.lower().replace("'s", "")
        if norm_product_cat in norm_node_name:
            return node
    norm_product_cat_no_and = norm_product_cat.replace('&', '').replace(' and ', '')
    for node_name, node in category_nodes.items():
        norm_node_name_no_and = node_name.lower().replace('&', '').replace(' and ', '')
        if norm_product_cat_no_and == norm_node_name_no_and:
            return node
    return None

def find_match_in_subtree(product_category, start_node):
    # This function remains unchanged
    if not start_node:
        return None
    queue = deque(start_node.children.values())
    while queue:
        node = queue.popleft()
        match = find_best_match(product_category, {node.name: node})
        if match:
            return match
        for child in node.children.values():
            queue.append(child)
    return None

# ###############################################
# MODIFIED FUNCTION
# ###############################################
def process_products(db_filepath, category_root, record_filepath, assigned_filepath):
    """Processes each product from a SQLite DB, assigns it to categories, and updates counts."""
    conn = None
    try:
        # Connect to the database and get all product data from the "raw" column
        conn = sqlite3.connect(db_filepath)
        cursor = conn.cursor()
        print(f"Querying products from {db_filepath}...")
        cursor.execute('SELECT "raw" FROM products')
        # fetchall() gets all rows from the query result
        product_rows = cursor.fetchall()
        print(f"Found {len(product_rows)} products to process.")

        with open(record_filepath, 'w', encoding='utf-8') as f_record, \
             open(assigned_filepath, 'w', encoding='utf-8') as f_assigned:

            for row_tuple in tqdm.tqdm(product_rows, desc="Processing Products", unit="product"):
                try:
                    # Each row is a tuple, the raw JSON string is the first element
                    raw_json_string = row_tuple[0]
                    product = json.loads(raw_json_string)

                    # --- From this point on, the logic is IDENTICAL to the original script ---
                    item_id = product.get("parent_asin") or product.get("asin", "N/A")
                    categories = product.get("categories", [])
                    
                    f_record.write(f"Processing item_id: {item_id}\n")
                    f_record.write(f"- Categories: {categories}\n")

                    if not categories or not category_root:
                        f_record.write("- Status: Failure (No categories provided or tree not loaded)\n---\n")
                        continue

                    current_node = category_root
                    path_found = True
                    
                    for product_cat in categories:
                        match = find_match_in_subtree(product_cat, current_node)
                        if match:
                            current_node = match
                        else:
                            path_found = False
                            break
                    
                    if path_found and current_node is not category_root:
                        full_path_nodes = []
                        temp_node = current_node
                        while temp_node and temp_node.parent:
                            full_path_nodes.insert(0, temp_node)
                            temp_node = temp_node.parent
                        
                        full_path_names = [n.name for n in full_path_nodes]
                        f_record.write(f"- Mapped Path: {' > '.join(full_path_names)}\n")
                        f_record.write("- Status: Success\n")

                        product['category_path'] = full_path_names
                        f_assigned.write(json.dumps(product) + '\n')
                        
                        node_to_increment = current_node
                        while node_to_increment and node_to_increment.parent:
                            node_to_increment.count += 1
                            node_to_increment = node_to_increment.parent
                    else:
                        f_record.write(f"- Mapped Path: Not found\n")
                        f_record.write(f"- Status: Failure (Could not find full path)\n")
                        
                    f_record.write("---\n")

                except Exception as e:
                    f_record.write(f"An error occurred processing a product row: {e}\n---\n")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred during product processing: {e}")
    finally:
        if conn:
            conn.close()


def write_output_counts(category_root, output_filepath):
    # This function remains unchanged
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            nodes_to_visit = deque(category_root.children.values())
            while nodes_to_visit:
                node = nodes_to_visit.popleft()
                indent = '    ' * node.level
                f.write(f"{indent}{node.name} [{node.count}]\n")
                children = list(node.children.values())
                for child in reversed(children):
                    nodes_to_visit.appendleft(child)
    except Exception as e:
        print(f"An error occurred while writing the output counts file: {e}")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Assigns products from a SQLite DB file to a category hierarchy and counts them."
    )
    parser.add_argument(
        "db_file", 
        help="Path to the input SQLite database file (e.g., raw_meta_Books.db)"
    )
    parser.add_argument(
        "category_file", 
        help="Path to the category hierarchy text file (e.g., AmazonCategory.txt)"
    )
    # The defaults are now just placeholders that we check against
    parser.add_argument(
        "--output_counts", 
        default="counted_category.txt", 
        help="Path for the output file with category counts."
    )
    parser.add_argument(
        "--output_record", 
        default="record.txt", 
        help="Path for the output log file."
    )
    parser.add_argument(
        "--output_assigned",
        default="metadata_assigned.jsonl",
        help="Path for the output JSONL file with successfully assigned products."
    )
    
    args = parser.parse_args()

    # ####################################################################
    # MODIFIED: Logic to create dynamic output filenames
    # ####################################################################
    
    # 1. Get the base name of the input file (e.g., "raw_meta_Books.db")
    basename = os.path.basename(args.db_file)
    
    # 2. Extract the unique identifier (e.g., "Books")
    identifier = os.path.splitext(basename)[0].replace('raw_meta_', '')

    # 3. If the user DID NOT provide custom output names, create them dynamically.
    #    If they did provide a name, we will use that instead.
    if args.output_counts == "counted_category.txt":
        args.output_counts = f"counted_category_{identifier}.txt"

    if args.output_record == "record.txt":
        args.output_record = f"record_{identifier}.txt"

    if args.output_assigned == "metadata_assigned.jsonl":
        args.output_assigned = f"metadata_assigned_{identifier}.jsonl"
    # ####################################################################

    print("Step 1: Parsing category hierarchy...")
    category_tree_root = parse_category_tree(args.category_file)
    
    if category_tree_root:
        print("Step 2: Processing products and assigning categories...")
        process_products(args.db_file, category_tree_root, args.output_record, args.output_assigned)
        
        print("Step 3: Writing final counts to output file...")
        write_output_counts(category_tree_root, args.output_counts)
        
        print("\nProcess complete.")
        print(f"Category counts saved to: {args.output_counts}")
        print(f"Assignment log saved to: {args.output_record}")
        print(f"Successfully assigned products saved to: {args.output_assigned}")
        
if __name__ == "__main__":
    main()