import json
import os
import argparse
import numpy as np
from rich.console import Console
from rich.tree import Tree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
except ImportError:
    print("Required libraries not found. Please run:")
    print("pip install sentence-transformers scikit-learn numpy rich")
    exit(1)

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
console = Console()


def load_products(file_path):
    console.print(f"[bold blue]Loading products from JSON Lines file '{file_path}'...[/bold blue]")
    products = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                try:
                    line_data = json.loads(line)
                    if not isinstance(line_data, dict) or 'metadata' not in line_data: continue
                    metadata = line_data['metadata']
                    if isinstance(metadata, dict) and 'title' in metadata:
                        products.append({
                            'id': line_data.get('item_id', f'unknown_id_{i}'),
                            'title': metadata['title'],
                        })
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        console.print(f"[bold red]Error: The file '{file_path}' was not found.[/bold red]")
        return None
    console.print(f"[green]Successfully loaded {len(products)} products.[/green]")
    return products

def generate_embeddings(titles, model):
    console.print(f"[bold blue]Generating embeddings for {len(titles)} titles...[/bold blue]")
    embeddings = model.encode(titles, show_progress_bar=True)
    console.print("[green]Embeddings generated successfully.[/green]")
    return embeddings

def build_cluster_structure(indices, all_products, embeddings, k_values, max_depth, min_cluster_size, current_depth=0):
    num_items = len(indices)

    k_for_this_level = k_values[min(current_depth, len(k_values) - 1)]

    if num_items < k_for_this_level:
        return None

    if num_items < min_cluster_size or current_depth >= max_depth:
        return None

    current_embeddings = embeddings[indices]
    
    kmeans = KMeans(n_clusters=k_for_this_level, random_state=42, n_init='auto')
    kmeans.fit(current_embeddings)
    
    children = []
    for i in range(k_for_this_level):
        sub_cluster_indices_relative = np.where(kmeans.labels_ == i)[0]
        if len(sub_cluster_indices_relative) == 0: continue
        
        sub_cluster_indices_original = [indices[j] for j in sub_cluster_indices_relative]
        
        sub_cluster_product_ids = [all_products[j]['id'] for j in sub_cluster_indices_original]
        sub_cluster_product_titles = [all_products[j]['title'] for j in sub_cluster_indices_original]
        
        node = {
            "cluster_name": f"Cluster-{current_depth}-{i}",
            "item_count": len(sub_cluster_product_ids),
            "item_ids": sub_cluster_product_ids,
            "item_titles": sub_cluster_product_titles,
            "children": build_cluster_structure(
                sub_cluster_indices_original, all_products, embeddings,
                k_values, max_depth, min_cluster_size, current_depth + 1
            )
        }
        children.append(node)
    return children

def main():
    parser = argparse.ArgumentParser(description="Cluster products and save the structure.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file with product data.")
    parser.add_argument("--output_file", type=str, default="clustering_results.json", help="File to save the cluster structure.")
    parser.add_argument("--k", type=int, nargs='+', default=[20, 10, 5, 3, 2], help="A space-separated list of k values for each hierarchy level (e.g., --k 5 10 8).")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth of the hierarchy.")
    parser.add_argument("--min_size", type=int, default=5, help="Minimum cluster size to continue recursion.")
    args = parser.parse_args()

    products = load_products(args.file_path)
    if not products: return

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    all_embeddings = generate_embeddings([p['title'] for p in products], embedding_model)

    console.print(f"[bold blue]Building cluster structure (k={args.k}, max_depth={args.max_depth})...[/bold blue]")
    initial_indices = list(range(len(products)))
    
    hierarchy = build_cluster_structure(
        indices=initial_indices,
        all_products=products,
        embeddings=all_embeddings,
        k_values=args.k, 
        max_depth=args.max_depth,
        min_cluster_size=args.min_size
    )

    if hierarchy:
        console.print(f"\n[bold green]Structure generated! Saving to '{args.output_file}'...[/bold green]")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(hierarchy, f, indent=2)
        console.print("[bold cyan]Next step: Run name.py to name the clusters.[/bold cyan]")
    else:
        console.print("[bold red]Failed to generate cluster structure.[/bold red]")

if __name__ == "__main__":
    main()