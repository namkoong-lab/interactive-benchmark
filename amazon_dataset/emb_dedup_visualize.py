import os
import time
import json
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns


try:
    client = OpenAI()
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

def load_products_from_jsonl(file_path):
    print(f"\nLoading data from local JSONL file: {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at '{file_path}'")
        return pd.DataFrame()
        
    all_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                all_data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {i+1} in {file_path}.")

    df = pd.DataFrame(all_data)
    print(f"Successfully loaded {len(df)} products.")
    return df


def generate_openai_embeddings(texts, model="text-embedding-3-small", batch_size=200):

    if client is None:
        print("OpenAI client not initialized. Cannot generate embeddings.")
        return np.array([], dtype=np.float32)
        
    print(f"\nGenerating embeddings with OpenAI's '{model}' model...")
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            print(f"Processing batch {i//batch_size + 1}/{num_batches}...")
            response = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"An error occurred while calling the OpenAI API: {e}")
            return np.array(all_embeddings, dtype=np.float32)
            
    return np.array(all_embeddings, dtype=np.float32)

def find_duplicates_faiss(df, embeddings, threshold=0.8, k=20):

    print(f"\nStarting duplicate search with FAISS (threshold={threshold})...")
    if embeddings.size == 0:
        print("Embeddings array is empty. Skipping deduplication.")
        return df.copy(), []

    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) 
    index.add(embeddings)
    
    print(f"Searching for the top {k} neighbors for each product...")
    distances, indices = index.search(embeddings, k + 1)
    
    indices_to_remove = set()
    duplicate_log = []
    processed_pairs = set()

    for i in range(len(indices)):
        for j in range(1, k + 1):
            if distances[i][j] > threshold:
                neighbor_index = indices[i][j]
                
                original_idx = min(i, neighbor_index)
                removed_idx = max(i, neighbor_index)
                
                if (original_idx, removed_idx) not in processed_pairs:
                    indices_to_remove.add(removed_idx)
                    processed_pairs.add((original_idx, removed_idx))
                    
                    original_title = df.loc[original_idx, 'title']
                    removed_title = df.loc[removed_idx, 'title']
                    
                    log_entry = (
                        f"REMOVED: '{removed_title}' (Index: {removed_idx})\n"
                        f"    AS DUPLICATE OF: '{original_title}' (Index: {original_idx})\n"
                        f"    (Similarity: {distances[i][j]:.4f})\n"
                        f"----------------------------------------------------\n"
                    )
                    duplicate_log.append(log_entry)

    unique_df_with_original_indices = df.drop(index=list(indices_to_remove))
    
    print("\n" + "-" * 50)
    print("DEDUPLICATION SUMMARY (FAISS)")
    print(f"Original number of products: {len(df)}")
    print(f"Number of duplicates found and removed: {len(indices_to_remove)}")
    print(f"Number of unique products remaining: {len(unique_df_with_original_indices)}")
    print("-" * 50)

    return unique_df_with_original_indices, duplicate_log

def find_duplicates_cosine(df, embeddings, threshold=0.9):

    print(f"\nStarting duplicate search with pairwise Cosine Similarity (threshold={threshold})...")
    if embeddings.size == 0:
        print("Embeddings array is empty. Skipping deduplication.")
        return df.copy(), []

    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norm
    
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    indices_to_remove = set()
    duplicate_log = []
    
    for i in range(similarity_matrix.shape[0]):
        for j in range(i + 1, similarity_matrix.shape[1]):
            if similarity_matrix[i, j] > threshold:
                indices_to_remove.add(j)
                
                original_title = df.loc[i, 'title']
                removed_title = df.loc[j, 'title']
                
                log_entry = (
                    f"REMOVED: '{removed_title}' (Index: {j})\n"
                    f"    AS DUPLICATE OF: '{original_title}' (Index: {i})\n"
                    f"    (Similarity: {similarity_matrix[i, j]:.4f})\n"
                    f"----------------------------------------------------\n"
                )
                duplicate_log.append(log_entry)

    unique_df_with_original_indices = df.drop(index=list(indices_to_remove))

    print("\n" + "-" * 50)
    print("DEDUPLICATION SUMMARY (Cosine Similarity)")
    print(f"Original number of products: {len(df)}")
    print(f"Number of duplicates found and removed: {len(indices_to_remove)}")
    print(f"Number of unique products remaining: {len(unique_df_with_original_indices)}")
    print("-" * 50)

    return unique_df_with_original_indices, duplicate_log


def create_comparison_visualizations(original_df, unique_df, output_filename):

    print(f"\nGenerating comparison visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Dataset Diversity Comparison: Original vs. Deduplicated', fontsize=24, y=1.02)

    # Plot 1: Main Category Distribution
    original_cat_dist = original_df['main_cat'].value_counts(normalize=True).mul(100)
    unique_cat_dist = unique_df['main_cat'].value_counts(normalize=True).mul(100)
    cat_df = pd.DataFrame({'Original': original_cat_dist, 'Deduplicated': unique_cat_dist}).reset_index()
    cat_df = cat_df.rename(columns={'index': 'Category'}).melt(id_vars='Category', var_name='Dataset', value_name='Percentage')
    sns.barplot(data=cat_df, y='Category', x='Percentage', hue='Dataset', ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Main Category Distribution', fontsize=16)

    # Plot 2: Price Distribution
    original_price = pd.to_numeric(original_df['price'], errors='coerce').dropna()
    unique_price = pd.to_numeric(unique_df['price'], errors='coerce').dropna()
    sns.kdeplot(original_price, ax=axes[0, 1], label='Original', fill=True, clip=(0, 200))
    sns.kdeplot(unique_price, ax=axes[0, 1], label='Deduplicated', fill=True, alpha=0.7, clip=(0, 200))
    axes[0, 1].set_title('Price Distribution (Clipped at $200)', fontsize=16)
    axes[0, 1].set_xlim(0, 200)
    axes[0, 1].legend()

    # Plot 3: Average Rating Distribution
    sns.kdeplot(original_df['average_rating'].dropna(), ax=axes[1, 0], label='Original', fill=True)
    sns.kdeplot(unique_df['average_rating'].dropna(), ax=axes[1, 0], label='Deduplicated', fill=True, alpha=0.7)
    axes[1, 0].set_title('Average Rating Distribution', fontsize=16)
    axes[1, 0].legend()
    
    # Plot 4: Top Brands Distribution
    top_brands = original_df['brand'].value_counts().nlargest(15).index
    original_brand_dist = original_df[original_df['brand'].isin(top_brands)]['brand'].value_counts(normalize=True).mul(100)
    unique_brand_dist = unique_df[unique_df['brand'].isin(top_brands)]['brand'].value_counts(normalize=True).mul(100)
    brand_df = pd.DataFrame({'Original': original_brand_dist, 'Deduplicated': unique_brand_dist}).reset_index()
    brand_df = brand_df.rename(columns={'index': 'Brand'}).melt(id_vars='Brand', var_name='Dataset', value_name='Percentage')
    sns.barplot(data=brand_df, y='Brand', x='Percentage', hue='Dataset', ax=axes[1, 1], palette='plasma')
    axes[1, 1].set_title('Top 15 Brand Distribution', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to '{output_filename}'")

if __name__ == "__main__":
    start_time = time.perf_counter()

    INPUT_FILE = 'benchmark_metadata.jsonl'
    
    UNIQUE_PRODUCTS_OUTPUT_FILE = 'unique_products.jsonl'
    DUPLICATE_LOG_OUTPUT_FILE = 'duplicate_log.txt'
    UNIQUE_EMBEDDINGS_OUTPUT_FILE = 'unique_embeddings.npy'
    VISUALIZATION_OUTPUT_FILE = 'dataset_diversity_comparison.png'

    DEDUPLICATION_THRESHOLD = 0.8 
    EMBEDDING_MODEL = "text-embedding-3-small"

    original_df = load_products_from_jsonl(INPUT_FILE)
    
    if original_df.empty:
        print("Exiting script due to loading error.")
        exit()

    original_df.dropna(subset=['title'], inplace=True)
    original_df.reset_index(drop=True, inplace=True)
    print(f"Product count after dropping items with no title: {len(original_df)}")
    
    def combine_product_fields(row):
        title = row.get('title', '') or ''
        features = ' '.join(row.get('features', [])) if isinstance(row.get('features'), list) else ''
        description = ' '.join(row.get('description', [])) if isinstance(row.get('description'), list) else ''
        categories = ' '.join(row.get('categories', [])) if isinstance(row.get('categories'), list) else ''
        price = str(row.get('price', ''))
        details = str(row.get('details', ''))
        combined_text = (
            f"Title: {title}. Categories: {categories}. Price: {price}. "
            f"Features: {features}. Description: {description}. Details: {details}"
        )
        return combined_text.strip()

    product_texts = original_df.apply(combine_product_fields, axis=1).tolist()
    all_embeddings = generate_openai_embeddings(product_texts, model=EMBEDDING_MODEL)
    
    if all_embeddings.size == 0:
        print("Embedding generation failed. Exiting script.")
        exit()

    unique_df_with_old_indices, duplicate_log = find_duplicates_faiss(
        original_df, 
        all_embeddings, 
        threshold=DEDUPLICATION_THRESHOLD
    )


    unique_indices = unique_df_with_old_indices.index.tolist()
    unique_embeddings = all_embeddings[unique_indices]
    
    print(f"\nSaving {len(unique_embeddings)} unique embeddings to '{UNIQUE_EMBEDDINGS_OUTPUT_FILE}'...")
    np.save(UNIQUE_EMBEDDINGS_OUTPUT_FILE, unique_embeddings)
    print("Save complete.")

    unique_df = unique_df_with_old_indices.reset_index(drop=True)

    print(f"\nSaving {len(unique_df)} unique products to '{UNIQUE_PRODUCTS_OUTPUT_FILE}'...")
    unique_df.to_json(UNIQUE_PRODUCTS_OUTPUT_FILE, orient='records', lines=True, force_ascii=False)
    print("Save complete.")
    
    print(f"\nSaving the duplicate pairs log to '{DUPLICATE_LOG_OUTPUT_FILE}'...")
    with open(DUPLICATE_LOG_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(duplicate_log)
    print("Save complete.")

    create_comparison_visualizations(original_df, unique_df, VISUALIZATION_OUTPUT_FILE)

    end_time = time.perf_counter()
    print(f"\n\n--- Workflow Complete ---")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print("Generated files:")
    print(f"  - Unique Products: {UNIQUE_PRODUCTS_OUTPUT_FILE}")
    print(f"  - Unique Embeddings: {UNIQUE_EMBEDDINGS_OUTPUT_FILE}")
    print(f"  - Duplicate Log: {DUPLICATE_LOG_OUTPUT_FILE}")
    print(f"  - Comparison Chart: {VISUALIZATION_OUTPUT_FILE}")