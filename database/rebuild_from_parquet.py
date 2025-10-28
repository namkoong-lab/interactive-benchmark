#!/usr/bin/env python3
"""
Download Parquet files from HuggingFace and rebuild SQLite database.
"""
import os
import sqlite3
import pandas as pd
from huggingface_hub import hf_hub_download
from typing import Optional

# Configuration
REPO_ID = "gilberty005/personas-product-database"  # Change to your HF username
DB_FILE = "products.db"
LOCAL_DB_DIR = os.path.dirname(__file__)
LOCAL_DB_PATH = os.path.join(LOCAL_DB_DIR, DB_FILE)

PARQUET_FILES = [
    'products.parquet',
    'categories.parquet',
    'product_category.parquet',
    'persona_scores.parquet'
]

def download_parquet_files(cache_dir: Optional[str] = None) -> dict:
    """Download all Parquet files from HuggingFace."""
    print(f"📥 Downloading Parquet files from {REPO_ID}...\n")
    
    downloaded_files = {}
    
    for filename in PARQUET_FILES:
        print(f"   Downloading {filename}...")
        try:
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir
            )
            downloaded_files[filename] = file_path
            print(f"   ✅ {filename}")
        except Exception as e:
            print(f"   ❌ Failed to download {filename}: {e}")
            raise
    
    print(f"\n✅ All files downloaded\n")
    return downloaded_files

def rebuild_database(parquet_files: dict, output_db_path: str = LOCAL_DB_PATH):
    """Rebuild SQLite database from Parquet files."""
    print(f"🔨 Building SQLite database: {output_db_path}\n")
    
    # Remove existing database if present
    if os.path.exists(output_db_path):
        print(f"   Removing existing database...")
        os.remove(output_db_path)
    
    # Create new database with schema
    conn = sqlite3.connect(output_db_path)
    cur = conn.cursor()
    
    # Enable foreign keys
    cur.execute("PRAGMA foreign_keys = ON")
    
    print("   Creating schema...")
    
    # Create tables
    cur.execute("""
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        )
    """)
    
    cur.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            external_id TEXT,
            title TEXT NOT NULL,
            main_category TEXT,
            store TEXT,
            price REAL,
            raw JSON
        )
    """)
    
    cur.execute("""
        CREATE UNIQUE INDEX idx_products_external_id
        ON products(external_id)
    """)
    
    cur.execute("""
        CREATE TABLE product_category (
            product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
            category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
            PRIMARY KEY (product_id, category_id)
        )
    """)
    
    cur.execute("""
        CREATE TABLE persona_scores (
            persona_index INTEGER NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
            product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
            score REAL NOT NULL,
            reason TEXT,
            model TEXT,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (persona_index, category_id, product_id)
        )
    """)
    
    conn.commit()
    
    # Import data from Parquet files
    table_map = {
        'categories.parquet': 'categories',
        'products.parquet': 'products',
        'product_category.parquet': 'product_category',
        'persona_scores.parquet': 'persona_scores'
    }
    
    for parquet_file, table_name in table_map.items():
        if parquet_file not in parquet_files:
            print(f"   ⚠️  Skipping {table_name} (file not found)")
            continue
        
        print(f"   Importing {table_name}...")
        df = pd.read_parquet(parquet_files[parquet_file])
        
        # Import to SQLite
        df.to_sql(table_name, conn, if_exists='append', index=False)
        print(f"      ✅ {len(df):,} rows imported")
    
    conn.commit()
    
    # Create indexes for performance
    print("   Creating indexes...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_product_category_product ON product_category(product_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_product_category_category ON product_category(category_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_persona_scores_lookup ON persona_scores(persona_index, category_id, product_id)")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Database rebuilt successfully!")
    print(f"   Location: {output_db_path}")
    
    # Show stats
    conn = sqlite3.connect(output_db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM products")
    num_products = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM categories")
    num_categories = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM persona_scores")
    num_scores = cur.fetchone()[0]
    
    conn.close()
    
    print(f"\n📊 Database Statistics:")
    print(f"   Products: {num_products:,}")
    print(f"   Categories: {num_categories:,}")
    print(f"   Cached Scores: {num_scores:,}")

def ensure_database(force_rebuild: bool = False) -> str:
    """
    Ensure database exists. Download and rebuild if needed.
    
    Args:
        force_rebuild: Force re-download and rebuild even if DB exists
        
    Returns:
        Path to products.db
    """
    # Check if database already exists
    if os.path.exists(LOCAL_DB_PATH) and not force_rebuild:
        return LOCAL_DB_PATH
    
    print(f"🔄 Database not found locally. Downloading from HuggingFace...\n")
    
    # Download Parquet files
    parquet_files = download_parquet_files()
    
    # Rebuild database
    rebuild_database(parquet_files, LOCAL_DB_PATH)
    
    return LOCAL_DB_PATH

if __name__ == "__main__":
    print("="*70)
    print("  Upload Parquet Files to HuggingFace")
    print("="*70)
    print()
    
    print("⚠️  Prerequisites:")
    print("   1. Run export_to_parquet.py first")
    print("   2. Login to HuggingFace: huggingface-cli login")
    print()
    
    input("Press Enter to continue or Ctrl+C to cancel...")
    print()
    
    upload_to_huggingface()

