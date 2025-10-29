#!/usr/bin/env python3
"""
Export products.db to Parquet format for HuggingFace upload.
"""
import sqlite3
import pandas as pd
import os
import json

DB_FILE = "products.db"
DB_PATH = os.path.join(os.path.dirname(__file__), DB_FILE)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "parquet_export")

def export_database_to_parquet():
    """Export all tables from products database to Parquet files."""
    
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "products.db")
    ]
    
    db_path = None
    for path in possible_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if db_path is None:
        print(f"❌ Error: No database found at:")
        for path in possible_paths:
            print(f"   - {path}")
        return
    
    print(f"✅ Using database: {os.path.basename(db_path)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📊 Exporting database to Parquet format...")
    print(f"   Input: {db_path}")
    print(f"   Output: {OUTPUT_DIR}/\n")
    
    conn = sqlite3.connect(db_path)
    
    # Export each table
    tables = {
        'products': 'products.parquet',
        'categories': 'categories.parquet',
        'product_category': 'product_category.parquet',
        'persona_scores': 'persona_scores.parquet'
    }
    
    total_size = 0
    
    for table_name, parquet_file in tables.items():
        print(f"📦 Exporting {table_name}...")
        
        # Check if table exists
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cur.fetchone():
            print(f"   ⚠️  Table {table_name} not found - creating empty file")
            # Create empty DataFrame with expected schema
            if table_name == 'persona_scores':
                df = pd.DataFrame(columns=['persona_index', 'category_id', 'product_id', 'score', 'reason', 'model', 'created_at'])
            else:
                print(f"   ❌ Skipping {table_name} (table missing and no default schema)")
                continue
        else:
            # Read table
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            
            # Handle JSON column in products table
            if table_name == 'products' and 'raw' in df.columns:
                # Keep raw as string (JSON text)
                print(f"   Keeping 'raw' column as JSON string")
        
        # Save to Parquet
        output_path = os.path.join(OUTPUT_DIR, parquet_file)
        row_group_size = 10000 if table_name == 'products' else None
        df.to_parquet(output_path, index=False, compression='snappy', row_group_size=row_group_size)
        
        # Stats
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        total_size += file_size
        print(f"   ✅ {table_name}: {len(df):,} rows, {file_size:.2f} MB")
        print(f"      Columns: {list(df.columns)}")
    
    conn.close()
    
    # Create metadata file
    metadata = {
        'version': '1.0',
        'source_db': os.path.basename(db_path),
        'tables': tables,
        'export_date': pd.Timestamp.now().isoformat(),
        'total_size_mb': round(total_size, 2),
        'statistics': {}
    }
    
    # Add table statistics
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM products")
    metadata['statistics']['num_products'] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM categories")
    metadata['statistics']['num_categories'] = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM product_category")
    metadata['statistics']['num_product_category_links'] = cur.fetchone()[0]
    
    # Check if persona_scores table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='persona_scores'")
    if cur.fetchone():
        cur.execute("SELECT COUNT(*) FROM persona_scores")
        metadata['statistics']['num_cached_scores'] = cur.fetchone()[0]
    else:
        metadata['statistics']['num_cached_scores'] = 0
    
    conn.close()
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Export complete!")
    print(f"   Total size: {total_size:.2f} MB")
    print(f"   Files saved to: {OUTPUT_DIR}/")
    print(f"\n📊 Statistics:")
    for key, value in metadata['statistics'].items():
        print(f"   {key}: {value:,}")
    
    return OUTPUT_DIR

if __name__ == "__main__":
    print("="*70)
    print("  Export Products Database to Parquet Format")
    print("="*70)
    print()
    
    export_database_to_parquet()
    
    print("\n📤 Next step:")
    print("   Run upload_parquet_to_hf.py to upload these files to HuggingFace")

