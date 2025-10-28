#!/usr/bin/env python3
"""
One-time setup script to download and build the database.
Users can run this manually or it happens automatically on first import.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.rebuild_from_parquet import ensure_database

if __name__ == "__main__":
    print("="*70)
    print("  Personas Product Database Setup")
    print("="*70)
    print()
    print("This will download the product database from HuggingFace")
    print("and build a local SQLite database for fast querying.")
    print()
    
    try:
        db_path = ensure_database(force_rebuild=False)
        print(f"\n✅ Setup complete!")
        print(f"   Database ready at: {db_path}")
        print(f"\nYou can now run experiments!")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check internet connection")
        print(f"2. Verify repository exists on HuggingFace")
        print(f"3. Install dependencies: pip install pandas pyarrow huggingface_hub")
        sys.exit(1)

