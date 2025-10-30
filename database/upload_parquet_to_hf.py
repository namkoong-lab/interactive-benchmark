#!/usr/bin/env python3
"""
Upload Parquet files to HuggingFace dataset repository.
"""
import os
import json
from huggingface_hub import HfApi

# Configuration
REPO_ID = "gilberty005/personas-product-database"  # Change to your HF username
PARQUET_DIR = os.path.join(os.path.dirname(__file__), "parquet_export")

def upload_to_huggingface():
    """Upload all Parquet files to HuggingFace."""
    
    if not os.path.exists(PARQUET_DIR):
        print(f"❌ Error: Parquet directory not found: {PARQUET_DIR}")
        print("Please run export_to_parquet.py first!")
        return
    
    # Get list of files (flat structure at root)
    parquet_files = [
        'products.parquet',
        'categories.parquet',
        'product_category.parquet',
        'persona_scores.parquet',
        'metadata.json'
    ]
    
    # Check all files exist
    missing = [f for f in parquet_files if not os.path.exists(os.path.join(PARQUET_DIR, f))]
    if missing:
        print(f"❌ Error: Missing files: {missing}")
        return
    
    print(f"📤 Uploading to HuggingFace dataset: {REPO_ID}")
    print()
    
    api = HfApi()
    
    # Create repo if needed
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            exist_ok=True,
            private=False
        )
        print(f"✅ Repository ready: {REPO_ID}\n")
    except Exception as e:
        print(f"ℹ️  Repository status: {e}\n")
    
    # Upload each file to root
    for filename in parquet_files:
        file_path = os.path.join(PARQUET_DIR, filename)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        print(f"📤 Uploading {filename} ({file_size_mb:.2f} MB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload {filename}"
            )
            print(f"   ✅ {filename} uploaded\n")
        except Exception as e:
            print(f"   ❌ Failed to upload {filename}: {e}\n")
            return
    
    # Create and upload README
    print("📝 Creating README...")
    create_and_upload_readme(api)
    
    print("="*70)
    print("✅ Upload Complete!")
    print("="*70)
    print(f"\n📍 View dataset: https://huggingface.co/datasets/{REPO_ID}")
    print(f"📍 Data viewer: https://huggingface.co/datasets/{REPO_ID}/viewer/")

def create_and_upload_readme(api: HfApi):
    """Create and upload README for the dataset."""
    
    # Load metadata for stats
    metadata_path = os.path.join(PARQUET_DIR, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    stats = metadata.get('statistics', {})
    
    readme_content = f"""---
license: mit
task_categories:
- other
pretty_name: Personas Product Database
size_categories:
- 10K<n<100K
---

# Personas Product Database

Product database for running recommendation experiments with the Personas framework.

## Dataset Summary

This dataset contains product information and categorization data used for conversational recommendation experiments.

**Statistics:**
- **Products:** {stats.get('num_products', 0):,}
- **Categories:** {stats.get('num_categories', 0):,}
- **Product-Category Links:** {stats.get('num_product_category_links', 0):,}
- **Cached Persona Scores:** {stats.get('num_cached_scores', 0):,}

## Files

- `products.parquet` - Product information (id, title, price, category, raw metadata)
- `categories.parquet` - Category names and IDs
- `product_category.parquet` - Many-to-many links between products and categories
- `persona_scores.parquet` - Cached LLM scores for persona-product pairs
- `metadata.json` - Export metadata and statistics

## Schema

### products
| Column | Type | Description |
|--------|------|-------------|
| id | int | Internal product ID |
| external_id | str | Amazon ASIN |
| title | str | Product name |
| main_category | str | Amazon's main category |
| store | str | Store name |
| price | float | Price in USD |
| raw | str | Full product metadata (JSON string) |

### categories
| Column | Type | Description |
|--------|------|-------------|
| id | int | Category ID |
| name | str | Category name |

### product_category
| Column | Type | Description |
|--------|------|-------------|
| product_id | int | References products.id |
| category_id | int | References categories.id |

### persona_scores
| Column | Type | Description |
|--------|------|-------------|
| persona_index | int | Persona ID (from Tianyi-Lab/Personas) |
| category_id | int | References categories.id |
| product_id | int | References products.id |
| score | float | Persona's score for product (0-100) |
| reason | str | Explanation for score |
| model | str | LLM model used for scoring |
| created_at | int | Unix timestamp |

## Usage

### Download with Python

```python
from datasets import load_dataset

# Load as HuggingFace dataset
dataset = load_dataset("{REPO_ID}")

# Access individual tables
products = dataset['products'].to_pandas()
categories = dataset['categories'].to_pandas()
```

### Rebuild SQLite Database

```python
# The experiment code includes auto-download and rebuild functionality
# Just run your experiment and the database will be created automatically!

from pipeline.core.simulate_interaction import ensure_database
ensure_database()  # Downloads Parquet files and builds SQLite DB
```

## Related Datasets

- **Personas:** [Tianyi-Lab/Personas](https://huggingface.co/datasets/Tianyi-Lab/Personas) - 200K synthetic customer personas

## Version

- **Version:** {metadata.get('version', '1.0')}
- **Export Date:** {metadata.get('export_date', 'N/A')}

## License

MIT License
"""
    
    # Save and upload README
    readme_path = os.path.join(PARQUET_DIR, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add README with dataset documentation"
        )
        print("   ✅ README uploaded")
    except Exception as e:
        print(f"   ⚠️  README upload failed: {e}")
    
    os.remove(readme_path)

if __name__ == "__main__":
    print("="*70)
    print("  Upload Parquet Files to HuggingFace")
    print("="*70)
    print()
    
    print("⚠️  Make sure you're logged in to HuggingFace:")
    print("   huggingface-cli login")
    print()
    
    upload_to_huggingface()

