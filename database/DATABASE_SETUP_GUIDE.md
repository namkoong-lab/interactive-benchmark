# Database Setup Guide

This guide explains how to share and setup the product database using HuggingFace.

---

## 🎯 Architecture Overview

```
Developer (You):                        End User (Other Computer):
┌─────────────────┐                    ┌──────────────────┐
│ products.db     │                    │ (empty)          │
│ - 71K products  │                    └──────────────────┘
│ - 2K categories │                              │
│ - Score cache   │                              │ First run
└─────────────────┘                              ↓
         │                            ┌──────────────────────┐
         │ Export                     │ Auto-downloads from  │
         ↓                            │ HuggingFace:         │
┌─────────────────┐                   │ - products.parquet   │
│ 4 Parquet files │                   │ - categories.parquet │
│ (compressed)    │                   │ - links.parquet      │
└─────────────────┘                   │ - scores.parquet     │
         │                            └──────────────────────┘
         │ Upload                               │
         ↓                                      │ Rebuild
┌─────────────────┐                            ↓
│  HuggingFace    │                   ┌──────────────────┐
│  Dataset Repo   │                   │ products.db      │
│  (browseable!)  │                   │ (ready to use!)  │
└─────────────────┘                   └──────────────────┘
```

---

## 📤 For Developers: Upload Database

### Step 1: Export to Parquet

```bash
cd database
python export_to_parquet.py
```

**Output:**
```
parquet_export/
  ├── products.parquet         (~XX MB)
  ├── categories.parquet        (~0.1 MB)
  ├── product_category.parquet  (~XX MB)
  ├── persona_scores.parquet    (~XX MB)
  └── metadata.json
```

### Step 2: Upload to HuggingFace

```bash
# Login first (one-time)
huggingface-cli login

# Upload
python upload_parquet_to_hf.py
```

**Creates:** `gilberty005/personas-product-database` on HuggingFace

**✅ Benefits:**
- Data is browseable on HuggingFace website
- Users can preview before downloading
- Standard data format (Parquet)

---

## 📥 For Users: Setup Database

### Option 1: Automatic (Recommended)

**Just run the code!** Database downloads automatically on first use:

```bash
python experiment_runners/run_experiment.py --config myconfig.yaml
```

**First run output:**
```
======================================================================
  🔄 Product Database Setup
======================================================================

Database not found at: /path/to/database/products.db
Downloading from HuggingFace and building local database...

📥 Downloading Parquet files from gilberty005/personas-product-database...
   ✅ products.parquet
   ✅ categories.parquet
   ✅ product_category.parquet
   ✅ persona_scores.parquet

🔨 Building SQLite database...
   ✅ 71,088 products imported
   ✅ 2,030 categories imported
   
✅ Database setup complete!
```

**Subsequent runs:** Instant (database already exists)

### Option 2: Manual Pre-Setup

```bash
cd database
python setup_database.py
```

Same process, but you control when it happens.

---

## 🔧 Technical Details

### What Gets Downloaded?

| File | Size | Content |
|------|------|---------|
| `products.parquet` | ~XX MB | All product information |
| `categories.parquet` | ~0.1 MB | Category names |
| `product_category.parquet` | ~XX MB | Product-category links |
| `persona_scores.parquet` | ~XX MB | Pre-cached LLM scores |

### What Gets Built Locally?

A SQLite database (`products.db`) with:
- 4 tables with proper indexes
- Foreign key constraints
- Optimized for fast queries

### Where Is It Stored?

```
your-repo/
  database/
    products.db  ← Built locally (71+ MB)
```

**Note:** `products.db` is in `.gitignore` - each user builds their own.

---

## 🌐 HuggingFace Dataset

**Repository:** https://huggingface.co/datasets/gilberty005/personas-product-database

---

## 🔄 Updating the Database

### For Developers:

If you add more products or scores:

```bash
cd database

# 1. Update products.db locally
python populate_database.py --jsonl new_products.jsonl

# 2. Re-export to Parquet
python export_to_parquet.py

# 3. Re-upload to HuggingFace
python upload_parquet_to_hf.py
```

### For Users:

Force re-download of updated data:

```python
from database.rebuild_from_parquet import ensure_database
ensure_database(force_rebuild=True)
```

Or delete `database/products.db` and re-run your experiment.
