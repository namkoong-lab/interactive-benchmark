Load a subset of data:
```bash
python3 load_products_sqlite_simple.py \
  --db simple_products.db \
  --jsonl benchmark_metadata_assigned.jsonl \
  --limit 100
```

### Parameters
- `--db`: Path to SQLite database file (will be created if doesn't exist)
- `--jsonl`: Path to input JSONL file with product data
- `--limit`: Optional limit on number of lines to process (for testing)

## Database Schema

### Tables

1. **`categories`** - Flat list of category names
   - `id` (PRIMARY KEY)
   - `name` (UNIQUE)

2. **`products`** - Product information
   - `id` (PRIMARY KEY)
   - `external_id` (UNIQUE, from ASIN)
   - `title`
   - `main_category`
   - `store`
   - `price`
   - `raw` (Full original JSON)

3. **`product_category`** - Many-to-many relationship
   - `product_id` (REFERENCES products.id)
   - `category_id` (REFERENCES categories.id)
   - PRIMARY KEY (product_id, category_id)

## Querying the Database

### Open SQLite Shell

```bash
sqlite3 simple_products.db
```

### Basic Setup Commands

```sql
.headers on
.mode box
```

### Basic Queries

#### View All Tables
```sql
.tables
```

#### Count Records
```sql
SELECT COUNT(*) AS products FROM products;
SELECT COUNT(*) AS categories FROM categories;
SELECT COUNT(*) AS links FROM product_category;
```

#### List All Categories
```sql
SELECT * FROM categories ORDER BY name;
```

#### Find Categories with Most Products
```sql
SELECT c.name, COUNT(*) as product_count
FROM categories c
JOIN product_category pc ON pc.category_id = c.id
GROUP BY c.id
ORDER BY product_count DESC
LIMIT 20;
```

#### Get All Products in a Category
```sql
SELECT p.id, p.title, p.price, p.store
FROM products p
JOIN product_category pc ON pc.product_id = p.id
JOIN categories c ON c.id = pc.category_id
WHERE c.name = 'Electronics'
ORDER BY p.title
LIMIT 50;
```

#### Show All Categories for a Product
```sql
SELECT c.name
FROM products p
JOIN product_category pc ON pc.product_id = p.id
JOIN categories c ON c.id = pc.category_id
WHERE p.id = 1;
```