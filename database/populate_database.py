#!/usr/bin/env python3
import argparse
import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Set, Tuple


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
STOPWORDS = {"and", "&"}


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        PRAGMA foreign_keys = ON;
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT,
            title TEXT NOT NULL,
            main_category TEXT,
            store TEXT,
            price REAL,
            raw JSON
        );
        """
    )

    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_products_external_id
        ON products(external_id);
        """
    )

    # Many-to-many link between products and categories
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS product_category (
            product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
            category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
            PRIMARY KEY (product_id, category_id)
        );
        """
    )

    conn.commit()


def get_or_create_category(conn: sqlite3.Connection, category_name: str) -> int:
    """Get category ID, create if doesn't exist"""
    cur = conn.cursor()

    # Try to find existing category
    cur.execute("SELECT id FROM categories WHERE name = ?", (category_name,))
    row = cur.fetchone()
    if row is not None:
        return int(row[0])

    # Create new category
    cur.execute("INSERT INTO categories(name) VALUES (?)", (category_name,))
    return cur.lastrowid


def normalize_token(token: str) -> str:
    token = token.lower().strip()
    token = NON_ALNUM_RE.sub("", token)
    # naive singularization: drop trailing 's' for tokens with length > 4
    if len(token) > 4 and token.endswith("s"):
        token = token[:-1]
    return token


def canonical_category_key(name: str) -> str:
    # tokenize on non-alnum, remove stopwords, normalize, sort tokens to match reordered variants
    tokens = re.split(r"[^a-zA-Z0-9]+", name.lower())
    norm_tokens: List[str] = []
    for t in tokens:
        if not t or t in STOPWORDS:
            continue
        nt = normalize_token(t)
        if nt and nt not in STOPWORDS:
            norm_tokens.append(nt)
    norm_tokens.sort()
    return " ".join(norm_tokens)


def select_canonical_name(candidates: Set[str]) -> str:
    # prefer shortest by length; tie-breaker: lexicographically
    return sorted(candidates, key=lambda s: (len(s), s.lower()))[0]


def extract_categories(row: Dict) -> Set[str]:
    """
    Extract category names from a product row, merging very similar names and
    ignoring the first two levels of category chains when possible.
    Returns a set of canonical category names (original strings) to store.
    """
    # 1) collect raw candidate names from provided fields
    raw_candidates: List[str] = []

    # Prefer list-of-lists fields as hierarchical chains
    maybe_multi = row.get("category_paths") or row.get("category_chains")
    if isinstance(maybe_multi, list) and maybe_multi and isinstance(maybe_multi[0], list):
        for chain in maybe_multi:
            flattened = [str(x).strip() for x in chain if x is not None and str(x).strip()]
            if not flattened:
                continue
            if len(flattened) > 2:
                # ignore first two levels, keep deeper ones
                raw_candidates.extend(flattened[2:])
            else:
                # chain length <= 2 -> keep the deepest available
                raw_candidates.append(flattened[-1])

    # Single chain fields
    if not raw_candidates:
        for key in ("category_path", "categories"):
            val = row.get(key)
            if isinstance(val, list) and val:
                flattened = [str(x).strip() for x in val if x is not None and str(x).strip()]
                if not flattened:
                    continue
                if len(flattened) > 2:
                    raw_candidates.extend(flattened[2:])
                else:
                    raw_candidates.append(flattened[-1])
                break

    # Fallback to main_category (single value)
    if not raw_candidates and row.get("main_category"):
        raw_candidates.append(str(row["main_category"]).strip())

    # If after ignoring first two levels we ended up with nothing but we do have
    # some hierarchical info, include the deepest available from the original fields
    if not raw_candidates:
        for field in ("category_paths", "category_chains", "category_path", "categories"):
            val = row.get(field)
            if isinstance(val, list) and val:
                if isinstance(val[0], list):
                    for chain in val:
                        flattened = [str(x).strip() for x in chain if x is not None and str(x).strip()]
                        if flattened:
                            raw_candidates.append(flattened[-1])
                else:
                    flattened = [str(x).strip() for x in val if x is not None and str(x).strip()]
                    if flattened:
                        raw_candidates.append(flattened[-1])
                if raw_candidates:
                    break

    # 2) merge similar names using canonical key
    key_to_originals: Dict[str, Set[str]] = {}
    for name in raw_candidates:
        key = canonical_category_key(name)
        if not key:
            continue
        key_to_originals.setdefault(key, set()).add(name)

    canonical_names: Set[str] = set()
    for originals in key_to_originals.values():
        canonical_names.add(select_canonical_name(originals))

    return canonical_names


def upsert_product(conn: sqlite3.Connection, raw: Dict) -> int:
    """Insert or update product, return product ID"""
    cur = conn.cursor()

    title = raw.get("title") or "Untitled"
    main_category = raw.get("main_category")
    store = raw.get("store")

    # Parse price
    price_val: Optional[float] = None
    price_raw = raw.get("price")
    if isinstance(price_raw, (int, float)):
        price_val = float(price_raw)
    elif isinstance(price_raw, str):
        try:
            price_val = float(price_raw.replace(",", ""))
        except ValueError:
            price_val = None

    external_id = raw.get("parent_asin") or raw.get("asin") or None

    # Try update by external id if present
    if external_id:
        cur.execute("SELECT id FROM products WHERE external_id = ?", (external_id,))
        row = cur.fetchone()
        if row is not None:
            prod_id = int(row[0])
            cur.execute(
                "UPDATE products SET title=?, main_category=?, store=?, price=?, raw=? WHERE id=?",
                (title, main_category, store, price_val, json.dumps(raw), prod_id),
            )
            return prod_id

    # Insert new product
    cur.execute(
        "INSERT INTO products(external_id, title, main_category, store, price, raw) VALUES (?, ?, ?, ?, ?, ?)",
        (external_id, title, main_category, store, price_val, json.dumps(raw)),
    )
    return cur.lastrowid


def link_product_to_categories(conn: sqlite3.Connection, product_id: int, category_names: Set[str]) -> int:
    """Link product to all categories, return number of links created"""
    cur = conn.cursor()
    links_created = 0

    for category_name in category_names:
        category_id = get_or_create_category(conn, category_name)
        cur.execute(
            "INSERT OR IGNORE INTO product_category(product_id, category_id) VALUES (?, ?)",
            (product_id, category_id),
        )
        if cur.rowcount > 0:
            links_created += 1

    return links_created


def load_jsonl(
    db_path: str, jsonl_path: str, limit: Optional[int] = None
) -> Dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)

        inserted_products = 0
        total_links = 0
        processed_lines = 0
        total_categories = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract all categories from this product
                category_names = extract_categories(row)

                # Insert/update product
                product_id = upsert_product(conn, row)

                # Link to all categories
                links_created = link_product_to_categories(conn, product_id, category_names)

                inserted_products += 1
                total_links += links_created
                processed_lines += 1

                if limit is not None and processed_lines >= limit:
                    break

        # Get final category count
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM categories")
        total_categories = cur.fetchone()[0]

        conn.commit()

        return {
            "products": inserted_products,
            "categories": total_categories,
            "links": total_links,
            "lines": processed_lines
        }

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load products and flat categories into SQLite (merged similar names, ignore top 2 levels)")
    parser.add_argument("--db", required=True, help="Path to SQLite DB file to create/update")
    parser.add_argument("--jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of lines to process")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    result = load_jsonl(args.db, args.jsonl, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
