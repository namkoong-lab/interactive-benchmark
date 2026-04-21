from openai import OpenAI
import os
import json
import re
import random
import sqlite3
import time
import hashlib
import statistics
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from .llm_providers import chat_completion

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "database", "products.db"))


def _repair_json_for_parse(s: str) -> str:
    """Remove trailing commas before ] or } (invalid JSON, but common in LLM output, especially Gemini)."""
    if not s:
        return s
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",(\s*[\]}])", r"\1", s)
    return s


def _extract_json_block(text: str) -> str:
    """Best-effort slice of first balanced JSON array or object."""
    if text.strip().startswith("```"):
        try:
            return text.split("\n", 1)[1].rsplit("\n", 1)[0]
        except Exception:
            pass
    s = text
    if "[" in s and (s.find("[") < s.find("{") or "{" not in s):
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = s[start : end + 1]
            stack = 0
            last_balanced_idx = -1
            for i, ch in enumerate(candidate):
                if ch == "[":
                    stack += 1
                elif ch == "]":
                    stack -= 1
                    if stack == 0:
                        last_balanced_idx = i
            if last_balanced_idx != -1:
                return candidate[: last_balanced_idx + 1]
            return candidate
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        stack = 0
        last_balanced_idx = -1
        for i, ch in enumerate(candidate):
            if ch == "{":
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    last_balanced_idx = i
        if last_balanced_idx != -1:
            return candidate[: last_balanced_idx + 1]
        return candidate
    return text


def _salvage_score_items_from_text(text: str) -> List[Dict[str, Any]]:
    """Recover {id, score} pairs from truncated or noisy model output."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for pat in (
        r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*([-+]?\d*\.?\d+)',
        r'\{\s*"score"\s*:\s*([-+]?\d*\.?\d+)\s*,\s*"id"\s*:\s*(\d+)',
    ):
        for m in re.finditer(pat, text):
            if m.lastindex == 2:
                a, b = m.group(1), m.group(2)
                if pat.startswith(r'\{\s*"id"'):
                    pid, sc = int(a), float(b)
                else:
                    sc, pid = float(a), int(b)
                if pid not in seen:
                    seen.add(pid)
                    out.append({"id": pid, "score": sc})
    return out


def _items_from_scoring_response(text: str) -> List[Dict[str, Any]]:
    """Parse model output into a list of {id, score} dicts."""
    if not (text or "").strip():
        return []
    t = _repair_json_for_parse(text.strip())
    try:
        parsed: Any = json.loads(t)
    except Exception:
        try:
            t2 = _repair_json_for_parse(_extract_json_block(text.strip()))
            parsed = json.loads(t2)
        except Exception:
            return _salvage_score_items_from_text(text)

    if isinstance(parsed, dict):
        if "results" in parsed:
            inner = parsed["results"]
        elif "result" in parsed:
            r = parsed["result"]
            inner = r if isinstance(r, list) else [r]
        else:
            inner = [parsed]
    elif isinstance(parsed, list):
        inner = parsed
    else:
        inner = [parsed]
    if not isinstance(inner, list):
        return []
    return inner


def _chunk_score_ids_match(chunk: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> bool:
    want = {int(p["id"]) for p in chunk if p.get("id") is not None}
    got = set()
    for it in items:
        try:
            got.add(int(it.get("id")))
        except (TypeError, ValueError):
            return False
    return want == got


_debug_mode: bool = False  # Global debug flag
_database_ensured: bool = False  # Track if we've checked for database

def set_debug_mode(debug: bool):
    """Set global debug mode for simulate_interaction."""
    global _debug_mode
    _debug_mode = debug

def _ensure_database_exists():
    """
    Ensure database exists. Downloads and builds from HuggingFace if needed.
    This is called automatically on first database access.
    """
    global _database_ensured
    
    # Only check once per process
    if _database_ensured:
        return
    
    if os.path.exists(DB_PATH):
        _database_ensured = True
        return
    
    # Database doesn't exist - try to download and rebuild
    print(f"\n{'='*70}")
    print(f"  🔄 Product Database Setup")
    print(f"{'='*70}")
    print(f"\nDatabase not found at: {DB_PATH}")
    print(f"Downloading from HuggingFace and building local database...\n")
    
    try:
        # Import here to avoid circular dependency
        import sys
        database_dir = os.path.dirname(DB_PATH)
        if database_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(database_dir))
        
        from database.rebuild_from_parquet import ensure_database
        ensure_database(force_rebuild=False)
        
        _database_ensured = True
        print(f"\n✅ Database setup complete!\n")
        
    except Exception as e:
        print(f"\n❌ Failed to setup database: {e}")
        print(f"\nManual setup:")
        print(f"  cd database")
        print(f"  python setup_database.py")
        raise RuntimeError(
            f"Database not found and auto-download failed. "
            f"Please run 'python database/setup_database.py' manually."
        )

def _get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    _ensure_database_exists()
    return sqlite3.connect(db_path)

def list_categories(db_path: str = DB_PATH) -> List[str]:
    conn = _get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM categories ORDER BY name ASC")
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

def get_products_by_category(
    category_name: str, 
    db_path: str = DB_PATH,
    limit: Optional[int] = None,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get products in a category, optionally limited to a random sample.
    
    Args:
        category_name: Category to query
        db_path: Path to database
        limit: If specified, randomly sample this many products (or all if fewer)
        seed: Random seed for reproducible product sampling (uses independent RNG)
    
    Returns:
        List of product dictionaries
    """
    conn = _get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.id, p.external_id, p.title, p.main_category, p.store, p.price, p.raw
            FROM products p
            JOIN product_category pc ON pc.product_id = p.id
            JOIN categories c ON c.id = pc.category_id
            WHERE c.name = ?
            ORDER BY p.id ASC
            """,
            (category_name,),
        )
        rows = cur.fetchall()
        products: List[Dict[str, Any]] = []
        for r in rows:
            raw = {}
            try:
                raw = json.loads(r[6]) if r[6] else {}
            except Exception:
                raw = {}
            products.append(
                {
                    "id": r[0],
                    "external_id": r[1],
                    "title": r[2],
                    "main_category": r[3],
                    "store": r[4],
                    "price": r[5],
                    "raw": raw,
                }
            )
        
        # Apply random sampling if limit is specified
        # Uses independent random state to avoid affecting experiment-level randomness
        if limit is not None and len(products) > limit:
            if seed is not None:
                # Create independent Random instance with deterministic seed
                # Derive unique seed from both the provided seed and category name
                # Use MD5 hash for cross-platform reproducibility (Python's hash() is not deterministic)
                category_hash = int(hashlib.md5(category_name.encode('utf-8')).hexdigest(), 16) % (2**31)
                derived_seed = (seed + category_hash) % (2**31)
                rng = random.Random(derived_seed)
            else:
                # Use independent Random instance without seed (truly random)
                rng = random.Random()
            
            total_before_sampling = len(products)
            products = rng.sample(products, limit)
            # Re-sort by id for consistency
            products.sort(key=lambda p: p["id"])
            print(f"[INFO] Sampled {limit} products from {total_before_sampling} total in category '{category_name}'")
        
        return products
    finally:
        conn.close()

def _ensure_scores_schema(conn: sqlite3.Connection) -> None:
    """Create cache table for persona/category/product scores if missing."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS persona_scores (
            persona_index INTEGER NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
            product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
            score REAL NOT NULL,
            reason TEXT,
            model TEXT,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
            PRIMARY KEY (persona_index, category_id, product_id)
        );
        """
    )
    conn.commit()

def _get_category_id(conn: sqlite3.Connection, category_name: str) -> Optional[int]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM categories WHERE name = ?", (category_name,))
    row = cur.fetchone()
    return int(row[0]) if row else None

def load_cached_scores(persona_index: int, category_name: str, product_ids: Optional[List[int]] = None, db_path: str = DB_PATH) -> List[Tuple[int, float]]:
    """Load cached scores for a persona/category. Optionally filter to product_ids.
    Returns list of (product_id, score), unsorted.
    """
    conn = _get_connection(db_path)
    try:
        _ensure_scores_schema(conn)
        category_id = _get_category_id(conn, category_name)
        if category_id is None:
            return []
        cur = conn.cursor()
        if product_ids:
            placeholders = ",".join(["?"] * len(product_ids))
            params = [persona_index, category_id, *product_ids]
            cur.execute(
                f"SELECT product_id, score FROM persona_scores WHERE persona_index = ? AND category_id = ? AND product_id IN ({placeholders})",
                params,
            )
        else:
            cur.execute(
                "SELECT product_id, score FROM persona_scores WHERE persona_index = ? AND category_id = ?",
                (persona_index, category_id),
            )
        return [(int(pid), float(score)) for pid, score in cur.fetchall()]
    finally:
        conn.close()

def save_scores(persona_index: int, category_name: str, scores: List[Tuple[int, float, str]], model: Optional[str] = None, db_path: str = DB_PATH) -> None:
    """Persist scores (product_id, score, reason) for a persona/category.
    Upserts by primary key.
    """
    if not scores:
        return
    conn = _get_connection(db_path)
    try:
        _ensure_scores_schema(conn)
        category_id = _get_category_id(conn, category_name)
        if category_id is None:
            return
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO persona_scores (persona_index, category_id, product_id, score, reason, model)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(persona_index, category_id, product_id) DO UPDATE SET
              score=excluded.score,
              reason=excluded.reason,
              model=excluded.model,
              created_at=strftime('%s','now')
            """,
            [
                (int(persona_index), int(category_id), int(pid), float(score), str(reason or ""), model)
                for (pid, score, reason) in scores
            ],
        )
        conn.commit()
    finally:
        conn.close()

def simulated_user_respond(persona_description: str, question: str, category: str, dialog_history: List[Tuple[str, str]] = None) -> str:
    """
    Simulate a user's answer given a natural-language persona description.
    The persona only knows their own characteristics and the product category,
    not the specific products or their scores.
    
    Args:
        persona_description: Natural language description of the persona
        question: The question being asked
        category: Product category being shopped
        dialog_history: List of (question, answer) tuples from previous conversation
    
    Returns:
        The persona's answer as a string
    """
    import time
    start_time = time.time()
    
    # Build conversation history
    history_str = ""
    if dialog_history:
        history_lines = []
        for q, a in dialog_history:
            history_lines.append(f"Q: {q}")
            history_lines.append(f"A: {a}")
        history_str = "\n".join(history_lines)
    else:
        history_str = "No previous conversation."
    
    # Simplified rules - no interest level check needed (categories are pre-filtered)
    rules = """**Rule #1: BE CONSISTENT.**
- Your answers must be 100% consistent with your persona description.
- Answer based on your general preferences, lifestyle, and needs described in your persona.

**Rule #2: BE HELPFUL & HONEST.**
- Answer questions truthfully as this persona would.
- If you don't have a strong preference, say so naturally.

**Rule #3: BE CONCISE.**
- Keep your answers brief (1-2 sentences) to encourage follow-up questions.
- Don't volunteer extra information beyond what's asked."""
    
    prompt = f"""You are role-playing as a customer shopping for {category} products.

{rules}
---
**Who You Are:**
{persona_description}
---
**Current Shopping Context:**
- You are shopping for {category} products for yourself
- Conversation history:
{history_str}
---
**Your Task:**
Answer the following question naturally and consistently with your persona. Base your answer on your general preferences, needs, and lifestyle.

**Question:** "{question}"

**Your Answer:**"""
    
    content = chat_completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You simulate a user based on a given persona description."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=100,
        json_mode=False,
    )
    
    elapsed = time.time() - start_time
    if _debug_mode:
        print(f"[TIMING] Persona question answering: {elapsed:.2f}s")
    return content


class DegenerateScoresError(RuntimeError):
    """Raised when ensemble scores still look unusable after one fresh rescoring attempt."""


def scores_look_degenerate(scores: List[float]) -> bool:
    """
    Heuristic: scoring failed or is unusable — too many near-zero scores, or almost no spread near zero.

    Used to trigger one full rescore (no cache) before skipping the category.
    """
    if not scores or len(scores) < 3:
        return False
    n = len(scores)
    vals = [float(s) for s in scores]
    zeroish = sum(1 for s in vals if s <= 0.51)
    if zeroish > max(3, int(0.12 * n)):
        return True
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    mx = max(vals)
    if mean <= 10.0 and std <= 5.0:
        return True
    if mx <= 15.0 and std <= 4.0:
        return True
    return False


def score_products_for_persona(persona_description: str, category: str, products: List[Dict[str, Any]], model: str = "gpt-4o") -> List[Tuple[int, float, str]]:
    """
    Score each product for the given persona using an ensemble of providers.
    Currently queries an OpenAI model and a Gemini model, then averages the scores
    for products present in both; if only one provider returns a product, use that score.
    Returns list of tuples: (product_id, score_0_100, reason_merged), sorted by score.
    """
    import time
    start_time = time.time()
    if _debug_mode:
        print(f"[TIMING] Starting product scoring for {len(products)} products...")
    condensed_products = [
        {
            "id": p.get("id"),
            "title": p.get("title"),
            "price": p.get("price"),
            "store": p.get("store"),
            "attributes": {
                k: v
                for k, v in (p.get("raw") or {}).items()
                if isinstance(v, (str, int, float)) and k not in {"description", "title"}
            },
        }
        for p in products
    ]

    def _score_once(target_model: str) -> Dict[int, Tuple[float, str]]:
        """
        Score all products with one provider: fixed-size chunks, one JSON shape for everyone.

        Always request {\"results\": [{\"id\", \"score\"}, ...]} so OpenAI json_object mode and Gemini
        schema stay aligned. No silent dropped chunks.
        """
        chunk_size = 20
        label = "Gemini" if target_model.startswith("gemini-") else "OpenAI"

        def _query_with_products(prod_subset: List[Dict[str, Any]]) -> str:
            n = len(prod_subset)
            instructions = (
                "You ARE the persona described. You are shopping for YOURSELF. "
                "Rate each listed product from 0 to 100 (integers) by how much you would like it for your own use. "
                f"Respond with ONLY a JSON object: key \"results\" maps to an array of exactly {n} objects, "
                "one object per product id in the request, each object {\"id\": number, \"score\": number}. "
                "Strict JSON: no trailing commas; no markdown; no extra keys or commentary."
            )
            user_payload = {
                "persona_description": persona_description,
                "category": category,
                "products": prod_subset,
                "instructions": instructions,
            }
            response_schema_local = {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "number"},
                                "score": {"type": "number"},
                            },
                            "required": ["id", "score"],
                        },
                    }
                },
                "required": ["results"],
            }
            max_out = min(8192, max(512, 100 * n + 500))
            if target_model.startswith("gemini-"):
                max_out = min(8192, max(max_out, int(max_out * 1.6) + 512))
            return chat_completion(
                model=target_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You evaluate product–persona fit for a shopping simulation. "
                            "Output strict JSON only. When products differ, scores should usually differ; "
                            "do not collapse everything to the same number unless truly indifferent."
                        ),
                    },
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                temperature=0.1,
                max_tokens=max_out,
                json_mode=True,
                response_schema=response_schema_local,
                count_usage=False,
            )

        aggregated_items: List[Dict[str, Any]] = []
        for start in range(0, len(condensed_products), chunk_size):
            chunk = condensed_products[start : start + chunk_size]
            want = {int(p["id"]) for p in chunk if p.get("id") is not None}
            if not want:
                continue
            if _debug_mode:
                print(f"[DEBUG] {label} scoring chunk products {start + 1}-{start + len(chunk)} ({len(chunk)} items)")
            last_err: Optional[Exception] = None
            chunk_ok = False
            for attempt in range(5):
                try:
                    raw = _query_with_products(chunk)
                    items = _items_from_scoring_response(raw)
                    if not _chunk_score_ids_match(chunk, items):
                        raise ValueError(
                            f"id mismatch: need exactly {len(want)} scores for chunk ids, "
                            f"parsed {len(items)} rows"
                        )
                    aggregated_items.extend(items)
                    if _debug_mode:
                        print(f"[DEBUG] {label} chunk ok, preview: {raw[:180]}...")
                    chunk_ok = True
                    break
                except Exception as e:
                    last_err = e
                    if attempt < 4:
                        delay = min(1.0 * (2**attempt), 30.0)
                        jitter = random.uniform(0, delay * 0.1)
                        print(
                            f"[WARN] {label} scoring chunk @ offset {start} attempt {attempt + 1}/5: {e}. "
                            f"Retry in {delay + jitter:.2f}s..."
                        )
                        time.sleep(delay + jitter)
            if not chunk_ok:
                raise RuntimeError(
                    f"{label} scoring failed for chunk starting at index {start} ({len(chunk)} products) "
                    f"after 5 attempts: {last_err}"
                ) from last_err

        raw_scores: List[float] = []
        for item in aggregated_items:
            try:
                raw_scores.append(float(item.get("score")))
            except Exception:
                continue
        scale_factor = 1.0
        if raw_scores:
            max_score = max(raw_scores)
            if max_score <= 1.0:
                scale_factor = 100.0
            elif max_score <= 10.0:
                scale_factor = 10.0

        out: Dict[int, Tuple[float, str]] = {}
        invalid_ids = []
        invalid_scores = []
        
        for item in aggregated_items:
            try:
                pid = int(item.get("id"))
            except (ValueError, TypeError):
                # Invalid product ID
                invalid_ids.append(str(item))
                continue
            
            # Handle score - STRICT: no defaults allowed
            score_raw = item.get("score")
            try:
                if score_raw is None or score_raw == "N/A" or score_raw == "":
                    raise ValueError(f"Score is None/N/A/empty for product {pid}")
                score = float(score_raw) * scale_factor
                if score < 0:
                    score = 0.0
                if score > 100:
                    score = 100.0
            except (ValueError, TypeError) as e:
                # CRITICAL: Invalid scores trigger retry
                invalid_scores.append((pid, score_raw))
                continue  # Skip this product, check others, then fail at end
            
            reason = str(item.get("reason") or "")
            out[pid] = (score, reason)
        
        # Raise error if any invalid scores detected (triggers retry)
        if invalid_scores:
            error_details = ", ".join([f"product {pid}: '{score}'" for pid, score in invalid_scores[:5]])
            raise ValueError(
                f"Invalid scores from LLM for {len(invalid_scores)} products ({error_details}...). "
                f"LLM must return valid numeric scores. Retrying..."
            )
        
        if invalid_ids:
            print(f"[WARN] {len(invalid_ids)} items had invalid product IDs and were skipped")
        
        return out

    openai_model = "gpt-4o"
    gemini_model = "gemini-3.1-flash-lite-preview"

    scores_openai: Dict[int, Tuple[float, str]] = {}
    scores_gemini: Dict[int, Tuple[float, str]] = {}
    n_products = len([p for p in products if p.get("id") is not None])
    try:
        print(f"[INFO] Starting OpenAI scoring with model: {openai_model}")
        scores_openai = _score_once(openai_model)
        if n_products and len(scores_openai) == 0:
            print(f"[WARN] OpenAI scoring returned 0 scores (expected ~{n_products}); check JSON/parsing.")
        else:
            print(f"[INFO] OpenAI scoring completed successfully, got {len(scores_openai)} scores")
    except Exception as e:
        print(f"[ERROR] OpenAI scoring failed: {type(e).__name__}: {e}")
        import traceback
        print(f"[ERROR] OpenAI traceback: {traceback.format_exc()}")
    try:
        print(f"[INFO] Starting Gemini scoring with model: {gemini_model}")
        scores_gemini = _score_once(gemini_model)
        if n_products and len(scores_gemini) == 0:
            print(f"[WARN] Gemini scoring returned 0 scores (expected ~{n_products}); often truncation or malformed JSON.")
        else:
            print(f"[INFO] Gemini scoring completed successfully, got {len(scores_gemini)} scores")
    except Exception as e:
        print(f"[ERROR] Gemini scoring failed: {type(e).__name__}: {e}")
        import traceback
        print(f"[ERROR] Gemini traceback: {traceback.format_exc()}")

    combined: List[Tuple[int, float, str]] = []
    all_ids = {int(p.get("id")) for p in products if p.get("id") is not None}
    products_missing_both_scores = []
    
    for pid in all_ids:
        have_o = pid in scores_openai
        have_g = pid in scores_gemini
        
        if not have_o and not have_g:
            products_missing_both_scores.append(pid)
        elif have_o and have_g:
            s_o, r_o = scores_openai[pid]
            s_g, r_g = scores_gemini[pid]
            avg = (s_o + s_g) / 2.0
            reason = f"OpenAI score: {s_o:.1f} | Gemini score: {s_g:.1f}"
            combined.append((pid, avg, reason))
        elif have_o:
            s_o, r_o = scores_openai[pid]
            avg = s_o
            reason = f"OpenAI score: {s_o:.1f} | Gemini score: failed (using OpenAI only)"
            combined.append((pid, avg, reason))
        else:
            s_g, r_g = scores_gemini[pid]
            avg = s_g
            reason = f"OpenAI score: failed | Gemini score: {s_g:.1f} (using Gemini only)"
            combined.append((pid, avg, reason))
    
    if products_missing_both_scores:
        raise ValueError(
            f"CRITICAL: {len(products_missing_both_scores)} products not scored by either provider. "
            f"Product IDs: {products_missing_both_scores[:10]}{'...' if len(products_missing_both_scores) > 10 else ''}. "
            f"This violates experiment consistency. Retrying..."
        )

    combined.sort(key=lambda t: t[1], reverse=True)

    if _debug_mode:
        id_to_title = {int(p.get("id")): p.get("title") for p in products if p.get("id") is not None}
        print("\n=== Persona Scoring (ensemble, highest to lowest) ===")
        for pid, score, reason in combined:
            title = id_to_title.get(pid, "<unknown title>")
            print(f"{score:6.1f} - {title} (id={pid})")
            print(f"        {reason}")
        print("=== End Persona Scoring ===\n")
    
    elapsed = time.time() - start_time
    if _debug_mode:
        print(f"[TIMING] Product scoring completed: {elapsed:.2f}s")
    return combined
