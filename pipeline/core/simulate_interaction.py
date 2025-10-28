from openai import OpenAI
import os
import json
import random
import sqlite3
import time
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv
from .llm_client import chat_completion

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "database", "products.db"))

_debug_mode: bool = False  # Global debug flag

def set_debug_mode(debug: bool):
    """Set global debug mode for simulate_interaction."""
    global _debug_mode
    _debug_mode = debug

def _get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
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
        def _query_with_products(prod_subset: List[Dict[str, Any]], array_only: bool = False) -> str:
            if array_only:
                user_payload = {
                    "persona_description": persona_description,
                    "category": category,
                    "products": prod_subset,
                    "instructions": "You ARE the persona described. You are shopping for YOURSELF (not for a friend or anyone else). Rate each product with a score from 0 to 100 (integers only) based on how much YOU would like it for YOUR OWN use. Return a JSON array containing exactly one object per product with 'id' and 'score' fields. Example: [{\"id\": 123, \"score\": 85}, {\"id\": 456, \"score\": 70}]. Do not return a single object, return an array.",
                }
                response_schema_local = {
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
            else:
                user_payload = {
                    "persona_description": persona_description,
                    "category": category,
                    "products": prod_subset,
                    "instructions": "You ARE the persona described. You are shopping for YOURSELF (not for a friend or anyone else). Rate each product with a score from 0 to 100 (integers only) based on how much YOU would like it for YOUR OWN use. Return a JSON object with key 'results' as an array of objects: {id, score}. Do not include any other keys or text.",
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
            return chat_completion(
                model=target_model,
                messages=[
                    {"role": "system", "content": "You evaluate product-persona fit for personal use and must return strict JSON only."},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=600 if array_only else 800,
                json_mode=True,
                response_schema=response_schema_local,
            )

        raw_outputs: List[str] = []
        if target_model.startswith("gemini-") and len(condensed_products) >= 15:
            chunk_size = 15
            for i in range(0, len(condensed_products), chunk_size):
                chunk = condensed_products[i:i+chunk_size]
                content_local = None
                for attempt in range(5):
                    try:
                        content_local = _query_with_products(chunk, array_only=True)
                        break
                    except Exception as e:
                        if attempt < 4:
                            delay = min(1.0 * (2 ** attempt), 30.0)  
                            jitter = random.uniform(0, delay * 0.1)  
                            print(f"Gemini chunk attempt {attempt + 1} failed: {e}. Retrying in {delay + jitter:.2f}s...")
                            time.sleep(delay + jitter)
                        else:
                            print(f"Gemini chunk failed after 5 attempts: {e}")
                            raise
                raw_outputs.append(content_local)
        else:   
            if not target_model.startswith("gemini-") and len(condensed_products) >= 30:
                chunk_size = 25
                for i in range(0, len(condensed_products), chunk_size):
                    chunk = condensed_products[i:i+chunk_size]
                    if _debug_mode:
                        print(f"[DEBUG] Processing chunk {i//chunk_size + 1} with {len(chunk)} products")
                    content_part = None
                    for attempt in range(5):
                        try:
                            content_part = _query_with_products(chunk, array_only=True)
                            break
                        except Exception as e:
                            if attempt < 4:
                                delay = min(1.0 * (2 ** attempt), 30.0)  
                                jitter = random.uniform(0, delay * 0.1) 
                                print(f"OpenAI chunk attempt {attempt + 1} failed: {e}. Retrying in {delay + jitter:.2f}s...")
                                time.sleep(delay + jitter)
                            else:
                                print(f"OpenAI chunk {i//chunk_size + 1} failed after 5 attempts: {e}. Skipping this chunk.")
                                content_part = None  
                                break
                    raw_outputs.append(content_part)
            else:
                content_local = None
                for attempt in range(5):
                    try:
                        content_local = _query_with_products(condensed_products, array_only=False)
                        break
                    except Exception as e:
                        if attempt < 4:
                            delay = min(1.0 * (2 ** attempt), 30.0) 
                            jitter = random.uniform(0, delay * 0.1)  
                            print(f"Model attempt {attempt + 1} failed: {e}. Retrying in {delay + jitter:.2f}s...")
                            time.sleep(delay + jitter)
                        else:
                            print(f"Model failed after 5 attempts: {e}")
                            raise
                raw_outputs.append(content_local)
            
        def _extract_json_block(text: str) -> str:
            if text.strip().startswith("```"):
                try:
                    return text.split("\n", 1)[1].rsplit("\n", 1)[0]
                except Exception:
                    pass
            s = text
            if '[' in s and (s.find('[') < s.find('{') or '{' not in s):
                start = s.find('[')
                end = s.rfind(']')
                if start != -1 and end != -1 and end > start:
                    candidate = s[start:end+1]
                    stack = 0
                    last_balanced_idx = -1
                    for i, ch in enumerate(candidate):
                        if ch == '[':
                            stack += 1
                        elif ch == ']':
                            stack -= 1
                            if stack == 0:
                                last_balanced_idx = i
                    if last_balanced_idx != -1:
                        return candidate[: last_balanced_idx + 1]
                    return candidate
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = s[start:end+1]
                stack = 0
                last_balanced_idx = -1
                for i, ch in enumerate(candidate):
                    if ch == '{':
                        stack += 1
                    elif ch == '}':
                        stack -= 1
                        if stack == 0:
                            last_balanced_idx = i
                if last_balanced_idx != -1:
                    return candidate[: last_balanced_idx + 1]
                return candidate
            return text

        aggregated_items: List[Dict[str, Any]] = []
        for i, content_local in enumerate(raw_outputs):
            if content_local is None or content_local.strip() == "":
                print(f"[WARN] Chunk {i} returned empty content, skipping")
                continue
            
            if i == 0:  
                if _debug_mode:
                    print(f"[DEBUG] Chunk {i} content preview: {content_local[:200]}...")
            try:
                parsed_local = json.loads(content_local)
                if isinstance(parsed_local, dict):
                    if "results" in parsed_local:
                        data_local = parsed_local["results"]
                    elif "result" in parsed_local:
                        data_local = parsed_local["result"]
                    else:
                        data_local = [parsed_local]
                elif isinstance(parsed_local, list):
                    data_local = parsed_local
                else:
                    data_local = [parsed_local]
            except Exception as e:
                print(f"[WARN] Failed to parse chunk {i} as JSON: {e}")
                try:
                    extracted = _extract_json_block(content_local)
                    parsed_local = json.loads(extracted)
                    if isinstance(parsed_local, dict):
                        if "results" in parsed_local:
                            data_local = parsed_local["results"]
                        elif "result" in parsed_local:
                            data_local = parsed_local["result"]
                        else:
                            data_local = [parsed_local]
                    elif isinstance(parsed_local, list):
                        data_local = parsed_local
                    else:
                        data_local = [parsed_local]
                except Exception as e2:
                    print(f"[WARN] Failed to extract JSON from chunk {i}: {e2}")
                    continue
            
            if isinstance(data_local, list):
                aggregated_items.extend(data_local)
            else:
                print(f"[WARN] Chunk {i} still not a list after processing, got: {type(data_local)}. Content: {str(data_local)[:100]}...")

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
        for item in aggregated_items:
            try:
                pid = int(item.get("id"))
                score = float(item.get("score")) * scale_factor
                if score < 0:
                    score = 0.0
                if score > 100:
                    score = 100.0
                reason = str(item.get("reason") or "")
                out[pid] = (score, reason)
            except Exception:
                continue
        return out

    openai_model = "gpt-4o"
    gemini_model = "gemini-2.0-flash-exp"

    scores_openai: Dict[int, Tuple[float, str]] = {}
    scores_gemini: Dict[int, Tuple[float, str]] = {}
    try:
        print(f"[INFO] Starting OpenAI scoring with model: {openai_model}")
        scores_openai = _score_once(openai_model)
        print(f"[INFO] OpenAI scoring completed successfully, got {len(scores_openai)} scores")
    except Exception as e:
        print(f"[ERROR] OpenAI scoring failed: {type(e).__name__}: {e}")
        import traceback
        print(f"[ERROR] OpenAI traceback: {traceback.format_exc()}")
    try:
        print(f"[INFO] Starting Gemini scoring with model: {gemini_model}")
        scores_gemini = _score_once(gemini_model)
        print(f"[INFO] Gemini scoring completed successfully, got {len(scores_gemini)} scores")
    except Exception as e:
        print(f"[ERROR] Gemini scoring failed: {type(e).__name__}: {e}")
        import traceback
        print(f"[ERROR] Gemini traceback: {traceback.format_exc()}")

    combined: List[Tuple[int, float, str]] = []
    all_ids = {int(p.get("id")) for p in products if p.get("id") is not None}
    for pid in all_ids:
        have_o = pid in scores_openai
        have_g = pid in scores_gemini
        if not have_o and not have_g:
            continue
        if have_o and have_g:
            s_o, r_o = scores_openai[pid]
            s_g, r_g = scores_gemini[pid]
            avg = (s_o + s_g) / 2.0
            reason = f"OpenAI score: {s_o:.1f} | Gemini score: {s_g:.1f}"
        elif have_o:
            s_o, r_o = scores_openai[pid]
            avg = s_o
            reason = f"OpenAI score: {s_o:.1f} | Gemini score: N/A"
        else:
            s_g, r_g = scores_gemini[pid]
            avg = s_g
            reason = f"OpenAI score: N/A | Gemini score: {s_g:.1f}"
        combined.append((pid, avg, reason))

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
