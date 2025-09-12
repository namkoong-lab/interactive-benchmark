from openai import OpenAI
import os
import json
import random
import sqlite3
import time
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
try:
    from llm_client import chat_completion
except Exception:
    from pipeline.llm_client import chat_completion

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "products.db"))

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

def get_products_by_category(category_name: str, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
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
        return products
    finally:
        conn.close()

def simulated_user_respond(persona_description: str, question: str) -> str:
    """
    Simulate a user's answer given a natural-language persona description.
    """
    prompt = f"""You simulate a user with the following persona description:
{persona_description}

Answer STRICTLY the question as this user would.
- Only answer the question asked
- Do not volunteer extra information
- Do not restate persona or add rationale
- If a choice is requested, give one choice only

Question: {question}

Answer:"""
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
    return content

def score_products_for_persona(persona_description: str, category: str, products: List[Dict[str, Any]], model: str = "gpt-4o") -> List[Tuple[int, float, str]]:
    """
    Score each product for the given persona using an ensemble of providers.
    Currently queries an OpenAI model and a Gemini model, then averages the scores
    for products present in both; if only one provider returns a product, use that score.
    Returns list of tuples: (product_id, score_0_100, reason_merged), sorted by score.
    """
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
        # Helper to query a subset of products (used for Gemini batching)
        def _query_with_products(prod_subset: List[Dict[str, Any]], array_only: bool = False) -> str:
            if array_only:
                user_payload = {
                    "persona_description": persona_description,
                    "category": category,
                    "products": prod_subset,
                    "instructions": "Return ONLY a JSON array of objects: {id, score}. Score must be an integer from 0 to 100. Do not include any surrounding text.",
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
                    "instructions": "You ARE the persona described. Rate each product with a score from 0 to 100 (integers only) based on how much YOU would like it. Return a JSON object with key 'results' as an array of objects: {id, score}. Do not include any other keys or text.",
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
                    {"role": "system", "content": "You evaluate product-persona fit and must return strict JSON only."},
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
                # Retry for transient errors
                content_local = None
                for attempt in range(3):
                    try:
                        content_local = _query_with_products(chunk, array_only=True)
                        break
                    except Exception:
                        if attempt < 2:
                            time.sleep(1.5 * (attempt + 1))
                        else:
                            raise
                raw_outputs.append(content_local)
        else:
            # Chunk OpenAI if very large to reduce payload / errors
            if not target_model.startswith("gemini-") and len(condensed_products) >= 100:
                chunk_size = 50
                for i in range(0, len(condensed_products), chunk_size):
                    chunk = condensed_products[i:i+chunk_size]
                    content_part = None
                    for attempt in range(3):
                        try:
                            content_part = _query_with_products(chunk, array_only=True)
                            break
                        except Exception:
                            if attempt < 2:
                                time.sleep(1.5 * (attempt + 1))
                            else:
                                raise
                    raw_outputs.append(content_part)
            else:
                content_local = None
                for attempt in range(3):
                    try:
                        content_local = _query_with_products(condensed_products, array_only=False)
                        break
                    except Exception:
                        if attempt < 2:
                            time.sleep(1.5 * (attempt + 1))
                        else:
                            raise
                raw_outputs.append(content_local)
            
        def _extract_json_block(text: str) -> str:
            # If fenced, strip fences
            if text.strip().startswith("```"):
                try:
                    return text.split("\n", 1)[1].rsplit("\n", 1)[0]
                except Exception:
                    pass
            s = text
            # Detect array JSON
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
            # Otherwise try object JSON
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

        # Parse and aggregate from one or multiple raw outputs
        aggregated_items: List[Dict[str, Any]] = []
        for content_local in raw_outputs:
            try:
                parsed_local = json.loads(content_local)
                # Support both array and object-with-results
                data_local = parsed_local.get("results", parsed_local) if isinstance(parsed_local, dict) else parsed_local
            except Exception:
                extracted = _extract_json_block(content_local)
                parsed_local = json.loads(extracted)
                data_local = parsed_local.get("results", parsed_local) if isinstance(parsed_local, dict) else parsed_local
            if isinstance(data_local, list):
                aggregated_items.extend(data_local)
        
        # Normalize scores to 0-100 if model under-ranges (e.g., 0-1 or 0-10)
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

    # Run scoring on OpenAI and Gemini
    openai_model = "gpt-4o"
    gemini_model = "gemini-1.5-pro"

    scores_openai: Dict[int, Tuple[float, str]] = {}
    scores_gemini: Dict[int, Tuple[float, str]] = {}
    try:
        scores_openai = _score_once(openai_model)
    except Exception as e:
        print("[WARN] OpenAI scoring failed:", e)
    try:
        scores_gemini = _score_once(gemini_model)
    except Exception as e:
        print("[WARN] Gemini scoring failed:", e)

    # Combine
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

    # Log product titles with scores (top to bottom)
    id_to_title = {int(p.get("id")): p.get("title") for p in products if p.get("id") is not None}
    print("\n=== Persona Scoring (ensemble, highest to lowest) ===")
    for pid, score, reason in combined:
        title = id_to_title.get(pid, "<unknown title>")
        print(f"{score:6.1f} - {title} (id={pid})")
        print(f"        {reason}")
    print("=== End Persona Scoring ===\n")
    return combined

def ai_recommender_interact(category: str, products: List[Dict[str, Any]], persona_description: str, llm_b_model: str = "gpt-4o", max_questions: int = 8) -> Tuple[int, str]:
    """
    Recommender interacts with the simulated persona to select the best product id.
    Returns (recommended_product_id, rationale_text).
    The recommender cannot see the persona description; it can only ask questions and see answers.
    The agent decides how many questions to ask, up to max_questions, and may stop early when confident.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous product recommender tasked with finding the single best product for the user in this category. "
                "Your objective is to choose the product that will most likely be the user's first choice over all other products.\n\n"
                "Operating rules:\n"
                "- You only see product summaries and the user's answers (you do NOT see the persona).\n"
                "- Ask at most the allowed number of concise questions. Each question must target exactly one attribute that most reduces uncertainty between the current top candidates.\n"
                "- Do not recommend until you are reasonably confident your choice is better than every other product in the list given the answers so far.\n"
                "- If you cannot reach that level of confidence yet, keep asking targeted questions (until you hit the maximum).\n"
                "- When you are sufficiently confident, you may stop asking and proceed to the final recommendation step."
            ),
        },
        {
            "role": "assistant",
            "content": f"Category: {category}\nProducts:\n{json.dumps([{'id': p['id'], 'title': p['title'], 'price': p['price'], 'store': p['store']} for p in products], indent=2)}",
        },
    ]

    conversation: List[Dict[str, str]] = []
    for i in range(max(1, max_questions)):
        raw_question = chat_completion(
            model=llm_b_model,
            messages=messages + conversation + [
                {
                    "role": "assistant",
                    "content": (
                        "If you are now reasonably confident which product is better than all the others for this user, output exactly: STOP.\n"
                        "Otherwise, ask one short question about exactly one attribute that best separates the remaining top candidates.\n"
                        "Output only the question text or STOP."
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=80,
            json_mode=False,
        )
        question = (raw_question or "").strip()
        if question.upper() == "STOP":
            break
        if not question:
            continue
        print(f"AI Recommender: {question}")
        answer = simulated_user_respond(persona_description, question)
        print(f"User: {answer}")
        conversation.append({"role": "assistant", "content": question})
        conversation.append({"role": "user", "content": answer})

    content = chat_completion(
        model=llm_b_model,
        messages=messages + conversation + [
            {
                "role": "assistant",
                "content": (
                    "Based on the answers so far, recommend exactly one product id from the list and give a brief rationale.\n"
                    "You must be reasonably confident that your chosen product is better than all other products in the list for this user.\n"
                    "Return a JSON object with keys 'id' and 'rationale'."
                ),
            }
        ],
        temperature=0.3,
        max_tokens=200,
        json_mode=True,
    )
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print("\n[DEBUG] Failed to parse JSON for final recommendation. Raw content follows:\n", content, "\n")
        m = None
        if content.startswith("```"):
            m = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        data = json.loads(m or content)
    # Validate id & rationale
    rec_id_val = data.get("id")
    try:
        rec_id = int(rec_id_val)
    except Exception:
        # Fallback: try to extract the first integer-looking token
        rec_id = None
        try:
            import re
            m = re.search(r"\d+", str(rec_id_val))
            if m:
                rec_id = int(m.group(0))
        except Exception:
            pass
        if rec_id is None:
            # Last-resort fallback to first product id in list
            try:
                rec_id = int(products[0]["id"]) if products else -1
            except Exception:
                rec_id = -1
    rationale = str(data.get("rationale") or "")
    
    return rec_id, rationale