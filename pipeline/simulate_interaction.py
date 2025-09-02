from openai import OpenAI
import os
import json
import random
import sqlite3
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

Answer the question as this user would. Be concise and natural. Only answer the question asked. Do not list multiple attributes. Do not add extra context.

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
                    "instructions": "Return ONLY a JSON array of objects: {id, score}. Do not include any surrounding text.",
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
                    "instructions": "You ARE the persona described. Rate each product from 0-100 based on how much YOU would like it. Return a JSON object with key 'results' as an array of objects: {id, score}. Do not include any other keys or text.",
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

        # For OpenAI, one shot is fine; for Gemini, batch to avoid long/truncated JSON
        raw_outputs: List[str] = []
        if target_model.startswith("gemini-") and len(condensed_products) > 25:
            chunk_size = 20
            for i in range(0, len(condensed_products), chunk_size):
                chunk = condensed_products[i:i+chunk_size]
                content_local = _query_with_products(chunk, array_only=True)
                raw_outputs.append(content_local)
        else:
            content_local = _query_with_products(condensed_products, array_only=False)
            raw_outputs.append(content_local)
            
        def _extract_json_block(text: str) -> str:
            # If fenced, strip fences
            if text.strip().startswith("```"):
                try:
                    return text.split("\n", 1)[1].rsplit("\n", 1)[0]
                except Exception:
                    pass
            # Heuristic: take substring from first '{' to last '}' using brace counting
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end+1]
                # Attempt to trim to a balanced block
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
        
        out: Dict[int, Tuple[float, str]] = {}
        for item in data_local:
            try:
                pid = int(item.get("id"))
                score = float(item.get("score"))
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
            "content": "You are a careful product recommender. You see a category and a list of product summaries. You may ask at most the provided number of concise questions about exactly one attribute each to infer the user's preferences, then recommend one product with rationale. You do NOT know the persona; you only see the user's answers.\nAsk few questions; prioritize attributes that maximally reduce uncertainty.",
        },
        {
            "role": "assistant",
            "content": f"Category: {category}\nProducts:\n{json.dumps([{'id': p['id'], 'title': p['title'], 'price': p['price'], 'store': p['store']} for p in products], indent=2)}",
        },
    ]

    conversation: List[Dict[str, str]] = []
    min_questions = max(1, min(2, max_questions))
    for i in range(max(1, max_questions)):
        raw_question = chat_completion(
            model=llm_b_model,
            messages=messages + conversation + [
                {"role": "assistant", "content": "If you can already make a recommendation that you are reasonable confident in being the first choice of the persona, output exactly the token STOP. Otherwise, ask one short question about exactly one attribute that helps choose among the products. Output only the question text or STOP."}
            ],
            temperature=0.3,
            max_tokens=80,
            json_mode=False,
        )
        question = (raw_question or "").strip()
        lower = question.lower()
        if i + 1 < min_questions and (not question or question.upper() == "STOP" or lower.startswith("i recommend") or "recommend" in lower[:80]):
            # Force at least min_questions by retrying once with a clarifier prompt
            raw_question = chat_completion(
                model=llm_b_model,
                messages=messages + conversation + [
                    {"role": "assistant", "content": "Ask one short, specific question about exactly one attribute. Do not recommend yet. Output only the question text."}
                ],
                temperature=0.3,
                max_tokens=80,
                json_mode=False,
            )
            question = (raw_question or "").strip()
            lower = question.lower()

        if not question or question.upper() == "STOP" or lower.startswith("i recommend") or "recommend" in lower[:80]:
            break
        print(f"AI Recommender: {question}")
        answer = simulated_user_respond(persona_description, question)
        print(f"User: {answer}")
        conversation.append({"role": "assistant", "content": question})
        conversation.append({"role": "user", "content": answer})

    content = chat_completion(
        model=llm_b_model,
        messages=messages + conversation + [
            {"role": "assistant", "content": "Based on the answers so far, recommend exactly one product id from the list and give a brief rationale. Return a JSON object with keys 'id' and 'rationale'."}
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
    rec_id = int(data.get("id"))
    rationale = str(data.get("rationale") or "")
    
    return rec_id, rationale