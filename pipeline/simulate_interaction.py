from openai import OpenAI
import os
import json
import random
import sqlite3
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "simple_products.db"))

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
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You simulate a user based on a given persona description."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()

def score_products_for_persona(persona_description: str, category: str, products: List[Dict[str, Any]], model: str = "gpt-4o") -> List[Tuple[int, float, str]]:
    """
    Ask the LLM to score each product for the given persona.
    Returns list of tuples: (product_id, score_0_100, short_reason), sorted descending by score.
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

    prompt = {
        "persona_description": persona_description,
        "category": category,
        "products": condensed_products,
        "instructions": "For each product, assign a score from 0 to 100 indicating how well it fits the persona. Provide a very brief reason. Return JSON as an array of objects: {id, score, reason}."
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You evaluate product-persona fit and must return strict JSON only."},
            {"role": "user", "content": json.dumps({
                "persona_description": persona_description,
                "category": category,
                "products": condensed_products,
                "instructions": "You ARE the persona described. Rate each product from 0-100 based on how much YOU would like it. Give a brief reason in YOUR voice as the persona, focusing on concrete traits and preferences from your persona description (like your background, values, lifestyle, hobbies, etc.). Return a JSON object with key 'results' as an array of objects: {id, score, reason}. Do not include any other keys or text."
            }, ensure_ascii=False)},
        ],
        temperature=0.2,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
        data = parsed.get("results", parsed)
    except json.JSONDecodeError:
        print("\n[DEBUG] Failed to parse JSON for persona scoring. Raw content follows:\n", content, "\n")
        # attempt to extract JSON block
        m = None
        if content.startswith("```"):
            m = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        parsed = json.loads(m or content)
        data = parsed.get("results", parsed)

    results: List[Tuple[int, float, str]] = []
    id_to_title = {int(p.get("id")): p.get("title") for p in products if p.get("id") is not None}
    for item in data:
        try:
            pid = int(item.get("id"))
            score = float(item.get("score"))
            reason = str(item.get("reason") or "")
            results.append((pid, score, reason))
        except Exception:
            continue
    results.sort(key=lambda t: t[1], reverse=True)
    # Log product titles with scores (top to bottom)
    print("\n=== Persona Scoring (highest to lowest) ===")
    for pid, score, reason in results:
        title = id_to_title.get(pid, "<unknown title>")
        print(f"{score:6.1f} - {title} (id={pid})")
        print(f"        Reason: {reason}")
    print("=== End Persona Scoring ===\n")
    return results

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
    for _ in range(max(1, max_questions)):
        q_resp = client.chat.completions.create(
            model=llm_b_model,
            messages=messages + conversation + [
                {"role": "assistant", "content": "If you can already make a reasonable recommendation, output exactly the token STOP. Otherwise, ask one short question about exactly one attribute that helps choose among the products. Output only the question text or STOP."}
            ],
            temperature=0.3,
            max_tokens=80,
        )
        question = q_resp.choices[0].message.content.strip()
        if question.upper() == "STOP":
            break
        print(f"AI Recommender: {question}")
        answer = simulated_user_respond(persona_description, question)
        print(f"User: {answer}")
        conversation.append({"role": "assistant", "content": question})
        conversation.append({"role": "user", "content": answer})

    rec_resp = client.chat.completions.create(
        model=llm_b_model,
        messages=messages + conversation + [
            {"role": "assistant", "content": "Based on the answers so far, recommend exactly one product id from the list and give a brief rationale. Return a JSON object with keys 'id' and 'rationale'."}
        ],
        temperature=0.3,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    content = rec_resp.choices[0].message.content.strip()
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