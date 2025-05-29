from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def simulated_user_respond(user_attributes, question):
    """
    LLM simulates the user based on user_attributes.
    """
    prompt = f"""You are simulating a user with the following attributes describing their preferences:
{json.dumps(user_attributes, indent=2)}

Answer the following question as this user would, providing exactly one attribute and its value in a short sentence:

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You simulate a user based on given attributes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def ai_recommender_interact(products, user_attributes, category, llm_b_model="gpt-4o", num_questions=1):
    """
    AI Recommender interacts with User iteratively to find a liked product.
    """
    # Initialize chat messages for AI Recommender 
    messages = [
        {"role": "system", "content": "You are a product recommender. Ask only about user preferences over product attributes; do not ask the user directly to name or pick a product by name."},
        {"role": "assistant", "content": f"The product category is: {category}. Here are the products and their attributes:\n{json.dumps(products, indent=2)}"}
    ]

    # Ask up to num_questions clarifying questions
    for _ in range(num_questions):
        q_resp = client.chat.completions.create(
            model=llm_b_model,
            messages=messages + [
                {"role": "assistant", "content": "Ask one concise question about exactly one product attribute to clarify the user's preference based on the conversation so far."}
            ],
            temperature=0.7,
            max_tokens=100
        )
        question = q_resp.choices[0].message.content.strip()
        print(f"AI Recommender: {question}")

        answer = simulated_user_respond(user_attributes, question)
        print(f"User: {answer}")

        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": answer})

    rec_resp = client.chat.completions.create(
        model=llm_b_model,
        messages=messages + [
            {"role": "assistant", "content": "Based on the conversation, recommend the single best product from the list. If there are multiple products that fit the description (ie. you didn't have enough questions to narrow it down), randomly select one. Provide the product details and a brief justification."}
        ],
        temperature=0.7,
        max_tokens=200
    )
    recommendation = rec_resp.choices[0].message.content.strip()
    return recommendation