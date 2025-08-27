import sys
import os
import random
from personas import get_persona_description
from simulate_interaction import (
    list_categories,
    get_products_by_category,
    score_products_for_persona,
    ai_recommender_interact,
)

def main():
    # Persona index prompt
    persona_index_str = input("Enter persona index (integer): ").strip()
    try:
        persona_index = int(persona_index_str)
    except ValueError:
        print("Invalid persona index; please enter an integer.")
        sys.exit(1)

    # Model selection
    available_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "o4-mini",
    ]
    print("\nAvailable AI recommender models:")
    for i, m in enumerate(available_models, start=1):
        print(f"  {i}. {m}")
    model_choice = input("Choose a model by number (or enter a custom model id): ").strip()
    ai_recommender_model = None
    if model_choice.isdigit():
        idx = int(model_choice)
        if 1 <= idx <= len(available_models):
            ai_recommender_model = available_models[idx - 1]
    if not ai_recommender_model:
        ai_recommender_model = model_choice if model_choice else "gpt-4o"

    # Categories
    categories = list_categories()
    if not categories:
        print("No categories found in database.")
        sys.exit(1)
    print("\nAvailable categories:")
    for i, c in enumerate(categories, start=1):
        print(f"  {i}. {c}")
    cat_choice = input("Enter category number, or press Enter for random: ").strip()
    if cat_choice == "":
        category_name = random.choice(categories)
        print(f"Randomly selected category: {category_name}")
    else:
        if not cat_choice.isdigit():
            print("Invalid category selection.")
            sys.exit(1)
        cat_idx = int(cat_choice)
        if not (1 <= cat_idx <= len(categories)):
            print("Category number out of range.")
            sys.exit(1)
        category_name = categories[cat_idx - 1]

    # Load persona description
    persona_description = get_persona_description(persona_index)
    print("\n=== Persona Description Selected ===")
    print(persona_description)
    print("=== End Persona Description ===\n")

    # Category validation (in case of legacy CLI provided name)
    categories = list_categories()
    if not categories:
        print("No categories found in database.")
        sys.exit(1)
    if category_name not in categories:
        print(f"Category '{category_name}' not found. Available examples: {', '.join(categories[:20])}...")
        sys.exit(1)

    # Fetch products
    products = get_products_by_category(category_name)
    if not products:
        print(f"No products found for category '{category_name}'.")
        sys.exit(1)

    # Persona scores products to form a hidden ground truth ranking
    persona_scores = score_products_for_persona(persona_description, category_name, products)
    if not persona_scores:
        print("Failed to obtain persona scores.")
        sys.exit(1)
    best_persona_product_id = persona_scores[0][0]

    # Recommender interacts with persona to infer best product (agent decides number of questions)
    rec_id, rationale = ai_recommender_interact(category_name, products, persona_description, ai_recommender_model)

    print("\nPersona top choice product id:", best_persona_product_id)
    print("Recommender prediction product id:", rec_id)
    print("Rationale:", rationale)

    if rec_id == best_persona_product_id:
        print("Simulation result: Correct recommendation.")
    else:
        print("Simulation result: Incorrect.")

if __name__ == "__main__":
    main()