import sys
import os
import json
from datasets import load_dataset
from personas import generate_benchmark_entry
from simulate_interaction import ai_recommender_interact

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python simulation.py <persona_index> [ai_recommender_model] [num_questions]\n")
        sys.exit(1)

    # Persona index to use
    idx = sys.argv[1]

    # AI Recommender model to use
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        ai_recommender_model = sys.argv[2]
    else:
        ai_recommender_model = "gpt-4o"

    # Number of questions to ask the user
    if len(sys.argv) >= 4 and sys.argv[3].strip():
        try:
            num_questions = int(sys.argv[3])
        except ValueError:
            print("Invalid num_questions; it must be an integer.")
            sys.exit(1)
    else:
        num_questions = 5

    file_path = os.path.join("benchmark_entries", f"persona_{idx}.json")
    if not os.path.exists(file_path):
        print(f"Benchmark entry file {file_path} not found. Generating benchmark entry.")
        os.makedirs("benchmark_entries", exist_ok=True)
        if __name__ == "__main__":
            dataset = load_dataset("Tianyi-Lab/Personas", split="train")
            persona = dataset[int(idx)]
            user_description = persona["Llama-3.1-70B-Instruct_descriptive_persona"]
            result = generate_benchmark_entry(user_description)
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved generated benchmark entry to {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    user_attributes = data.get("user_attributes", [])
    products = data.get("products", [])
    category = data.get("category")

    # Remove any 'user_preference' metadata so the recommender doesn't see it
    for product in products:
        if 'user_preference' in product:
            del product['user_preference']

    # Debug: show exactly what the recommender sees
    print("=== Recommender Input ===")
    print("Category:", category)
    print("Products and attributes:")
    print(json.dumps(products, indent=2))
    print("=========================")

    if not user_attributes or not products:
        print("Invalid benchmark entry format: missing 'user_attributes' or 'products'.")
        sys.exit(1)

    recommendation = ai_recommender_interact(products, user_attributes, category, ai_recommender_model, num_questions)
    print("\nAI Recommender: " + recommendation)

    correct = data.get("correct_product", {})
    correct_name = correct.get("name")
    if correct_name and correct_name in recommendation:
        print("Simulation result: Correct recommendation.")
    else:
        print(f"Simulation result: Incorrect. Expected recommendation: {correct_name}")

if __name__ == "__main__":
    main()
