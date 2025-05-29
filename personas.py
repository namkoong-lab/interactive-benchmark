from openai import OpenAI
import re
from datasets import load_dataset
import json
import sys
import os
from dotenv import load_dotenv

INSTRUCTIONS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "instructions", "benchmark_prompt.txt"))

def generate_benchmark_entry(user_description):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(INSTRUCTIONS_FILE, "r") as f:
        prompt = f.read()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    content = response.choices[0].message.content.strip()
    try:
        # Extract JSON 
        match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = content
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON returned:\n{content}\n--- Error: {e}")
        raise

if __name__ == "__main__":
    dataset = load_dataset("Tianyi-Lab/Personas", split="train")

    if len(sys.argv) < 2:
        print("Usage: python personas.py <comma_separated_indices>")
        sys.exit(1)

    try:
        persona_indices = list(map(int, sys.argv[1].split(',')))
        print(f"Generating benchmark entries for personas: {persona_indices}")
    except ValueError:
        print("Invalid input. Provide comma-separated integer indices like 4,5,6.")
        sys.exit(1)

    output_dir = "benchmark_entries"
    os.makedirs(output_dir, exist_ok=True)

    for idx in persona_indices:
        try:
            file_path = os.path.join(output_dir, f"persona_{idx}.json")
            if len(persona_indices) == 1:
                # Single index: check if file exists
                if os.path.exists(file_path):
                    print(f"Benchmark entry for persona {idx} already exists at {file_path}. Skipping generation.")
                    continue
                else:
                    print(f"Generating benchmark entry for persona {idx} as file does not exist.")
            entry = dataset[int(idx)]
            user_description = entry["Llama-3.1-70B-Instruct_descriptive_persona"]
            result = generate_benchmark_entry(user_description)

            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"Saved persona {idx} to {file_path}")

        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            continue