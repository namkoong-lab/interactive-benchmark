# Interactive Persona-Based Recommender Simulation

This repo generates persona entries, simulates a user (LLM A) interacting with a recommender (LLM B), and verifies whether the model’s recommendation matches the ground truth.

## Project Structure

 - **personas.py**  
   Generates one or more “benchmark entry” JSON files for given persona indices. Must be run as:  
   ```bash
   python personas.py <comma_separated_indices>
   ```
   This will create files named `benchmark_entries/persona_<idx>.json` for each index. Each file includes:
     - `products`: a list of ~30 items with attributes and a `"user_preference"` label  
     - `user_attributes`: decisive filters that rule out disliked items  
     - `correct_product` & `noise_traits` for evaluation (hidden from the recommender LLM)

- **simulation_logic.py**  
  Core simulation functions:
  - `llm_a_respond(user_attributes, question)` → simulates user answers  
  - `llm_b_interact(products, user_attributes, model_name)` → runs 5 rounds of dynamic questioning and issues a recommendation

- **simulation.py**  
  Command-line driver for running the recommender simulation. Usage:
  ```bash
  python simulation.py <persona_index> [llm_b_model] [num_questions]
  ```
  1. `<persona_index>` (required) selects which persona JSON to load or auto-generate if missing.  
  2. `[llm_b_model]` (optional): model name for the AI recommender (default: `gpt-4o`; pass `""` to use default).  
  3. `[num_questions]` (optional): number of interactive clarification questions to ask (default: 5).  
  The script then:
    - Ensures `benchmark_entries/persona_<idx>.json` exists, generating it if necessary.  
    - Loads the persona file, strips any hidden `user_preference` fields.  
    - Runs `llm_b_interact` to ask the specified number of questions.  
    - Checks the final recommendation against `correct_product` for evaluation.

## Setup

1. **Clone** this repo and `cd` into it.  
2. **Install** dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install openai datasets python-dotenv