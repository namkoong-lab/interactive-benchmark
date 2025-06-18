# Interactive Persona-Based Recommender System

This repository contains two main components:

## 1. Amazon Product Categorization (amazon_dataset/)

Tools for processing and categorizing Amazon product data into a hierarchical category structure. The pipeline:
- Downloads product metadata from the Amazon dataset
- Processes product information and reviews
- Categorizes products into a hierarchical structure (e.g., "Beauty & Personal Care > Makeup > Eye Makeup > Eyeliner")
- Saves categorized products for use in the recommender system

### Running the Amazon Categorization Pipeline

```bash
cd amazon_dataset
python run_pipeline.py
```

This will:
1. Download metadata for up to 3000 products
2. Process and categorize the products
3. Save results in `categorized_products/` directory

Key files:
- `categorize.py`: Implements hierarchical product categorization
- `meta.py`: Handles product metadata processing
- `review.py`: Processes product reviews
- `run_pipeline.py`: Orchestrates the categorization pipeline

## 2. Persona-Based Recommender Simulation (pipeline/)

Simulates interactions between a user (LLM A) and a recommender (LLM B):
- `personas.py`: Generates benchmark entries for personas
- `simulate_interaction.py`: Core simulation functions for user-recommender interaction
- `simulation.py`: Command-line driver for running simulations

### Running the Recommender Simulation

```bash
python pipeline/simulation.py <persona_index> [llm_b_model] [num_questions]
```

### Setup

1. Clone this repo and `cd` into it
2. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## Project Structure

 - **personas.py**  
   Generates one or more "benchmark entry" JSON files for given persona indices. Must be run as:  
   ```bash
   python personas.py <comma_separated_indices>
   ```
   This will create files named `benchmark_entries/persona_<idx>.json` for each index. Each file includes:
     - `products`: a list of ~30 items with attributes and a `"user_preference"` label  
     - `user_attributes`: decisive filters that rule out disliked items  
     - `correct_product` & `noise_traits` for evaluation (hidden from the recommender LLM)

- **simulate_interaction.py**  
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