import json
import os
import glob
import time
import sys
import argparse
import random
import threading
from typing import List, Dict, Any

import openai
import together
import google.generativeai as genai

from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

OUTPUT_DIR = "categorization_output6k"
PRODUCT_TYPES_FILE = "product_types.json"
LOG_FILE = "categorization_log.jsonl"
INITIAL_SAMPLE_SIZE = 20 

OPENAI_CLIENT = None
TOGETHER_CLIENT = None

def get_llm_response(model_name: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    """
    Calls the appropriate LLM API based on the model name prefix.
    Expected format: "provider/model-name".
    """
    provider, model_id = model_name.split('/', 1)

    if provider == "ollama":
        if not OPENAI_CLIENT:
            raise ValueError("Ollama client not initialized.")
        response = OPENAI_CLIENT.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    elif provider == "openai":
        if not OPENAI_CLIENT:
            raise ValueError("OpenAI client not initialized. Check your OPENAI_API_KEY.")
        response = OPENAI_CLIENT.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    elif provider == "together":
        if not TOGETHER_CLIENT:
            raise ValueError("Together AI client not initialized. Check your TOGETHER_API_KEY.")
        response = TOGETHER_CLIENT.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    elif provider == "gemini":
        user_prompt += "\n\nImportant: Respond with ONLY the JSON object, without any markdown formatting like ```json."
        if not genai.get_model(f'models/{model_id}'):
            raise ValueError("Gemini client not initialized or model not found. Check GOOGLE_API_KEY.")
        model_instance = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                response_mime_type="application/json"
            )
        )
        response = model_instance.generate_content(user_prompt)
        return response.text

    else:
        raise ValueError(f"Unsupported provider '{provider}'. Use 'openai', 'together', 'gemini', or 'ollama'.")

def load_products_from_local_repo(data_path: str, limit: int = None) -> List[Dict[str, Any]]:
    print(f"Loading files from local path: {data_path}...")
    all_data = []
    if os.path.isdir(data_path):
        file_paths = glob.glob(os.path.join(data_path, '*.jsonl'))
        if not file_paths:
            print(f"Warning: No .jsonl files found in directory: {data_path}")
    elif os.path.isfile(data_path):
        file_paths = [data_path]
    else:
        print(f"Error: Path not found or is not a file/directory: {data_path}")
        return []

    for file_path in file_paths:
        print(f"Reading {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {i+1} in {os.path.basename(file_path)}. Skipping.")
        if limit and len(all_data) >= limit:
            all_data = all_data[:limit]
            break
    print(f"Successfully loaded {len(all_data)} products.")
    return all_data

def load_product_types() -> Dict[str, Any]:
    types_path = os.path.join(OUTPUT_DIR, PRODUCT_TYPES_FILE)
    if os.path.exists(types_path):
        with open(types_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_product_types(types_dict: Dict[str, Any]):
    types_path = os.path.join(OUTPUT_DIR, PRODUCT_TYPES_FILE)
    with open(types_path, 'w', encoding='utf-8') as f:
        json.dump(types_dict, f, indent=4, ensure_ascii=False)

def generate_initial_category_list(products: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    print(f"\n=== Generating Initial List of Fine-Grained Product Types from {INITIAL_SAMPLE_SIZE} products ===")
    sample_products = products[:INITIAL_SAMPLE_SIZE]
    products_text = "\n".join([
        f"- {p['metadata'].get('title', 'N/A')}"
        for p in sample_products
    ])
    
    system_prompt = "You are a product categorization expert. Your task is to create a list of specific, fine-grained product types (leaf nodes). Respond with ONLY the JSON object."
    user_prompt = f"""Based on the following list of product titles, generate a comprehensive list of fine-grained, specific product types. Avoid broad categories like "Electronics" or "Clothing". Instead, use specific types like "Wireless Bluetooth Headphones", "Men's Graphic T-Shirt", or "Stainless Steel French Press".

PRODUCT TITLES:
{products_text}

Provide your response in the following JSON format:
{{
  "product_types": [
    "Product Type 1",
    "Product Type 2",
    "Product Type 3"
  ]
}}
"""
    response_content = get_llm_response(
        model_name=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=2048
    )
    
    try:
        product_types_data = json.loads(response_content)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from model response for initial types.")
        print("Response:", response_content)
        sys.exit(1)

    if 'product_types' not in product_types_data or not isinstance(product_types_data['product_types'], list):
        print("Error: The model's response for initial types is not in the expected format.")
        print("Response:", product_types_data)
        sys.exit(1)

    # Add metadata
    product_types_data['version'] = "1.0"
    product_types_data['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S")

    save_product_types(product_types_data)
    print(f"\nGenerated and saved {len(product_types_data['product_types'])} initial product types.")
    return product_types_data


def categorize_product(product_data: Dict[str, Any], model: str, product_types: List[str], max_retries: int = 3) -> Dict[str, Any]:
    metadata = product_data.get('metadata', {})
    product_info = f"""- Title: {metadata.get('title', 'N/A')}
- Description: {metadata.get('description', 'N/A')}
- Brand: {metadata.get('brand', 'N/A')}"""

    system_prompt = "You are an expert product categorization AI. Your goal is to assign the most accurate, fine-grained product type to the given product. You must respond with only a single JSON object."
    user_prompt = f"""Analyze the product information and select the single most appropriate, fine-grained product type from the provided list.

**Product Information:**
{product_info}

**Available Product Types:**
{json.dumps(product_types, indent=2)}

**Instructions & Output Format:**
1.  Choose the best existing type.
2.  If NO existing type is a good fit, set `is_new` to `true` and provide a new, specific product type in `product_type`.
3.  Respond with a single JSON object in the following format. Do not add any explanations.

{{
  "product_type": "Selected or New Product Type",
  "confidence": 0.9,
  "is_new": false
}}
"""
    last_error = None
    for attempt in range(max_retries):
        try:
            response_content = get_llm_response(
                model_name=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=256
            )
            
            if not response_content or not response_content.strip():
                raise ValueError("Empty response from the API")

            result = json.loads(response_content)
            # Basic validation
            if 'product_type' in result and 'confidence' in result and 'is_new' in result:
                return result
            else:
                raise ValueError(f"Model returned malformed JSON: {result}")

        except Exception as e:
            last_error = e
            print(f"API call error for item (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)

    # If all retries fail, return an error structure
    return {
        "product_type": "categorization_failed",
        "confidence": 0.0,
        "is_new": False,
        "error": str(last_error)
    }

def categorize_products(data_path: str, start_from: int, num_products: int, random_sample: bool, seed: int, num_workers: int, model: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_lock = threading.Lock()
    log_path = os.path.join(OUTPUT_DIR, LOG_FILE)

    print(f"Using model: {model}")
    print(f"Results will be logged to: {log_path}")
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")

    dataset = load_products_from_local_repo(data_path, limit=num_products)
    if not dataset:
        print("No products loaded. Exiting.")
        return

    # Prepare product list with original item_id
    products_to_process = []
    for i, item in enumerate(dataset):
        product_data = item
        item_id = item.get('item_id', f"line_{i+1}")
        products_to_process.append((item_id, product_data))
    
    if random_sample:
        print("Randomly sampling products...")
        random.shuffle(products_to_process)

    # Slice the dataset according to start_from and num_products
    products_to_process = products_to_process[start_from:]

    # Load or generate product types
    product_types_data = load_product_types()
    if not product_types_data:
        # Pass only the product data part to the generation function
        product_list_for_init = [p[1] for p in products_to_process]
        product_types_data = generate_initial_category_list(product_list_for_init, model=model)
    else:
        print("\nLoaded existing product types list.")

    current_product_types = product_types_data.get('product_types', [])

    # Open log file in append mode to write results
    with open(log_path, 'a', encoding='utf-8') as log_f:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit jobs to the thread pool
            futures = {
                executor.submit(categorize_product, prod_data, model, current_product_types): item_id
                for item_id, prod_data in products_to_process
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Categorizing products"):
                item_id = futures[future]
                try:
                    categorization_result = future.result()
                    
                    with log_lock:
                        result_type = categorization_result.get('product_type')
                        
                        if result_type not in current_product_types:
                            categorization_result['is_new'] = True
                        else:
                            categorization_result['is_new'] = False

                        log_entry = {
                            "item_id": item_id,
                            "title": products_to_process[[p[0] for p in products_to_process].index(item_id)][1]['metadata'].get('title', 'N/A'),
                            "categorization": categorization_result # Log the corrected result
                        }

                        # Write the corrected result to the log file
                        log_f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                        
                        if categorization_result.get('is_new') and result_type:
                            current_product_types.append(result_type)
                            print(f"\nDiscovered new product type: '{result_type}' for item {item_id}")

                except Exception as exc:
                    print(f'\nItem {item_id} generated an exception: {exc}')
                    # Finally, save the potentially updated list of product types
                product_types_data['product_types'] = sorted(list(set(current_product_types))) # Sort and de-duplicate
                product_types_data['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S")
                save_product_types(product_types_data)
            print(f"\nProcessing complete. Final product type list saved to {PRODUCT_TYPES_FILE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Categorize products using various LLM providers.')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the local data file or directory (.jsonl).')
    parser.add_argument('--model', type=str, default='ollama/llama3:8b', help='Model to use. Format: provider/model_name.')
    parser.add_argument('--num-products', type=int, default=None, help='Number of products to process from the file.')
    parser.add_argument('--start-from', type=int, default=0, help='Start index for processing products.')
    parser.add_argument('--random-sample', action='store_true', help='Randomly sample products before processing.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers for API calls.')

    args = parser.parse_args()

    provider, model_id = args.model.split('/', 1)
    if provider == 'ollama':
        OPENAI_CLIENT = openai.OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        print("Using local Ollama provider.")
    elif provider == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: sys.exit("Error: OPENAI_API_KEY not found in .env file.")
        OPENAI_CLIENT = openai.OpenAI(api_key=api_key)
        print("Using OpenAI provider.")
    elif provider == 'together':
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key: sys.exit("Error: TOGETHER_API_KEY not found in .env file.")
        TOGETHER_CLIENT = together.Client(api_key=api_key)
        print("Using Together AI provider.")
    elif provider == 'gemini':
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: sys.exit("Error: GOOGLE_API_KEY not found in .env file.")
        genai.configure(api_key=api_key)
        print("Using Google Gemini provider.")
    else:
        sys.exit(f"Error: Invalid provider '{provider}' in model name '{args.model}'.")

    categorize_products(
        data_path=args.data_path,
        start_from=args.start_from,
        num_products=args.num_products,
        random_sample=args.random_sample,
        seed=args.seed,
        num_workers=args.num_workers,
        model=args.model
    )