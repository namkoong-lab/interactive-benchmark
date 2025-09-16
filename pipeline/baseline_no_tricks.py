#!/usr/bin/env python3
"""
Baseline Experiment: No Prompting tricks + Everything in context.

This script is the control group for Baseline 4. It uses a simple, direct
prompt for the final recommendation, without any Chain-of-Thought elements.
"""

import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, List, Any
from datetime import datetime

from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
from .wrappers.metrics_wrapper import MetricsWrapper

class NoTricksAgent:
    """An LLM-based agent that uses a simple, direct prompt for its recommendation."""
    
    def __init__(self, model: str = "gpt-4o", max_questions: int = 8):
        self.model = model
        self.max_questions = max_questions
        self.current_episode_info = None
        self.questions_asked_count = 0

    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Decide whether to ask a question or make a recommendation."""
        # Detect the start of a new episode to reset the internal counter.
        if 'num_products' in info and self.current_episode_info is None:
            self.current_episode_info = info
            self.questions_asked_count = 0

        num_products = self.current_episode_info.get('num_products', 0)
        
        # This agent will ask max_questions number of questions. On the turn AFTER
        # the final question, it will make its recommendation.
        if self.questions_asked_count < self.max_questions:
            self.questions_asked_count += 1
            return num_products # Ask a question
        else:
            return self._choose_recommendation(obs, info)
    
    def _choose_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Choose a product using an LLM with a simple, direct prompt."""
        products = self.current_episode_info.get('products', [])
        
        if not products:
            print("[WARN] _choose_recommendation was called with an empty product list. Cannot make a recommendation.")
            return 0 

        dialog = info.get('dialog_history_text', [])
        
        product_summaries = [{'id': p['id'], 'title': p['title'], 'price': p['price']} for p in products]

        # Construct the simple, direct prompt without Chain-of-Thought
        prompt_lines = [
            "You are an expert product recommender.",
            "Based on the conversation history and the product list below, choose the single best product for the user.",
            "Conversation History:",
            *dialog,
            "\nProduct List:",
            json.dumps(product_summaries, indent=2),
            "\nYour task: You MUST choose a product from the list. Output a JSON object with your final choice: {\"id\": <product_id>, \"rationale\": \"A brief explanation.\"}",
            "Output only the JSON object."
        ]
        
        content = chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful product recommendation assistant that always follows instructions and returns only JSON."},
                {"role": "user", "content": "\n".join(prompt_lines)}
            ],
            temperature=0.2,
            max_tokens=400,
            json_mode=True
        )
        
        rec_id = -1
        try:
            data = json.loads(content)
            id_val = data.get("id")
            if id_val is not None:
                rec_id = int(id_val)
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"[WARN] Failed to parse LLM JSON output for recommendation. Content: {content}")
        
        product_ids = [p['id'] for p in products]
        try:
            return product_ids.index(rec_id)
        except ValueError:
            print(f"[WARN] LLM recommended a product ID ({rec_id}) not in the list. Defaulting to a random choice.")
            if products:
                return np.random.randint(0, len(products))
            return 0

    def update_preferences(self, episode_result: Dict[str, Any]):
        """This baseline does not learn across episodes. Reset state for next episode."""
        self.current_episode_info = None
        self.questions_asked_count = 0
        pass

def run_baseline_no_tricks(persona_index: int = 42, 
                           categories: List[str] = None,
                           episodes_per_category: int = 5,
                           max_questions: int = 3,
                           model: str = "gpt-4o",
                           output_dir: str = "baseline_no_tricks_results"):
    """
    Run the 'No Prompting Tricks' baseline experiment.
    """
    
    print(f"=== Baseline Experiment: No Prompting Tricks ===")
    print(f"Persona: {persona_index}, Model: {model}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    env_id = "RecoEnv-NoTricks-v0"
    if env_id not in gym.envs.registry:
        gym.register(env_id, entry_point="pipeline.envs.reco_env:RecoEnv")
    
    agent = NoTricksAgent(model=model, max_questions=max_questions)
    
    from .core.simulate_interaction import list_categories
    available_categories = list_categories()
    if categories is None:
        categories = available_categories[:5]
    else:
        categories = [cat for cat in categories if cat in available_categories]
    
    all_results = []
    category_results = {cat: [] for cat in categories}
    
    total_episodes = len(categories) * episodes_per_category
    episode_num = 0
    
    for category in categories:
        print(f"\n--- Testing Category: {category} ---")
        
        for episode in range(episodes_per_category):
            episode_num += 1
            print(f"Episode {episode_num}/{total_episodes} (Category: {category})")
            
            env = gym.make(env_id, 
                           persona_index=persona_index,
                           # Environment allows one more step for the final recommendation
                           max_questions=(max_questions + 1),
                           categories=[category])
            
            metrics_wrapper = MetricsWrapper(env, 
                                             output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
            
            obs, initial_info = metrics_wrapper.reset()
            
            if 'products' not in initial_info:
                unwrapped_env = metrics_wrapper.unwrapped
                if hasattr(unwrapped_env, 'products'):
                    initial_info['products'] = unwrapped_env.products

            terminated = False
            truncated = False
            step_count = 0
            current_info = initial_info
            
            while not terminated and not truncated and step_count < 20:
                action = agent.get_action(obs, current_info)
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                current_info = info
                step_count += 1
                
                if info['action_type'] == 'recommend':
                    print(f"  Step {step_count}: Recommended product {info['chosen_product_id']}")
                    print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}")
                    break
            
            episode_result = {
                'episode': episode_num, 'category': category,
                'final_info': info
            }
            all_results.append(episode_result)
            category_results[category].append(episode_result)
            
            agent.update_preferences(episode_result)
            
            metrics_wrapper.close()

    # --- Results Analysis ---
    print(f"\n=== Results Analysis for No Tricks Baseline ===")
    
    print("\nPerformance by Category:")
    for category, results in category_results.items():
        # Filter for episodes that actually ended with a recommendation
        rec_results = [r for r in results if r['final_info'].get('action_type') == 'recommend']
        scores = [r['final_info'].get('chosen_score', 0) for r in rec_results]
        top1_rates = [r['final_info'].get('top1', False) for r in rec_results]
        
        if scores:
            avg_score = np.mean(scores)
            top1_rate = np.mean(top1_rates)
            print(f"  {category}:")
            print(f"    Avg Score: {avg_score:.1f}")
            print(f"    Top-1 Rate: {top1_rate:.1%}")
            print(f"    Episodes: {len(scores)}")
        else:
            print(f"  {category}: No successful recommendations recorded.")

    print("\nLearning Progression (should be flat for this baseline):")
    for category, results in category_results.items():
        rec_results = [r for r in results if r['final_info'].get('action_type') == 'recommend']
        scores = [r['final_info'].get('chosen_score', 0) for r in rec_results]
        if len(scores) >= 2:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            improvement = second_half - first_half
            print(f"  {category}: {first_half:.1f} → {second_half:.1f} (Δ{improvement:+.1f})")

    results_file = os.path.join(output_dir, "baseline_no_tricks_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Baseline: No Prompting Tricks',
            'timestamp': datetime.now().isoformat(),
            'config': {'persona_index': persona_index, 'model': model, 'categories': categories, 'episodes_per_category': episodes_per_category},
            'results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return all_results, category_results

