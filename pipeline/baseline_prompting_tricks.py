#!/usr/bin/env python3
"""
Baseline Experiment: Prompting tricks + Everything in context.

This script runs the Experiment 1 setup with an LLM-based agent.
The agent asks a few questions and then uses a Chain-of-Thought style prompt
to make its final recommendation based on the full context.
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

class PromptingTricksAgent:
    """An LLM-based agent that uses advanced prompting for its final recommendation."""
    
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
        """Choose a product using an LLM with a Chain-of-Thought style prompt."""
        products = self.current_episode_info.get('products', [])
        
        if not products:
            print("[WARN] _choose_recommendation was called with an empty product list. Cannot make a recommendation.")
            return 0 

        dialog = info.get('dialog_history_text', [])
        
        product_summaries = [{'id': p['id'], 'title': p['title'], 'price': p['price']} for p in products]

        # Construct the advanced "prompting tricks" (Chain-of-Thought) prompt
        prompt_lines = [
            "You are an expert product recommender. Your goal is to choose the single best product for the user based on the conversation.",
            "Conversation History:",
            *dialog,
            "\nProduct List:",
            json.dumps(product_summaries, indent=2),
            "\nInstructions:",
            "1. First, in a <thinking> block, summarize the user's preferences based on the conversation.",
            "2. Second, based on your thinking, analyze which product best fits these preferences.",
            "3. Finally, you MUST choose the product from the list that is the best fit, even if no product is a perfect match. Output a JSON object with your final choice: {\"id\": <product_id>, \"rationale\": \"A brief explanation for your choice.\"}",
            "You must output only the JSON object and nothing else."
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

def run_baseline_prompting_tricks(persona_index: int = 42, 
                                  categories: List[str] = None,
                                  episodes_per_category: int = 5,
                                  max_questions: int = 3,
                                  model: str = "gpt-4o",
                                  output_dir: str = "baseline_prompting_tricks_results"):
    """
    Run the 'Prompting Tricks' baseline experiment.
    """
    
    print(f"=== Baseline Experiment: Prompting Tricks ===")
    print(f"Persona: {persona_index}, Model: {model}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    env_id = "RecoEnv-PromptingTricks-v0"
    if env_id not in gym.envs.registry:
        gym.register(env_id, entry_point="pipeline.envs.reco_env:RecoEnv")
    
    agent = PromptingTricksAgent(model=model, max_questions=max_questions)
    
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
            
            # Manually inject the full product list into the initial info dict
            # to ensure the agent has the data it needs for LLM prompts.
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
    print(f"\n=== Results Analysis for Prompting Tricks Baseline ===")
    
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

    results_file = os.path.join(output_dir, "baseline_prompting_tricks_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Baseline: Prompting Tricks',
            'timestamp': datetime.now().isoformat(),
            'config': {'persona_index': persona_index, 'model': model, 'categories': categories, 'episodes_per_category': episodes_per_category},
            'results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return all_results, category_results

