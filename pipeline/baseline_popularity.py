#!/usr/bin/env python3
"""
Baseline Experiment: Popularity Recommendation.

This script runs the Experiment 1 setup with a simple, non-learning agent
that recommends the most "popular" product, defined as the one with the
highest rating, or lowest price as a fallback.
"""

import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, List, Any
from datetime import datetime

# Import modules from the project
from .envs.reco_env import RecoEnv
from .wrappers.metrics_wrapper import MetricsWrapper

class PopularityAgent:
    """
    An agent that recommends the most popular product.
    It does not ask questions and does not learn.
    """
    
    def __init__(self, model: str = "popularity", max_questions: int = 0):
        self.model = model
        self.max_questions = max_questions
        # This instance variable is no longer needed, removing it for clarity.

    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Always chooses to make a recommendation immediately."""
        return self._choose_recommendation(obs, info)
    
    def _choose_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Choose the most popular product to recommend.
        Strategy:
        1. Find the product with the highest 'rating' in its raw attributes.
        2. If no products have a rating, fall back to the one with the lowest price.
        """
        products = info.get('products', [])
        if not products:
            return 0 # Safe fallback

        best_product_by_rating = None
        max_rating = -1.0
        for p in products:
            try:
                rating = float(p.get("raw", {}).get("rating", -1))
                if rating > max_rating:
                    max_rating = rating
                    best_product_by_rating = p
            except (ValueError, TypeError):
                continue
        
        if best_product_by_rating is not None:
            chosen_product = best_product_by_rating
        else:
            # Fallback: choose the cheapest product
            chosen_product = min(products, key=lambda p: p.get("price", float('inf')))

        # Find the index of the chosen product to return as the action
        product_ids = [p['id'] for p in products]
        try:
            return product_ids.index(chosen_product['id'])
        except (ValueError, KeyError):
            return 0 # Fallback to first product if something goes wrong

    def update_preferences(self, episode_result: Dict[str, Any]):
        """A non-learning agent, so this method does nothing."""
        pass

def run_baseline_popularity(persona_index: int = 42, 
                            categories: List[str] = None,
                            episodes_per_category: int = 5,
                            max_questions: int = 8,
                            output_dir: str = "baseline_popularity_results"):
    """
    Run the Popularity Recommendation baseline experiment.
    """
    
    print(f"=== Baseline Experiment: Popularity Recommendation ===")
    print(f"Persona: {persona_index}")
    print(f"Episodes per category: {episodes_per_category}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if "RecoEnv-v0" not in gym.envs.registry:
        gym.register("RecoEnv-v0", entry_point="pipeline.envs.reco_env:RecoEnv")
    
    agent = PopularityAgent()
    
    from .core.simulate_interaction import list_categories
    available_categories = list_categories()
    if categories is None:
        categories = available_categories[:5]
    else:
        categories = [cat for cat in categories if cat in available_categories]
    
    print(f"Categories: {categories}")
    
    all_results = []
    category_results = {cat: [] for cat in categories}
    
    total_episodes = len(categories) * episodes_per_category
    episode_num = 0
    
    for category in categories:
        print(f"\n--- Testing Category: {category} ---")
        
        for episode in range(episodes_per_category):
            episode_num += 1
            print(f"Episode {episode_num}/{total_episodes} (Category: {category})")
            
            env = gym.make("RecoEnv-v0", 
                           persona_index=persona_index,
                           max_questions=max_questions,
                           categories=[category])
            
            metrics_wrapper = MetricsWrapper(env, 
                                             output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
            
            obs, initial_info = metrics_wrapper.reset()
            
            action = agent.get_action(obs, initial_info)
            obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
            
            if info['action_type'] == 'recommend':
                print(f"  Step 1: Recommended product {info['chosen_product_id']} (Popularity)")
                print(f"    Score: {info['chosen_score']:.1f}, Best Possible: {info['best_score']:.1f}")
            
            episode_result = {
                'episode': episode_num, 'category': category,
                'episode_in_category': episode + 1, 'steps': 1,
                'terminated': terminated, 'truncated': truncated,
                'final_info': info
            }
            
            all_results.append(episode_result)
            category_results[category].append(episode_result)
            
            metrics_wrapper.close()

    # Analyze and save results
    print(f"\n=== Results Analysis for Popularity Baseline ===")
    
    # Performance by category
    print("\nPerformance by Category:")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'final_info' in r]
        top1_rates = [r['final_info'].get('top1', False) for r in results if 'final_info' in r]
        
        if scores:
            avg_score = np.mean(scores)
            top1_rate = np.mean(top1_rates)
            print(f"  {category}:")
            print(f"    Avg Score: {avg_score:.1f}")
            print(f"    Top-1 Accuracy: {top1_rate:.1%}")
            print(f"    Episodes: {len(scores)}")

    # Learning progression (should be flat for a non-learning baseline)
    print("\nLearning Progression (should be flat for this baseline):")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'final_info' in r]
        if len(scores) >= 2:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            improvement = second_half - first_half
            print(f"  {category}: {first_half:.1f} → {second_half:.1f} (Δ{improvement:+.1f})")

    results_file = os.path.join(output_dir, "baseline_popularity_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Baseline: Popularity Recommendation',
            'timestamp': datetime.now().isoformat(),
            'config': {'persona_index': persona_index, 'categories': categories, 'episodes_per_category': episodes_per_category},
            'results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return all_results, category_results

