#!/usr/bin/env python3
"""
Baseline Experiment: Random Recommendation.

This script runs the same experimental setup as Experiment 1, but uses a
simple, non-learning agent that makes a random recommendation without asking questions.

This establishes a performance lower-bound.
"""

import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime

# Import modules from the project
from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
from .core.metrics import EpisodeRecord, MetricsRecorder
from .wrappers.metrics_wrapper import MetricsWrapper


class RandomAgent:
    """
    An agent that makes a random recommendation on its first turn.
    It does not ask questions and does not learn from past episodes.
    """
    
    def __init__(self, model: str = "random", max_questions: int = 0):
        # Parameters are kept for signature consistency with the experiment runner
        self.model = model
        self.max_questions = max_questions
        self.current_episode_info = None

    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Always chooses to make a recommendation immediately.
        """
        # A random agent does not need to ask questions.
        # It proceeds directly to making a recommendation.
        return self._choose_recommendation(obs, info)
    
    def _choose_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Choose a random product to recommend.
        """
        # Determine the number of available products from the info dict or observation
        if 'num_products' in info:
            num_products = info['num_products']
        else:
            # Fallback for subsequent steps if info is not fully populated
            num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
            
        if num_products == 0:
            return 0 # Should not happen, but as a safe fallback
            
        # CORE LOGIC: Choose a random product index
        best_product_idx = np.random.randint(0, num_products)
        
        return int(best_product_idx)
    
    def update_preferences(self, episode_result: Dict[str, Any]):
        """
        A random agent does not learn, so this method does nothing.
        """
        pass


def run_baseline_random(persona_index: int = 42, 
                        categories: List[str] = None,
                        episodes_per_category: int = 5,
                        max_questions: int = 8,
                        output_dir: str = "exp1_baseline_random_results"):
    """
    Run the Random Recommendation baseline experiment.
    The structure is identical to run_experiment1 for fair comparison.
    """
    
    print(f"=== Baseline Experiment1: Random Recommendation ===")
    print(f"Persona: {persona_index}")
    print(f"Episodes per category: {episodes_per_category}")
    print(f"Categories: {categories or 'Default'}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if "RecoEnv-v0" not in gym.envs.registry:
        gym.register("RecoEnv-v0", entry_point="pipeline.envs.reco_env:RecoEnv")
    
    # Create the RandomAgent
    agent = RandomAgent()
    
    from .core.simulate_interaction import list_categories
    available_categories = list_categories()
    if categories is None:
        categories = available_categories[:5]
    else:
        categories = [cat for cat in categories if cat in available_categories]
    
    print(f"Running on categories: {categories}")
    
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
            
            # Since RandomAgent recommends immediately, the loop will run only once.
            terminated = False
            truncated = False
            step_count = 0
            
            while not terminated and not truncated and step_count < 5: # Safety limit
                action = agent.get_action(obs, initial_info)
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                step_count += 1
                
                if info['action_type'] == 'recommend':
                    print(f"  Step {step_count}: Recommended product {info['chosen_product_id']} (Randomly)")
                    print(f"    Score: {info['chosen_score']:.1f}, Best Possible: {info['best_score']:.1f}")
                    break
            
            episode_result = {
                'episode': episode_num, 'category': category,
                'episode_in_category': episode + 1, 'steps': step_count,
                'terminated': terminated, 'truncated': truncated,
                'final_info': info
            }
            
            all_results.append(episode_result)
            category_results[category].append(episode_result)
            
            agent.update_preferences(episode_result) # Does nothing, but called for consistency
            
            metrics_wrapper.close()
    
    # Analyze results (the same way as in experiment1)
    print(f"\n=== Results Analysis for Random Baseline ===")
    
    print("\nPerformance by Category:")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        if scores:
            avg_score = np.mean(scores)
            print(f"  {category}: Avg Score: {avg_score:.1f}")
    
    # The learning progression for a random agent should be flat (around 0).
    print("\nLearning Progression (should be flat for random):")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        if len(scores) >= 2:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            improvement = second_half - first_half
            print(f"  {category}: {first_half:.1f} → {second_half:.1f} (Δ{improvement:+.1f})")
    
    results_file = os.path.join(output_dir, "exp1_baseline_random_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment1': 'Baseline: Random Recommendation',
            'timestamp': datetime.now().isoformat(),
            'config': {'persona_index': persona_index, 'categories': categories, 'episodes_per_category': episodes_per_category},
            'results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return all_results, category_results
