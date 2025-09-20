#!/usr/bin/env python3
"""
Baseline 2: Random Recommendation (with Checkpointing).

This script runs a baseline where the agent makes a purely random recommendation
without asking any questions. It serves as a performance lower-bound.
The structure is identical to other baselines for fair comparison.
"""

import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import random
import yaml
import re

# Import modules from the project
from .envs.reco_env import RecoEnv
from .wrappers.metrics_wrapper import MetricsWrapper
from .core.feedback_system import FeedbackSystem
from .core.simulate_interaction import list_categories, get_products_by_category
from .core.user_model import UserModel

class RandomAgent:
    """
    An agent that makes a random recommendation. It does not ask questions
    and is STATELESS.
    """
    def __init__(self):
        self.model = "random"
        self.current_env = None

    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Immediately makes a recommendation."""
        return self._choose_recommendation(obs, info)

    def _choose_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """CORE LOGIC: Choose a random product index."""
        num_products = info.get('num_products', 0)
        if num_products == 0:
            return 0 
        return np.random.randint(0, num_products)

    def update_preferences(self, episode_result: Dict[str, Any]):
        """This agent does not learn."""
        pass

def save_checkpoint(all_results, category_results, agent, output_dir, feedback_type, episode_num, seed):
    """Saves the current state of the experiment to a JSON file."""
    feedback_safe_name = feedback_type.replace(" ", "_")
    checkpoint_path = os.path.join(output_dir, f"checkpoint_random_{feedback_safe_name}.json")
    
    checkpoint_data = {
        'results': all_results,
        'category_results': category_results,
        'last_episode_num': episode_num,
        'seed': seed
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"  Checkpoint saved at episode {episode_num} to {checkpoint_path}")

def load_checkpoint(checkpoint_file: str) -> Tuple[List, Dict, Dict]:
    """Loads the experiment state from a checkpoint file."""
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
    return (
        checkpoint_data.get('results', []),
        checkpoint_data.get('category_results', {}),
        checkpoint_data.get('agent_state', {})
    )




def run_baseline_random(
    categories: List[str] = None, 
    num_categories: int = 5, 
    episodes_per_category: int = 5, 
    feedback_type: str = "none", 
    min_score_threshold: float = 50.0,
    output_dir: str = "baseline_random_results",
    checkpoint_file: str = None,
    seed: Optional[int] = None):
    """
    Run the 'Random Recommendation' baseline with checkpointing.
    """
    print(f"=== Baseline 2: Random Recommendation (with Checkpointing) ===")

    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    # Randomly select a persona based on the seed (consistent across episodes)
    persona_index = random.randint(0, 47000)  
    print(f"Selected persona: {persona_index}")
    
    os.makedirs(output_dir, exist_ok=True)
    if "RecoEnv-v0" not in gym.envs.registry:
        gym.register("RecoEnv-v0", entry_point="pipeline.envs.reco_env:RecoEnv")
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        all_results, category_results, _ = load_checkpoint(checkpoint_file)
        start_episode = len(all_results) + 1
        print(f"Resuming from episode {start_episode}")
    else:
        print("Starting fresh experiment")
        all_results, category_results, start_episode = [], {}, 1
    
    agent = RandomAgent()
    feedback_system = FeedbackSystem(feedback_type=feedback_type)
    
    available_categories = list_categories()
    
    def is_category_relevant_for_persona(category, persona_index, min_score_threshold):
        try:
            products = get_products_by_category(category)
            if not products: return False, 0.0, []
            user_model = UserModel(persona_index)
            scores = user_model.score_products(category, products)
            if scores:
                max_score = max(score for _, score in scores)
                return max_score > min_score_threshold, max_score, scores
            return False, 0.0, []
        except Exception as e:
            print(f"  Error checking category {category}: {e}")
            return False, 0.0, []

    # NEW: Select exactly num_categories that pass the relevance filter
    def select_relevant_categories(available_categories, num_categories, persona_index, min_score_threshold):
        """Select exactly num_categories that pass the relevance filter."""
        relevant_categories = []
        tested_categories = set()
        
        # Shuffle available categories for randomness
        shuffled_categories = available_categories.copy()
        random.shuffle(shuffled_categories)
        
        print(f"Searching for {num_categories} relevant categories...")
        
        for category in shuffled_categories:
            if len(relevant_categories) >= num_categories:
                break
                
            if category in tested_categories:
                continue
                
            tested_categories.add(category)
            print(f"  Testing category: {category}")
            
            is_relevant, max_score, cached_scores = is_category_relevant_for_persona(category, persona_index, min_score_threshold)
            if is_relevant:
                relevant_categories.append((category, cached_scores))
                print(f"    ✓ Category {category}: Max score {max_score:.1f} > {min_score_threshold}")
            else:
                print(f"    ✗ Category {category}: Max score {max_score:.1f} ≤ {min_score_threshold}")
        
        if len(relevant_categories) < num_categories:
            print(f"WARNING: Only found {len(relevant_categories)} relevant categories out of {num_categories} requested")
            print(f"Tested {len(tested_categories)} categories total")
        
        return [cat for cat, _ in relevant_categories], {cat: scores for cat, scores in relevant_categories}

    # Initialize category selection with new strategy
    if categories is None:
        if not checkpoint_file or not os.path.exists(checkpoint_file):
            # Fresh start: select exactly num_categories that pass the filter
            selected_categories, cached_scores_map = select_relevant_categories(
                available_categories, num_categories, persona_index, min_score_threshold
            )
            print(f"Selected {len(selected_categories)} relevant categories: {selected_categories}")
        else:
            # Resuming from checkpoint: use the original logic for remaining categories
            if len(available_categories) >= num_categories:
                selected_categories = random.sample(available_categories, num_categories)
            else:
                selected_categories = available_categories.copy()
            cached_scores_map = {}
    else:
        # Use provided categories, filtered by availability
        selected_categories = [cat for cat in categories if cat in available_categories]
        cached_scores_map = {}
    
    print(f"Categories to test: {selected_categories}")
    
    used_categories = set(category_results.keys())
    total_episodes = len(selected_categories) * episodes_per_category
    episode_num = start_episode - 1

    for category in selected_categories:
        if category in category_results and len(category_results[category]) >= episodes_per_category:
            print(f"\n--- Category: {category} (already completed, skipping) ---")
            continue
            
        print(f"\n--- Testing Category: {category} ---")
        
        # Use cached scores if available, otherwise check relevance
        if category in cached_scores_map:
            cached_scores = cached_scores_map[category]
            print(f"  Category {category}: Using cached scores (already verified as relevant)")
        else:
            # Check if this category is relevant for the persona
            is_relevant, max_score, cached_scores = is_category_relevant_for_persona(category, persona_index, min_score_threshold)
            if not is_relevant:
                print(f"  Category {category}: Max score {max_score:.1f} <= {min_score_threshold}, skipping.")
                continue
            
            print(f"  Category {category}: Max score {max_score:.1f} > {min_score_threshold}, proceeding.")
        
        used_categories.add(category)
        if category not in category_results:
            category_results[category] = []
        
        for episode in range(episodes_per_category):
            episode_num += 1
            print(f"Episode {episode_num}/{total_episodes} (Category: {category})")
            
            try:
                env = RecoEnv(
                    persona_index=persona_index, max_questions=0, 
                    categories=[category], agent=agent, 
                    feedback_system=feedback_system, cached_scores=cached_scores
                )
                
                metrics_wrapper = MetricsWrapper(env, output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
                obs, initial_info = metrics_wrapper.reset()
                agent.current_env = env
                
                action = agent.get_action(obs, initial_info)
                _, _, _, _, info = metrics_wrapper.step(action)
                
                print(f"  Step 1: Recommended product {info['chosen_product_id']}")
                print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}, Regret: {info.get('regret', 'N/A'):.1f}")
                
                episode_result = {
                    'episode': episode_num, 'category': category, 'episode_in_category': episode + 1,
                    'steps': 1, 'terminated': True, 'truncated': False, 
                    'final_info': info, 'full_dialog': [], 'product_info': {}
                }
                
                all_results.append(episode_result)
                category_results[category].append(episode_result)
                agent.update_preferences(episode_result)
                metrics_wrapper.close()

            except Exception as e:
                print(f"  ERROR in episode {episode_num}: {e}")
                continue
        
        save_checkpoint(all_results, category_results, agent, output_dir, feedback_type, episode_num, seed)
        
    print(f"\n=== Final Results Analysis ===")
    feedback_safe_name = feedback_type.replace(" ", "_")
    final_results_file = os.path.join(output_dir, f"baseline_random_final_{feedback_safe_name}.json")
    
    with open(final_results_file, 'w') as f:
        json.dump({
            'experiment': 'Baseline 2: Random Recommendation',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'regret_progression': {
                    'episode_regrets': [r['final_info'].get('regret') for r in all_results if 'regret' in r.get('final_info', {})],
                    'avg_regret': np.mean([r['final_info'].get('regret') for r in all_results if 'regret' in r.get('final_info', {})]) if any('regret' in r.get('final_info', {}) for r in all_results) else 0.0,
                },
                'categories_tested': list(used_categories), 'total_episodes': len(all_results)
            },
            'config': {
                'persona_index': persona_index, 'categories': selected_categories,
                'episodes_per_category': episodes_per_category, 'feedback_type': feedback_type, 'seed': seed
            },
            'category_summary': {
                cat: {
                    'avg_score': np.mean([r['final_info'].get('chosen_score', 0) for r in res if 'final_info' in r]),
                    'avg_regret': np.mean([r['final_info'].get('regret', 100) for r in res if 'final_info' in r]),
                    'top1_rate': np.mean([r['final_info'].get('top1', False) for r in res if 'final_info' in r]),
                } for cat, res in category_results.items() if res
            },
            'results': all_results
        }, f, indent=2)

    print(f"\nFinal results saved to: {final_results_file}")
    return all_results, category_results