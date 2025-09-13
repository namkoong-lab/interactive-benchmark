#!/usr/bin/env python3
"""
Experiment 1: Same user LLM, learning across different categories.

This experiment tests whether an LLM can learn consistent user preferences
across different product categories and improve its recommendation performance
over multiple episodes.

Key questions:
1. Can the LLM learn latent user preferences that transfer across categories?
2. Does recommendation performance improve over episodes?
3. Is the feedback signal sufficient for learning?
"""

import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime

# Import our modules
from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
from .core.metrics import EpisodeRecord, MetricsRecorder
from .wrappers.metrics_wrapper import MetricsWrapper


class LLMAgent:
    """
    LLM-based agent that can ask questions and make recommendations.
    This agent should learn user preferences across episodes.
    """
    
    def __init__(self, model: str = "gpt-4o", max_questions: int = 8):
        self.model = model
        self.max_questions = max_questions
        self.episode_count = 0
        self.learned_preferences = {}  # Store learned user preferences
        self.current_episode_info = None  # Store current episode info
        
    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Decide whether to ask a question or make a recommendation.
        
        Args:
            obs: Current observation (product features, dialog history, etc.)
            info: Environment info (category, product_ids, etc.)
            
        Returns:
            Action: 0 to num_products-1 for recommend, num_products for ask question
        """
        # Debug: print available keys
        print(f"[DEBUG] Available info keys: {list(info.keys())}")
        
        # Handle different info formats from reset() vs step()
        if 'num_products' in info:
            # From reset() - has full environment info, store it
            self.current_episode_info = info
            num_products = info['num_products']
            category = info['category']
        else:
            # From step() - use stored episode info
            if self.current_episode_info is None:
                # Fallback: count non-zero product features
                num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
                category = "unknown"
            else:
                num_products = self.current_episode_info['num_products']
                category = self.current_episode_info['category']
        
        budget_remaining = obs['budget_remaining'][0]
        
        # Extract dialog history
        dialog_history = self._extract_dialog_history(obs)
        
        # Simple strategy: ask questions if we have budget and haven't learned enough
        if budget_remaining > 0 and len(dialog_history) < 3:
            return num_products  # Ask a question
        else:
            # Make a recommendation based on learned preferences
            return self._choose_recommendation(obs, info, dialog_history)
    
    def _extract_dialog_history(self, obs: Dict[str, np.ndarray]) -> List[Tuple[str, str]]:
        """Extract dialog history from observation."""
        # This is a simplified extraction - in practice you'd decode the character embeddings
        dialog_history = []
        dialog_array = obs['dialog_history']
        
        # Count non-zero entries to estimate dialog length
        non_zero_count = np.count_nonzero(dialog_array)
        estimated_dialog_length = non_zero_count // 50  # Rough estimate
        
        return [("placeholder_q", "placeholder_a")] * (estimated_dialog_length // 2)
    
    def _choose_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                             dialog_history: List[Tuple[str, str]]) -> int:
        """Choose which product to recommend based on learned preferences."""
        # Use stored episode info for num_products
        if self.current_episode_info is not None:
            num_products = self.current_episode_info['num_products']
        else:
            # Fallback: count non-zero product features
            num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
        
        # Simple strategy: choose based on price preferences learned from dialog
        # In a real implementation, this would use the LLM to analyze preferences
        
        # For now, use a simple heuristic based on product features
        product_features = obs['product_features']
        
        # Extract price information (feature 0 is normalized price)
        prices = product_features[:num_products, 0]
        
        # Simple preference: prefer mid-range products (avoid extremes)
        price_scores = 1.0 - np.abs(prices - 0.5)  # Higher score for prices closer to 0.5
        best_product_idx = np.argmax(price_scores)
        
        return int(best_product_idx)
    
    def update_preferences(self, episode_result: Dict[str, Any]):
        """Update learned preferences based on episode outcome."""
        self.episode_count += 1
        
        # Store episode results for learning
        if 'chosen_score' in episode_result:
            score = episode_result['chosen_score']
            category = episode_result.get('category', 'unknown')
            
            # Simple learning: track performance by category
            if category not in self.learned_preferences:
                self.learned_preferences[category] = []
            self.learned_preferences[category].append(score)


def run_experiment1(persona_index: int = 42, 
                   categories: List[str] = None,
                   episodes_per_category: int = 5,
                   max_questions: int = 8,
                   model: str = "gpt-4o",
                   output_dir: str = "experiment1_results"):
    """
    Run Experiment 1: LLM learning across categories.
    
    Args:
        persona_index: Which persona to use (consistent across episodes)
        categories: List of categories to test (None = use all available)
        episodes_per_category: Number of episodes per category
        max_questions: Maximum questions per episode
        model: LLM model to use
        output_dir: Directory to save results
    """
    
    print(f"=== Experiment 1: LLM Learning Across Categories ===")
    print(f"Persona: {persona_index}")
    print(f"Episodes per category: {episodes_per_category}")
    print(f"Max questions: {max_questions}")
    print(f"Model: {model}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Register environment
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    # Create LLM agent
    agent = LLMAgent(model=model, max_questions=max_questions)
    
    # Get available categories
    from .core.simulate_interaction import list_categories
    available_categories = list_categories()
    if categories is None:
        categories = available_categories[:5]  # Use first 5 categories
    else:
        categories = [cat for cat in categories if cat in available_categories]
    
    print(f"Categories: {categories}")
    
    # Track results
    all_results = []
    category_results = {cat: [] for cat in categories}
    
    # Run episodes
    total_episodes = len(categories) * episodes_per_category
    episode_num = 0
    
    for category in categories:
        print(f"\n--- Testing Category: {category} ---")
        
        for episode in range(episodes_per_category):
            episode_num += 1
            print(f"Episode {episode_num}/{total_episodes} (Category: {category})")
            
            # Create environment for this episode
            env = gym.make("RecoEnv-v0", 
                          persona_index=persona_index,
                          max_questions=max_questions,
                          categories=[category])  # Single category per episode
            
            # Wrap with metrics
            metrics_wrapper = MetricsWrapper(env, 
                                           output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
            
            # Reset environment
            obs, initial_info = metrics_wrapper.reset()
            
            # Run episode
            terminated = False
            truncated = False
            step_count = 0
            current_info = initial_info  # Use initial info for first action
            
            while not terminated and not truncated and step_count < 20:  # Safety limit
                # Get action from agent
                action = agent.get_action(obs, current_info)
                
                # Take step
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                current_info = info  # Update info for next iteration
                step_count += 1
                
                # Print progress
                if info['action_type'] == 'ask':
                    print(f"  Step {step_count}: Asked question")
                elif info['action_type'] == 'recommend':
                    print(f"  Step {step_count}: Recommended product {info['chosen_product_id']}")
                    print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}")
                    print(f"    Top1: {info['top1']}, Top3: {info['top3']}")
                    break
            
            # Store episode results
            episode_result = {
                'episode': episode_num,
                'category': category,
                'episode_in_category': episode + 1,
                'steps': step_count,
                'terminated': terminated,
                'truncated': truncated,
                'final_info': info
            }
            
            all_results.append(episode_result)
            category_results[category].append(episode_result)
            
            # Update agent preferences
            agent.update_preferences(episode_result)
            
            # Close environment
            metrics_wrapper.close()
    
    # Analyze results
    print(f"\n=== Results Analysis ===")
    
    # Performance by category
    print("\nPerformance by Category:")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        top1_rates = [r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]
        
        if scores:
            avg_score = np.mean(scores)
            top1_rate = np.mean(top1_rates)
            print(f"  {category}:")
            print(f"    Avg Score: {avg_score:.1f}")
            print(f"    Top1 Rate: {top1_rate:.1%}")
            print(f"    Episodes: {len(scores)}")
    
    # Learning progression (performance over episodes within each category)
    print("\nLearning Progression (Performance over episodes):")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        if len(scores) >= 2:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            improvement = second_half - first_half
            print(f"  {category}: {first_half:.1f} → {second_half:.1f} (Δ{improvement:+.1f})")
    
    # Save detailed results
    results_file = os.path.join(output_dir, "experiment1_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Experiment 1: LLM Learning Across Categories',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'persona_index': persona_index,
                'categories': categories,
                'episodes_per_category': episodes_per_category,
                'max_questions': max_questions,
                'model': model
            },
            'results': all_results,
            'category_summary': {
                cat: {
                    'avg_score': np.mean([r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]),
                    'top1_rate': np.mean([r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]),
                    'episode_count': len(results)
                }
                for cat, results in category_results.items()
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Individual episode metrics saved to: {output_dir}/episode_*.jsonl")
    
    return all_results, category_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1: LLM Learning Across Categories")
    parser.add_argument("--persona_index", type=int, default=42, help="Persona index to use")
    parser.add_argument("--categories", nargs="+", default=None, help="Categories to test")
    parser.add_argument("--episodes_per_category", type=int, default=5, help="Episodes per category")
    parser.add_argument("--max_questions", type=int, default=8, help="Max questions per episode")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--output_dir", type=str, default="experiment1_results", help="Output directory")
    
    args = parser.parse_args()
    
    run_experiment1(
        persona_index=args.persona_index,
        categories=args.categories,
        episodes_per_category=args.episodes_per_category,
        max_questions=args.max_questions,
        model=args.model,
        output_dir=args.output_dir
    )
