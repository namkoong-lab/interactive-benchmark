#!/usr/bin/env python3
"""
Baseline 5: No Prompting Tricks (with Checkpointing).

This script runs a baseline where the LLM agent asks a fixed number of
meaningful questions but uses a simple, direct prompt for the final
recommendation, without any Chain-of-Thought or other reasoning tricks.
The structure is identical to baseline_prompting_tricks.py for fair comparison.
"""

import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple, Optional
import argparse
from datetime import datetime
import random
import yaml
import re

# Import modules from the project
from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
from .wrappers.metrics_wrapper import MetricsWrapper
from .core.feedback_system import FeedbackSystem
from .core.simulate_interaction import list_categories, get_products_by_category
from .core.user_model import UserModel
from .experiment1_with_checkpoints import save_checkpoint, load_checkpoint

# --- Load prompts from the central YAML file ---
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'core', 'prompts.yaml')
with open(PROMPT_FILE_PATH, 'r') as f:
    PROMPTS = yaml.safe_load(f)

class NoTricksAgent:
    """
    An LLM-based agent that asks meaningful questions but uses a simple, direct
    prompt for the final recommendation. This agent is STATELESS.
    """
    
    def __init__(self, model: str = "gpt-4o", max_questions: int = 3):
        self.model = model
        self.max_questions = max_questions
        self.current_env = None
        self.questions_asked_count = 0
        self.last_response = ""
        self.last_thinking_block = None 
        self.learned_preferences = {}
        self.feedback_history = []

    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Decide whether to ask a meaningful question or make a final recommendation."""
        num_products = info.get('num_products', 0)
        
        if self.questions_asked_count < self.max_questions:
            return self._ask_question(obs, info)
        else:
            return self._choose_recommendation(obs, info)

    def _ask_question(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Generate a single, context-aware follow-up question using an LLM.
        This part is IDENTICAL to Baseline 4 to ensure fair comparison.
        """
        self.questions_asked_count += 1
        num_products = info.get('num_products', 0)
        category = info.get('category', 'this product category')
        dialog_history = self.current_env.dialog_history
        
        dialog_text = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(dialog_history)]) if dialog_history else "No questions have been asked yet."

        prompt_template = PROMPTS.get("baseline_4_ask_question")
        if not prompt_template:
            raise ValueError("Could not find 'baseline_4_ask_question' prompt in prompts.yaml")
        
        final_prompt = prompt_template.format(category=category, dialog_text=dialog_text)

        try:
            response = chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.7,
                max_tokens=100
            )
            self.last_response = response.strip()
        except Exception as e:
            print(f"[WARN] Failed to generate a question: {e}")
            self.last_response = "QUESTION: What is your budget?"

        return num_products

    def _choose_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Choose a product using the simple, direct "no tricks" prompt.
        """
        if not self.current_env or not hasattr(self.current_env, 'products'):
            return 0
        
        products = self.current_env.products
        if not products:
            return 0 

        dialog = self.current_env.dialog_history
        dialog_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in dialog]) if dialog else "No questions have been asked yet."
        product_summaries = [{'id': p['id'], 'title': p['title'], 'price': p.get('price')} for p in products]
        product_json = json.dumps(product_summaries, indent=2)

        prompt_template = PROMPTS.get("baseline_5_no_tricks")
        if not prompt_template:
            raise ValueError("Could not find 'baseline_5_no_tricks' prompt in prompts.yaml")
        
        final_prompt = prompt_template.format(dialog_text=dialog_text, product_json=product_json)
        
        content = chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful product recommendation assistant that always returns only JSON."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.2,
            max_tokens=500,
            json_mode=True 
        )
        
        rec_id = -1
        try:
            data = json.loads(content)
            self.last_response = data.get("rationale", "LLM Recommendation")
            id_val = data.get("id")
            if id_val is not None:
                rec_id = int(id_val)
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"[WARN] Failed to parse LLM JSON output. Content: {content}")
        
        product_ids = [p['id'] for p in products]
        try:
            return product_ids.index(rec_id)
        except ValueError:
            print(f"[WARN] LLM recommended a product ID ({rec_id}) not in list. Defaulting to random.")
            return np.random.randint(0, len(products)) if products else 0

    def update_preferences(self, episode_result: Dict[str, Any]):
        """This baseline does not learn, so it just resets for the next episode."""
        self.questions_asked_count = 0
        self.current_env = None
        pass

def save_checkpoint(all_results, category_results, agent, output_dir, model, feedback_type, episode_num, seed):
    """Saves the current state of the experiment to a JSON file."""
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace(" ", "_")
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{model_safe_name}_{feedback_safe_name}.json")
    
    agent_state = {
        'episode_count': agent.episode_count if hasattr(agent, 'episode_count') else 0,
        'learned_preferences': agent.learned_preferences,
        'feedback_history': agent.feedback_history,
    }

    checkpoint_data = {
        'results': all_results,
        'category_results': category_results,
        'agent_state': agent_state,
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



def run_baseline_no_tricks(
    persona_index: int, 
    categories: List[str] = None, 
    num_categories: int = 5, 
    episodes_per_category: int = 5, 
    max_questions: int = 3, 
    model: str = "gpt-4o",
    feedback_type: str = "none", 
    min_score_threshold: float = 50.0,
    output_dir: str = "baseline_no_tricks_results",
    checkpoint_file: str = None,
    seed: Optional[int] = None):
    """
    Run the 'No Prompting Tricks' baseline with checkpointing, structured identically to Experiment 1.
    """
    
    print(f"=== Baseline 5: No Prompting Tricks (with Checkpointing) ===")
    print(f"Persona: {persona_index}, Episodes per category: {episodes_per_category}")
    print(f"Max questions: {max_questions}, Model: {model}, Feedback: {feedback_type}")
    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
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
        all_results = []
        category_results = {}
        start_episode = 1
    
    agent = NoTricksAgent(model=model, max_questions=max_questions)
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

    if categories is None:
        if len(available_categories) >= num_categories:
            selected_categories = random.sample(available_categories, num_categories)
        else:
            selected_categories = available_categories.copy()
    else:
        selected_categories = [cat for cat in categories if cat in available_categories]
    
    print(f"Initial categories to potentially run: {selected_categories}")
    
    used_categories = set(category_results.keys())
    total_episodes = len(selected_categories) * episodes_per_category
    episode_num = start_episode - 1

    for category in selected_categories:
        if category in category_results and len(category_results[category]) >= episodes_per_category:
            print(f"\n--- Category: {category} (already completed, skipping) ---")
            continue
            
        print(f"\n--- Testing Category: {category} ---")
        
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
                    persona_index=persona_index, max_questions=max_questions + 1,
                    categories=[category], agent=agent, 
                    feedback_system=feedback_system, cached_scores=cached_scores
                )
                
                metrics_wrapper = MetricsWrapper(env, output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
                obs, initial_info = metrics_wrapper.reset()
                agent.current_env = env
                
                terminated, truncated, step_count = False, False, 0
                while not terminated and not truncated and step_count < (max_questions + 5):
                    action = agent.get_action(obs, initial_info)
                    obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                    step_count += 1
                    if info['action_type'] == 'ask':
                        print(f"  Step {step_count}: Asked question")
                    elif info['action_type'] == 'recommend':
                        print(f"  Step {step_count}: Recommended product {info['chosen_product_id']}")
                        print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}, Regret: {info.get('regret', 'N/A'):.1f}")
                        break
                
                full_dialog = env.dialog_history if hasattr(env, 'dialog_history') else []
                product_info = {
                    'num_products': len(env.products) if hasattr(env, 'products') else 0,
                    'products_with_scores': []
                }
                if hasattr(env, 'products') and hasattr(env, 'oracle_scores'):
                    id_to_product = {p['id']: p for p in env.products}
                    for product_id, avg_score in env.oracle_scores:
                        product = id_to_product.get(product_id)
                        if product:
                            product_info['products_with_scores'].append({
                                'id': product_id, 'name': product.get('title', 'Unknown'),
                                'price': product.get('price', 'Unknown'), 'average_score': float(avg_score)
                            })
                
                episode_result = {
                    'episode': episode_num, 'category': category, 'episode_in_category': episode + 1,
                    'steps': step_count, 'terminated': terminated, 'truncated': truncated, 
                    'final_info': info, 'full_dialog': full_dialog, 'product_info': product_info,
                    'thinking_block': agent.last_thinking_block
                }
                
                all_results.append(episode_result)
                category_results[category].append(episode_result)
                agent.update_preferences(episode_result)
                metrics_wrapper.close()

            except Exception as e:
                print(f"  ERROR in episode {episode_num}: {e}")
                continue
        
        save_checkpoint(all_results, category_results, agent, output_dir, model, feedback_type, episode_num, seed)
        
    print(f"\n=== Final Results Analysis ===")
    

    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace(" ", "_")
    final_results_file = os.path.join(output_dir, f"baseline_no_tricks_final_{model_safe_name}_{feedback_safe_name}.json")
    
    with open(final_results_file, 'w') as f:
        json.dump({
            'experiment': 'Baseline 5: LLM With No Prompting Tricks',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'regret_progression': {
                    'episode_regrets': [r['final_info'].get('regret') for r in all_results if 'regret' in r.get('final_info', {})],
                    'avg_regret': np.mean([r['final_info'].get('regret') for r in all_results if 'regret' in r.get('final_info', {})]) if any('regret' in r.get('final_info', {}) for r in all_results) else 0.0,
                },
                'categories_tested': list(used_categories),
                'total_episodes': len(all_results)
            },
            'config': {
                'persona_index': persona_index, 'categories': selected_categories,
                'episodes_per_category': episodes_per_category, 'max_questions': max_questions,
                'model': model, 'feedback_type': feedback_type, 'seed': seed
            },
            'category_summary': {
                cat: {
                    'avg_score': np.mean([r['final_info'].get('chosen_score', 0) for r in res if 'final_info' in r]),
                    'avg_regret': np.mean([r['final_info'].get('regret', 100) for r in res if 'final_info' in r]),
                    'top1_rate': np.mean([r['final_info'].get('top1', False) for r in res if 'final_info' in r]),
                    'num_products': res[0]['product_info']['num_products'] if res else 0,
                } for cat, res in category_results.items() if res
            },
            'results': all_results
        }, f, indent=2)

    print(f"\nFinal results saved to: {final_results_file}")
    return all_results, category_results