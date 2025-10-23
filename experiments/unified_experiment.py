#!/usr/bin/env python3
"""
Unified experiment orchestrator that works with all experiment types.

This module provides a single UnifiedExperiment class that can run:
- variable_category experiments
- variable_persona experiments  
- variable_settings experiments
- planning_no_strat experiments
- planning_greedy experiments
- planning_dp experiments

It uses the UnifiedAgent class for all experiment types.
"""

import gymnasium as gym
import numpy as np
import json
import os
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from pipeline.envs.reco_env import RecoEnv
from pipeline.wrappers.metrics_wrapper import MetricsWrapper
from pipeline.core.feedback_system import FeedbackSystem
from pipeline.core.user_model import UserModel
from pipeline.core.simulate_interaction import list_categories, get_products_by_category
from pipeline.core.unified_agent import UnifiedAgent


class UnifiedExperiment:
    """
    Unified experiment orchestrator.
    
    Handles any experiment type with a single implementation.
    Uses UnifiedAgent for all experiment types.
    """
    
    def __init__(self, config):
        """
        Initialize unified experiment.
        
        Args:
            config: ExperimentConfig object with all settings
        """
        self.config = config
        self.agent = None
        self.all_results = []
        self.grouped_results = {}  # Category-grouped or persona-grouped results
        
        # Register environment if needed
        if "RecoEnv-v0" not in gym.envs.registry:
            gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    def _create_agent(self):
        """Create UnifiedAgent based on experiment type."""
        if self.config.experiment_type == "variable_category":
            return UnifiedAgent(
                model=self.config.model,
                max_questions=self.config.max_questions,
                context_mode=self.config.context_mode,
                prompting_tricks=self.config.prompting_tricks,
                force_all_questions=False,
                strategy="none",
                vary_persona=False,    # Same customer
                vary_category=True     # Different categories
            )
        
        elif self.config.experiment_type == "variable_persona":
            return UnifiedAgent(
                model=self.config.model,
                max_questions=self.config.max_questions,
                context_mode=self.config.context_mode,
                prompting_tricks=self.config.prompting_tricks,
                force_all_questions=False,
                strategy="none",
                vary_persona=True,     # Different customers
                vary_category=False    # Same category
            )
        
        elif self.config.experiment_type == "variable_settings":
            return UnifiedAgent(
                model=self.config.model,
                max_questions=self.config.max_questions,
                context_mode=self.config.context_mode,
                prompting_tricks=self.config.prompting_tricks,
                force_all_questions=False,
                strategy="none",
                vary_persona=True,     # Different customers
                vary_category=True     # Different categories
            )
        
        elif self.config.experiment_type == "planning_no_strat":
            return UnifiedAgent(
                model=self.config.model,
                max_questions=self.config.max_questions,
                context_mode=self.config.context_mode,
                prompting_tricks=self.config.prompting_tricks,
                force_all_questions=True,
                strategy="none",
                vary_persona=True,
                vary_category=True,
                track_regret_progression=self.config.track_regret_progression
            )
        
        elif self.config.experiment_type == "planning_greedy":
            return UnifiedAgent(
                model=self.config.model,
                max_questions=self.config.max_questions,
                context_mode=self.config.context_mode,
                prompting_tricks=self.config.prompting_tricks,
                force_all_questions=True,
                strategy="greedy",
                vary_persona=True,
                vary_category=True,
                track_regret_progression=self.config.track_regret_progression
            )
        
        elif self.config.experiment_type == "planning_dp":
            return UnifiedAgent(
                model=self.config.model,
                max_questions=self.config.max_questions,
                context_mode=self.config.context_mode,
                prompting_tricks=self.config.prompting_tricks,
                force_all_questions=True,
                strategy="pomdp",
                vary_persona=True,
                vary_category=True,
                track_regret_progression=self.config.track_regret_progression
            )
        
        else:
            raise ValueError(f"Unknown experiment type: {self.config.experiment_type}")
    
    def _is_category_relevant_for_persona(self, category: str, persona_index: int) -> Tuple[bool, float, List]:
        """Check if a category is relevant for a persona."""
        try:
            products = get_products_by_category(
                category, 
                limit=self.config.max_products_per_category,
                seed=self.config.get_seeds()[0]
            )
            if not products:
                return False, 0.0, []
                
            user_model = UserModel(persona_index)
            scores = user_model.score_products(category, products)
            if scores:
                max_score = max(score for _, score in scores)
                return max_score > self.config.min_score_threshold, max_score, scores
            return False, 0.0, []
        except Exception as e:
            print(f"  Error checking category {category} for persona {persona_index}: {e}")
            return False, 0.0, []
    
    def _plan_trajectories(self) -> List[Tuple[int, List[str], List[int], List[Tuple[int, int, str]]]]:
        """
        Plan trajectories based on experiment type.
        
        Returns:
            List of (trajectory_num, categories, personas, episode_plan)
            episode_plan is list of (episode_num, persona_idx, category)
        """
        trajectories = []
        
        for traj_num in range(self.config.total_trajectories):
            if self.config.experiment_type == "variable_category":
                # Same persona, different categories
                personas = self.config.get_persona_indices()
                categories = self.config.get_categories()
                
                # Filter categories for relevance
                relevant_categories = []
                for cat in categories:
                    if len(relevant_categories) >= self.config.get_num_categories():
                        break
                    is_relevant, max_score, _ = self._is_category_relevant_for_persona(cat, personas[0])
                    if is_relevant:
                        relevant_categories.append(cat)
                
                # Create episode plan for this trajectory
                episode_plan = []
                for episode_num in range(self.config.episodes_per_trajectory):
                    # Cycle through categories
                    category = relevant_categories[episode_num % len(relevant_categories)]
                    episode_plan.append((episode_num + 1, personas[0], category))
                
                trajectories.append((traj_num + 1, relevant_categories, personas, episode_plan))
            
            elif self.config.experiment_type == "variable_persona":
                # Different personas, same category
                personas = self.config.get_persona_indices()
                categories = self.config.get_categories()
                
                # Create episode plan for this trajectory
                episode_plan = []
                for episode_num in range(self.config.episodes_per_trajectory):
                    # Cycle through personas
                    persona = personas[episode_num % len(personas)]
                    episode_plan.append((episode_num + 1, persona, categories[0]))
                
                trajectories.append((traj_num + 1, categories, personas, episode_plan))
            
            else:  # variable_settings or planning experiments
                # Both personas and categories vary
                personas = self.config.get_persona_indices()
                categories = self.config.get_categories()
                
                # Create episode plan for this trajectory
                episode_plan = []
                for episode_num in range(self.config.episodes_per_trajectory):
                    # Cycle through persona/category combinations
                    persona = personas[episode_num % len(personas)]
                    category = categories[(episode_num // len(personas)) % len(categories)]
                    episode_plan.append((episode_num + 1, persona, category))
                
                trajectories.append((traj_num + 1, categories, personas, episode_plan))
        
        return trajectories
    
    def _run_single_episode(self, episode_num: int, persona_index: int, category: str, 
                           cached_scores: Optional[List[Tuple[int, float]]] = None) -> Optional[Dict[str, Any]]:
        """
        Run a single episode.
        
        Args:
            episode_num: Episode number
            persona_index: Persona index to use
            category: Category to use
            cached_scores: Optional pre-computed scores
            
        Returns:
            Episode result dictionary or None if failed
        """
        print(f"\nEpisode {episode_num}: Persona #{persona_index}, Category: {category}")
        
        try:
            # Create feedback system for this persona
            if self.config.feedback_type == "persona":
                persona_agent = UserModel(persona_index)
                feedback_system = FeedbackSystem(feedback_type="persona", persona_agent=persona_agent)
            else:
                feedback_system = FeedbackSystem(feedback_type=self.config.feedback_type)
            
            # Create environment
            env = RecoEnv(
                persona_index=persona_index,
                max_questions=self.config.max_questions,
                categories=[category],
                agent=self.agent,
                feedback_system=feedback_system,
                cached_scores=cached_scores,
                max_products_per_category=self.config.max_products_per_category,
                seed=self.config.get_seeds()[0]
            )
            
            # Wrap with metrics
            metrics_wrapper = MetricsWrapper(
                env, 
                output_path=os.path.join(self.output_path, f"episode_{episode_num}.jsonl")
            )
            
            # Reset environment
            obs, info = metrics_wrapper.reset()
            self.agent.current_env = env
            
            # Run episode
            terminated = False
            truncated = False
            step_count = 0
            current_info = info
            
            while not terminated and not truncated and step_count <= 20:
                action = self.agent.get_action(obs, current_info)
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                current_info = info
                step_count += 1
                
                if info['action_type'] == 'ask':
                    print(f"  Step {step_count}: Asked question")
                elif info['action_type'] == 'recommend':
                    print(f"  Step {step_count}: Recommended product {info['chosen_product_id']}")
                    print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}, Regret: {info.get('regret', 0):.1f}")
                    break
            
            # Get dialog history
            full_dialog = env.dialog_history if hasattr(env, 'dialog_history') else []
            
            # Build product info
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
                            'id': product_id,
                            'name': product.get('title', 'Unknown'),
                            'price': product.get('price', 'Unknown'),
                            'average_score': float(avg_score)
                        })
            
            # Build episode result
            if info.get('action_type') == 'recommend' and 'chosen_score' in info:
                episode_result = {
                    'episode': episode_num,
                    'category': category,
                    'persona_index': persona_index,
                    'steps': step_count,
                    'terminated': terminated,
                    'truncated': truncated,
                    'final_info': info,
                    'full_dialog': full_dialog,
                    'product_info': product_info
                }
                
                # Update agent
                self.agent.update_preferences(episode_result)
                
                metrics_wrapper.close()
                return episode_result
            else:
                print(f"  Episode {episode_num}: Skipped - No recommendation made")
                metrics_wrapper.close()
                return None
                
        except Exception as e:
            print(f"  Error in episode {episode_num}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment based on configuration.
        
        Returns:
            Results dictionary with all episodes
        """
        print(f"\n{'='*70}")
        print(f"  {self.config.experiment_type.upper()} EXPERIMENT")
        print(f"{'='*70}")
        print(f"Model: {self.config.model}")
        print(f"Max questions: {self.config.max_questions}")
        print(f"Context mode: {self.config.context_mode}")
        print(f"Feedback type: {self.config.feedback_type}")
        print(f"Prompting tricks: {self.config.prompting_tricks}")
        print(f"Seed: {self.config.get_seeds()[0]}")
        
        # Set seed if provided
        current_seed = self.config.get_seeds()[0]
        if current_seed is not None:
            random.seed(current_seed)
            np.random.seed(current_seed)
        
        # Create output directory
        self.output_path = self.config.get_output_path(self.config.get_seeds()[0])
        os.makedirs(self.output_path, exist_ok=True)
        
        # Save config
        self.config.to_json(os.path.join(self.output_path, "config.json"))
        
        # Create agent
        self.agent = self._create_agent()
        
        # Track used categories and personas
        used_categories = set()
        used_persona_indices = set()
        
        # Plan trajectories
        trajectories = self._plan_trajectories()
        
        print(f"\nExperiment Plan:")
        print(f"  Total trajectories: {len(trajectories)}")
        print(f"  Episodes per trajectory: {self.config.episodes_per_trajectory}")
        print(f"  Total episodes: {len(trajectories) * self.config.episodes_per_trajectory}")
        
        # Initialize results tracking
        self.all_results = []
        self.grouped_results = {}
        
        # Run trajectories
        successful_count = 0
        for traj_num, categories, personas, episode_plan in trajectories:
            print(f"\n{'='*50}")
            print(f"TRAJECTORY {traj_num}/{len(trajectories)}")
            print(f"{'='*50}")
            
            # Track used values
            used_categories.update(categories)
            used_persona_indices.update(personas)
            
            # Run episodes in this trajectory
            trajectory_results = []
            for episode_num, persona_idx, category in episode_plan:
                print(f"\n  Episode {episode_num}/{len(episode_plan)}: Persona {persona_idx}, Category {category}")
                
                # Check if we should use cached scores (first episode of first trajectory)
                cached_scores = None
                if traj_num == 1 and episode_num == 1:
                    _, _, cached_scores = self._is_category_relevant_for_persona(category, persona_idx)
                
                result = self._run_single_episode(episode_num, persona_idx, category, cached_scores)
                
                if result:
                    trajectory_results.append(result)
                    self.all_results.append(result)
                    successful_count += 1
                else:
                    print(f"    Episode failed")
            
            # Store trajectory results
            self.grouped_results[f"trajectory_{traj_num}"] = trajectory_results
        
        # Update config with used values
        self.config._used_categories = list(used_categories) if used_categories else None
        self.config._used_persona_indices = list(used_persona_indices) if used_persona_indices else None
        
        # Save results
        return self._save_results()
    
    def _save_results(self) -> Dict[str, Any]:
        """Save experiment results to file."""
        # Calculate statistics
        episode_regrets = [r['final_info']['regret'] for r in self.all_results if 'regret' in r['final_info']]
        episode_scores = [r['final_info']['chosen_score'] for r in self.all_results if 'chosen_score' in r['final_info']]
        total_questions = sum(r.get('steps', 0) for r in self.all_results)
        
        # Build summary
        summary = {
            'experiment_type': self.config.experiment_type,
            'model': self.config.model,
            'total_trajectories': self.config.total_trajectories,
            'episodes_per_trajectory': self.config.episodes_per_trajectory,
            'total_episodes': len(self.all_results),
            'successful_episodes': len(self.all_results),
            'total_questions_asked': total_questions,
            'avg_regret': float(np.mean(episode_regrets)) if episode_regrets else 0.0,
            'avg_score': float(np.mean(episode_scores)) if episode_scores else 0.0,
            'regret_std': float(np.std(episode_regrets)) if episode_regrets else 0.0
        }
        
        # Add trajectory-level statistics
        trajectory_stats = []
        for traj_key, traj_results in self.grouped_results.items():
            if traj_results:
                traj_regrets = [r['final_info']['regret'] for r in traj_results if 'regret' in r['final_info']]
                traj_scores = [r['final_info']['chosen_score'] for r in traj_results if 'chosen_score' in r['final_info']]
                trajectory_stats.append({
                    'trajectory': traj_key,
                    'episodes': len(traj_results),
                    'avg_regret': float(np.mean(traj_regrets)) if traj_regrets else 0.0,
                    'avg_score': float(np.mean(traj_scores)) if traj_scores else 0.0,
                    'total_questions': sum(r.get('steps', 0) for r in traj_results)
                })
        
        summary['trajectory_stats'] = trajectory_stats
        
        # Save to file
        results_data = {
            'config': self.config.to_dict_complete(),
            'summary': summary,
            'results': self.all_results,
            'agent_history': self.agent.episode_history if hasattr(self.agent, 'episode_history') else [],
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = os.path.join(self.output_path, "results.json")
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        print(f"Total episodes: {len(self.all_results)}")
        print(f"Average regret: {summary['avg_regret']:.2f}")
        print(f"Average score: {summary['avg_score']:.2f}")
        print(f"Results saved to: {results_file}")
        
        return results_data
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'UnifiedExperiment':
        """Create experiment from YAML or JSON config file."""
        from config.experiment_config import ExperimentConfig
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = ExperimentConfig.from_yaml(config_path)
        elif config_path.endswith('.json'):
            config = ExperimentConfig.from_json(config_path)
        else:
            raise ValueError(f"Unknown config file format: {config_path}")
        
        return cls(config)

