#!/usr/bin/env python3
"""
Unified experiment orchestrator for all experiment types.
Uses UnifiedAgent and supports variable_category, variable_persona, and variable_settings modes.
"""

import gymnasium as gym
import numpy as np
import json
import os
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from pipeline.envs.reco_env import RecoEnv
from pipeline.wrappers.metrics_wrapper import MetricsWrapper
from pipeline.core.feedback_system import FeedbackSystem
from pipeline.core.user_model import UserModel
from pipeline.core.simulate_interaction import list_categories, get_products_by_category
from pipeline.core.unified_agent import UnifiedAgent
from pipeline.core import llm_providers
from pipeline.core import simulate_interaction


class UnifiedExperiment:
    """Unified experiment orchestrator for all experiment types."""
    
    def __init__(self, config):
        self.config = config
        self.agent = None
        self.all_results = []
        self.grouped_results = {}
        
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
                vary_persona=False,
                vary_category=True
            )
        
        elif self.config.experiment_type == "variable_persona":
            return UnifiedAgent(
                model=self.config.model,
                max_questions=self.config.max_questions,
                context_mode=self.config.context_mode,
                prompting_tricks=self.config.prompting_tricks,
                force_all_questions=False,
                strategy="none",
                vary_persona=True,
                vary_category=False
            )
        
        elif self.config.experiment_type == "variable_settings":
            return UnifiedAgent(
                model=self.config.model,
                max_questions=self.config.max_questions,
                context_mode=self.config.context_mode,
                prompting_tricks=self.config.prompting_tricks,
                force_all_questions=False,
                strategy="none",
                vary_persona=True,
                vary_category=True
            )
        
        
        else:
            raise ValueError(f"Unknown experiment type: {self.config.experiment_type}")
    
    def _is_category_relevant_for_persona(self, category: str, persona_index: int, 
                                          seed: Optional[int] = None) -> Tuple[bool, float, List]:
        """Check if category is relevant for persona (max score > threshold)."""
        try:
            products = get_products_by_category(category, limit=self.config.max_products_per_category, seed=seed)
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
    
    def _generate_categories_for_trajectory(self, traj_idx: int) -> List[str]:
        """Generate categories for a trajectory using current random state."""
        from pipeline.core.simulate_interaction import list_categories
        
        if self.config.categories is not None:
            return self.config.categories[traj_idx]
        
        all_categories = list_categories()
        
        if self.config.persona_indices is not None:
            if self.config.experiment_type == "variable_category":
                num_cats = self.config.episodes_per_trajectory
            elif self.config.experiment_type == "variable_persona":
                num_cats = 1
            else:
                num_cats = len(self.config.persona_indices[traj_idx])
        else:
            if self.config.experiment_type == "variable_category":
                num_cats = self.config.episodes_per_trajectory
            elif self.config.experiment_type == "variable_persona":
                num_cats = 1
            else:
                num_cats = self.config.episodes_per_trajectory
        
        sampled_categories = random.sample(all_categories, min(num_cats, len(all_categories)))
        
        if self.config.debug_mode:
            print(f"[DEBUG] Trajectory {traj_idx + 1}: Generated {len(sampled_categories)} categories (requested {num_cats}): {sampled_categories}")
        
        return sampled_categories
    
    def _generate_personas_for_trajectory(self, traj_idx: int) -> List[int]:
        """Generate personas for a trajectory using current random state."""
        from pipeline.core.personas import get_persona_description
        
        if self.config.persona_indices is not None:
            return self.config.persona_indices[traj_idx]
        
        max_persona_index = 0
        while True:
            try:
                get_persona_description(max_persona_index)
                max_persona_index += 1
            except:
                break
        all_persona_indices = list(range(max_persona_index))
        
        if self.config.categories is not None:
            if self.config.experiment_type == "variable_persona":
                num_personas = self.config.episodes_per_trajectory
            elif self.config.experiment_type == "variable_category":
                num_personas = 1
            else:
                num_personas = len(self.config.categories[traj_idx])
        else:
            if self.config.experiment_type == "variable_persona":
                num_personas = self.config.episodes_per_trajectory
            elif self.config.experiment_type == "variable_category":
                num_personas = 1
            else:
                num_personas = self.config.episodes_per_trajectory
        
        sampled_personas = random.sample(all_persona_indices, min(num_personas, len(all_persona_indices)))
        
        if self.config.debug_mode:
            print(f"[DEBUG] Trajectory {traj_idx + 1}: Generated {len(sampled_personas)} personas (requested {num_personas}): {sampled_personas}")
        
        return sampled_personas
    
    def _plan_trajectories(self) -> List[Tuple[int, List[str], List[int], List[Tuple[int, int, str]]]]:
        """Plan trajectories - each generates its own categories/personas with its own seed."""
        trajectories = []
        
        for traj_idx in range(self.config.total_trajectories):
            traj_num = traj_idx + 1
            
            traj_seed = self.trajectory_seeds[traj_idx]
            if traj_seed is not None:
                random.seed(traj_seed)
                np.random.seed(traj_seed)
            
            traj_categories = self._generate_categories_for_trajectory(traj_idx)
            traj_personas = self._generate_personas_for_trajectory(traj_idx)
            
            episode_plan = []
            
            if self.config.experiment_type == "variable_category":
                persona = traj_personas[0]
                for episode_num in range(len(traj_categories)):
                    category = traj_categories[episode_num]
                    episode_plan.append((episode_num + 1, persona, category))
            
            elif self.config.experiment_type == "variable_persona":
                category = traj_categories[0]
                for episode_num in range(len(traj_personas)):
                    persona = traj_personas[episode_num]
                    episode_plan.append((episode_num + 1, persona, category))
            
            else:  # variable_settings
                # Balance categories and personas if lengths differ
                if len(traj_categories) != len(traj_personas):
                    if len(traj_categories) < len(traj_personas):
                        # Need more categories - sample additional ones
                        needed = len(traj_personas) - len(traj_categories)
                        all_categories = list_categories()
                        available = [c for c in all_categories if c not in traj_categories]
                        if available:
                            additional = random.sample(available, min(needed, len(available)))
                            traj_categories.extend(additional)
                    elif len(traj_personas) < len(traj_categories):
                        # Need more personas - sample additional ones
                        from pipeline.core.personas import get_persona_description
                        max_persona_index = 0
                        while True:
                            try:
                                get_persona_description(max_persona_index)
                                max_persona_index += 1
                            except:
                                break
                        needed = len(traj_categories) - len(traj_personas)
                        available = [p for p in range(max_persona_index) if p not in traj_personas]
                        if available:
                            additional = random.sample(available, min(needed, len(available)))
                            traj_personas.extend(additional)
                
                # Now create episode plan with balanced lists
                num_episodes = max(len(traj_categories), len(traj_personas))
                for episode_num in range(num_episodes):
                    persona = traj_personas[episode_num % len(traj_personas)]
                    category = traj_categories[episode_num % len(traj_categories)]
                    episode_plan.append((episode_num + 1, persona, category))
                
            trajectories.append((traj_num, traj_categories, traj_personas, episode_plan))
        
        return trajectories
    
    def _is_planning_episode(self, episode_num_in_trajectory: int) -> bool:
        """Check if this episode should use planning mode (regret tracking)."""
        if self.config.planning_mode == "none":
            return False
        return (episode_num_in_trajectory - 1) % self.config.planning_interval == 0
    
    def _get_planning_strategy(self) -> str:
        """Get strategy name for planning episodes."""
        if self.config.planning_mode == "greedy":
            return "greedy"
        elif self.config.planning_mode == "pomdp":
            return "pomdp"
        else:
            return "none"
    
    def _run_single_episode(self, episode_num: int, persona_index: int, category: str, 
                           cached_scores: Optional[List[Tuple[int, float]]] = None,
                           trajectory_seed: Optional[int] = None,
                           episode_num_in_trajectory: int = None,
                           trajectory_num: int = None) -> Optional[Dict[str, Any]]:
        """Run a single episode. Delegates to planning or regular mode based on episode number."""
        # Determine if this is a planning episode
        if episode_num_in_trajectory and self._is_planning_episode(episode_num_in_trajectory):
            return self._run_planning_episode(episode_num, persona_index, category, cached_scores, trajectory_seed, trajectory_num)
        else:
            return self._run_regular_episode(episode_num, persona_index, category, cached_scores, trajectory_seed, trajectory_num)
    
    def _run_regular_episode(self, episode_num: int, persona_index: int, category: str, 
                           cached_scores: Optional[List[Tuple[int, float]]] = None,
                           trajectory_seed: Optional[int] = None,
                           trajectory_num: int = None) -> Optional[Dict[str, Any]]:
        """Run a regular (non-planning) episode."""
        if self.config.debug_mode:
            print(f"\nEpisode {episode_num}: Persona #{persona_index}, Category: {category}")
        
        try:
            if self.config.feedback_type == "persona":
                persona_agent = UserModel(persona_index)
                feedback_system = FeedbackSystem(feedback_type="persona", persona_agent=persona_agent)
            else:
                feedback_system = FeedbackSystem(feedback_type=self.config.feedback_type)
            
            env = RecoEnv(
                persona_index=persona_index,
                max_questions=self.config.max_questions,
                categories=[category],
                agent=self.agent,
                feedback_system=feedback_system,
                cached_scores=cached_scores,
                max_products_per_category=self.config.max_products_per_category,
                seed=trajectory_seed,
                debug_mode=self.config.debug_mode
            )
            
            # Include trajectory number in filename to prevent collisions across trajectories
            filename = f"trajectory_{trajectory_num}_episode_{episode_num}.jsonl" if trajectory_num else f"episode_{episode_num}.jsonl"
            metrics_wrapper = MetricsWrapper(env, output_path=os.path.join(self.output_path, filename))
            
            obs, info = metrics_wrapper.reset()
            self.agent.current_env = env
            
            if not self.config.debug_mode:
                print(f"    → Running episode...")
            
            terminated = False
            truncated = False
            step_count = 0
            current_info = info
            
            while not terminated and not truncated and step_count <= 20:
                action = self.agent.get_action(obs, current_info)
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                current_info = info
                step_count += 1
                
                if self.config.debug_mode:
                    if info['action_type'] == 'ask':
                        print(f"  Step {step_count}: Asked question")
                    elif info['action_type'] == 'recommend':
                        print(f"  Step {step_count}: Recommended product {info['chosen_product_id']}")
                        print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}, Regret: {info.get('regret', 0):.1f}")
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
                            'id': product_id,
                            'name': product.get('title', 'Unknown'),
                            'price': product.get('price', 'Unknown'),
                            'average_score': float(avg_score)
                        })
            
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
                
                self.agent.update_preferences(episode_result)
                metrics_wrapper.close()
                return episode_result
            else:
                if self.config.debug_mode:
                    print(f"  Episode {episode_num}: Skipped - No recommendation made")
                metrics_wrapper.close()
                return None
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"  Error in episode {episode_num}: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _run_planning_episode(self, episode_num: int, persona_index: int, category: str,
                             cached_scores: Optional[List[Tuple[int, float]]] = None,
                             trajectory_seed: Optional[int] = None,
                             trajectory_num: int = None) -> Optional[Dict[str, Any]]:
        """
        Run a planning episode with regret tracking after each question.
        
        Flow:
        1. Ask Question 1 → Get intermediate recommendation → Track regret (hidden from agent)
        2. Ask Question 2 → Get intermediate recommendation → Track regret (hidden from agent)
        ...
        N. Ask Question N → Get final recommendation → Track regret + Give feedback
        
        The agent doesn't know about intermediate recommendations and only gets feedback for the final one.
        """
        if self.config.debug_mode:
            strategy_name = self._get_planning_strategy()
            print(f"\n[PLANNING MODE - {strategy_name.upper()}] Episode {episode_num}: Persona #{persona_index}, Category: {category}")
        
        # Temporarily set agent strategy for this episode
        original_strategy = self.agent.strategy
        self.agent.strategy = self._get_planning_strategy()
        
        try:
            # Setup environment
            if self.config.feedback_type == "persona":
                persona_agent = UserModel(persona_index)
                feedback_system = FeedbackSystem(feedback_type="persona", persona_agent=persona_agent)
            else:
                feedback_system = FeedbackSystem(feedback_type=self.config.feedback_type)
            
            env = RecoEnv(
                persona_index=persona_index,
                max_questions=self.config.max_questions,
                categories=[category],
                agent=self.agent,
                feedback_system=feedback_system,
                cached_scores=cached_scores,
                max_products_per_category=self.config.max_products_per_category,
                seed=trajectory_seed,
                debug_mode=self.config.debug_mode
            )
            
            # Include trajectory number in filename to prevent collisions across trajectories
            filename = f"trajectory_{trajectory_num}_episode_{episode_num}_planning.jsonl" if trajectory_num else f"episode_{episode_num}_planning.jsonl"
            metrics_wrapper = MetricsWrapper(env, output_path=os.path.join(self.output_path, filename))
            obs, info = metrics_wrapper.reset()
            self.agent.current_env = env
            
            # Regret progression tracking
            regret_progression = []
            question_count = 0
            terminated = False
            truncated = False
            num_products = len(env.products) if hasattr(env, 'products') else 0
            
            # Planning mode: Force agent to ask max_questions, secretly track recommendations
            while question_count < self.config.max_questions and not terminated and not truncated:
                # Let agent choose action, but override if it tries to recommend early
                action = self.agent.get_action(obs, info)
                
                if action < num_products:  # Agent wants to recommend
                    action = num_products  # Force ask question instead
                
                # Execute the question
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                question_count += 1
                
                if self.config.debug_mode:
                    print(f"  Step {question_count}: Asked question")
                
                # Secretly ask agent for recommendation (agent doesn't remember this)
                intermediate_recommendation = self._get_intermediate_recommendation(env, obs, info)
                
                if intermediate_recommendation is not None:
                    # Calculate regret for this hidden recommendation
                    best_score = max(score for _, score in env.oracle_scores) if hasattr(env, 'oracle_scores') else 0
                    recommended_product_id = env.products[intermediate_recommendation]['id']
                    recommended_score = 0
                    for pid, score in env.oracle_scores:
                        if pid == recommended_product_id:
                            recommended_score = score
                            break
                    
                    intermediate_regret = best_score - recommended_score
                    regret_progression.append({
                        'question_number': question_count,
                        'recommended_product_id': recommended_product_id,
                        'recommended_score': recommended_score,
                        'best_score': best_score,
                        'regret': intermediate_regret
                    })
                    
                    if self.config.debug_mode:
                        print(f"    → Hidden recommendation: Product {recommended_product_id}, Regret: {intermediate_regret:.1f}")
            
            # After max_questions, let agent make FINAL recommendation
            final_action = self.agent.get_action(obs, info)
            if final_action >= num_products:
                final_action = 0  # Fallback
            
            # Execute final recommendation and give feedback
            obs, reward, terminated, truncated, info = metrics_wrapper.step(final_action)
            
            if self.config.debug_mode:
                print(f"  Final Recommendation: Product {info.get('chosen_product_id')}")
                print(f"    Score: {info.get('chosen_score', 0):.1f}, Best: {info.get('best_score', 0):.1f}, Regret: {info.get('regret', 0):.1f}")
            
            # Build result
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
                            'id': product_id,
                            'name': product.get('title', 'Unknown'),
                            'price': product.get('price', 'Unknown'),
                            'average_score': float(avg_score)
                        })
            
            if info.get('action_type') == 'recommend' and 'chosen_score' in info:
                episode_result = {
                    'episode': episode_num,
                    'category': category,
                    'persona_index': persona_index,
                    'steps': question_count + 1,  # All questions + final recommendation
                    'terminated': terminated,
                    'truncated': truncated,
                    'final_info': info,
                    'full_dialog': full_dialog,
                    'product_info': product_info,
                    'planning_mode': True,
                    'regret_progression': regret_progression  # Track how regret improves with more questions
                }
                
                self.agent.update_preferences(episode_result)
                metrics_wrapper.close()
                return episode_result
            else:
                metrics_wrapper.close()
                return None
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"  Error in planning episode {episode_num}: {e}")
                import traceback
                traceback.print_exc()
            return None
        finally:
            # Restore original strategy
            self.agent.strategy = original_strategy
    
    def _get_intermediate_recommendation(self, env, obs, info):
        """
        Get agent's actual recommendation using its real decision-making logic.
        Saves and restores agent state to make this transparent (agent doesn't remember).
        """
        try:
            import copy
            
            # Save agent's current state
            saved_last_response = self.agent.last_response
            saved_current_episode_info = copy.deepcopy(self.agent.current_episode_info) if self.agent.current_episode_info else None
            
            # Get agent's REAL recommendation (uses its full logic, strategy, and context)
            action = self.agent.get_action(obs, info)
            
            # Restore agent's state (pretend this recommendation never happened)
            self.agent.last_response = saved_last_response
            self.agent.current_episode_info = saved_current_episode_info
            
            # If action is "ask question", fallback to recommending first product
            num_products = len(env.products) if hasattr(env, 'products') else 0
            if action >= num_products:
                return 0  # Agent chose to ask question, use first product as fallback
            
            return action
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"    Error getting intermediate recommendation: {e}")
                import traceback
                traceback.print_exc()
            return 0  # Fallback to first product
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment based on configuration."""
        # Validate configuration BEFORE doing any work
        try:
            self.config.validate()
        except ValueError as e:
            print(f"\n❌ Configuration Error:")
            print(f"{e}\n")
            raise
        
        llm_providers.set_debug_mode(self.config.debug_mode)
        simulate_interaction.set_debug_mode(self.config.debug_mode)
        
        if self.config.debug_mode:
            print(f"\n{'='*70}")
            print(f"  {self.config.experiment_type.upper()} EXPERIMENT")
            print(f"{'='*70}")
            print(f"Model: {self.config.model}")
            print(f"Max questions: {self.config.max_questions}")
            print(f"Context mode: {self.config.context_mode}")
            print(f"Feedback type: {self.config.feedback_type}")
            print(f"Prompting tricks: {self.config.prompting_tricks}")
            print(f"Seed: {self.config.get_seeds()[0]}")
        else:
            print(f"\n🚀 Running {self.config.experiment_type} experiment with {self.config.model}")
        
        self.trajectory_seeds = self.config.get_seeds()
        self.output_path = self.config.get_output_path()
        os.makedirs(self.output_path, exist_ok=True)
        self.config.to_json(os.path.join(self.output_path, "config.json"))
        self.agent = self._create_agent()
        
        executed_categories_per_traj = []
        executed_personas_per_traj = []
        trajectories = self._plan_trajectories()
        
        if self.config.debug_mode:
            print(f"\nExperiment Plan:")
            print(f"  Total trajectories: {len(trajectories)}")
            print(f"  Episodes per trajectory: {self.config.episodes_per_trajectory}")
            print(f"  Total episodes: {len(trajectories) * self.config.episodes_per_trajectory}")
        
        self.all_results = []
        self.grouped_results = {}
        self.planning_regret_data = {}  # Track planning episode regret progressions
        successful_count = 0
        
        for traj_idx, (traj_num, categories, personas, episode_plan) in enumerate(trajectories):
            trajectory_seed = self.trajectory_seeds[traj_idx] if traj_idx < len(self.trajectory_seeds) else None
            if trajectory_seed is not None:
                random.seed(trajectory_seed)
                np.random.seed(trajectory_seed)
            
            if self.config.debug_mode:
                print(f"\n{'='*50}")
                print(f"TRAJECTORY {traj_num}/{len(trajectories)} (seed: {trajectory_seed})")
                print(f"{'='*50}")
            else:
                print(f"\n📍 Trajectory {traj_num}/{len(trajectories)} (seed: {trajectory_seed})")
            
            trajectory_results = []
            trajectory_executed_categories = []
            trajectory_executed_personas = []
            trajectory_tried_categories = set()
            
            for episode_num, persona_idx, category in episode_plan:
                if self.config.debug_mode:
                    print(f"\n  Episode {episode_num}/{len(episode_plan)}: Persona {persona_idx}, Category {category}")
                else:
                    print(f"  Episode {episode_num}/{len(episode_plan)}: Persona {persona_idx}, Category {category}")
                
                result = None
                episode_tried_categories = set()
                episode_tried_personas = set()
                failed_category_prefixes = set()  # Track semantic clusters that failed
                current_category = category
                current_persona = persona_idx
                
                from pipeline.core.simulate_interaction import list_categories
                all_available_categories = set(list_categories())
                
                # Get all available personas
                from pipeline.core.personas import get_persona_description
                max_persona_index = 0
                while True:
                    try:
                        get_persona_description(max_persona_index)
                        max_persona_index += 1
                    except:
                        break
                all_available_personas = set(range(max_persona_index))
                
                # Determine retry strategy based on experiment type
                if self.config.experiment_type == "variable_persona":
                    max_retries = min(10, len(all_available_personas))
                else:  # variable_category or variable_settings
                    max_retries = min(10, len(all_available_categories))
                
                for retry in range(max_retries):
                    # Decide what to retry based on experiment type
                    if self.config.experiment_type == "variable_persona":
                        # Fixed category, varying personas → retry with new persona
                        if current_persona in episode_tried_personas:
                            untried = [p for p in all_available_personas 
                                      if p not in episode_tried_personas]
                            if not untried:
                                break
                            current_persona = random.choice(sorted(untried))
                        episode_tried_personas.add(current_persona)
                    else:
                        # variable_category or variable_settings → retry with new category
                        if current_category in episode_tried_categories:
                            untried = [c for c in all_available_categories 
                                      if c not in episode_tried_categories and c not in trajectory_tried_categories]
                            if not untried:
                                break
                            
                            weighted_choices = []
                            for cat in untried:
                                cat_prefix = cat.split()[0] if ' ' in cat else cat
                                if cat_prefix in failed_category_prefixes:
                                    weighted_choices.extend([cat] * 1)
                                else:
                                    weighted_choices.extend([cat] * 9)
                            
                            if weighted_choices:
                                current_category = random.choice(weighted_choices)
                            else:
                                current_category = random.choice(sorted(untried))
                        episode_tried_categories.add(current_category)
                    
                    if retry > 0 and self.config.debug_mode:
                        if self.config.experiment_type == "variable_persona":
                            print(f"    Retry {retry}: Trying persona {current_persona}")
                        else:
                            print(f"    Retry {retry}: Trying category {current_category}")
                    
                    is_relevant, max_score, scores = self._is_category_relevant_for_persona(
                        current_category, current_persona, seed=trajectory_seed
                    )
                    
                    if not is_relevant:
                        # Track semantic cluster (first word) of failed category (for category retries)
                        if self.config.experiment_type != "variable_persona":
                            category_prefix = current_category.split()[0] if ' ' in current_category else current_category
                            failed_category_prefixes.add(category_prefix)
                        
                        if self.config.debug_mode:
                            if self.config.experiment_type == "variable_persona":
                                print(f"    Persona {current_persona} not interested in {current_category} (max score: {max_score:.1f} < {self.config.min_score_threshold})")
                            else:
                                print(f"    Category {current_category} not relevant for persona {current_persona} (max score: {max_score:.1f} < {self.config.min_score_threshold})")
                                if self.config.experiment_type != "variable_persona":
                                    print(f"    Downweighting categories starting with '{category_prefix}'")
                        continue
                    
                    result = self._run_single_episode(episode_num, current_persona, current_category, 
                                                     cached_scores=scores, trajectory_seed=trajectory_seed,
                                                     episode_num_in_trajectory=episode_num, trajectory_num=traj_num)
                    
                    if result:
                        break
                    elif self.config.debug_mode:
                        if self.config.experiment_type == "variable_persona":
                            print(f"    Episode failed for other reasons, trying another persona...")
                        else:
                            print(f"    Episode failed for other reasons, trying another category...")
                
                if result:
                    trajectory_results.append(result)
                    self.all_results.append(result)
                    successful_count += 1
                    
                    # Track planning episode regret progression
                    if result.get('planning_mode') and 'regret_progression' in result:
                        ep_key = f"EP{episode_num}"
                        self.planning_regret_data[ep_key] = {
                            f"Q{item['question_number']}": item['regret'] 
                            for item in result['regret_progression']
                        }
                    
                    if self.config.experiment_type == "variable_category":
                        if result['category'] not in trajectory_executed_categories:
                            trajectory_executed_categories.append(result['category'])
                        if not trajectory_executed_personas:
                            trajectory_executed_personas.append(result['persona_index'])
                    elif self.config.experiment_type == "variable_persona":
                        if result['persona_index'] not in trajectory_executed_personas:
                            trajectory_executed_personas.append(result['persona_index'])
                        if not trajectory_executed_categories:
                            trajectory_executed_categories.append(result['category'])
                    else:  # variable_settings
                        if result['category'] not in trajectory_executed_categories:
                            trajectory_executed_categories.append(result['category'])
                        if result['persona_index'] not in trajectory_executed_personas:
                            trajectory_executed_personas.append(result['persona_index'])
                    
                    trajectory_tried_categories.add(result['category'])
                else:
                    if self.config.debug_mode:
                        print(f"    Episode {episode_num} failed after {max_retries} attempts")
            
            self.grouped_results[f"trajectory_{traj_num}"] = trajectory_results
            executed_categories_per_traj.append(trajectory_executed_categories)
            executed_personas_per_traj.append(trajectory_executed_personas)
        
        self.config._used_categories = executed_categories_per_traj if any(executed_categories_per_traj) else None
        self.config._used_persona_indices = executed_personas_per_traj if any(executed_personas_per_traj) else None
        
        return self._save_results()
    
    def _save_results(self) -> Dict[str, Any]:
        """Save experiment results to file."""
        episode_regrets = [r['final_info']['regret'] for r in self.all_results if 'regret' in r['final_info']]
        episode_scores = [r['final_info']['chosen_score'] for r in self.all_results if 'chosen_score' in r['final_info']]
        total_questions = sum(r.get('steps', 0) for r in self.all_results)
        
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
        
        # Always save config separately
        config_file = os.path.join(self.output_path, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict_complete(), f, indent=2)
        
        if self.config.debug_mode:
            processed_results = []
            for result in self.all_results:
                result_copy = result.copy()
                
                # Add feedback from final_info
                if 'final_info' in result and 'feedback' in result['final_info']:
                    result_copy['feedback'] = result['final_info']['feedback']
                
                # Add episode summary if in summary mode
                if self.config.context_mode == "summary" and hasattr(self.agent, 'episode_summaries'):
                    episode_idx = result.get('episode', 0) - 1  # episode numbers are 1-indexed
                    if 0 <= episode_idx < len(self.agent.episode_summaries):
                        result_copy['episode_summary'] = self.agent.episode_summaries[episode_idx]
                
                # Add internal prompts if show_internal_prompts is True
                if self.config.show_internal_prompts and hasattr(self.agent, 'episode_prompts'):
                    episode_idx = result.get('episode', 0) - 1
                    if 0 <= episode_idx < len(self.agent.episode_prompts):
                        prompts = self.agent.episode_prompts[episode_idx]
                        # Also add feedback prompt from final_info if available
                        if 'final_info' in result and 'feedback_prompt' in result['final_info']:
                            if prompts:  # prompts is a dict
                                prompts = prompts.copy()
                                prompts['feedback_generation_prompt'] = result['final_info']['feedback_prompt']
                        result_copy['internal_prompts'] = prompts
                
                # Remove product_info unless show_product_scores is True
                if not self.config.show_product_scores and 'product_info' in result_copy:
                    del result_copy['product_info']
                
                processed_results.append(result_copy)
            
            full_results_data = {
                'summary': summary,
                'results': processed_results,
                'timestamp': datetime.now().isoformat()
            }
            if self.planning_regret_data:
                full_results_data['planning_regret_progression'] = self.planning_regret_data
            
            results_file = os.path.join(self.output_path, "results.json")
            with open(results_file, 'w') as f:
                json.dump(full_results_data, f, indent=2)
            
            # 2. Aggregated metrics (results_summary.json) - same as non-debug mode
            regret_progression = self._calculate_regret_progression()
            questions_progression = self._calculate_questions_progression()
            summary_data = {
                'regret_progression': regret_progression,
                'questions_progression': questions_progression
            }
            if self.planning_regret_data:
                summary_data['planning_regret_progression'] = self.planning_regret_data
            
            summary_file = os.path.join(self.output_path, "results_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"\n{'='*70}")
            print(f"  EXPERIMENT COMPLETE")
            print(f"{'='*70}")
            print(f"Total episodes: {len(self.all_results)}")
            print(f"Average regret: {summary['avg_regret']:.2f}")
            print(f"Average score: {summary['avg_score']:.2f}")
            print(f"\nFiles saved:")
            print(f"  - Config:           {config_file}")
            print(f"  - Full results:     {results_file}")
            print(f"  - Summary metrics:  {summary_file}")
            
            return full_results_data
            
        else:
            # Non-debug mode: Save ONE aggregated results file
            regret_progression = self._calculate_regret_progression()
            questions_progression = self._calculate_questions_progression()
            results_data = {
                'regret_progression': regret_progression,
                'questions_progression': questions_progression
            }
            if self.planning_regret_data:
                results_data['planning_regret_progression'] = self.planning_regret_data
            
            results_file = os.path.join(self.output_path, "results.json")
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"\n✅ Experiment complete!")
            print(f"Files saved:")
            print(f"  - Config:   {config_file}")
            print(f"  - Results:  {results_file}")
            
            return results_data
    
    def _calculate_regret_progression(self) -> Dict[str, Any]:
        """Calculate regret progression across trajectories."""
        all_seed_data = []
        
        for traj_key, traj_results in self.grouped_results.items():
            if traj_results:
                trajectory_regrets = [r['final_info'].get('regret', 0.0) for r in traj_results]
                all_seed_data.append(trajectory_regrets)
        
        if all_seed_data:
            max_length = max(len(traj) for traj in all_seed_data)
            mean_values = []
            std_values = []
            for i in range(max_length):
                # Get values from trajectories that have an episode at position i
                values_at_i = [all_seed_data[j][i] for j in range(len(all_seed_data)) if i < len(all_seed_data[j])]
                if values_at_i:
                    mean_values.append(float(np.mean(values_at_i)))
                    std_values.append(float(np.std(values_at_i)))
                else:
                    mean_values.append(0.0)
                    std_values.append(0.0)
            
            return {'all_seed_data': all_seed_data, 'mean': mean_values, 'standard_error': std_values}
        
        return {'all_seed_data': [], 'mean': [], 'standard_error': []}
    
    def _calculate_questions_progression(self) -> Dict[str, Any]:
        """Calculate questions progression across trajectories."""
        all_seed_data = []
        
        for traj_key, traj_results in self.grouped_results.items():
            if traj_results:
                trajectory_questions = [r.get('steps', 0) for r in traj_results]
                all_seed_data.append(trajectory_questions)
        
        if all_seed_data:
            max_length = max(len(traj) for traj in all_seed_data)
            mean_values = []
            std_values = []
            for i in range(max_length):
                # Get values from trajectories that have an episode at position i
                values_at_i = [all_seed_data[j][i] for j in range(len(all_seed_data)) if i < len(all_seed_data[j])]
                if values_at_i:
                    mean_values.append(float(np.mean(values_at_i)))
                    std_values.append(float(np.std(values_at_i)))
                else:
                    mean_values.append(0.0)
                    std_values.append(0.0)
            
            return {'all_seed_data': all_seed_data, 'mean': mean_values, 'standard_error': std_values}
        
        return {'all_seed_data': [], 'mean': [], 'standard_error': []}
    
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