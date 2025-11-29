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
    
    # ===================================================================
    # CHECKPOINT METHODS
    # ===================================================================
    
    def save_checkpoint(self, trajectory_idx: int, reason: str = "periodic", trajectory_seed: Optional[int] = None) -> str:
        """
        Save checkpoint of current experiment state.
        
        Args:
            trajectory_idx: Current trajectory index (0-based)
            reason: "periodic", "trajectory_complete", "interrupt", "error"
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_dir = os.path.join(self.output_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_num = trajectory_idx + 1
        ep_num = len(self.all_results)
        seed_str = f"_seed{trajectory_seed}" if trajectory_seed is not None else ""
        checkpoint_filename = f"checkpoint_traj{traj_num:03d}_{seed_str}_{timestamp}.json"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        if reason == "trajectory_complete":
            completed_trajectories = trajectory_idx + 1
            next_trajectory_idx = trajectory_idx + 1
        else: # "periodic", "interrupt", "error"
            completed_trajectories = trajectory_idx
            next_trajectory_idx = trajectory_idx
        # Build checkpoint data
        cumulative_data = {
            "checkpoint_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            
            "config": self.config.to_dict_complete(),
            
            "progress": {
                "completed_trajectories": completed_trajectories,
                "completed_episodes": len(self.all_results),
                "next_trajectory_idx": next_trajectory_idx,
                "next_episode_num": len(self.all_results) + 1,
            },
            
            "all_results": self.all_results,
            "grouped_results": {str(k): v for k, v in self.grouped_results.items()},
            "planning_regret_data": getattr(self, 'planning_regret_data', {}),
            
            "agent_state": self._serialize_agent_state(), 
            
            "execution_history": {
                "trajectory_seeds": getattr(self, 'trajectory_seeds', []),
                "executed_categories_per_traj": getattr(self, 'executed_categories_per_traj', []),
                "executed_personas_per_traj": getattr(self, 'executed_personas_per_traj', []),
            },
            
            "stats": self._compute_checkpoint_stats()
        }
        
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.json")
        with open(latest_path, 'w') as f:
            json.dump(cumulative_data, f, indent=2)
            
        import copy
        standalone_data = copy.deepcopy(cumulative_data)
        
        traj_key = f"trajectory_{traj_num}"
        current_traj_results = self.grouped_results.get(traj_key, [])
        
        standalone_data["all_results"] = current_traj_results
        standalone_data["grouped_results"] = {traj_key: current_traj_results}
        
        standalone_data["progress"]["completed_trajectories"] = 1 if current_traj_results else 0
        standalone_data["progress"]["completed_episodes"] = len(current_traj_results)
        
        
        with open(checkpoint_path, 'w') as f:
            json.dump(standalone_data, f, indent=2)
        
        # Cleanup old checkpoints if configured
        # if self.config.checkpoint_keep_last is not None:
        #     self._cleanup_old_checkpoints(checkpoint_dir)
        
        if self.config.debug_mode:
            print(f"💾 Checkpoint saved: {checkpoint_filename} (reason: {reason})")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_file: str) -> int:
        """
        Load checkpoint and restore experiment state.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Next trajectory index to resume from (0-based)
        """
        print(f"📂 Loading checkpoint: {os.path.basename(checkpoint_file)}")
        
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Validate version
        if checkpoint_data.get("checkpoint_version") != "1.0":
            raise ValueError(f"Incompatible checkpoint version: {checkpoint_data.get('checkpoint_version')}")
        
        progress = checkpoint_data["progress"]
        reason = progress.get("reason", "trajectory_complete")
        completed_trajectories = progress["completed_trajectories"]
        next_traj_idx = progress["next_trajectory_idx"]
        
        all_results_from_file = checkpoint_data["all_results"]
        grouped_results_from_file = checkpoint_data["grouped_results"]
        start_trajectory_idx = 0
        
        if reason != "trajectory_complete":            
            episodes_in_completed_trajs = completed_trajectories * self.config.episodes_per_trajectory
            
            self.all_results = all_results_from_file[:episodes_in_completed_trajs]
            
            print(f"   Rollback: Interruption detected. Purging {len(all_results_from_file) - len(self.all_results)} orphaned episodes.")
            
            start_trajectory_idx = next_traj_idx
            
        else:
            self.all_results = all_results_from_file
            self.grouped_results = grouped_results_from_file
            start_trajectory_idx = next_traj_idx

        # self.grouped_results = checkpoint_data["grouped_results"]
        self.planning_regret_data = checkpoint_data.get("planning_regret_data", {})
        
        # Restore execution history
        exec_history = checkpoint_data.get("execution_history", {})
        self.trajectory_seeds = exec_history.get("trajectory_seeds", [])
        self.executed_categories_per_traj = exec_history.get("executed_categories_per_traj", [])
        self.executed_personas_per_traj = exec_history.get("executed_personas_per_traj", [])
        
        print(f"✅ Checkpoint loaded:")
        print(f"   Timestamp: {checkpoint_data['timestamp']}")
        print(f"   Episodes completed (after rollback): {len(self.all_results)}")
        print(f"   Trajectories completed: {completed_trajectories}")
        print(f"   Resuming from trajectory {start_trajectory_idx + 1}")
        
        return start_trajectory_idx
    
    def _serialize_agent_state(self) -> dict:
        """Extract serializable agent state."""
        if not self.agent:
            return {}
        
        return {
            "episode_history": self.agent.episode_history,
            "episode_summaries": self.agent.episode_summaries,
            "episode_prompts": getattr(self.agent, 'episode_prompts', []),
            "episode_count": self.agent.episode_count,
        }
    
    def _restore_agent_state(self, agent_state: dict):
        """Restore agent state from checkpoint."""
        if not self.agent:
            self.agent = self._create_agent()
        
        self.agent.episode_history = agent_state.get("episode_history", [])
        self.agent.episode_summaries = agent_state.get("episode_summaries", [])
        self.agent.episode_prompts = agent_state.get("episode_prompts", [])
        self.agent.episode_count = agent_state.get("episode_count", 0)
    
    def _compute_checkpoint_stats(self) -> dict:
        """Compute summary statistics for checkpoint."""
        if not self.all_results:
            return {}
        
        regrets = [r['final_info']['regret'] for r in self.all_results if 'regret' in r['final_info']]
        scores = [r['final_info']['chosen_score'] for r in self.all_results if 'chosen_score' in r['final_info']]
        
        return {
            "avg_regret": float(np.mean(regrets)) if regrets else 0.0,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "total_questions_asked": sum(r.get('steps', 0) for r in self.all_results),
            "top1_rate": float(np.mean([r['final_info'].get('top1', False) for r in self.all_results])),
        }
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: str):
        """Remove old checkpoints, keeping only the most recent N."""
        import glob
        
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_traj*.json"))
        
        if len(checkpoint_files) <= self.config.checkpoint_keep_last:
            return
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # Delete old ones
        for file_path in checkpoint_files[self.config.checkpoint_keep_last:]:
            try:
                os.remove(file_path)
            except Exception as e:
                if self.config.debug_mode:
                    print(f"   Warning: Could not delete old checkpoint: {e}")
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file."""
        checkpoint_dir = os.path.join(self.output_path, "checkpoints")
        
        if not os.path.exists(checkpoint_dir):
            return None
        
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.json")
        if os.path.exists(latest_path):
            return latest_path
        
        return None
    
    # ===================================================================
    # END CHECKPOINT METHODS
    # ===================================================================
    
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
                return max_score >= self.config.min_score_threshold, max_score, scores
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
        if episode_num_in_trajectory == 1:
            return True  
        if self.config.planning_interval <= 0:
            return False           
        return (episode_num_in_trajectory % self.config.planning_interval == 0)
    
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
                    'trajectory': f"trajectory_{trajectory_num}" if trajectory_num else "unknown",
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
        1. Ask Question 1 → Get intermediate recommendation + confidence → Track metrics (hidden from agent)
        2. Ask Question 2 → Get intermediate recommendation + confidence → Track metrics (hidden from agent)
        ...
        N. Ask Question N → Get final recommendation → Track metrics + Give feedback
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
            
            filename = f"trajectory_{trajectory_num}_episode_{episode_num}_planning.jsonl" if trajectory_num else f"episode_{episode_num}_planning.jsonl"
            metrics_wrapper = MetricsWrapper(env, output_path=os.path.join(self.output_path, filename))
            obs, info = metrics_wrapper.reset()
            self.agent.current_env = env
            
            regret_progression = []
            rank_progression = []
            confidence_progression = []
            
            question_count = 0
            terminated = False
            truncated = False
            num_products = len(env.products) if hasattr(env, 'products') else 0
            
            # Planning mode: Force agent to ask max_questions
            while question_count < self.config.max_questions and not terminated and not truncated:
                # 1. AGENT ASKS A QUESTION (Forced)
                action_ask = num_products
                self.agent.set_tracking_episode(True) 
                _ = self.agent.get_action(obs, info) 
                self.agent.set_tracking_episode(False)
                
                # 2. ENVIRONMENT RESPONDS
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action_ask)
                question_count += 1
                
                if self.config.debug_mode:
                    print(f"  Step {question_count}: Asked question")
                
                # --- START: NEW METRIC LOGGING BLOCK ---
                
                # 3. GET INTERMEDIATE RECOMMENDATION AND BELIEFS
                intermediate_recommendation_idx, current_belief_state = self.agent._make_recommendation_with_confidence(
                    obs, info, env.dialog_history, category, num_products
                )
                confidence_progression.append(current_belief_state)

                # 4. GET GROUND-TRUTH (FACTS) FOR THIS RECOMMENDATION
                best_id, best_score = env.oracle_scores[0] if hasattr(env, 'oracle_scores') and env.oracle_scores else (None, 0)
                recommended_product_id = -1
                recommended_score = 0.0
                intermediate_regret = best_score 
                intermediate_rank = -1

                if intermediate_recommendation_idx is not None and hasattr(env, 'products') and intermediate_recommendation_idx < len(env.products):
                    try:
                        recommended_product_id = env.products[intermediate_recommendation_idx]['id']
                        for i, (pid, score) in enumerate(env.oracle_scores):
                            if pid == recommended_product_id:
                                recommended_score = score
                                intermediate_rank = i + 1 # 1-indexed rank
                                break
                        intermediate_regret = max(0.0, best_score - recommended_score)
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"    [WARN] Error calculating intermediate stats: {e}")

                # 5. LOG GROUND-TRUTH (FACTS)
                regret_progression.append({
                    'question_number': question_count,
                    'recommended_product_id': recommended_product_id,
                    'recommended_score': recommended_score,
                    'best_score': best_score,
                    'regret': intermediate_regret # Factual Regret
                })
                
                rank_progression.append({
                    'question_number': question_count,
                    'recommended_product_id': recommended_product_id,
                    'actual_rank': intermediate_rank # Factual Rank
                })
                
                # 6. PRINT LOG (FACTS vs BELIEFS)
                if self.config.debug_mode:
                    print(f"    → Hidden Rec (Q{question_count}): Rank {intermediate_rank}, Fact Regret: {intermediate_regret:.1f}")
                    if current_belief_state:
                         print(f"    → Agent Belief: Predicted Regret: {current_belief_state.get('predicted_regret', 0):.1f}, Conf (Fav): {current_belief_state.get('confidence_favorite_prob', 0):.1%}")


            # After max_questions, get FINAL recommendation
            final_action_idx, final_beliefs = self.agent._make_recommendation_with_confidence(
                obs, info, env.dialog_history, category, num_products
            )
            
            # Execute final recommendation and give feedback
            obs, reward, terminated, truncated, info = metrics_wrapper.step(final_action_idx)
            
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
                
                regret_progression.append({
                    'question_number': question_count + 1, # (Final rec step)
                    'recommended_product_id': info.get('chosen_product_id'),
                    'recommended_score': info.get('chosen_score'),
                    'best_score': info.get('best_score'),
                    'regret': info.get('regret', 0) 
                })
                
                rank_progression.append({
                    'question_number': question_count + 1,
                    'recommended_product_id': info.get('chosen_product_id'),
                    'actual_rank': info.get('Actual Rank', -1) 
                })
                
                confidence_progression.append(final_beliefs)

                episode_result = {
                    'episode': episode_num,
                    'trajectory': f"trajectory_{trajectory_num}" if trajectory_num else "unknown",
                    'category': category,
                    'persona_index': persona_index,
                    'steps': question_count + 1,  # All questions + final recommendation
                    'terminated': terminated,
                    'truncated': truncated,
                    'final_info': info,
                    'full_dialog': full_dialog,
                    'product_info': product_info,
                    'planning_mode': True,
                    'regret_progression': regret_progression,
                    'rank_progression': rank_progression,
                    'confidence_progression': confidence_progression
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
    
    def run_interactive(self) -> Dict[str, Any]:
        """
        Run in interactive mode: Generate multiple variants of the same episode
        for manual curation to build a golden trajectory.
        
        Workflow:
        1. If no input file: Generate N variants of Episode 1
        2. If input file provided: Load it, extract settings, generate N variants of next episode
        3. Save each variant to separate file for manual review
        4. User picks their favorite and re-runs with that file as input
        """
        print(f"\n{'='*70}")
        print(f"  INTERACTIVE EPISODE MODE")
        print(f"{'='*70}")
        
        base_seed = self.config.get_seeds()[0]
        random.seed(base_seed)
        np.random.seed(base_seed)
        
        # Setup output directory
        self.output_path = self.config.get_output_path()
        os.makedirs(self.output_path, exist_ok=True)
        
        output_base = os.path.join(self.output_path, self.config.interactive_output_dir)
        os.makedirs(output_base, exist_ok=True)
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Determine episode context
        previous_episode = None  # Initialize for scope
        
        if self.config.interactive_input_file is None:
            # Starting Episode 1 - need to find a relevant persona/category combo
            episode_num = 1
            
            print(f"\n🚀 Starting new trajectory - Finding relevant persona/category combination...")
            
            # Get available personas and categories
            from pipeline.core.simulate_interaction import list_categories
            from pipeline.core.personas import get_persona_description
            
            all_categories = list_categories()
            
            # Get max persona index
            max_persona_index = 0
            while True:
                try:
                    get_persona_description(max_persona_index)
                    max_persona_index += 1
                except:
                    break
            all_personas = list(range(max_persona_index))
            
            # Try to find a relevant combo (like normal experiment retry logic)
            max_retries = 50
            persona_id = None
            category = None
            scores = None
            
            for retry in range(max_retries):
                # Sample a persona and category
                test_persona = random.choice(all_personas)
                test_category = random.choice(all_categories)
                
                # Check if relevant
                is_relevant, max_score, test_scores = self._is_category_relevant_for_persona(
                    test_category, test_persona, seed=self.config.get_seeds()[0]
                )
                
                if is_relevant:
                    persona_id = test_persona
                    category = test_category
                    scores = test_scores
                    print(f"   ✅ Found match after {retry + 1} attempt(s)")
                    print(f"   Persona: {persona_id}")
                    print(f"   Category: {category}")
                    print(f"   Max score: {max_score:.1f}")
                    break
            
            if persona_id is None:
                raise ValueError(
                    f"Could not find relevant persona/category combination after {max_retries} attempts. "
                    f"Try lowering min_score_threshold (current: {self.config.min_score_threshold})"
                )
            
            print(f"\n   Generating {self.config.interactive_variants} variants of Episode 1...")
            
        else:
            # Continuing from previous episode
            print(f"\n📂 Loading previous episode from: {self.config.interactive_input_file}")
            previous_episode = self._load_variant_file(self.config.interactive_input_file)
            
            episode_num = previous_episode['episode'] + 1
            
            # Restore agent state
            self.agent.update_preferences(previous_episode)
            
            print(f"   Continuing to Episode {episode_num}")
            
            # Determine what varies based on experiment_type
            from pipeline.core.simulate_interaction import list_categories
            from pipeline.core.personas import get_persona_description
            
            if self.config.experiment_type == "variable_persona":
                # Keep category constant, find new persona
                category = previous_episode['category']
                print(f"   Category: {category} (unchanged)")
                print(f"   Finding new persona...")
                
                # Get all personas
                max_persona_index = 0
                while True:
                    try:
                        get_persona_description(max_persona_index)
                        max_persona_index += 1
                    except:
                        break
                all_personas = list(range(max_persona_index))
                
                # Try to find a relevant persona
                max_retries = 50
                persona_id = None
                scores = None
                
                for retry in range(max_retries):
                    test_persona = random.choice(all_personas)
                    is_relevant, max_score, test_scores = self._is_category_relevant_for_persona(
                        category, test_persona, seed=self.config.get_seeds()[0]
                    )
                    
                    if is_relevant:
                        persona_id = test_persona
                        scores = test_scores
                        print(f"   ✅ Found persona {persona_id} after {retry + 1} attempt(s) (score: {max_score:.1f})")
                        break
                
                if persona_id is None:
                    raise ValueError(
                        f"Could not find relevant persona for category '{category}' after {max_retries} attempts"
                    )
                    
            elif self.config.experiment_type == "variable_category":
                # Keep persona constant, find new category
                persona_id = previous_episode['persona_index']
                print(f"   Persona: {persona_id} (unchanged)")
                print(f"   Finding new category...")
                
                all_categories = list_categories()
                
                # Try to find a relevant category
                max_retries = 50
                category = None
                scores = None
                
                for retry in range(max_retries):
                    test_category = random.choice(all_categories)
                    is_relevant, max_score, test_scores = self._is_category_relevant_for_persona(
                        test_category, persona_id, seed=self.config.get_seeds()[0]
                    )
                    
                    if is_relevant:
                        category = test_category
                        scores = test_scores
                        print(f"   ✅ Found category '{category}' after {retry + 1} attempt(s) (score: {max_score:.1f})")
                        break
                
                if category is None:
                    raise ValueError(
                        f"Could not find relevant category for persona {persona_id} after {max_retries} attempts"
                    )
                    
            else:  # variable_settings
                # Both vary - find new persona AND category
                print(f"   Finding new persona and category...")
                
                all_categories = list_categories()
                max_persona_index = 0
                while True:
                    try:
                        get_persona_description(max_persona_index)
                        max_persona_index += 1
                    except:
                        break
                all_personas = list(range(max_persona_index))
                
                # Try to find a relevant combo
                max_retries = 50
                persona_id = None
                category = None
                scores = None
                
                for retry in range(max_retries):
                    test_persona = random.choice(all_personas)
                    test_category = random.choice(all_categories)
                    is_relevant, max_score, test_scores = self._is_category_relevant_for_persona(
                        test_category, test_persona, seed=self.config.get_seeds()[0]
                    )
                    
                    if is_relevant:
                        persona_id = test_persona
                        category = test_category
                        scores = test_scores
                        print(f"   ✅ Found match after {retry + 1} attempt(s)")
                        print(f"      Persona: {persona_id}, Category: {category} (score: {max_score:.1f})")
                        break
                
                if persona_id is None:
                    raise ValueError(
                        f"Could not find relevant persona/category combination after {max_retries} attempts"
                    )
            
            print(f"   Generating {self.config.interactive_variants} variants...")
        
        # Generate N variants
        variants = []
        base_seed = self.config.get_seeds()[0]
        
        print(f"\n{'='*70}")
        print(f"  GENERATING VARIANTS")
        print(f"{'='*70}\n")
        
        for variant_idx in range(self.config.interactive_variants):
            # Use different seed for each variant
            variant_seed = base_seed + variant_idx
            random.seed(variant_seed)
            np.random.seed(variant_seed)
            
            print(f"  Variant {variant_idx + 1}/{self.config.interactive_variants}...")
            
            # Reset agent for this variant (start fresh)
            # But restore memory from previous episode if continuing
            if self.config.interactive_input_file is not None:
                # Recreate agent and restore state from previous episode
                self.agent = self._create_agent()
                self.agent.update_preferences(previous_episode)
            else:
                # Fresh agent for Episode 1
                self.agent = self._create_agent()
            
            # Run episode
            result = self._run_regular_episode(
                episode_num=episode_num,
                persona_index=persona_id,
                category=category,
                cached_scores=scores,
                trajectory_seed=variant_seed,
                trajectory_num=1
            )
            
            if result:
                # Add variant metadata
                result['variant_id'] = variant_idx + 1
                result['variant_seed'] = variant_seed
                variants.append(result)
                
                # Save variant to file
                variant_filename = f"episode_{episode_num:02d}_variant_{variant_idx + 1:03d}.json"
                variant_path = os.path.join(output_base, variant_filename)
                self._save_variant_file(variant_path, result)
                
                print(f"    ✅ Saved: {variant_filename} (regret: {result['final_info'].get('regret', 0):.1f})")
            else:
                print(f"    ❌ Failed to generate variant {variant_idx + 1}")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"  GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nGenerated {len(variants)}/{self.config.interactive_variants} variants of Episode {episode_num}")
        print(f"Location: {output_base}/")
        print(f"\n📋 Next Steps:")
        print(f"   1. Review variant files: episode_{episode_num:02d}_variant_001.json ... episode_{episode_num:02d}_variant_{self.config.interactive_variants:03d}.json")
        print(f"   2. Delete variants you don't like")
        print(f"   3. Pick your favorite variant")
        
        if episode_num < self.config.episodes_per_trajectory:
            print(f"   4. Re-run with: --interactive_input_file {output_base}/episode_{episode_num:02d}_variant_XXX.json")
            print(f"      (This will generate Episode {episode_num + 1} variants)")
        else:
            print(f"   4. Trajectory complete! ({self.config.episodes_per_trajectory} episodes)")
        
        print()
        
        # Return summary
        return {
            'mode': 'interactive',
            'episode_num': episode_num,
            'variants_generated': len(variants),
            'output_directory': output_base,
            'variants': variants
        }
    
    def _get_persona_for_episode(self, episode_num: int) -> int:
        """Get persona ID for an episode in interactive mode."""
        personas_list = self.config.get_persona_indices()
        if personas_list and len(personas_list) > 0:
            # Use first trajectory's first persona
            if len(personas_list[0]) > 0:
                return personas_list[0][0]
        
        # Fallback: random persona
        from pipeline.core.personas import get_persona_description
        max_persona_index = 0
        while True:
            try:
                get_persona_description(max_persona_index)
                max_persona_index += 1
            except:
                break
        return random.randint(0, max_persona_index - 1)
    
    def _get_category_for_episode(self, episode_num: int) -> str:
        """Get category for an episode in interactive mode."""
        categories_list = self.config.get_categories()
        if categories_list and len(categories_list) > 0:
            # Use first trajectory's first category
            if len(categories_list[0]) > 0:
                return categories_list[0][0]
        
        # Fallback: random category
        from pipeline.core.simulate_interaction import list_categories
        all_categories = list_categories()
        return random.choice(all_categories)
    
    def _load_variant_file(self, filepath: str) -> Dict[str, Any]:
        """Load a variant file from previous episode."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    
    def _save_variant_file(self, filepath: str, variant: Dict[str, Any]):
        """Save a single variant to file."""
        # Save with nice formatting
        with open(filepath, 'w') as f:
            json.dump(variant, f, indent=2)
    
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
        
        # Route to interactive mode if enabled
        if self.config.interactive_mode:
            return self.run_interactive()
        
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
        # self.agent = self._create_agent()
        
        # === CHECKPOINT LOADING ===
        start_trajectory_idx = 0
        if self.config.checkpoint_enabled and self.config.resume_from_checkpoint:
            if os.path.exists(self.config.resume_from_checkpoint):
                start_trajectory_idx = self.load_checkpoint(self.config.resume_from_checkpoint)
            else:
                print(f"⚠️  Warning: Checkpoint file not found: {self.config.resume_from_checkpoint}")
                print("   Starting from beginning...")
        
        executed_categories_per_traj = []
        executed_personas_per_traj = []
        trajectories = self._plan_trajectories()
        
        if self.config.debug_mode:
            print(f"\nExperiment Plan:")
            print(f"  Total trajectories: {len(trajectories)}")
            print(f"  Episodes per trajectory: {self.config.episodes_per_trajectory}")
            print(f"  Total episodes: {len(trajectories) * self.config.episodes_per_trajectory}")
        
        # Skip completed trajectories if resuming
        if start_trajectory_idx > 0:
            print(f"\n⏭️  Skipping {start_trajectory_idx} completed trajectories")
            trajectories = trajectories[start_trajectory_idx:]
        
        self.all_results = self.all_results if start_trajectory_idx > 0 else []
        self.grouped_results = self.grouped_results if start_trajectory_idx > 0 else {}
        self.planning_regret_data = getattr(self, 'planning_regret_data', {})
        self.executed_categories_per_traj = getattr(self, 'executed_categories_per_traj', [])
        self.executed_personas_per_traj = getattr(self, 'executed_personas_per_traj', [])
        successful_count = 0
        
        try:
            for traj_idx, (traj_num, categories, personas, episode_plan) in enumerate(trajectories):
                actual_traj_idx = start_trajectory_idx + traj_idx
                self.agent = self._create_agent()
                
                if hasattr(self, 'restored_agent_state') and self.restored_agent_state:
                    
                    resume_traj_idx = getattr(self, 'resume_progress', {}).get('next_trajectory_idx', 0)
                    resume_reason = getattr(self, 'resume_progress', {}).get('reason', 'trajectory_complete')
                    
                    
                    
                    if actual_traj_idx == resume_traj_idx and resume_reason != "trajectory_complete":
                        print(f"   Resuming agent state for in-progress trajectory {actual_traj_idx + 1}...")
                        self._restore_agent_state(self.restored_agent_state)
                    
                    self.restored_agent_state = None
                    self.resume_progress = None
                # Adjust trajectory index if resuming
                # actual_traj_idx = start_trajectory_idx + traj_idx
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
                        
                        # === PERIODIC CHECKPOINT ===
                        if (self.config.checkpoint_enabled and 
                            self.config.checkpoint_every_n_episodes and
                            len(self.all_results) % self.config.checkpoint_every_n_episodes == 0):
                            self.save_checkpoint(trajectory_idx=actual_traj_idx, reason="periodic",trajectory_seed=trajectory_seed)
                        
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
                
                # End of episode loop - save trajectory results
                self.grouped_results[f"trajectory_{traj_num}"] = trajectory_results
                executed_categories_per_traj.append(trajectory_executed_categories)
                executed_personas_per_traj.append(trajectory_executed_personas)
                
                # === TRAJECTORY CHECKPOINT ===
                if self.config.checkpoint_enabled and self.config.checkpoint_after_each_trajectory:
                    self.save_checkpoint(trajectory_idx=actual_traj_idx, reason="trajectory_complete",trajectory_seed=trajectory_seed)
            
            # All trajectories complete - save results
            self.config._used_categories = executed_categories_per_traj if any(executed_categories_per_traj) else None
            self.config._used_persona_indices = executed_personas_per_traj if any(executed_personas_per_traj) else None
            
            return self._save_results()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Experiment interrupted by user (Ctrl+C)")
            
            if self.config.checkpoint_enabled and self.config.checkpoint_on_interrupt and len(self.all_results) > 0:
                print("💾 Saving interrupt checkpoint...")
                # actual_traj_idx = start_trajectory_idx + len(self.grouped_results) - 1
                checkpoint_path = self.save_checkpoint(trajectory_idx=actual_traj_idx, reason="interrupt",trajectory_seed=trajectory_seed)
                print(f"✅ Progress saved to: {os.path.basename(checkpoint_path)}")
                print(f"\nTo resume, add to your config:")
                print(f"  resume_from_checkpoint: {checkpoint_path}")
            
            raise
        
        except Exception as e:
            print(f"\n❌ Experiment failed with error: {e}")
            
            # Emergency checkpoint on crash
            if self.config.checkpoint_enabled and len(self.all_results) > 0:
                print("💾 Saving emergency checkpoint...")
                try:
                    # actual_traj_idx = start_trajectory_idx + len(self.grouped_results) - 1
                    checkpoint_path = self.save_checkpoint(trajectory_idx=max(0, actual_traj_idx), reason="error",trajectory_seed=trajectory_seed)
                    print(f"✅ Emergency checkpoint saved: {os.path.basename(checkpoint_path)}")
                except Exception as save_error:
                    print(f"⚠️  Could not save emergency checkpoint: {save_error}")
            
            raise
    
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
        token_usage_stats = None
        try:
            if hasattr(llm_providers, 'get_total_usage_stats'):
                token_usage_stats = llm_providers.get_total_usage_stats()
                summary['token_usage'] = token_usage_stats  
        except Exception as e:
            if self.config.debug_mode:
                print(f"   Warning: Could not retrieve token usage: {e}")
        
        
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
                'timestamp': datetime.now().isoformat(),
                'config_file_path': self.config.config_file_path  # Reference to source config file
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
                'experiment_type': self.config.experiment_type,
                'model': self.config.model,
                'context_mode': self.config.context_mode,
                'feedback_type': self.config.feedback_type,
                'prompting_tricks': self.config.prompting_tricks,
                'config_file_path': self.config.config_file_path,  # Reference to source config file
                'regret_progression': regret_progression,
                'questions_progression': questions_progression
            }
            if self.planning_regret_data:
                summary_data['planning_regret_progression'] = self.planning_regret_data
            if token_usage_stats:
                summary_data['token_usage'] = token_usage_stats
            summary_file = os.path.join(self.output_path, "results_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"\n{'='*70}")
            print(f"  EXPERIMENT COMPLETE")
            print(f"{'='*70}")
            print(f"Total episodes: {len(self.all_results)}")
            print(f"Average regret: {summary['avg_regret']:.2f}")
            print(f"Average score: {summary['avg_score']:.2f}")
            
            if token_usage_stats:
                print(f"Token Usage (Input):  {token_usage_stats['input_tokens']}")
                print(f"Token Usage (Output): {token_usage_stats['output_tokens']}")
                
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
                'experiment_type': self.config.experiment_type,
                'model': self.config.model,
                'context_mode': self.config.context_mode,
                'feedback_type': self.config.feedback_type,
                'prompting_tricks': self.config.prompting_tricks,
                'config_file_path': self.config.config_file_path,  # Reference to source config file
                'regret_progression': regret_progression,
                'questions_progression': questions_progression
            }
            if self.planning_regret_data:
                results_data['planning_regret_progression'] = self.planning_regret_data
            if token_usage_stats:
                results_data['token_usage'] = token_usage_stats
            results_file = os.path.join(self.output_path, "results.json")
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"\n✅ Experiment complete!")
            
            # if token_usage_stats:
            print(f"Token Usage (Input):  {token_usage_stats['input_tokens']}")
            print(f"Token Usage (Output): {token_usage_stats['output_tokens']}")
                
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