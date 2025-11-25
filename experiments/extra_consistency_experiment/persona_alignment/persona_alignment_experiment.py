#!/usr/bin/env python3
"""
Persona Alignment Experiment

Temporary experiment to measure alignment between persona agent responses
and persona descriptions throughout trajectories.

This module will be deleted after the experiment is complete.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from experiments.unified_experiment import UnifiedExperiment
from pipeline.envs.reco_env import RecoEnv
from pipeline.core.user_model import UserModel
from pipeline.core.feedback_system import FeedbackSystem
from pipeline.wrappers.metrics_wrapper import MetricsWrapper
from .persona_alignment_evaluator import PersonaAlignmentEvaluator


class AlignmentTrackingRecoEnv(RecoEnv):
    """
    Wrapper around RecoEnv that tracks alignment after each question.
    Non-intrusive - only adds tracking, doesn't change behavior.
    """
    
    def __init__(self, alignment_evaluator: PersonaAlignmentEvaluator, 
                 persona_index: int, alignment_callback, *args, **kwargs):
        """
        Initialize with alignment tracking.
        
        Args:
            alignment_evaluator: PersonaAlignmentEvaluator instance
            persona_index: Persona index for this episode
            alignment_callback: Function to call with alignment results
            *args, **kwargs: Passed to RecoEnv
        """
        # Ensure persona_index is passed to parent RecoEnv
        kwargs['persona_index'] = persona_index
        super().__init__(*args, **kwargs)
        self.alignment_evaluator = alignment_evaluator
        self.persona_index = persona_index
        self.alignment_callback = alignment_callback
        self.persona_description = None  # Will be set after user_model is created
        
    def step(self, action: int):
        """Override step to track alignment after questions."""
        # Get persona description if not set (use parent's user_model which now has correct persona_index)
        if self.persona_description is None:
            if hasattr(self, 'user_model') and self.user_model:
                self.persona_description = self.user_model.get_persona_text()
            else:
                # Fallback: create UserModel to get description
                from pipeline.core.user_model import UserModel
                user_model = UserModel(self.persona_index)
                self.persona_description = user_model.get_persona_text()
        
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # If a question was just asked and answered, evaluate alignment
        if info.get('action_type') == 'ask':
            question_text = info.get('question_text', '')
            answer = info.get('answer', '')
            
            # Evaluate alignment (only needs persona description and answer)
            # Only evaluate if we have both persona description and a non-empty answer
            if self.persona_description and self.persona_description.strip() and answer and answer.strip():
                alignment_result = self.alignment_evaluator.evaluate_alignment(
                    persona_description=self.persona_description,
                    answer=answer
                )
                
                # Call callback with alignment data
                if self.alignment_callback:
                    self.alignment_callback(
                        question=question_text,
                        answer=answer,
                        alignment_result=alignment_result,
                        question_num=len(self.dialog_history),
                        persona_index=self.persona_index,
                        category=self.current_category
                    )
        
        return obs, reward, terminated, truncated, info


class PersonaAlignmentExperiment:
    """
    Experiment runner that measures persona alignment throughout trajectories.
    Wraps UnifiedExperiment to add alignment tracking.
    """
    
    def __init__(self, config, evaluator_model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the alignment experiment.
        
        Args:
            config: ExperimentConfig instance (same as UnifiedExperiment)
            evaluator_model: Claude model to use for alignment evaluation
        """
        self.config = config
        self.base_experiment = UnifiedExperiment(config)
        self.alignment_evaluator = PersonaAlignmentEvaluator(model=evaluator_model)
        
        # Alignment data storage
        self.alignment_data = []  # List of alignment results
        self.current_trajectory_data = None
        self.current_episode_data = None
        
        # Output directory
        self.output_dir = "persona_alignment_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _on_alignment_check(self, question: str, answer: str, alignment_result: dict,
                           question_num: int, persona_index: int, category: str):
        """Callback when alignment is evaluated after a question."""
        if self.current_episode_data is None:
            return
        
        alignment_check = {
            'question_num': question_num,
            'question': question,
            'answer': answer,
            'alignment': alignment_result['alignment'],
            'evaluator_response': alignment_result['evaluator_response'],
            'timestamp': alignment_result['timestamp']
        }
        
        self.current_episode_data['alignment_checks'].append(alignment_check)
    
    def _run_episode_with_alignment_tracking(
        self, episode_num: int, persona_index: int, category: str,
        cached_scores: Optional[List[Tuple[int, float]]] = None,
        trajectory_seed: Optional[int] = None,
        trajectory_num: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single episode with alignment tracking.
        Based on UnifiedExperiment._run_regular_episode but with alignment tracking.
        """
        if self.config.debug_mode:
            print(f"\nEpisode {episode_num}: Persona #{persona_index}, Category: {category}")
        
        # Initialize episode data
        self.current_episode_data = {
            'episode_num': episode_num,
            'alignment_checks': []
        }
        
        try:
            # Create feedback system
            if self.config.feedback_type == "persona":
                persona_agent = UserModel(persona_index)
                feedback_system = FeedbackSystem(feedback_type="persona", persona_agent=persona_agent)
            else:
                feedback_system = FeedbackSystem(feedback_type=self.config.feedback_type)
            
            # Create environment with alignment tracking
            env = AlignmentTrackingRecoEnv(
                alignment_evaluator=self.alignment_evaluator,
                persona_index=persona_index,
                alignment_callback=self._on_alignment_check,
                max_questions=self.config.max_questions,
                categories=[category],
                agent=self.base_experiment.agent,
                feedback_system=feedback_system,
                cached_scores=cached_scores,
                max_products_per_category=self.config.max_products_per_category,
                seed=trajectory_seed,
                debug_mode=self.config.debug_mode
            )
            
            # Persona description will be retrieved from env.user_model in step() method
            # No need to set it here since parent RecoEnv now has correct user_model
            
            # Create metrics wrapper
            filename = f"trajectory_{trajectory_num}_episode_{episode_num}.jsonl" if trajectory_num else f"episode_{episode_num}.jsonl"
            metrics_wrapper = MetricsWrapper(env, output_path=os.path.join(self.base_experiment.output_path, filename))
            
            obs, info = metrics_wrapper.reset()
            self.base_experiment.agent.current_env = env
            
            if not self.config.debug_mode:
                print(f"    → Running episode...")
            
            terminated = False
            truncated = False
            step_count = 0
            current_info = info
            
            while not terminated and not truncated and step_count <= 20:
                action = self.base_experiment.agent.get_action(obs, current_info)
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
            
            # Store episode data
            if self.current_trajectory_data:
                self.current_trajectory_data['episodes'].append(self.current_episode_data)
            
            # Return standard episode result (for compatibility)
            full_dialog = env.dialog_history if hasattr(env, 'dialog_history') else []
            
            episode_result = {
                'episode': episode_num,
                'trajectory': f"trajectory_{trajectory_num}" if trajectory_num else "unknown",
                'category': category,
                'persona_index': persona_index,
                'steps': step_count,
                'terminated': terminated,
                'truncated': truncated,
                'final_info': info,
                'full_dialog': full_dialog
            }
            
            self.current_episode_data = None
            return episode_result
            
        except Exception as e:
            print(f"Error in episode {episode_num}: {e}")
            if self.config.debug_mode:
                import traceback
                traceback.print_exc()
            
            # Store partial episode data if we have any alignment checks
            if self.current_episode_data and self.current_trajectory_data:
                if len(self.current_episode_data.get('alignment_checks', [])) > 0:
                    self.current_trajectory_data['episodes'].append(self.current_episode_data)
            
            self.current_episode_data = None
            return None
    
    def run(self):
        """Run the full alignment experiment."""
        print(f"\n{'='*70}")
        print(f"  PERSONA ALIGNMENT EXPERIMENT")
        print(f"{'='*70}")
        print(f"Model: {self.config.model}")
        print(f"Evaluator: {self.alignment_evaluator.model}")
        print(f"Experiment Type: {self.config.experiment_type}")
        print(f"Total Trajectories: {self.config.total_trajectories}")
        print(f"Episodes per Trajectory: {self.config.episodes_per_trajectory}")
        print(f"{'='*70}\n")
        
        # Initialize agent
        self.base_experiment.agent = self.base_experiment._create_agent()
        self.base_experiment.output_path = self.config.get_output_path()
        os.makedirs(self.base_experiment.output_path, exist_ok=True)
        
        # Set trajectory seeds (required before _plan_trajectories)
        self.base_experiment.trajectory_seeds = self.config.get_seeds()
        
        # Generate trajectories
        trajectories = self.base_experiment._plan_trajectories()
        
        # Get seeds
        seeds = self.config.get_seeds()
        
        # Run each trajectory
        for traj_idx, (traj_num, traj_categories, traj_personas, episode_plan) in enumerate(trajectories):
            print(f"\n{'='*70}")
            print(f"  TRAJECTORY {traj_num} / {len(trajectories)}")
            print(f"{'='*70}")
            
            # Initialize trajectory data
            self.current_trajectory_data = {
                'trajectory_num': traj_num,
                'persona_indices': traj_personas,
                'categories': traj_categories,
                'episodes': []
            }
            
            trajectory_seed = seeds[traj_idx] if traj_idx < len(seeds) else None
            
            # Run each episode in trajectory
            for episode_num, persona_index, category in episode_plan:
                # Check if category is relevant for persona (min_score_threshold check)
                is_relevant, max_score, test_scores = self.base_experiment._is_category_relevant_for_persona(
                    category, persona_index, seed=trajectory_seed
                )
                
                if not is_relevant:
                    if self.config.debug_mode:
                        print(f"  Episode {episode_num}: Skipping - Category {category} not relevant for persona {persona_index}")
                        print(f"    Max score: {max_score:.1f} < threshold: {self.config.min_score_threshold}")
                    continue
                
                # Get cached scores if available
                cached_scores = None
                if test_scores:
                    # Use scores from relevance check
                    cached_scores = [(pid, score) for pid, score in test_scores]
                else:
                    try:
                        from pipeline.core.simulate_interaction import load_cached_scores, get_products_by_category
                        products = get_products_by_category(
                            category, 
                            limit=self.config.max_products_per_category, 
                            seed=trajectory_seed
                        )
                        if products:
                            product_ids = [p.get('id') for p in products if p.get('id') is not None]
                            cached_pairs = load_cached_scores(persona_index, category, product_ids)
                            cached_scores = cached_pairs if cached_pairs else None
                    except:
                        cached_scores = None
                
                # Run episode
                self._run_episode_with_alignment_tracking(
                    episode_num=episode_num,
                    persona_index=persona_index,
                    category=category,
                    cached_scores=cached_scores,
                    trajectory_seed=trajectory_seed,
                    trajectory_num=traj_num
                )
            
            # Calculate summary for trajectory
            all_checks = []
            for ep in self.current_trajectory_data['episodes']:
                all_checks.extend(ep['alignment_checks'])
            
            aligned_count = sum(1 for c in all_checks if c['alignment'] == 'Aligned')
            conflict_count = sum(1 for c in all_checks if c['alignment'] == 'Conflict')
            neutral_count = sum(1 for c in all_checks if c['alignment'] == 'Neutral')
            total_questions = len(all_checks)
            
            self.current_trajectory_data['summary'] = {
                'total_questions': total_questions,
                'aligned_count': aligned_count,
                'conflict_count': conflict_count,
                'neutral_count': neutral_count,
                'alignment_rate': aligned_count / total_questions if total_questions > 0 else 0.0
            }
            
            # Save trajectory results
            output_file = os.path.join(self.output_dir, f"alignment_trajectory_{traj_num}_results.json")
            with open(output_file, 'w') as f:
                json.dump(self.current_trajectory_data, f, indent=2)
            
            print(f"\n  Trajectory {traj_num} complete:")
            print(f"    Total Questions: {total_questions}")
            print(f"    Aligned: {aligned_count} ({aligned_count/total_questions*100:.1f}%)" if total_questions > 0 else "    Aligned: 0")
            print(f"    Conflict: {conflict_count} ({conflict_count/total_questions*100:.1f}%)" if total_questions > 0 else "    Conflict: 0")
            print(f"    Neutral: {neutral_count} ({neutral_count/total_questions*100:.1f}%)" if total_questions > 0 else "    Neutral: 0")
            print(f"    Results saved to: {output_file}")
            
            # Print JSON results
            print(f"\n  Trajectory {traj_num} JSON Results:")
            print("=" * 70)
            print(json.dumps(self.current_trajectory_data, indent=2))
            print("=" * 70)
            
            # Store for overall summary
            self.alignment_data.append(self.current_trajectory_data)
            self.current_trajectory_data = None
        
        # Generate overall summary
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        
        all_checks_all_traj = []
        for traj_data in self.alignment_data:
            for ep in traj_data['episodes']:
                all_checks_all_traj.extend(ep['alignment_checks'])
        
        if all_checks_all_traj:
            total_aligned = sum(1 for c in all_checks_all_traj if c['alignment'] == 'Aligned')
            total_conflict = sum(1 for c in all_checks_all_traj if c['alignment'] == 'Conflict')
            total_neutral = sum(1 for c in all_checks_all_traj if c['alignment'] == 'Neutral')
            total_all = len(all_checks_all_traj)
            
            print(f"Overall Statistics:")
            print(f"  Total Questions Evaluated: {total_all}")
            print(f"  Aligned: {total_aligned} ({total_aligned/total_all*100:.1f}%)")
            print(f"  Conflict: {total_conflict} ({total_conflict/total_all*100:.1f}%)")
            print(f"  Neutral: {total_neutral} ({total_neutral/total_all*100:.1f}%)")
        
        print(f"\nAll results saved to: {self.output_dir}/")
        print(f"{'='*70}\n")

