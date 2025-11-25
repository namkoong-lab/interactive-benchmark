#!/usr/bin/env python3
"""
Persona Consistency Experiment

Temporary experiment to measure consistency of persona agent responses
across multiple runs of the same question.

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
from .persona_consistency_evaluator import PersonaConsistencyEvaluator


class ConsistencyTrackingRecoEnv(RecoEnv):
    """
    Wrapper around RecoEnv that tracks questions and allows pausing for consistency checks.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with consistency tracking."""
        super().__init__(*args, **kwargs)
        self.questions_asked = []  # Track questions for consistency checking
    
    def step(self, action: int):
        """Override step to track questions."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track questions
        if info.get('action_type') == 'ask':
            question_text = info.get('question_text', '')
            answer = info.get('answer', '')
            self.questions_asked.append({
                'question': question_text,
                'answer': answer,
                'question_num': len(self.dialog_history)
            })
        
        return obs, reward, terminated, truncated, info


class PersonaConsistencyExperiment:
    """
    Experiment runner that measures persona consistency across multiple runs.
    """
    
    def __init__(self, config, evaluator_model: str = "claude-sonnet-4-20250514", num_runs: int = 10):
        """
        Initialize the consistency experiment.
        
        Args:
            config: ExperimentConfig instance
            evaluator_model: Claude model to use for consistency evaluation
            num_runs: Number of times to ask the same question (default: 10)
        """
        self.config = config
        self.base_experiment = UnifiedExperiment(config)
        self.consistency_evaluator = PersonaConsistencyEvaluator(model=evaluator_model)
        self.num_runs = num_runs
        
        # Consistency data storage
        self.consistency_data = []
        self.current_trajectory_data = None
        self.current_episode_data = None
        
        # Output directory
        self.output_dir = "persona_consistency_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_persona_answers(self, persona_index: int, question: str, category: str, 
                             dialog_history: List[Tuple[str, str]], num_runs: int = 10) -> List[str]:
        """
        Get multiple answers from persona agent by refreshing it each time.
        
        Args:
            persona_index: Persona index
            question: Question to ask
            category: Product category
            dialog_history: Previous conversation history
            num_runs: Number of times to ask (default: 10)
        
        Returns:
            List of answers (length = num_runs)
        """
        answers = []
        for run in range(num_runs):
            # Create fresh UserModel instance each time (refreshing the persona agent)
            user_model = UserModel(persona_index)
            answer = user_model.respond(question, category, dialog_history)
            answers.append(answer)
        return answers
    
    def _run_episode_with_consistency_tracking(
        self, episode_num: int, persona_index: int, category: str,
        cached_scores: Optional[List[Tuple[int, float]]] = None,
        trajectory_seed: Optional[int] = None,
        trajectory_num: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single episode with consistency tracking.
        Pauses at each question to run consistency checks.
        """
        if self.config.debug_mode:
            print(f"\nEpisode {episode_num}: Persona #{persona_index}, Category: {category}")
        
        # Initialize episode data
        self.current_episode_data = {
            'episode_num': episode_num,
            'consistency_checks': []
        }
        
        try:
            # Create feedback system
            if self.config.feedback_type == "persona":
                persona_agent = UserModel(persona_index)
                feedback_system = FeedbackSystem(feedback_type="persona", persona_agent=persona_agent)
            else:
                feedback_system = FeedbackSystem(feedback_type=self.config.feedback_type)
            
            # Create environment with consistency tracking
            env = ConsistencyTrackingRecoEnv(
                persona_index=persona_index,
                max_questions=self.config.max_questions,
                categories=[category],
                agent=self.base_experiment.agent,
                feedback_system=feedback_system,
                cached_scores=cached_scores,
                max_products_per_category=self.config.max_products_per_category,
                seed=trajectory_seed,
                debug_mode=self.config.debug_mode
            )
            
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
            dialog_history = []
            
            while not terminated and not truncated and step_count <= 20:
                action = self.base_experiment.agent.get_action(obs, current_info)
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                current_info = info
                step_count += 1
                
                # If a question was just asked, pause and run consistency check
                if info.get('action_type') == 'ask':
                    question_text = info.get('question_text', '')
                    answer = info.get('answer', '')
                    
                    # Get dialog history BEFORE the current Q&A was added
                    # The environment's step() already added it, so we need history before that
                    dialog_history = env.dialog_history[:-1] if hasattr(env, 'dialog_history') and len(env.dialog_history) > 0 else []
                    
                    print(f"\n  Question {len(dialog_history) + 1}: {question_text}")
                    print(f"    → Running consistency check ({self.num_runs} runs)...")
                    
                    # Get multiple answers by refreshing persona agent each time
                    # Use the dialog history BEFORE the current question
                    answers = self._get_persona_answers(
                        persona_index=persona_index,
                        question=question_text,
                        category=category,
                        dialog_history=dialog_history,
                        num_runs=self.num_runs
                    )
                    
                    # Evaluate consistency
                    consistency_result = self.consistency_evaluator.evaluate_consistency(
                        answers=answers
                    )
                    
                    # Store consistency check
                    consistency_check = {
                        'question_num': len(dialog_history) + 1,
                        'question': question_text,
                        'answers': answers,  # All 10 answers
                        'majority_position': consistency_result['majority_position'],
                        'aligned_with_majority_count': consistency_result['aligned_with_majority_count'],
                        'misaligned_with_majority_count': consistency_result['misaligned_with_majority_count'],
                        'consistency_rate': consistency_result['consistency_rate'],
                        'breakdown': consistency_result['breakdown'],
                        'timestamp': consistency_result['timestamp']
                    }
                    
                    self.current_episode_data['consistency_checks'].append(consistency_check)
                    
                    print(f"    → Consistency: {consistency_result['aligned_with_majority_count']}/{self.num_runs} aligned with majority "
                          f"({consistency_result['consistency_rate']*100:.1f}%)")
                    print(f"      Misaligned: {consistency_result['misaligned_with_majority_count']}")
                    print(f"      Majority position: {consistency_result['majority_position'][:100]}...")
                    
                    # Continue with the episode (use the first answer as the actual answer)
                    # The environment already has the answer from the first run
                
                if self.config.debug_mode:
                    if info['action_type'] == 'ask':
                        print(f"  Step {step_count}: Asked question")
                    elif info['action_type'] == 'recommend':
                        print(f"  Step {step_count}: Recommended product {info['chosen_product_id']}")
                        print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}, Regret: {info.get('regret', 0):.1f}")
                        break
            
            # Get full dialog history (all questions and answers)
            full_dialog = env.dialog_history if hasattr(env, 'dialog_history') else []
            
            # Add full dialog history to episode data
            self.current_episode_data['full_dialog_history'] = [
                {'question': q, 'answer': a} for q, a in full_dialog
            ]
            self.current_episode_data['category'] = category
            self.current_episode_data['persona_index'] = persona_index
            
            # Store episode data
            if self.current_trajectory_data:
                self.current_trajectory_data['episodes'].append(self.current_episode_data)
            
            # Return standard episode result
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
            
            # Store partial episode data if we have any consistency checks
            if self.current_episode_data and self.current_trajectory_data:
                if len(self.current_episode_data.get('consistency_checks', [])) > 0:
                    self.current_trajectory_data['episodes'].append(self.current_episode_data)
            
            self.current_episode_data = None
            return None
    
    def run(self):
        """Run the full consistency experiment."""
        print(f"\n{'='*70}")
        print(f"  PERSONA CONSISTENCY EXPERIMENT")
        print(f"{'='*70}")
        print(f"Model: {self.config.model}")
        print(f"Evaluator: {self.consistency_evaluator.model}")
        print(f"Experiment Type: {self.config.experiment_type}")
        print(f"Total Trajectories: {self.config.total_trajectories}")
        print(f"Episodes per Trajectory: {self.config.episodes_per_trajectory}")
        print(f"Consistency Runs per Question: {self.num_runs}")
        print(f"{'='*70}\n")
        
        # Initialize agent
        self.base_experiment.agent = self.base_experiment._create_agent()
        self.base_experiment.output_path = self.config.get_output_path()
        os.makedirs(self.base_experiment.output_path, exist_ok=True)
        
        # Set trajectory seeds
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
                self._run_episode_with_consistency_tracking(
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
                all_checks.extend(ep['consistency_checks'])
            
            total_questions = len(all_checks)
            total_aligned = sum(c['aligned_with_majority_count'] for c in all_checks)
            total_misaligned = sum(c['misaligned_with_majority_count'] for c in all_checks)
            total_answers = sum(c['aligned_with_majority_count'] + c['misaligned_with_majority_count'] for c in all_checks)
            
            avg_consistency_rate = sum(c['consistency_rate'] for c in all_checks) / total_questions if total_questions > 0 else 0.0
            
            self.current_trajectory_data['summary'] = {
                'total_questions': total_questions,
                'total_answers_evaluated': total_answers,
                'total_aligned_with_majority': total_aligned,
                'total_misaligned_with_majority': total_misaligned,
                'overall_consistency_rate': total_aligned / total_answers if total_answers > 0 else 0.0,
                'avg_consistency_rate_per_question': avg_consistency_rate
            }
            
            # Save trajectory results
            output_file = os.path.join(self.output_dir, f"consistency_trajectory_{traj_num}_results.json")
            with open(output_file, 'w') as f:
                json.dump(self.current_trajectory_data, f, indent=2)
            
            print(f"\n  Trajectory {traj_num} complete:")
            print(f"    Total Questions: {total_questions}")
            print(f"    Total Answers Evaluated: {total_answers}")
            print(f"    Overall Aligned with Majority: {total_aligned} ({total_aligned/total_answers*100:.1f}%)" if total_answers > 0 else "    Overall Aligned with Majority: 0")
            print(f"    Overall Misaligned with Majority: {total_misaligned} ({total_misaligned/total_answers*100:.1f}%)" if total_answers > 0 else "    Overall Misaligned with Majority: 0")
            print(f"    Average Consistency Rate: {avg_consistency_rate*100:.1f}%")
            print(f"    Results saved to: {output_file}")
            
            # Print JSON results
            print(f"\n  Trajectory {traj_num} JSON Results:")
            print("=" * 70)
            print(json.dumps(self.current_trajectory_data, indent=2))
            print("=" * 70)
            
            # Store for overall summary
            self.consistency_data.append(self.current_trajectory_data)
            self.current_trajectory_data = None
        
        # Generate overall summary
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        
        all_checks_all_traj = []
        for traj_data in self.consistency_data:
            for ep in traj_data['episodes']:
                all_checks_all_traj.extend(ep['consistency_checks'])
        
        if all_checks_all_traj:
            total_all_aligned = sum(c['aligned_with_majority_count'] for c in all_checks_all_traj)
            total_all_misaligned = sum(c['misaligned_with_majority_count'] for c in all_checks_all_traj)
            total_all_answers = sum(c['aligned_with_majority_count'] + c['misaligned_with_majority_count'] for c in all_checks_all_traj)
            avg_consistency = sum(c['consistency_rate'] for c in all_checks_all_traj) / len(all_checks_all_traj) if all_checks_all_traj else 0.0
            
            print(f"Overall Statistics:")
            print(f"  Total Questions: {len(all_checks_all_traj)}")
            print(f"  Total Answers Evaluated: {total_all_answers}")
            print(f"  Overall Aligned with Majority: {total_all_aligned} ({total_all_aligned/total_all_answers*100:.1f}%)")
            print(f"  Overall Misaligned with Majority: {total_all_misaligned} ({total_all_misaligned/total_all_answers*100:.1f}%)")
            print(f"  Average Consistency Rate: {avg_consistency*100:.1f}%")
        
        print(f"\nAll results saved to: {self.output_dir}/")
        print(f"{'='*70}\n")

