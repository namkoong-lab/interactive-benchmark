#!/usr/bin/env python3
"""
Experiment 2: Cross-User Learning with Same Category.

This experiment tests whether an LLM can learn optimal questioning strategies 
that work across different user personas within the same product category.

Key questions:
1. Can the LLM learn optimal questioning strategies for a category across different users?
2. Do questioning strategies improve as the agent experiences more diverse users?
3. Are there consistent questioning patterns that work well for a category regardless of user?
4. Do we need to set particular user groups with consistent preferences?

Setup: Different user personas, same category tested sequentially.
Hypothesis: Agent should learn category-specific questioning strategies that 
work across diverse user types (e.g., price-focused questions for electronics, 
style questions for clothing).
"""

import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime
import random
from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
from .wrappers.metrics_wrapper import MetricsWrapper


class LLMAgentExperiment2:
    """
    LLM-based agent for Experiment 2 that learns questioning strategies
    across different users within the same category.
    """
    
    def __init__(self, model: str = "gpt-4o", max_questions: int = 8):
        self.model = model
        self.max_questions = max_questions
        self.episode_count = 0
        self.learned_questioning_strategies = {}  # Store learned questioning patterns
        self.current_episode_info = None
        self.last_response = None
        self.category_question_history = []  # Track questions asked in this category
        
    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Decide whether to ask a question or make a recommendation using LLM."""
        if 'num_products' in info:
            self.current_episode_info = info
            num_products = info['num_products']
            category = info['category']
        else:
            if self.current_episode_info is None:
                num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
                category = "unknown"
            else:
                num_products = self.current_episode_info['num_products']
                category = self.current_episode_info['category']
        
        dialog_history = []
        if hasattr(self, 'current_env') and self.current_env and hasattr(self.current_env, 'dialog_history'):
            dialog_history = self.current_env.dialog_history
        
        return self._llm_decide_action(obs, info, dialog_history, category, num_products)
    
    def _llm_decide_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                          dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Use LLM to decide whether to ask a question or make a recommendation."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        
        # Add learned questioning strategies to context
        strategy_context = self._build_strategy_context(category)
        
        unified_prompt = f"""You are a product recommendation agent learning optimal questioning strategies for {category} products.

Context:
{context}

{strategy_context}

Task:
Based on the conversation so far and learned questioning patterns, either:
- Ask one short, consumer-friendly question to clarify user preferences, or
- If sufficiently confident, recommend one product by index. 

Output format (MUST be exactly one line, no extra text):
- To ask: QUESTION: <your question>
- To recommend: RECOMMEND: <number 0-{num_products-1}>

Rules:
- Do not include explanations, reasoning, bullets, or multiple questions
- Avoid jargon; use everyday language a shopper understands
- Keep questions specific and helpful for {category} products
- Consider what types of questions have worked well for this category before
- No meta commentary, only the question or recommendation
"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": unified_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=400
            )
            
            self.last_response = response.strip()
            response = self.last_response.upper()
            
            if response.startswith("RECOMMEND:"):
                import re
                numbers = re.findall(r'\d+', response)
                if numbers:
                    product_idx = int(numbers[0])
                    if 0 <= product_idx < num_products:
                        return product_idx
                print(f"[WARN] Could not parse recommendation '{response}', using product 0")
                return 0
                
            elif response.startswith("QUESTION:"):
                return num_products
            else:
                import re
                numbers = re.findall(r'\d+', response)
                if numbers:
                    product_idx = int(numbers[0])
                    if 0 <= product_idx < num_products:
                        print(f"[INFO] Interpreted response as recommendation: {product_idx}")
                        return product_idx
                print(f"[WARN] Unclear response '{response}', asking question")
                return num_products
                
        except Exception as e:
            raise RuntimeError(f"LLM decision failed: {e}")
    
    def _get_product_info(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], num_products: int) -> List[Dict]:
        """Extract product information from observation."""
        products = []
        product_features = obs['product_features']
        
        for i in range(num_products):
            if i < len(info.get('product_ids', [])):
                product_id = info['product_ids'][i]
                features = product_features[i]
                
                price = features[0] * 1000
                store_hash = features[1]
                title_length = features[2] * 100
                
                products.append({
                    'id': product_id,
                    'price': f"${price:.0f}",
                    'store_hash': f"{store_hash:.2f}",
                    'title_length': f"{title_length:.0f} chars"
                })
        
        return products
    
    def _build_llm_context(self, products: List[Dict], dialog_history: List[Tuple[str, str]], category: str) -> str:
        """Build context string for LLM decision making."""
        product_list = f"Available {category} products:\n"
        for i, product in enumerate(products):
            product_list += f"{i}: Product ID {product['id']} - Price: {product['price']}, Store: {product['store_hash']}, Title: {product['title_length']}\n"
        
        dialog_text = "Conversation so far:\n"
        if dialog_history:
            for i, (question, answer) in enumerate(dialog_history):
                dialog_text += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"
        else:
            dialog_text += "No questions asked yet.\n"
        
        return f"{product_list}\n{dialog_text}"
    
    def _build_strategy_context(self, category: str) -> str:
        """Build context about learned questioning strategies for this category."""
        if category not in self.learned_questioning_strategies:
            return f"No previous questioning strategies learned for {category} yet."
        
        strategies = self.learned_questioning_strategies[category]
        if not strategies:
            return f"No effective questioning strategies identified for {category} yet."
        
        strategy_text = f"Learned questioning strategies for {category}:\n"
        for i, (question_type, effectiveness) in enumerate(strategies[:3]):  # Top 3 strategies
            strategy_text += f"- {question_type}: {effectiveness:.1f}% effective\n"
        
        return strategy_text
    
    def update_strategies(self, episode_result: Dict[str, Any]):
        """Update learned questioning strategies based on episode outcome."""
        self.episode_count += 1
        
        if 'final_info' in episode_result and 'chosen_score' in episode_result['final_info']:
            score = episode_result['final_info']['chosen_score']
            category = episode_result.get('category', 'unknown')
            dialog_history = episode_result.get('full_dialog', [])
            feedback = episode_result['final_info'].get('feedback', '')
            feedback_type = episode_result['final_info'].get('feedback_type', 'regret')
            
            # Analyze which questions led to good outcomes
            if dialog_history:
                for question, answer in dialog_history:
                    question_type = self._categorize_question(question)
                    effectiveness = score  # Use final score as effectiveness measure
                    
                    if category not in self.learned_questioning_strategies:
                        self.learned_questioning_strategies[category] = {}
                    
                    if question_type not in self.learned_questioning_strategies[category]:
                        self.learned_questioning_strategies[category][question_type] = []
                    
                    self.learned_questioning_strategies[category][question_type].append(effectiveness)
            
            # Process feedback for strategy learning
            self._process_feedback_for_strategies(feedback, feedback_type, score, category, dialog_history)
    
    def _process_feedback_for_strategies(self, feedback: str, feedback_type: str, score: float, 
                                        category: str, dialog_history: list):
        """Process feedback specifically for questioning strategy learning."""
        if not feedback:
            return
        
        # Store feedback for strategy analysis
        if not hasattr(self, 'strategy_feedback_history'):
            self.strategy_feedback_history = []
        
        self.strategy_feedback_history.append({
            'feedback': feedback,
            'feedback_type': feedback_type,
            'score': score,
            'category': category,
            'dialog_length': len(dialog_history),
            'episode': self.episode_count
        })
            
    def _analyze_question_effectiveness_from_feedback(self, feedback: str, dialog_history: list, 
                                                     score: float, category: str):
        """Analyze which questions were most effective based on quality feedback."""
        # Simple analysis based on quality feedback
        feedback_lower = feedback.lower()
        
        # Determine overall effectiveness based on quality feedback
        if 'great' in feedback_lower:
            overall_effectiveness = 'high'
        elif 'ok' in feedback_lower:
            overall_effectiveness = 'medium'
        elif 'bad' in feedback_lower:
            overall_effectiveness = 'low'
        else:
            overall_effectiveness = 'unknown'
        
        # Store effectiveness insights
        if not hasattr(self, 'question_effectiveness_insights'):
            self.question_effectiveness_insights = {}
        if category not in self.question_effectiveness_insights:
            self.question_effectiveness_insights[category] = []
        
        self.question_effectiveness_insights[category].append({
            'overall_effectiveness': overall_effectiveness,
            'score': score,
            'feedback': feedback,
            'dialog_length': len(dialog_history)
        })
    
    def _categorize_question(self, question: str) -> str:
        """Categorize question type for strategy learning."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['price', 'budget', 'cost', 'expensive', 'cheap']):
            return "price_budget"
        elif any(word in question_lower for word in ['brand', 'manufacturer', 'company']):
            return "brand_preference"
        elif any(word in question_lower for word in ['size', 'dimension', 'measurement']):
            return "size_specifications"
        elif any(word in question_lower for word in ['color', 'style', 'design', 'appearance']):
            return "style_appearance"
        elif any(word in question_lower for word in ['feature', 'function', 'capability']):
            return "features_functionality"
        elif any(word in question_lower for word in ['quality', 'durability', 'reliability']):
            return "quality_durability"
        else:
            return "general_preference"


def save_checkpoint(all_results: List[Dict], persona_results: Dict, agent: LLMAgentExperiment2, 
                   output_dir: str, model: str, feedback_type: str, episode_num: int, seed: Optional[int] = None):
    """Save incremental checkpoint every 5 personas."""
    
    # Create checkpoint filename
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace(" ", "_")
    completed_personas = len([pid for pid, results in persona_results.items() if len(results) > 0])
    checkpoint_file = os.path.join(output_dir, f"checkpoint_personas_{completed_personas:02d}_episode_{episode_num:03d}_{model_safe_name}_{feedback_safe_name}.json")
    
    # Prepare checkpoint data
    checkpoint_data = {
        'experiment': 'Experiment 2: LLM Learning Questioning Strategies Across Users (Checkpoint)',
        'timestamp': datetime.now().isoformat(),
        'episode_num': episode_num,
        'model': model,
        'feedback_type': feedback_type,
        'seed': seed,
        'agent_state': {
            'episode_count': agent.episode_count,
            'learned_questioning_strategies': agent.learned_questioning_strategies,
            'category_question_history': agent.category_question_history
        },
        'episodes_completed': len(all_results),
        'all_results': all_results,
        'persona_results': persona_results,
        'summary': {
            'personas_tested': list(persona_results.keys()),
            'total_episodes': len(all_results),
            'episodes_by_persona': {pid: len(results) for pid, results in persona_results.items()},
            'product_counts_by_persona': {
                pid: {
                    'num_products': results[0]['product_info']['num_products'] if results else 0,
                    'episodes': len(results)
                } for pid, results in persona_results.items()
            }
        }
    }
    
    # Save checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"  Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(checkpoint_file: str) -> Tuple[List[Dict], Dict, Dict]:
    """Load experiment from checkpoint file."""
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded checkpoint from episode {data['episode_num']}")
    print(f"  Personas tested: {data['summary']['personas_tested']}")
    print(f"  Total episodes: {data['summary']['total_episodes']}")
    
    return data['all_results'], data['persona_results'], data['agent_state']


def run_experiment2(category: str = "Electronics",
                   persona_indices: List[int] = None,
                   num_personas: int = 10,
                   episodes_per_persona: int = 3,
                   max_questions: int = 8,
                   model: str = "gpt-4o",
                   feedback_type: str = "none",
                   min_score_threshold: float = 50.0,
                   output_dir: str = "experiment2_results",
                   seed: Optional[int] = None,
                   checkpoint_file: str = None):
    """
    Run Experiment 2: LLM learning questioning strategies across users in same category.
    
    Args:
        category: Single category to test across different users
        persona_indices: List of persona indices to use (None = randomly choose)
        num_personas: Number of personas to randomly select (if persona_indices is None)
        episodes_per_persona: Number of episodes per persona
        max_questions: Maximum questions per episode
        model: LLM model to use
        feedback_type: Type of feedback to provide ("none", "regret", "persona")
        min_score_threshold: Minimum score threshold for category relevance (default: 50.0)
        output_dir: Directory to save results
        seed: Random seed for reproducible persona selection (None = no seeding)
        checkpoint_file: Optional checkpoint file to resume from
    """
    
    print(f"=== Experiment 2: LLM Learning Questioning Strategies Across Users ===")
    print(f"Category: {category}, Episodes per persona: {episodes_per_persona}")
    print(f"Max questions: {max_questions}, Model: {model}, Feedback: {feedback_type}")
    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    # Initialize or load from checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        all_results, persona_results, agent_state = load_checkpoint(checkpoint_file)
        
        # Recreate agent with saved state
        agent = LLMAgentExperiment2(model=model, max_questions=max_questions)
        agent.episode_count = agent_state['episode_count']
        agent.learned_questioning_strategies = agent_state['learned_questioning_strategies']
        agent.category_question_history = agent_state.get('category_question_history', [])
        
        # Calculate starting episode number
        start_episode = len(all_results) + 1
        print(f"Resuming from episode {start_episode}")
    else:
        print("Starting fresh experiment")
        all_results = []
        persona_results = {}
        agent = LLMAgentExperiment2(model=model, max_questions=max_questions)
        start_episode = 1
    
    # Create feedback system
    from .core.feedback_system import FeedbackSystem
    from .core.personas import get_persona_description
    
    if feedback_type == "persona":
        # Get persona description for persona feedback
        persona_description = get_persona_description(persona_index)
        feedback_system = FeedbackSystem(feedback_type=feedback_type, persona_description=persona_description)
    else:
        feedback_system = FeedbackSystem(feedback_type=feedback_type)
    
    from .core.simulate_interaction import list_categories, get_products_by_category
    available_categories = list_categories()
    
    if category not in available_categories:
        print(f"Category '{category}' not found. Available categories: {available_categories[:5]}...")
        category = available_categories[0] if available_categories else "Electronics"
        print(f"Using category: {category}")
    
    # Check if category is relevant (has products with score > threshold for at least some personas)
    def check_category_relevance(category, sample_personas=10, min_score_threshold=min_score_threshold):
        """Check if a category has relevant products for at least some personas."""
        from .core.user_model import UserModel
        
        try:
            products = get_products_by_category(category)
            if not products:
                print(f"Category '{category}' has no products")
                return False
                
            # Test with a sample of personas
            test_personas = random.sample(range(0, 1000), min(sample_personas, 1000))
            relevant_personas = 0
            
            print(f"Checking category relevance for '{category}' (testing {len(test_personas)} personas)...")
            
            for persona_idx in test_personas:
                try:
                    user_model = UserModel(persona_idx)
                    scores = user_model.score_products(category, products)
                    if scores:
                        max_score = max(score for _, score in scores)
                        if max_score > min_score_threshold:
                            relevant_personas += 1
                except Exception as e:
                    continue
            
            relevance_ratio = relevant_personas / len(test_personas)
            print(f"Category '{category}': {relevant_personas}/{len(test_personas)} personas find it relevant (score > {min_score_threshold})")
            
            # Consider category relevant if at least 20% of test personas find it relevant
            return relevance_ratio >= 0.2
            
        except Exception as e:
            print(f"Error checking category relevance: {e}")
            return False
    
    # Optional: Check if the specified category is relevant to a sample of personas
    # (This is just informational - we'll check each persona individually during the experiment)
    sample_relevance = check_category_relevance(category)
    if not sample_relevance:
        print(f"Note: Category '{category}' may not be relevant to many personas.")
        print("Individual personas will be checked and skipped if irrelevant.")
    
    # Select personas
    if persona_indices is None:
        # Use a diverse set of personas
        persona_indices = random.sample(range(0, 1000), min(num_personas, 1000))
    else:
        persona_indices = persona_indices[:num_personas]
    
    print(f"Personas: {persona_indices}")
    
    # Initialize results if not loaded from checkpoint
    if not checkpoint_file or not os.path.exists(checkpoint_file):
        all_results = []
        persona_results = {pid: [] for pid in persona_indices}
    
    total_episodes = len(persona_indices) * episodes_per_persona
    episode_num = 0
    
    # Dynamic persona filtering - check relevance as we encounter personas
    def check_persona_category_relevance(persona_idx, category, min_score_threshold):
        """Check if a specific persona finds the category relevant."""
        from .core.user_model import UserModel
        try:
            products = get_products_by_category(category)
            if not products:
                return False, 0.0
                
            user_model = UserModel(persona_idx)
            scores = user_model.score_products(category, products)
            if scores:
                max_score = max(score for _, score in scores)
                return max_score > min_score_threshold, max_score
            return False, 0.0
        except Exception as e:
            print(f"  Error checking persona relevance: {e}")
            return False, 0.0
    
    for persona_index in persona_indices:
        print(f"\n--- Testing Persona: {persona_index} ---")
        
        # Check relevance for this persona
        is_relevant, max_score = check_persona_category_relevance(persona_index, category, min_score_threshold)
        if not is_relevant:
            print(f"  Persona {persona_index}: Max score {max_score:.1f} ≤ {min_score_threshold}, skipping persona")
            # Skip all episodes for this persona
            for episode in range(episodes_per_persona):
                episode_num += 1
            continue
        
        print(f"  Persona {persona_index}: Max score {max_score:.1f} > {min_score_threshold}, proceeding")
        
        for episode in range(episodes_per_persona):
            episode_num += 1
            print(f"Episode {episode_num}/{total_episodes} (Persona: {persona_index})")
            
            env = RecoEnv(
                persona_index=persona_index,
                max_questions=max_questions,
                categories=[category],  # Same category for all episodes
                agent=agent,
                feedback_system=feedback_system
            )
            
            metrics_wrapper = MetricsWrapper(env, 
                                           output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
            
            obs, initial_info = metrics_wrapper.reset()
            
            if episode == 0:
                print(f"  Products in {category}: {len(env.products)}")
                print(f"  Top 3 products by score:")
                for i, (product_id, score) in enumerate(env.oracle_scores[:3]):
                    product = next((p for p in env.products if p['id'] == product_id), None)
                    if product:
                        title = product.get('title', 'Unknown')[:50] + "..." if len(product.get('title', '')) > 50 else product.get('title', 'Unknown')
                        print(f"    {i+1}. {title} (Score: {score:.1f})")
            
            agent.current_env = env
            
            terminated = False
            truncated = False
            step_count = 0
            current_info = initial_info
            
            while not terminated and not truncated and step_count < 20:
                action = agent.get_action(obs, current_info)
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                current_info = info
                step_count += 1
                
                if info['action_type'] == 'ask':
                    print(f"  Step {step_count}: Asked question")
                elif info['action_type'] == 'recommend':
                    print(f"  Step {step_count}: Recommended product {info['chosen_product_id']}")
                    print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}")
                    print(f"    Top1: {info['top1']}, Top3: {info['top3']}")
                    if 'feedback' in info and info['feedback']:
                        print(f"    Feedback: {info['feedback']}")
                    break
            
            full_dialog = []
            if hasattr(env, 'dialog_history'):
                full_dialog = env.dialog_history
            
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
            
            episode_result = {
                'episode': episode_num,
                'category': category,
                'persona_index': persona_index,
                'episode_in_persona': episode + 1,
                'steps': step_count,
                'terminated': terminated,
                'truncated': truncated,
                'final_info': info,
                'full_dialog': full_dialog,
                'product_info': product_info
            }
            
            all_results.append(episode_result)
            persona_results[persona_index].append(episode_result)
            agent.update_strategies(episode_result)
            metrics_wrapper.close()
            
            # Save checkpoint every 5 personas (at the end of each persona)
            if episode == episodes_per_persona - 1:  # Last episode of this persona
                # Check if we've completed 5 personas (or all personas)
                completed_personas = len([pid for pid, results in persona_results.items() if len(results) == episodes_per_persona])
                if completed_personas % 5 == 0 or completed_personas == len(persona_indices):
                    save_checkpoint(all_results, persona_results, agent, output_dir, model, feedback_type, episode_num, seed)
    
    print(f"\n=== Results Analysis ===")
    
    print("\nPerformance by Persona:")
    for persona_index, results in persona_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        top1_rates = [r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]
        
        if scores:
            avg_score = np.mean(scores)
            top1_rate = np.mean(top1_rates)
            print(f"  Persona {persona_index}: Avg Score: {avg_score:.1f}, Top1 Rate: {top1_rate:.1%}, Episodes: {len(scores)}")
    
    print("\nLearning Progression (Questioning Strategy Effectiveness):")
    for persona_index, results in persona_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        if len(scores) >= 2:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            improvement = second_half - first_half
            print(f"  Persona {persona_index}: {first_half:.1f} → {second_half:.1f} (Δ{improvement:+.1f})")
    
    # Analyze learned questioning strategies
    print("\nLearned Questioning Strategies:")
    if category in agent.learned_questioning_strategies:
        strategies = agent.learned_questioning_strategies[category]
        for question_type, effectiveness_scores in strategies.items():
            avg_effectiveness = np.mean(effectiveness_scores)
            print(f"  {question_type}: {avg_effectiveness:.1f} avg effectiveness ({len(effectiveness_scores)} examples)")
    
    episode_regrets = []
    episode_questions = []
    episode_scores = []
    
    for result in all_results:
        if 'chosen_score' in result['final_info']:
            episode_scores.append(result['final_info']['chosen_score'])
        if 'regret' in result['final_info']:
            episode_regrets.append(result['final_info']['regret'])
        if 'questions_asked' in result['final_info']:
            episode_questions.append(result['final_info']['questions_asked'])
    
    regret_progression = {
        'episode_regrets': episode_regrets,
        'avg_regret': np.mean(episode_regrets) if episode_regrets else 0,
        'regret_trend': 'improving' if len(episode_regrets) > 1 and episode_regrets[-1] < episode_regrets[0] else 'stable'
    }
    
    questions_progression = {
        'episode_questions': episode_questions,
        'avg_questions': np.mean(episode_questions) if episode_questions else 0,
        'total_questions': sum(episode_questions) if episode_questions else 0
    }
    
    # Persona information
    persona_info = {}
    for pid, results in persona_results.items():
        if results:
            persona_info[pid] = {
                'num_products': results[0]['product_info']['num_products'],
                'episodes': len(results)
            }
    
    # Create model-specific filename
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace("/", "_").replace(":", "_")
    results_file = os.path.join(output_dir, f"experiment2_results_{model_safe_name}_{feedback_safe_name}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Experiment 2: LLM Learning Questioning Strategies Across Users',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'regret_progression': regret_progression,
                'questions_progression': questions_progression,
                'persona_info': persona_info,
                'learned_strategies': agent.learned_questioning_strategies,
                'overall_performance': {
                    'avg_score': np.mean(episode_scores) if episode_scores else 0,
                    'total_episodes': len(all_results),
                    'successful_episodes': len([r for r in all_results if 'chosen_score' in r['final_info']])
                }
            },
            'config': {
                'category': category,
                'persona_indices': persona_indices,
                'episodes_per_persona': episodes_per_persona,
                'max_questions': max_questions,
                'model': model,
                'feedback_type': feedback_type,
                'seed': seed
            },
            'results': all_results,
            'persona_summary': {
                pid: {
                    'avg_score': np.mean([r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]),
                    'top1_rate': np.mean([r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]),
                    'episode_count': len(results),
                    'num_products': results[0]['product_info']['num_products'] if results else 0,
                    'products_with_scores': results[0]['product_info']['products_with_scores'] if results else []
                }
                for pid, results in persona_results.items()
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Individual episode metrics saved to: {output_dir}/episode_*.jsonl")
    print(f"Checkpoints saved to: {output_dir}/checkpoint_personas_*.json")
    
    return all_results, persona_results, agent.learned_questioning_strategies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 2: LLM Learning Questioning Strategies Across Users")
    parser.add_argument("--category", type=str, default="Electronics", help="Category to test")
    parser.add_argument("--persona_indices", nargs="+", type=int, default=None, help="Persona indices to use")
    parser.add_argument("--num_personas", type=int, default=10, help="Number of personas to randomly select")
    parser.add_argument("--episodes_per_persona", type=int, default=3, help="Episodes per persona")
    parser.add_argument("--max_questions", type=int, default=8, help="Max questions per episode")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--feedback_type", type=str, default="none", choices=["none", "regret", "persona"], help="Type of feedback to provide")
    parser.add_argument("--min_score_threshold", type=float, default=50.0, help="Minimum score threshold for category relevance")
    parser.add_argument("--output_dir", type=str, default="experiment2_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible persona selection")
    parser.add_argument("--resume_from", help="Checkpoint file to resume from")
    
    args = parser.parse_args()
    
    run_experiment2(
        category=args.category,
        persona_indices=args.persona_indices,
        num_personas=args.num_personas,
        episodes_per_persona=args.episodes_per_persona,
        max_questions=args.max_questions,
        model=args.model,
        feedback_type=args.feedback_type,
        min_score_threshold=args.min_score_threshold,
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint_file=args.resume_from
    )
