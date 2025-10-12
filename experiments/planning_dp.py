#!/usr/bin/env python3
"""
Dynamic Programming variant of Planning Experiment 

This implements POMDP planning prompting where the agent is explicitly told to:
"Think like a planner solving a POMDP with a single terminal reward from the score 
that the customer would assign to what you recommend. Maintain a belief state, 
estimate expected value of information for each question, and choose the question 
with highest expected value."
"""

import gymnasium as gym
import numpy as np
import json
import os
import random
import re
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime
from pipeline.envs.reco_env import RecoEnv
from pipeline.core.llm_client import chat_completion
from pipeline.wrappers.metrics_wrapper import MetricsWrapper


class DPFixedQuestionsAgent:
    """
    Agent that asks exactly 10 questions with POMDP planning prompting, then makes a recommendation.
    On tracking episodes (1, 5, 10), forces recommendation after each question to measure regret progression.
    On normal episodes, proceed normally like experiment1 but with POMDP planning prompting.
    """
    
    def __init__(self, model: str = "gpt-4o", context_mode: str = "raw"):
        self.model = model
        self.fixed_questions = 10 
        self.context_mode = context_mode
        self.episode_count = 0
        self.episode_history = []  
        self.current_episode_info = None
        self.current_question_count = 0
        self.last_response = None
        self.is_tracking_episode = False  
        
    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Decide action based on current question count and episode type."""
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
        questions_remaining = self.fixed_questions
        if hasattr(self, 'current_env') and self.current_env and hasattr(self.current_env, 'dialog_history'):
            dialog_history = self.current_env.dialog_history
            self.current_question_count = len(dialog_history)
            if hasattr(self.current_env, 'questions_remaining'):
                questions_remaining = self.current_env.questions_remaining
        
        if questions_remaining <= 0:
            return self._make_recommendation(obs, info, dialog_history, category, num_products)
        else:
            if self.is_tracking_episode:   
                question_action = self._ask_question(obs, info, dialog_history, category, num_products)
                return question_action
            else:
                return self._choose_action(obs, info, dialog_history, category, num_products)
    
    def _choose_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                       dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category)
        previous_questions = [q for q, a in dialog_history]
        questions_context = ""
        if previous_questions:
            questions_context = f"Previous questions asked:\n" + "\n".join([f"- {q}" for q in previous_questions]) + "\n\n"
        
        questions_remaining = self.fixed_questions - len(dialog_history)
        
        base_prompt = f"""You are a product recommendation agent. Your goal is to find the best product for this user.

Context:
{context}

{feedback_context}

{questions_context}

INTERNAL REASONING (do not share with customer):
1. Think about which products the customer might like based on what you know so far
2. Identify what information would be most valuable to distinguish between these candidates
3. Consider what preference question would help you make the best final recommendation

CUSTOMER INTERACTION:
You can either ask another question to learn more about the user's preferences, or make a recommendation if you feel you have enough information.

IMPORTANT: Only ask the customer about their preferences, needs, and requirements. Do NOT ask about specific products, product numbers, or product details.

STRICT OUTPUT FORMAT - Choose ONE:
Either: QUESTION: [your question here]
Or: RECOMMEND: [product number 0-{num_products-1}]

Rules:
- If you want to ask a question, start your response with "QUESTION:"
- If you want to recommend, start your response with "RECOMMEND:"
- Do NOT include any explanations, reasoning, or additional text
- Do NOT use multiple lines or formatting
- Just the format above, nothing else
- Ask about preferences, not about specific products

Examples:
QUESTION: What's your budget range for this purchase?
QUESTION: What style do you prefer - casual or formal?
RECOMMEND: 5

Your choice:"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": base_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=200
            )
            self.last_response = response.strip()
            response_upper = response.upper().strip()
            if response_upper.startswith("RECOMMEND:"):
                try:
                    rec_part = response_upper.split("RECOMMEND:", 1)[1].strip()
                    numbers = re.findall(r'\d+', rec_part)
                    if numbers:
                        product_num = int(numbers[0])
                        if 0 <= product_num < num_products:
                            print(f"Agent chose to recommend product {product_num}")
                            return product_num
                        else:
                            print(f"Warning: Invalid product number {product_num} (valid range: 0-{num_products-1}), asking question instead")
                            return self._ask_question(obs, info, dialog_history, category, num_products)
                    else:
                        print(f"Warning: No number found in recommendation: {response}")
                        return self._ask_question(obs, info, dialog_history, category, num_products)
                except (ValueError, IndexError):
                    print(f"Warning: Failed to parse recommendation from response: {response}")
                    return self._ask_question(obs, info, dialog_history, category, num_products)
            elif response_upper.startswith("QUESTION:"):
                print(f"Agent chose to ask another question")
                return self._ask_question(obs, info, dialog_history, category, num_products)
            else:
                print(f"Warning: Failed to parse action from response (expected QUESTION: or RECOMMEND:): {response[:100]}...")
                return self._ask_question(obs, info, dialog_history, category, num_products)
                
        except Exception as e:
            print(f"Error choosing action: {e}")
            return self._ask_question(obs, info, dialog_history, category, num_products)
    
    def set_tracking_episode(self, is_tracking: bool):
        """Set whether this episode should track regret after each question."""
        self.is_tracking_episode = is_tracking
    
    def should_force_recommendation_after_question(self) -> bool:
        """Check if we should force a recommendation after the current question (for tracking episodes)."""
        return self.is_tracking_episode and self.current_question_count < self.fixed_questions
    
    def force_recommendation_after_question(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Force a recommendation after asking a question (for regret tracking)."""
        if 'num_products' in info:
            num_products = info['num_products']
            category = info['category']
        else:
            num_products = self.current_episode_info['num_products']
            category = self.current_episode_info['category']
        
        dialog_history = []
        if hasattr(self, 'current_env') and self.current_env and hasattr(self.current_env, 'dialog_history'):
            dialog_history = self.current_env.dialog_history
        
        return self._make_recommendation(obs, info, dialog_history, category, num_products)
    
    def _ask_question(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                     dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Ask the next question in the sequence with greedy prompting."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category)
        
        previous_questions = [q for q, a in dialog_history]
        questions_context = ""
        if previous_questions:
            questions_context = f"Previous questions asked:\n" + "\n".join([f"- {q}" for q in previous_questions]) + "\n\n"
        
        questions_remaining = self.fixed_questions - len(dialog_history)
        
        base_prompt = f"""You are a product recommendation agent. Your goal is to find the best product for this user.

Context:
{context}

{feedback_context}

{questions_context}

Think like a planner solving a POMDP with a **single terminal reward** from the score that the customer would assign to what you recommend. 
Note that you have {questions_remaining} turn(s) left. 

1. **Maintain a belief state** — a probability distribution over possible customer preferences given past answers.
2. For each possible next question:
   - Predict how each possible answer will **update your belief**.
   - Estimate how that updated belief will affect your **final recommendation quality**.
   - Compute the **expected value of information (EVI)** for that question.
3. Choose the question with the **highest expected value**, even if it has no immediate payoff.

CUSTOMER INTERACTION:
Ask the question that maximizes expected value of information for your final recommendation.

Your question should:
- Maximize the expected value of information for your final recommendation
- Help you update your belief state about customer preferences most effectively
- Consider how different answers would change your final recommendation strategy
- Ask about preferences, needs, and requirements - NOT about specific products or product numbers

CRITICAL OUTPUT FORMAT (MUST FOLLOW EXACTLY):
QUESTION: [your question here]

STRICT RULES:
- Your response must start with "QUESTION:"
- Do NOT include any explanations, reasoning, or additional text
- Do NOT use bullets, multiple lines, or formatting
- Just the question format above, nothing else
- Ask about preferences, not about specific products
- Example: QUESTION: What's your budget range for this purchase?

Ask your highest expected value question:"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": base_prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=200
            )
            
            self.last_response = response.strip()
            
            if "QUESTION:" in response:
                return num_products
            else:
                print(f"Warning: Failed to parse question from response: {response}")
                return num_products
                
        except Exception as e:
            print(f"Error asking question: {e}")
            return num_products
    
    def _make_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                           dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Make final recommendation after asking all questions."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category)
        
        base_prompt = f"""You are a product recommendation agent for {category} products.

Context:
{context}

{feedback_context}

Task:
You have gathered information about the user's preferences through conversation. Now make your final recommendation based on all the information you've collected.

Output format (MUST be exactly one line, no extra text):
RECOMMEND: <number 0-{num_products-1}>

Rules:
- Choose the product that best matches the user's expressed preferences
- Consider all the information gathered from your questions
- No explanations, just the recommendation
- You must recommend exactly one product"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": base_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=100
            )
            self.last_response = response.strip()
            if "RECOMMEND:" in response:
                try:
                    product_idx = int(response.split("RECOMMEND:")[-1].strip())
                    if 0 <= product_idx < num_products:
                        return product_idx
                except (ValueError, IndexError):
                    pass
            
            print(f"Warning: Failed to parse recommendation from response: {response}")
            return 0
            
        except Exception as e:
            print(f"Error making recommendation: {e}")
            return 0
    
    
    def _get_product_info(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], num_products: int) -> List[Dict]:
        """Extract product information from observation."""
        products = []
        product_descriptions = info.get('product_descriptions', [])
        
        for i in range(num_products):
            if i < obs['product_features'].shape[0]:
                features = obs['product_features'][i]
                product = {
                    'id': info.get('product_ids', [])[i] if i < len(info.get('product_ids', [])) else i,
                    'price': float(features[0]) if not np.isnan(features[0]) else 0.0,
                    'store_hash': int(features[1]) if not np.isnan(features[1]) else 0,
                    'title_length': int(features[2]) if not np.isnan(features[2]) else 0,
                    'features': [float(f) for f in features[3:] if not np.isnan(f)],
                    'description': product_descriptions[i] if i < len(product_descriptions) else "No description available"
                }
                products.append(product)
        
        return products
    
    def _build_llm_context(self, products: List[Dict], dialog_history: List[Tuple[str, str]], category: str) -> str:
        """Build context string for LLM decision making."""
        product_list = f"Available {category} products:\n"
        for i, product in enumerate(products):
            description = product.get('description', 'No description available')
            product_list += f"{i}: Product ID {product['id']} - Price: ${product['price']:.2f}\n"
            product_list += f"   Description: {description}\n\n"
        
        dialog_text = "Conversation so far:\n"
        if dialog_history:
            for i, (question, answer) in enumerate(dialog_history):
                dialog_text += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"
        else:
            dialog_text += "No questions asked yet.\n"
        
        return f"{product_list}\n{dialog_text}"
    
    def _build_feedback_context(self, category: str) -> str:
        """Build context string from previous episodes."""
        if self.context_mode == "none" or not self.episode_history:
            return ""
        
        context_parts = ["Previous episodes:"]
        
        recent_episodes = self.episode_history[-3:] 
        
        for episode_data in recent_episodes:
            episode_num = episode_data.get('episode', 0)
            episode_category = episode_data.get('category', 'unknown')
            dialog = episode_data.get('dialog', [])
            selected_product_name = episode_data.get('selected_product_name', 'Unknown Product')
            feedback = episode_data.get('feedback', '')
            regret = episode_data.get('regret', 0.0)
            
            context_parts.append(f"Episode {episode_num}: [{episode_category}] (Regret: {regret:.1f})")
            
            if dialog:
                for i, (question, answer) in enumerate(dialog):
                    context_parts.append(f"  Q{i+1}: {question}")
                    context_parts.append(f"  A{i+1}: {answer}")
            
            context_parts.append(f"  Selected: {selected_product_name}")
            if feedback:
                context_parts.append(f"  Feedback: {feedback}")
            
            context_parts.append("")  
        
        return "\n".join(context_parts)
    
    def update_preferences(self, episode_result: Dict[str, Any]):
        """Store episode context for future episodes."""
        self.episode_count += 1
        
        if 'final_info' in episode_result:
            category = episode_result.get('category', 'unknown')
            feedback = episode_result['final_info'].get('feedback', '')
            full_dialog = episode_result.get('full_dialog', [])
            chosen_product_id = episode_result['final_info'].get('chosen_product_id', None)
            regret = episode_result['final_info'].get('regret', 0.0)
            
            filtered_dialog = []
            for entry in full_dialog:
                if isinstance(entry, tuple) and len(entry) == 2:
                    filtered_dialog.append(entry)
            
            product_name = "Unknown Product"
            if 'product_info' in episode_result and 'products_with_scores' in episode_result['product_info']:
                for product in episode_result['product_info']['products_with_scores']:
                    if product.get('id') == chosen_product_id:
                        product_name = product.get('name', 'Unknown Product')
                        break
            
            episode_data = {
                'episode': self.episode_count,
                'category': category,
                'dialog': filtered_dialog,  
                'selected_product_id': chosen_product_id,
                'selected_product_name': product_name,
                'feedback': feedback,
                'regret': regret
            }
            
            self.episode_history.append(episode_data)


def run_fixed_questions_experiment_dp(
    categories: List[str] = None,
    num_categories: int = 10,
    episodes_per_category: int = 1,  # 10 episodes total (10 categories * 1 episode each)
    model: str = "gpt-4o",
    feedback_type: str = "persona",
    min_score_threshold: float = 60.0,
    output_dir: str = "fixed_questions_dp_results",
    seed: Optional[int] = None,
    context_mode: str = "raw"
) -> Dict[str, Any]:
    """
    Run dynamic programming fixed questions experiment following experiment1 pattern.
    
    - Fixed 10 questions per episode with POMDP PLANNING prompting
    - 10 episodes total (10 categories * 1 episode each)
    - Fixed persona, changing categories based on seed
    - On episodes 1, 5, 10: force recommendation after each question to track regret progression
    - On episodes 2,3,4,6,7,8,9: run normally like experiment1 but with POMDP planning prompting
    - Between episodes: provide regret feedback from final recommendation after 10th question
    
    Args:
        categories: List of categories to test (None = randomly choose)
        num_categories: Number of categories to randomly select (total episodes = 10)
        episodes_per_category: Number of episodes per category (should be 1 for 10 total episodes)
        model: LLM model to use
        feedback_type: Type of feedback to provide
        min_score_threshold: Minimum score threshold for category relevance
        output_dir: Directory to save results
        seed: Random seed for reproducible selection
        context_mode: How to carry context between episodes
    """
    
    print(f"=== Dynamic Programming Fixed Questions Experiment: Persona Elicitation Effectiveness ===")
    print(f"Fixed questions per episode: 10 (with POMDP PLANNING prompting)")
    target_successful_episodes = num_categories * episodes_per_category
    print(f"Total episodes planned: {target_successful_episodes} (1 per category)")
    print(f"Tracking episodes: 1, 5, 10 (regret after each question)")
    normal_set = [e for e in range(1, target_successful_episodes + 1) if e not in {1, 5, 10}]
    print(f"Normal episodes: {', '.join(map(str, normal_set))} (standard experiment1 behavior with POMDP planning prompting)")
    print(f"Model: {model}, Feedback: {feedback_type}, Context: {context_mode}")
    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    persona_index = random.randint(0, 47000)
    print(f"Selected persona: {persona_index}")
    
    os.makedirs(output_dir, exist_ok=True)
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    from pipeline.core.feedback_system import FeedbackSystem
    from pipeline.core.user_model import UserModel
    
    if feedback_type == "persona":
        persona_agent = UserModel(persona_index)
        feedback_system = FeedbackSystem(feedback_type=feedback_type, persona_agent=persona_agent)
    else:
        feedback_system = FeedbackSystem(feedback_type=feedback_type)
    
    from pipeline.core.simulate_interaction import list_categories, get_products_by_category
    
    available_categories = list_categories()
    
    def is_category_relevant_for_persona(category, persona_index, min_score_threshold):
        """Check if a category is relevant for a specific persona."""
        try:
            products = get_products_by_category(category)
            if not products:
                return False, 0.0, []
                
            user_model = UserModel(persona_index)
            scores = user_model.score_products(category, products)
            if scores:
                max_score = max(score for _, score in scores)
                cached_scores = [(pid, score) for pid, score in scores]
                return max_score > min_score_threshold, max_score, cached_scores
            return False, 0.0, []
        except Exception as e:
            print(f"  Error checking category {category}: {e}")
            return False, 0.0, []

    def get_categories_for_seed(available_categories, seed):
        """Get categories in randomized order based on seed."""
        random.seed(seed)
        categories = available_categories.copy()
        random.shuffle(categories)
        random.seed() 
        return categories

    if categories is None:
        selected_categories = get_categories_for_seed(available_categories, seed)
        print(f"Categories selected with randomization from seed {seed}")
    else:
        selected_categories = [cat for cat in categories if cat in available_categories]
    
    print(f"Categories to test: {len(selected_categories)}")
    
    agent = DPFixedQuestionsAgent(
        model=model,
        context_mode=context_mode
    )
    
    all_results = []
    category_results = {}
    tracking_episodes = {1, 5, 10}  
    episode_num = 0
    successful_episodes_count = 0   
    planned_episodes_started = 0  
    print(f"\n=== Starting Episode Execution ===")
    print(f"Target: {target_successful_episodes} total episodes (exactly 10)")
    
    for category in selected_categories:
        if planned_episodes_started >= target_successful_episodes:
            break
            
        print(f"\n--- Testing Category: {category} ---")
        
        is_relevant, max_score, cached_scores = is_category_relevant_for_persona(category, persona_index, min_score_threshold)
        
        if not is_relevant:
            print(f"  ✗ Category {category}: Max score {max_score:.1f} ≤ {min_score_threshold}, skipping")
            continue
            
        print(f"  ✓ Category {category}: Max score {max_score:.1f} > {min_score_threshold}, proceeding")
        
        if category not in category_results:
            category_results[category] = []
        
        for episode in range(episodes_per_category):
            if planned_episodes_started >= target_successful_episodes:
                break

            episode_num += 1
            planned_episodes_started += 1
            planned_episode_index = planned_episodes_started
            is_tracking = planned_episode_index in tracking_episodes
            agent.set_tracking_episode(is_tracking)
            episode_type = "TRACKING" if is_tracking else "NORMAL"
            print(f"Episode {planned_episode_index} (Category: {category}, Type: {episode_type}) - {planned_episode_index}/{target_successful_episodes} planned episodes")

            try:
                env = RecoEnv(
                    persona_index=persona_index,
                    max_questions=10,  
                    categories=[category],  
                    agent=agent,
                    feedback_system=feedback_system,
                    cached_scores=cached_scores
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
                regret_progression = []  
                rank_progression = []  
                
                while not terminated and not truncated and step_count <= 21:  # 10 questions + 10 forced recs + 1 final recommendation
                    action = agent.get_action(obs, current_info)
                    obs, reward, terminated, truncated, info = metrics_wrapper.step(action)

                    current_info = info
                    step_count += 1

                    if info['action_type'] == 'ask':
                        if env.dialog_history:
                            last_qa = env.dialog_history[-1]
                            if isinstance(last_qa, tuple) and len(last_qa) >= 2:
                                question = last_qa[0]
                                answer = last_qa[1]
                                print(f"  Step {step_count}: Asked question: {question}")
                                print(f"  Step {step_count}: Received answer: {answer}")
                            else:
                                print(f"  Step {step_count}: Asked question")
                        else:
                            print(f"  Step {step_count}: Asked question")

                        if is_tracking and len(env.dialog_history) <= 10:
                            rec_action = agent.force_recommendation_after_question(obs, current_info)
                            chosen_product_id = env.product_ids[rec_action]
                            chosen_score = next((score for pid, score in env.oracle_scores if pid == chosen_product_id), 0.0)
                            best_id, best_score = env.oracle_scores[0] if env.oracle_scores else (chosen_product_id, chosen_score)
                            regret = max(0.0, best_score - chosen_score)
                            try:
                                ordered_product_ids = [pid for pid, _ in env.oracle_scores] if hasattr(env, 'oracle_scores') else []
                                if chosen_product_id in ordered_product_ids:
                                    chosen_rank = ordered_product_ids.index(chosen_product_id) + 1
                                else:
                                    chosen_rank = None
                            except Exception:
                                chosen_rank = None
                            print(f"    Forced recommendation after question (Regret: {regret:.1f}, Rank: {chosen_rank})")
                            regret_progression.append(regret)
                            rank_progression.append(chosen_rank)
                            if len(env.dialog_history) >= 10:
                                print(f"  Reached 10 questions, forcing final recommendation...")
                                final_action = agent._make_recommendation(obs, current_info, env.dialog_history, category, len(env.product_ids))
                                obs, reward, terminated, truncated, final_info = metrics_wrapper.step(final_action)
                                current_info = final_info
                                step_count += 1

                                print(f"  Step {step_count}: Final recommendation - Product {final_info['chosen_product_id']}")
                                print(f"    Score: {final_info['chosen_score']:.1f}, Best: {final_info['best_score']:.1f}")
                                print(f"    Top1: {final_info['top1']}, Top3: {final_info['top3']}")
                                if 'feedback' in final_info and final_info['feedback']:
                                    print(f"    Feedback: {final_info['feedback']}")
                                break
                            else:
                                continue
                        elif len(env.dialog_history) >= 10:
                            print(f"  Reached 10 questions, forcing final recommendation...")
                            final_action = agent._make_recommendation(obs, current_info, env.dialog_history, category, len(env.product_ids))
                            obs, reward, terminated, truncated, final_info = metrics_wrapper.step(final_action)
                            current_info = final_info
                            step_count += 1

                            print(f"  Step {step_count}: Final recommendation - Product {final_info['chosen_product_id']}")
                            print(f"    Score: {final_info['chosen_score']:.1f}, Best: {final_info['best_score']:.1f}")
                            print(f"    Top1: {final_info['top1']}, Top3: {final_info['top3']}")
                            if 'feedback' in final_info and final_info['feedback']:
                                print(f"    Feedback: {final_info['feedback']}")
                            break

                    elif info['action_type'] == 'recommend':
                        print(f"  Step {step_count}: Final recommendation - Product {info['chosen_product_id']}")
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
                
                if current_info.get('action_type') == 'recommend' and 'chosen_score' in current_info:
                   
                    try:
                        ordered_product_ids = [pid for pid, _ in env.oracle_scores] if hasattr(env, 'oracle_scores') else []
                        if current_info.get('chosen_product_id') in ordered_product_ids:
                            chosen_rank = ordered_product_ids.index(current_info.get('chosen_product_id')) + 1
                        else:
                            chosen_rank = None
                    except Exception:
                        chosen_rank = None

                    current_info['chosen_rank'] = chosen_rank
                    current_info['total_products'] = len(env.products) if hasattr(env, 'products') else None

                    episode_result = {
                        'episode': episode_num,
                        'category': category,
                        'persona_index': persona_index,
                        'episode_in_category': episode + 1,
                        'is_tracking_episode': is_tracking,
                        'steps': step_count,
                        'terminated': terminated,
                        'truncated': truncated,
                        'final_info': current_info,
                        'full_dialog': full_dialog,
                        'product_info': product_info,
                        'regret_progression': regret_progression if is_tracking else [],
                        'rank_progression': rank_progression if is_tracking else [],
                        'planned_episode_index': planned_episode_index,
                        'attempt_index': episode_num,
                        'questions_asked': [qa[0] for qa in env.dialog_history if isinstance(qa, tuple) and len(qa) >= 2],
                        'answers_received': [qa[1] for qa in env.dialog_history if isinstance(qa, tuple) and len(qa) >= 2],
                        'qa_pairs': [qa for qa in env.dialog_history if isinstance(qa, tuple) and len(qa) >= 2]
                    }

                    all_results.append(episode_result)
                    category_results[category].append(episode_result)
                    agent.update_preferences(episode_result)

                    print(f"  Episode {episode_num}: Successfully completed (Score: {current_info.get('chosen_score', 0):.1f})")
                    
                    qa_pairs = [qa for qa in env.dialog_history if isinstance(qa, tuple) and len(qa) >= 2]
                    print(f"    Questions and answers ({len(qa_pairs)}):")
                    for i, (question, answer) in enumerate(qa_pairs, 1):
                        print(f"      {i}. Q: {question}")
                        print(f"         A: {answer}")

                    successful_episodes_count += 1
                else:
                    print(f"  Episode {episode_num}: Skipped - No recommendation made or missing score data")

                metrics_wrapper.close()

            except Exception as e:
                print(f"  Error in episode {episode_num}: {e}")
                continue
    
    print(f"\n=== Results Analysis ===")
    print(f"Target successful episodes: {target_successful_episodes}")
    print(f"Actual successful episodes: {successful_episodes_count}")
    print(f"Categories tested: {len([cat for cat, results in category_results.items() if len(results) > 0])}")
    
    tracking_episodes_data = [r for r in all_results if r.get('is_tracking_episode', False)]
    normal_episodes_data = [r for r in all_results if not r.get('is_tracking_episode', False)]
    
    print(f"\nTracking episodes (1, 5, 10): {len(tracking_episodes_data)}")
    print(f"Normal episodes (2, 3, 4, 6, 7, 8, 9): {len(normal_episodes_data)}")
    
    tracking_episodes_analysis = []
    for episode in tracking_episodes_data:
        episode_analysis = {
            'episode': episode['episode'],
            'category': episode['category'],
            'final_regret': episode['final_info'].get('regret', 0),
            'final_rank': episode['final_info'].get('chosen_rank', None),
            'regret_progression': episode.get('regret_progression', []),
            'rank_progression': episode.get('rank_progression', []),
            'questions_and_answers': episode.get('qa_pairs', [])
        }
        tracking_episodes_analysis.append(episode_analysis)
    
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace(" ", "_")
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    results_file = os.path.join(output_dir, f"fixed_questions_dp_experiment_{model_safe_name}_{feedback_safe_name}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Dynamic Programming Fixed Questions: Persona Elicitation Effectiveness',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_episodes': len(all_results),
                'tracking_episodes': len(tracking_episodes_data),
                'normal_episodes': len(normal_episodes_data),
                'categories_tested': list(category_results.keys()),
                'avg_final_regret': np.mean([r['final_info'].get('regret', 0) for r in all_results if 'regret' in r['final_info']]),
                'tracking_episodes_avg_regret': np.mean([r['final_info'].get('regret', 0) for r in tracking_episodes_data if 'regret' in r['final_info']]),
                'normal_episodes_avg_regret': np.mean([r['final_info'].get('regret', 0) for r in normal_episodes_data if 'regret' in r['final_info']]),
                'avg_chosen_rank': float(np.mean([r['final_info'].get('chosen_rank', np.nan) for r in all_results if r['final_info'].get('chosen_rank') is not None])) if any(r['final_info'].get('chosen_rank') is not None for r in all_results) else None,
                'total_episodes_planned': target_successful_episodes,
                'tracking_episode_indices': [1, 5, 10]
            },
            'tracking_episodes_analysis': tracking_episodes_analysis,
            'config': {
                'persona_index': persona_index,
                'categories': categories,
                'episodes_per_category': episodes_per_category,
                'model': model,
                'feedback_type': feedback_type,
                'context_mode': context_mode,
                'seed': seed,
                'tracking_episodes': list(tracking_episodes)
            },
            'agent_episode_history': agent.episode_history,
            'category_results': {
                cat: {
                    'avg_score': np.mean([r['final_info'].get('chosen_score', 0) for r in results]),
                    'top1_rate': np.mean([r['final_info'].get('top1', False) for r in results]),
                    'episode_count': len(results),
                    'num_products': results[0]['product_info']['num_products'] if results else 0
                }
                for cat, results in category_results.items()
            },
            'all_episodes': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Individual episode metrics saved to: {output_dir}/episode_*.jsonl")
    
    return {
        'experiment': 'Dynamic Programming Fixed Questions: Persona Elicitation Effectiveness',
        'all_results': all_results,
        'category_results': category_results,
        'tracking_episodes_analysis': tracking_episodes_analysis
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Programming Fixed Questions Experiment")
    parser.add_argument("--categories", nargs="+", help="Categories to test")
    parser.add_argument("--num_categories", type=int, default=10, help="Number of categories (total episodes)")
    parser.add_argument("--episodes_per_category", type=int, default=1, help="Episodes per category (should be 1)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model")
    parser.add_argument("--feedback_type", default="persona", help="Feedback type")
    parser.add_argument("--min_score_threshold", type=float, default=60.0, help="Min score threshold")
    parser.add_argument("--output_dir", default="fixed_questions_dp_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--context_mode", choices=["raw", "summary", "none"], default="raw", 
                       help="Context mode between episodes")
    
    args = parser.parse_args()
    
    run_fixed_questions_experiment_dp(
        categories=args.categories,
        num_categories=args.num_categories,
        episodes_per_category=args.episodes_per_category,
        model=args.model,
        feedback_type=args.feedback_type,
        min_score_threshold=args.min_score_threshold,
        output_dir=args.output_dir,
        seed=args.seed,
        context_mode=args.context_mode
    )