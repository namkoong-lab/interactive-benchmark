#!/usr/bin/env python3
"""
Experiment: Fixed Questions for Persona Elicitation

Research Question: How many questions do LLMs need to fully elicit persona info?
- Fix number of questions = K
- Explicitly prompt agent to ask next question, then choose product after every question
- Measure regret at every turn over multiple episodes
- Compare against Oracle baseline
- Test if context from previous episodes reduces required questions
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
from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
from .wrappers.metrics_wrapper import MetricsWrapper


class FixedQuestionsAgent:
    """
    Agent that asks exactly 10 questions, then makes a recommendation.
    On tracking episodes (1, 5, 10), forces recommendation after each question to measure regret progression.
    On normal episodes, proceed normally like experiment1.
    """
    
    def __init__(self, model: str = "gpt-4o", context_mode: str = "raw"):
        self.model = model
        self.fixed_questions = 10  # Always 10 questions
        self.context_mode = context_mode
        self.episode_count = 0
        self.episode_history = []  # Store context from previous episodes
        self.current_episode_info = None
        self.current_question_count = 0
        self.last_response = None
        self.is_tracking_episode = False  # Whether this episode should track regret after each question
        
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
        
        # Get current dialog history
        dialog_history = []
        questions_remaining = self.fixed_questions
        if hasattr(self, 'current_env') and self.current_env and hasattr(self.current_env, 'dialog_history'):
            dialog_history = self.current_env.dialog_history
            self.current_question_count = len(dialog_history)
            if hasattr(self.current_env, 'questions_remaining'):
                questions_remaining = self.current_env.questions_remaining
        
        # If we've asked the fixed number of questions, make final recommendation
        if questions_remaining <= 0:
            return self._make_recommendation(obs, info, dialog_history, category, num_products)
        else:
            # On tracking episodes, after asking question, force a recommendation to measure regret
            if self.is_tracking_episode:
                # First ask the question
                question_action = self._ask_question(obs, info, dialog_history, category, num_products)
                # The environment will handle the question, then we'll force recommendation in next step
                return question_action
            else:
                # Normal episode - let agent choose between asking question or making recommendation (like experiment1)
                return self._choose_action(obs, info, dialog_history, category, num_products)
    
    def _choose_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                       dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Choose between asking a question or making a recommendation (like experiment1)."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category)
        
        # Build context about previous questions to avoid repetition
        previous_questions = [q for q, a in dialog_history]
        questions_context = ""
        if previous_questions:
            questions_context = f"Previous questions asked:\n" + "\n".join([f"- {q}" for q in previous_questions]) + "\n\n"
        
        base_prompt = f"""You are a product recommendation agent. Your goal is to find the best product for this user.

Context:
{context}

{feedback_context}

{questions_context}

Task:
You can either ask another question to learn more about the user's preferences, or make a recommendation if you feel you have enough information.

STRICT OUTPUT FORMAT - Choose ONE:
Either: QUESTION: [your question here]
Or: RECOMMEND: [product number 0-{num_products-1}]

Rules:
- If you want to ask a question, start your response with "QUESTION:"
- If you want to recommend, start your response with "RECOMMEND:"
- Do NOT include any explanations, reasoning, or additional text
- Do NOT use multiple lines or formatting
- Just the format above, nothing else

Examples:
QUESTION: What's your budget range for this purchase?
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
            
            # Parse response - look for exact format
            response_upper = response.upper().strip()
            
            if response_upper.startswith("RECOMMEND:"):
                # Extract product number
                try:
                    rec_part = response_upper.split("RECOMMEND:", 1)[1].strip()
                    # Extract first number from the recommendation
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
                # Fallback - ask a question
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
    
    def force_recommendation_after_question(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
        """Force a recommendation after asking a question (for regret tracking) and get confidence scores."""
        if 'num_products' in info:
            num_products = info['num_products']
            category = info['category']
        else:
            num_products = self.current_episode_info['num_products']
            category = self.current_episode_info['category']
        
        dialog_history = []
        if hasattr(self, 'current_env') and self.current_env and hasattr(self.current_env, 'dialog_history'):
            dialog_history = self.current_env.dialog_history
        
        # Force recommendation with confidence scores
        return self._make_recommendation_with_confidence(obs, info, dialog_history, category, num_products)
    
    def _ask_question(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                     dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Ask the next question in the sequence."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category)
        
        # Build context about previous questions to avoid repetition
        previous_questions = [q for q, a in dialog_history]
        questions_context = ""
        if previous_questions:
            questions_context = f"Previous questions asked:\n" + "\n".join([f"- {q}" for q in previous_questions]) + "\n\n"
        
        base_prompt = f"""You are a product recommendation agent. Your goal is to find the best product for this user.

Context:
{context}

{feedback_context}

{questions_context}

Task:
Ask ONE specific, helpful question to learn more about the user's preferences for {category} products. Make sure your question:
- Is different from previous questions
- Helps narrow down which product would be best
- Focuses on key decision factors (budget, style, features, usage, etc.)
- Uses everyday language a shopper understands

CRITICAL OUTPUT FORMAT (MUST FOLLOW EXACTLY):
QUESTION: [your question here]

STRICT RULES:
- Your response must start with "QUESTION:"
- Do NOT include any explanations, reasoning, or additional text
- Do NOT use bullets, multiple lines, or formatting
- Just the question format above, nothing else
- Example: QUESTION: What's your budget range for this purchase?

Ask your question:"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": base_prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=200
            )
            
            self.last_response = response.strip()
            
            # Parse question
            if "QUESTION:" in response:
                return num_products  # Ask question action
            else:
                # Fallback if parsing fails
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
            
            # Parse recommendation
            if "RECOMMEND:" in response:
                try:
                    product_idx = int(response.split("RECOMMEND:")[-1].strip())
                    if 0 <= product_idx < num_products:
                        return product_idx
                except (ValueError, IndexError):
                    pass
            
            # Fallback: return first product if parsing fails
            print(f"Warning: Failed to parse recommendation from response: {response}")
            return 0
            
        except Exception as e:
            print(f"Error making recommendation: {e}")
            return 0
    
    def _make_recommendation_with_confidence(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                                           dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> Tuple[int, Dict[str, float]]:
        """Make recommendation with confidence scores for tracking episodes."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category)
        
        base_prompt = f"""You are a product recommendation agent for {category} products.

Context:
{context}

{feedback_context}

Task:
Based on the information you've gathered so far, make a recommendation and provide confidence scores.

Output format (MUST be exactly as shown, no extra text):
RECOMMEND: <number 0-{num_products-1}>
CONFIDENCE_FAVORITE: <probability 0.0-1.0>
CONFIDENCE_TOP5: <probability 0.0-1.0>
CONFIDENCE_EXPECTED_SCORE: <expected score 0.0-100.0>
CONFIDENCE_EXPECTED_REGRET: <expected regret 0.0-100.0>

Rules:
- Choose the product that best matches the user's expressed preferences
- CONFIDENCE_FAVORITE: Probability this product will be the user's favorite (top 1)
- CONFIDENCE_TOP5: Probability this product will be in the user's top 5 favorites
- CONFIDENCE_EXPECTED_SCORE: Expected score the user would give this product (0-100)
- CONFIDENCE_EXPECTED_REGRET: Expected regret if this product is chosen (0-100, where 0 = no regret)
- All values must be numeric, no explanations"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": base_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=300
            )
            
            self.last_response = response.strip()
            
            # Parse recommendation and confidence scores
            product_idx = 0
            confidence_scores = {
                'confidence_favorite_prob': 0.0,
                'confidence_top5_prob': 0.0,
                'confidence_expected_score': 0.0,
                'confidence_expected_regret': 0.0
            }
            
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("RECOMMEND:"):
                    try:
                        product_idx = int(line.split("RECOMMEND:")[-1].strip())
                        if not (0 <= product_idx < num_products):
                            product_idx = 0
                    except (ValueError, IndexError):
                        product_idx = 0
                elif line.startswith("CONFIDENCE_FAVORITE:"):
                    try:
                        confidence_scores['confidence_favorite_prob'] = float(line.split("CONFIDENCE_FAVORITE:")[-1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("CONFIDENCE_TOP5:"):
                    try:
                        confidence_scores['confidence_top5_prob'] = float(line.split("CONFIDENCE_TOP5:")[-1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("CONFIDENCE_EXPECTED_SCORE:"):
                    try:
                        confidence_scores['confidence_expected_score'] = float(line.split("CONFIDENCE_EXPECTED_SCORE:")[-1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("CONFIDENCE_EXPECTED_REGRET:"):
                    try:
                        confidence_scores['confidence_expected_regret'] = float(line.split("CONFIDENCE_EXPECTED_REGRET:")[-1].strip())
                    except (ValueError, IndexError):
                        pass
            
            # Clamp confidence values to valid ranges
            confidence_scores['confidence_favorite_prob'] = max(0.0, min(1.0, confidence_scores['confidence_favorite_prob']))
            confidence_scores['confidence_top5_prob'] = max(0.0, min(1.0, confidence_scores['confidence_top5_prob']))
            confidence_scores['confidence_expected_score'] = max(0.0, min(100.0, confidence_scores['confidence_expected_score']))
            confidence_scores['confidence_expected_regret'] = max(0.0, min(100.0, confidence_scores['confidence_expected_regret']))
            
            return product_idx, confidence_scores
            
        except Exception as e:
            print(f"Error making recommendation with confidence: {e}")
            return 0, {
                'confidence_favorite_prob': 0.0,
                'confidence_top5_prob': 0.0,
                'confidence_expected_score': 0.0,
                'confidence_expected_regret': 0.0
            }
    
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
        
        # Show recent episodes for context
        recent_episodes = self.episode_history[-3:]  # Last 3 episodes
        
        for episode_data in recent_episodes:
            episode_num = episode_data.get('episode', 0)
            episode_category = episode_data.get('category', 'unknown')
            dialog = episode_data.get('dialog', [])
            selected_product_name = episode_data.get('selected_product_name', 'Unknown Product')
            feedback = episode_data.get('feedback', '')
            regret = episode_data.get('regret', 0.0)
            
            context_parts.append(f"Episode {episode_num}: [{episode_category}] (Regret: {regret:.1f})")
            
            # Add dialog summary
            if dialog:
                for i, (question, answer) in enumerate(dialog):
                    context_parts.append(f"  Q{i+1}: {question}")
                    context_parts.append(f"  A{i+1}: {answer}")
            
            # Add outcome
            context_parts.append(f"  Selected: {selected_product_name}")
            if feedback:
                context_parts.append(f"  Feedback: {feedback}")
            
            context_parts.append("")  # Empty line for readability
        
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
            
            # Filter dialog to only include Q&A pairs, not intermediate recommendations
            # The dialog should only contain actual questions and answers, not forced recommendations
            filtered_dialog = []
            for entry in full_dialog:
                if isinstance(entry, tuple) and len(entry) == 2:
                    # This is a Q&A pair
                    filtered_dialog.append(entry)
                # Skip any forced recommendation entries
            
            # Get product name from product_info
            product_name = "Unknown Product"
            if 'product_info' in episode_result and 'products_with_scores' in episode_result['product_info']:
                for product in episode_result['product_info']['products_with_scores']:
                    if product.get('id') == chosen_product_id:
                        product_name = product.get('name', 'Unknown Product')
                        break
            
            # Store episode context with filtered dialog (only Q&A, no intermediate recommendations)
            episode_data = {
                'episode': self.episode_count,
                'category': category,
                'dialog': filtered_dialog,  # Only Q&A pairs, no forced recommendations
                'selected_product_id': chosen_product_id,
                'selected_product_name': product_name,
                'feedback': feedback,
                'regret': regret
            }
            
            self.episode_history.append(episode_data)


def run_fixed_questions_experiment(
    categories: List[str] = None,
    num_categories: int = 10,
    episodes_per_category: int = 1,  # 10 episodes total (10 categories * 1 episode each)
    model: str = "gpt-4o",
    feedback_type: str = "persona",
    min_score_threshold: float = 60.0,
    output_dir: str = "fixed_questions_results",
    seed: Optional[int] = None,
    context_mode: str = "raw"
) -> Dict[str, Any]:
    """
    Run fixed questions experiment following experiment1 pattern.
    
    - Fixed 10 questions per episode
    - 10 episodes total (10 categories * 1 episode each)
    - Fixed persona, changing categories based on seed
    - On episodes 1, 5, 10: force recommendation after each question to track regret progression
    - On episodes 2,3,4,6,7,8,9: run normally like experiment1
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
    
    print(f"=== Fixed Questions Experiment: Persona Elicitation Effectiveness ===")
    print(f"Fixed questions per episode: 10")
    target_successful_episodes = num_categories * episodes_per_category
    print(f"Total episodes planned: {target_successful_episodes} (1 per category)")
    print(f"Tracking episodes: 1, 5, 10 (regret after each question)")
    normal_set = [e for e in range(1, target_successful_episodes + 1) if e not in {1, 5, 10}]
    print(f"Normal episodes: {', '.join(map(str, normal_set))} (standard experiment1 behavior)")
    print(f"Model: {model}, Feedback: {feedback_type}, Context: {context_mode}")
    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    # Select persona (fixed for all episodes like experiment1)
    persona_index = random.randint(0, 47000)
    print(f"Selected persona: {persona_index}")
    
    os.makedirs(output_dir, exist_ok=True)
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    # Create feedback system
    from .core.feedback_system import FeedbackSystem
    from .core.user_model import UserModel
    
    if feedback_type == "persona":
        persona_agent = UserModel(persona_index)
        feedback_system = FeedbackSystem(feedback_type=feedback_type, persona_agent=persona_agent)
    else:
        feedback_system = FeedbackSystem(feedback_type=feedback_type)
    
    from .core.simulate_interaction import list_categories, get_products_by_category
    
    # Get available categories (following experiment1 pattern)
    available_categories = list_categories()
    
    # Dynamic category filtering - check relevance as we encounter categories
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
                # Convert to the format expected by RecoEnv: (product_id, score) tuples
                cached_scores = [(pid, score) for pid, score in scores]
                return max_score > min_score_threshold, max_score, cached_scores
            return False, 0.0, []
        except Exception as e:
            print(f"  Error checking category {category}: {e}")
            return False, 0.0, []
    
    # Category selection with proper seed-based randomization (like experiment1)
    def get_categories_for_seed(available_categories, seed):
        """Get categories in randomized order based on seed."""
        # Set seed for reproducible category ordering
        random.seed(seed)
        categories = available_categories.copy()
        # Shuffle categories based on seed for proper randomization
        random.shuffle(categories)
        random.seed()  # Reset seed after use
        return categories

    # Initialize category selection with proper randomization
    if categories is None:
        # Get all categories in randomized order based on seed
        selected_categories = get_categories_for_seed(available_categories, seed)
        print(f"Categories selected with randomization from seed {seed}")
    else:
        # Use provided categories, filtered by availability
        selected_categories = [cat for cat in categories if cat in available_categories]
    
    print(f"Categories to test: {len(selected_categories)}")
    
    # Create agent
    agent = FixedQuestionsAgent(
        model=model,
        context_mode=context_mode
    )
    
                # Track results
    all_results = []
    category_results = {}
    tracking_episodes = {1, 5, 10}  # Episodes that track regret after each question
    episode_num = 0
    successful_episodes_count = 0
    planned_episodes_started = 0  # counts episodes we actually start (including failures)
    confidence_progression_data = []  # Store confidence scores for tracking episodes
    print(f"\n=== Starting Episode Execution ===")
    print(f"Target: {target_successful_episodes} total episodes (exactly 10)")
    
    # Simple approach: test categories one by one until we complete exactly 10 episodes
    for category in selected_categories:
        if planned_episodes_started >= target_successful_episodes:
            break
            
        print(f"\n--- Testing Category: {category} ---")
        
        # Check if category is relevant for this persona
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
                    max_questions=10,  # Always 10 questions
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
                confidence_progression = []  # Store confidence scores for this episode
                rank_progression = []  # Store ranks for forced recommendations in tracking episodes
                
                while not terminated and not truncated and step_count <= 21:  # 10 questions + 10 forced recs + 1 final recommendation
                    action = agent.get_action(obs, current_info)
                    obs, reward, terminated, truncated, info = metrics_wrapper.step(action)

                    current_info = info
                    step_count += 1

                    if info['action_type'] == 'ask':
                        # Print the question that was asked and the answer received
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

                        # For tracking episodes, force recommendation after each question (including the 10th)
                        if is_tracking and len(env.dialog_history) <= 10:
                            rec_action, confidence_scores = agent.force_recommendation_after_question(obs, current_info)

                            # Calculate regret manually using the same logic as the environment
                            chosen_product_id = env.product_ids[rec_action]
                            chosen_score = next((score for pid, score in env.oracle_scores if pid == chosen_product_id), 0.0)
                            best_id, best_score = env.oracle_scores[0] if env.oracle_scores else (chosen_product_id, chosen_score)
                            regret = max(0.0, best_score - chosen_score)

                            # Calculate rank of chosen product (1 = best)
                            try:
                                ordered_product_ids = [pid for pid, _ in env.oracle_scores] if hasattr(env, 'oracle_scores') else []
                                if chosen_product_id in ordered_product_ids:
                                    chosen_rank = ordered_product_ids.index(chosen_product_id) + 1
                                else:
                                    chosen_rank = None
                            except Exception:
                                chosen_rank = None

                            # Do NOT increment step_count here: forced recommendation is not an env step
                            print(f"    Forced recommendation after question (Regret: {regret:.1f}, Rank: {chosen_rank})")
                            print(f"    Confidence - Favorite: {confidence_scores['confidence_favorite_prob']:.2f}, Top5: {confidence_scores['confidence_top5_prob']:.2f}, Expected Score: {confidence_scores['confidence_expected_score']:.1f}, Expected Regret: {confidence_scores['confidence_expected_regret']:.1f}")
                            regret_progression.append(regret)
                            confidence_progression.append(confidence_scores.copy())
                            rank_progression.append(chosen_rank)

                            # If this was the 10th question, now do the final recommendation
                            if len(env.dialog_history) >= 10:
                                print(f"  Reached 10 questions, forcing final recommendation...")
                                # Force final recommendation
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
                                # Continue the loop to ask the next question (don't break)
                                continue

                        # Check if we've reached 10 questions and need to force final recommendation (for non-tracking episodes)
                        elif len(env.dialog_history) >= 10:
                            print(f"  Reached 10 questions, forcing final recommendation...")
                            # Force final recommendation
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
                
                # To keep regret and displayed scores consistent, use the environment's
                # oracle_scores (already averaged across providers) instead of
                # re-scoring here.
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
                
                # Only count episodes that successfully made a recommendation
                if current_info.get('action_type') == 'recommend' and 'chosen_score' in current_info:
                    # Compute chosen product's rank among oracle-ranked products (1 = best)
                    try:
                        ordered_product_ids = [pid for pid, _ in env.oracle_scores] if hasattr(env, 'oracle_scores') else []
                        if current_info.get('chosen_product_id') in ordered_product_ids:
                            chosen_rank = ordered_product_ids.index(current_info.get('chosen_product_id')) + 1
                        else:
                            chosen_rank = None
                    except Exception:
                        chosen_rank = None

                    # Enhance final_info with rank and total products for clarity
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
                        'confidence_progression': confidence_progression if is_tracking else [],
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
                    
                    # Print summary of questions and answers
                    qa_pairs = [qa for qa in env.dialog_history if isinstance(qa, tuple) and len(qa) >= 2]
                    print(f"    Questions and answers ({len(qa_pairs)}):")
                    for i, (question, answer) in enumerate(qa_pairs, 1):
                        print(f"      {i}. Q: {question}")
                        print(f"         A: {answer}")

                    # Only increment after episode succeeds
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
    
    # Analyze tracking vs normal episodes
    tracking_episodes_data = [r for r in all_results if r.get('is_tracking_episode', False)]
    normal_episodes_data = [r for r in all_results if not r.get('is_tracking_episode', False)]
    
    print(f"\nTracking episodes (1, 5, 10): {len(tracking_episodes_data)}")
    print(f"Normal episodes (2, 3, 4, 6, 7, 8, 9): {len(normal_episodes_data)}")
    
    # Calculate regret and confidence progression for tracking episodes
    tracking_episodes_analysis = []
    for episode in tracking_episodes_data:
        episode_analysis = {
            'episode': episode['episode'],
            'category': episode['category'],
            'final_regret': episode['final_info'].get('regret', 0),
            'final_rank': episode['final_info'].get('chosen_rank', None),
            'regret_progression': episode.get('regret_progression', []),
            'confidence_progression': episode.get('confidence_progression', []),
            'rank_progression': episode.get('rank_progression', []),
            'questions_and_answers': episode.get('qa_pairs', [])
        }
        tracking_episodes_analysis.append(episode_analysis)
    
    # Save final results
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace(" ", "_")
    results_file = os.path.join(output_dir, f"fixed_questions_experiment_{model_safe_name}_{feedback_safe_name}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Fixed Questions: Persona Elicitation Effectiveness',
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
        'experiment': 'Fixed Questions: Persona Elicitation Effectiveness',
        'all_results': all_results,
        'category_results': category_results,
        'tracking_episodes_analysis': tracking_episodes_analysis
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Questions Experiment")
    parser.add_argument("--categories", nargs="+", help="Categories to test")
    parser.add_argument("--num_categories", type=int, default=10, help="Number of categories (total episodes)")
    parser.add_argument("--episodes_per_category", type=int, default=1, help="Episodes per category (should be 1)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model")
    parser.add_argument("--feedback_type", default="persona", help="Feedback type")
    parser.add_argument("--min_score_threshold", type=float, default=60.0, help="Min score threshold")
    parser.add_argument("--output_dir", default="fixed_questions_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--context_mode", choices=["raw", "summary", "none"], default="raw", 
                       help="Context mode between episodes")
    
    args = parser.parse_args()
    
    run_fixed_questions_experiment(
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
