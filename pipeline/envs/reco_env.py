import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import json
import random

from ..core.simulate_interaction import get_products_by_category, list_categories
from ..core.user_model import UserModel
from ..core.feedback_system import FeedbackSystem


class RecoEnv(gym.Env):
    """
    Interactive recommendation environment that tests question-asking ability.
    
    Actions:
    - 0 to (num_products-1): recommend product i
    - num_products: ask a question (LLM generates question dynamically)
    
    Observations:
    - Product features (price, store, title_length, etc.)
    - Dialog history (asked questions and answers)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, 
                 persona_index: int = 0,
                 max_questions: int = 20,
                 categories: Optional[List[str]] = None,
                 seed: Optional[int] = None,
                 agent: Optional[Any] = None,
                 feedback_system: Optional[FeedbackSystem] = None,
                 cached_scores: Optional[List[Tuple[int, float]]] = None):
        super().__init__()
        
        self.persona_index = persona_index
        self.max_questions = max_questions
        self.categories = categories or list_categories()
        self.seed = seed
        self.agent = agent
        self.last_agent_response = None  # Store agent's last response for question extraction
        self.cached_scores = cached_scores  # Pre-computed scores to avoid re-scoring
        
        # Initialize user model
        self.user_model = UserModel(persona_index)
        
        # Initialize feedback system
        self.feedback_system = feedback_system or FeedbackSystem(feedback_type="regret")
        
        # Will be set during reset
        self.current_category = None
        self.products = []
        self.product_ids = []
        self.oracle_scores = []
        self.dialog_history = []  # List of (question, answer) tuples
        self.questions_remaining = max_questions
        self.episode_step = 0
        
        # Action space: recommend products + ask question
        self.action_space = spaces.Discrete(1000)  # Will be resized in reset, increased for larger product sets
        
        # Observation space: product features + dialog state
        self.observation_space = spaces.Dict({
            "product_features": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(500, 10), dtype=np.float32  # max 500 products, 10 features each
            ),
            "dialog_history": spaces.Box(
                low=-1, high=1, shape=(max_questions * 2, 50), dtype=np.float32  # questions + answers, max 50 chars each
            ),
            "category_encoded": spaces.Box(
                low=0, high=1, shape=(10,), dtype=np.float32  # one-hot category
            )
        })
        
        # Episode tracking
        self.episode_count = 0
        self.category_episode_counts = {cat: 0 for cat in self.categories}
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
        
        # Select next category (round-robin)
        self.current_category = self.categories[self.episode_count % len(self.categories)]
        self.category_episode_counts[self.current_category] += 1
        
        # Fetch products for this category
        self.products = get_products_by_category(self.current_category)
        if not self.products:
            # Fallback to any category with products
            for cat in self.categories:
                self.products = get_products_by_category(cat)
                if self.products:
                    self.current_category = cat
                    break
        
        if not self.products:
            raise RuntimeError("No products found in any category")
        
        # Use all products from the category (no sampling limit)
        # Note: This may result in large product sets for some categories
        
        self.product_ids = [p["id"] for p in self.products]
        
        # Compute oracle scores (hidden ground truth) - use cached scores if available
        if self.cached_scores:
            # Use pre-computed scores to avoid re-scoring
            # cached_scores is a list of (product_id, score) tuples
            self.oracle_scores = [(pid, score) for pid, score in self.cached_scores 
                                 if pid in self.product_ids]
            self.oracle_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
        else:
            # Fallback to scoring if no cached scores provided
            self.oracle_scores = self.user_model.score_products(self.current_category, self.products)
            self.oracle_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
        
        # Reset episode state
        self.dialog_history = []
        self.questions_remaining = self.max_questions
        self.episode_step = 0
        
        # Resize action space: products + 1 for "ask question"
        num_products = len(self.products)
        self.action_space = spaces.Discrete(num_products + 1)
        
        # Build initial observation
        obs = self._build_observation()
        
        info = {
            "category": self.current_category,
            "num_products": num_products,
            "action_map": {
                "recommend": list(range(num_products)),
                "ask": [num_products]
            },
            "product_ids": self.product_ids,
            "oracle_best_id": self.oracle_scores[0][0] if self.oracle_scores else None,
            "oracle_best_score": self.oracle_scores[0][1] if self.oracle_scores else 0.0
        }
        
        self.episode_count += 1
        return obs, info
    
    def step(self, action: int):
        self.episode_step += 1
        
        num_products = len(self.products)
        
        # Debug output
        
        if action < num_products:
            # Recommend action
            chosen_product_id = self.product_ids[action]
            chosen_score = next((score for pid, score in self.oracle_scores if pid == chosen_product_id), 0.0)
            best_id, best_score = self.oracle_scores[0] if self.oracle_scores else (chosen_product_id, chosen_score)
            
            # Calculate regret for feedback (no reward system)
            regret = max(0.0, best_score - chosen_score)
            
            # Get chosen product information
            chosen_product = next((p for p in self.products if p["id"] == chosen_product_id), None)
            
            # Generate feedback using the feedback system
            feedback = self.feedback_system.generate_feedback(
                chosen_score=chosen_score,
                best_score=best_score,
                regret=regret,
                chosen_product=chosen_product,
                available_products=self.products,
                category=self.current_category
            )
            
            # No reward system - agent motivated only by prompts
            reward = 0.0
            
            terminated = True
            truncated = False
            
            info = {
                "action_type": "recommend",
                "chosen_product_id": chosen_product_id,
                "chosen_score": chosen_score,
                "best_product_id": best_id,
                "best_score": best_score,
                "regret": regret,
                "top1": chosen_product_id == best_id,
                "top3": chosen_product_id in [pid for pid, _ in self.oracle_scores[:3]],
                "questions_asked": len(self.dialog_history),
                "feedback": feedback,
                "feedback_type": self.feedback_system.get_feedback_type()
            }
            
            # Return zero observation for terminal state
            obs = self._build_observation()
            for key in obs:
                obs[key].fill(0)
            
        elif action == num_products:
            if self.questions_remaining <= 0:
                reward = 0.0
                terminated = False
                truncated = False  
                info = {"action_type": "force_recommendation", "message": "No more questions allowed, must recommend now"}
            else:
                # Extract question from agent's last response
                if self.agent and hasattr(self.agent, 'last_response') and self.agent.last_response:
                    response = self.agent.last_response
                    if "QUESTION:" in response.upper():
                        question_text = response.split("QUESTION:", 1)[1].strip()
                        # Clean up the question
                        if not question_text.endswith('?'):
                            question_text += "?"
                    else:
                        question_text = "What are your preferences for this product category?"
                else:
                    # Fallback to simple question
                    question_text = "What are your preferences for this product category?"
                
                # Get user response
                answer = self.user_model.respond(question_text)
                
                # Add to dialog history
                self.dialog_history.append((question_text, answer))
                self.questions_remaining -= 1
                
                reward = 0.0
                terminated = False
                truncated = False  # Don't truncate immediately, let agent make recommendation next
                
                info = {
                    "action_type": "ask",
                    "question_text": question_text,
                    "answer": answer,
                    "dialog_length": len(self.dialog_history)
                }
            
            obs = self._build_observation()
        
        else:
            reward = 0.0
            terminated = False
            truncated = False
            info = {"action_type": "invalid", "action": action}
            obs = self._build_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Build observation from current state."""
        num_products = len(self.products)
        max_products = 500  
        max_dialog_entries = self.max_questions * 2  # questions + answers
        
        # Product features
        product_features = np.zeros((max_products, 10), dtype=np.float32)
        for i, product in enumerate(self.products[:max_products]):
            # Feature 0: normalized price
            price = product.get("price", 0) or 0
            product_features[i, 0] = min(price / 1000.0, 1.0)  # Normalize to [0,1]
            
            # Feature 1: store encoding (simple hash)
            store = str(product.get("store", ""))
            product_features[i, 1] = hash(store) % 100 / 100.0
            
            # Feature 2: title length
            title = str(product.get("title", ""))
            product_features[i, 2] = min(len(title) / 100.0, 1.0)
            
            # Features 3-9: raw attributes (simplified)
            raw = product.get("raw", {})
            for j, (key, value) in enumerate(list(raw.items())[:7]):
                if isinstance(value, (int, float)):
                    product_features[i, 3 + j] = min(abs(value) / 100.0, 1.0)
                elif isinstance(value, str):
                    product_features[i, 3 + j] = len(value) / 50.0
        
        # Dialog history (questions and answers as character embeddings)
        dialog_history = np.zeros((max_dialog_entries, 50), dtype=np.float32)
        for i, (question, answer) in enumerate(self.dialog_history[:self.max_questions]):
            # Encode question (even indices)
            question_chars = [ord(c) % 128 for c in question[:50]]
            dialog_history[i * 2, :len(question_chars)] = np.array(question_chars, dtype=np.float32) / 127.0
            
            # Encode answer (odd indices)
            answer_chars = [ord(c) % 128 for c in answer[:50]]
            dialog_history[i * 2 + 1, :len(answer_chars)] = np.array(answer_chars, dtype=np.float32) / 127.0
        
        # Category encoding (simple hash)
        category_encoded = np.zeros(10, dtype=np.float32)
        if self.current_category:
            cat_hash = hash(self.current_category) % 10
            category_encoded[cat_hash] = 1.0
        
        return {
            "product_features": product_features,
            "dialog_history": dialog_history,
            "category_encoded": category_encoded
        }
    
    def render(self, mode="human"):
        if mode == "human":
            print(f"\n=== Episode {self.episode_count} ===")
            print(f"Category: {self.current_category}")
            print(f"Products: {len(self.products)}")
            print(f"Questions asked: {len(self.dialog_history)}")
            if self.dialog_history:
                print("Dialog history:")
                for i, (q, a) in enumerate(self.dialog_history):
                    print(f"  Q{i+1}: {q}")
                    print(f"  A{i+1}: {a}")
            if self.oracle_scores:
                best_id, best_score = self.oracle_scores[0]
                print(f"Oracle best: {best_id} (score: {best_score:.1f})")
    
    def close(self):
        pass
