import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import json
import random

try:
    from ..simulate_interaction import get_products_by_category, list_categories
    from ..user_model import UserModel
except ImportError:
    from simulate_interaction import get_products_by_category, list_categories
    from user_model import UserModel


class QuestionTemplate:
    """Template for generating questions from product attributes."""
    
    def __init__(self, template_id: int, template_text: str, attribute_key: str, 
                 attribute_type: str, values: List[Any] = None):
        self.template_id = template_id
        self.template_text = template_text
        self.attribute_key = attribute_key
        self.attribute_type = attribute_type  # "price_bin", "store", "boolean", "numeric_bin"
        self.values = values or []
    
    def generate_question(self, value: Any = None) -> str:
        if self.attribute_type == "price_bin":
            return self.template_text.format(price=value)
        elif self.attribute_type == "store":
            return self.template_text.format(store=value)
        elif self.attribute_type == "boolean":
            return self.template_text.format(feature=value)
        elif self.attribute_type == "numeric_bin":
            return self.template_text.format(feature=value, threshold=value)
        else:
            return self.template_text


def build_question_templates(products: List[Dict[str, Any]]) -> List[QuestionTemplate]:
    """Build question templates from product attributes in the category."""
    templates = []
    template_id = 0
    
    # Price-based questions
    prices = [p.get("price", 0) for p in products if p.get("price") is not None]
    if prices:
        price_quantiles = np.percentile(prices, [25, 50, 75])
        for i, q in enumerate(price_quantiles):
            templates.append(QuestionTemplate(
                template_id=template_id,
                template_text="Do you prefer products under ${price:.0f}?",
                attribute_key="price",
                attribute_type="price_bin",
                values=[q]
            ))
            template_id += 1
    
    # Store-based questions
    stores = list(set(p.get("store") for p in products if p.get("store")))
    for store in stores[:5]:  # Limit to top 5 stores
        templates.append(QuestionTemplate(
            template_id=template_id,
            template_text="Do you prefer products from {store}?",
            attribute_key="store",
            attribute_type="store",
            values=[store]
        ))
        template_id += 1
    
    # Feature-based questions from raw attributes
    all_attrs = set()
    for p in products:
        raw = p.get("raw", {})
        for k, v in raw.items():
            if isinstance(v, (str, bool)) and k not in {"title", "description"}:
                all_attrs.add(k)
    
    for attr in list(all_attrs)[:10]:  # Limit to 10 features
        templates.append(QuestionTemplate(
            template_id=template_id,
            template_text="Do you care about {feature}?",
            attribute_key=attr,
            attribute_type="boolean",
            values=[attr]
        ))
        template_id += 1
    
    return templates


class RecoEnv(gym.Env):
    """
    Interactive recommendation environment that tests question-asking ability.
    
    Actions:
    - 0 to (num_products-1): recommend product i
    - num_products to (num_products+num_questions-1): ask question j
    
    Observations:
    - Product features (price, store, title_length, etc.)
    - Asked questions mask (binary vector)
    - Question answers (encoded responses)
    - Remaining question budget
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, 
                 persona_index: int = 0,
                 max_questions: int = 8,
                 categories: Optional[List[str]] = None,
                 seed: Optional[int] = None):
        super().__init__()
        
        self.persona_index = persona_index
        self.max_questions = max_questions
        self.categories = categories or list_categories()
        self.seed = seed
        
        # Initialize user model
        self.user_model = UserModel(persona_index)
        
        # Will be set during reset
        self.current_category = None
        self.products = []
        self.product_ids = []
        self.question_templates = []
        self.oracle_scores = []
        self.asked_questions = set()
        self.question_answers = {}
        self.questions_remaining = max_questions
        self.episode_step = 0
        
        # Action space: recommend products + ask questions
        self.action_space = spaces.Discrete(100)  # Will be resized in reset
        
        # Observation space: product features + dialog state
        self.observation_space = spaces.Dict({
            "product_features": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(50, 10), dtype=np.float32  # max 50 products, 10 features each
            ),
            "asked_questions": spaces.Box(
                low=0, high=1, shape=(20,), dtype=np.float32  # max 20 questions
            ),
            "question_answers": spaces.Box(
                low=-1, high=1, shape=(20,), dtype=np.float32  # -1=unknown, 0=no, 1=yes
            ),
            "budget_remaining": spaces.Box(
                low=0, high=max_questions, shape=(1,), dtype=np.float32
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
        
        # Limit to reasonable number of products
        if len(self.products) > 50:
            self.products = random.sample(self.products, 50)
        
        self.product_ids = [p["id"] for p in self.products]
        
        # Build question templates for this category
        self.question_templates = build_question_templates(self.products)
        
        # Compute oracle scores (hidden ground truth)
        self.oracle_scores = self.user_model.score_products(self.current_category, self.products)
        self.oracle_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
        
        # Reset episode state
        self.asked_questions = set()
        self.question_answers = {}
        self.questions_remaining = self.max_questions
        self.episode_step = 0
        
        # Resize action space
        num_products = len(self.products)
        num_questions = len(self.question_templates)
        self.action_space = spaces.Discrete(num_products + num_questions)
        
        # Build initial observation
        obs = self._build_observation()
        
        info = {
            "category": self.current_category,
            "num_products": num_products,
            "num_questions": num_questions,
            "action_map": {
                "recommend": list(range(num_products)),
                "ask": list(range(num_products, num_products + num_questions))
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
        num_questions = len(self.question_templates)
        
        # Debug output
        print(f"[DEBUG] Action: {action}, num_products: {num_products}, num_questions: {num_questions}")
        
        if action < num_products:
            # Recommend action
            chosen_product_id = self.product_ids[action]
            chosen_score = next((score for pid, score in self.oracle_scores if pid == chosen_product_id), 0.0)
            best_id, best_score = self.oracle_scores[0] if self.oracle_scores else (chosen_product_id, chosen_score)
            
            # Compute reward
            regret = max(0.0, best_score - chosen_score)
            score_reward = chosen_score / 100.0  # Normalize to [0,1]
            regret_reward = -regret / 100.0  # Negative regret reward
            
            # Efficiency bonus for unused questions
            efficiency_bonus = (self.questions_remaining / self.max_questions) * 0.1
            
            reward = score_reward + efficiency_bonus
            
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
                "questions_asked": len(self.asked_questions),
                "efficiency_bonus": efficiency_bonus,
                "score_reward": score_reward,
                "regret_reward": regret_reward
            }
            
            # Return zero observation for terminal state
            obs = self._build_observation()
            for key in obs:
                obs[key].fill(0)
            
        else:
            # Ask question action
            question_idx = action - num_products
            if question_idx >= num_questions or question_idx in self.asked_questions:
                # Invalid question - small penalty
                reward = -0.1
                terminated = False
                truncated = False
                info = {"action_type": "invalid_question", "question_idx": question_idx}
            else:
                # Ask the question
                template = self.question_templates[question_idx]
                question_text = template.generate_question()
                
                # Get user response
                answer = self.user_model.respond(question_text)
                
                # Simple answer encoding: -1=unknown, 0=no, 1=yes
                answer_encoded = 0.0  # Default to "no"
                if any(word in answer.lower() for word in ["yes", "yep", "sure", "like", "prefer"]):
                    answer_encoded = 1.0
                elif any(word in answer.lower() for word in ["no", "nope", "don't", "not"]):
                    answer_encoded = 0.0
                else:
                    answer_encoded = 0.5  # Neutral/uncertain
                
                self.asked_questions.add(question_idx)
                self.question_answers[question_idx] = answer_encoded
                self.questions_remaining -= 1
                
                # Small step penalty to encourage efficiency
                reward = -0.01
                terminated = False
                truncated = (self.questions_remaining <= 0)
                
                info = {
                    "action_type": "ask",
                    "question_idx": question_idx,
                    "question_text": question_text,
                    "answer": answer,
                    "answer_encoded": answer_encoded,
                    "questions_remaining": self.questions_remaining
                }
            
            obs = self._build_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Build observation from current state."""
        num_products = len(self.products)
        max_products = 50
        max_questions = 20
        
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
        
        # Asked questions mask
        asked_questions = np.zeros(max_questions, dtype=np.float32)
        for q_idx in self.asked_questions:
            if q_idx < max_questions:
                asked_questions[q_idx] = 1.0
        
        # Question answers
        question_answers = np.full(max_questions, -1.0, dtype=np.float32)  # -1 = unknown
        for q_idx, answer in self.question_answers.items():
            if q_idx < max_questions:
                question_answers[q_idx] = answer
        
        # Budget remaining
        budget_remaining = np.array([self.questions_remaining / self.max_questions], dtype=np.float32)
        
        # Category encoding (simple hash)
        category_encoded = np.zeros(10, dtype=np.float32)
        if self.current_category:
            cat_hash = hash(self.current_category) % 10
            category_encoded[cat_hash] = 1.0
        
        return {
            "product_features": product_features,
            "asked_questions": asked_questions,
            "question_answers": question_answers,
            "budget_remaining": budget_remaining,
            "category_encoded": category_encoded
        }
    
    def render(self, mode="human"):
        if mode == "human":
            print(f"\n=== Episode {self.episode_count} ===")
            print(f"Category: {self.current_category}")
            print(f"Products: {len(self.products)}")
            print(f"Questions asked: {len(self.asked_questions)}")
            print(f"Questions remaining: {self.questions_remaining}")
            if self.oracle_scores:
                best_id, best_score = self.oracle_scores[0]
                print(f"Oracle best: {best_id} (score: {best_score:.1f})")
    
    def close(self):
        pass
