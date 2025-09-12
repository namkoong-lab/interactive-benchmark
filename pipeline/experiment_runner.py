import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional

from .agents import RecommenderAgent, RandomAgent
from .metrics import MetricsRecorder, EpisodeRecord
from .user_model import UserModel

try:
    from simulate_interaction import get_products_by_category, list_categories
except Exception:
    from pipeline.simulate_interaction import get_products_by_category, list_categories


@dataclass
class ExperimentConfig:
    persona_index: int
    categories: List[str]
    episodes_per_category: int
    feedback_type: str  # "rank" or "score"
    agent_model: str
    max_questions: int
    seed: int = 42


def _compute_metrics(
    category: str,
    chosen_id: int,
    rationale: str,
    num_questions: int,
    agent_model: str,
    oracle: List[tuple],  # list of (id, score)
) -> dict:
    # oracle is sorted descending by score
    best_id, best_score = int(oracle[0][0]), float(oracle[0][1])
    chosen_score = next((float(s) for pid, s in oracle if int(pid) == int(chosen_id)), 0.0)
    regret = max(0.0, best_score - chosen_score)
    ids_in_order = [int(pid) for pid, _ in oracle]
    top1 = (chosen_id == best_id)
    top3 = (chosen_id in ids_in_order[:3])
    return {
        "category": category,
        "chosen_id": int(chosen_id),
        "best_id": int(best_id),
        "chosen_score": float(chosen_score),
        "best_score": float(best_score),
        "regret": float(regret),
        "top1": bool(top1),
        "top3": bool(top3),
        "num_questions": int(num_questions),
        "rationale": str(rationale),
        "agent_model": str(agent_model),
    }


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        random.seed(config.seed)
        self.user = UserModel(config.persona_index)
        self.agent = RecommenderAgent(model=config.agent_model, max_questions=config.max_questions)
        self.recorder = MetricsRecorder()

    def run(self) -> MetricsRecorder:
        episode_idx = 0
        for category in self.config.categories:
            products = get_products_by_category(category)
            if not products:
                continue
            oracle_scores = self.user.score_products(category, products)
            for _ in range(self.config.episodes_per_category):
                episode_idx += 1
                rec_id, rationale, num_q = self.agent.recommend(category, products, self.user)
                m = _compute_metrics(
                    category=category,
                    chosen_id=rec_id,
                    rationale=rationale,
                    num_questions=num_q,
                    agent_model=self.config.agent_model,
                    oracle=oracle_scores,
                )
                record = EpisodeRecord(
                    episode=episode_idx,
                    **m,
                )
                self.recorder.log(record)
        return self.recorder


def _load_config(path: Optional[str], args: argparse.Namespace) -> ExperimentConfig:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentConfig(
            persona_index=int(data["persona_index"]),
            categories=list(data["categories"]),
            episodes_per_category=int(data["episodes_per_category"]),
            feedback_type=str(data.get("feedback_type", "score")),
            agent_model=str(data.get("agent_model", "gpt-4o")),
            max_questions=int(data.get("max_questions", 8)),
            seed=int(data.get("seed", 42)),
        )

    # From CLI flags
    categories = args.categories or list_categories()
    return ExperimentConfig(
        persona_index=int(args.persona_index),
        categories=categories,
        episodes_per_category=int(args.episodes_per_category),
        feedback_type=str(args.feedback_type),
        agent_model=str(args.agent_model),
        max_questions=int(args.max_questions),
        seed=int(args.seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1: fixed user across categories")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--persona_index", type=int, default=0)
    parser.add_argument("--categories", nargs="*", default=None, help="Category names (space-separated). If omitted, all categories")
    parser.add_argument("--episodes_per_category", type=int, default=10)
    parser.add_argument("--feedback_type", type=str, default="score", choices=["score", "rank"]) 
    parser.add_argument("--agent_model", type=str, default="gpt-4o")
    parser.add_argument("--max_questions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_jsonl", type=str, default="experiment1.jsonl")
    parser.add_argument("--out_csv", type=str, default="experiment1.csv")
    args = parser.parse_args()

    cfg = _load_config(args.config, args)
    runner = ExperimentRunner(cfg)
    rec = runner.run()
    rec.save_jsonl(args.out_jsonl)
    rec.save_csv(args.out_csv)


if __name__ == "__main__":
    main()


