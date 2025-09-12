import gymnasium as gym
from gymnasium.core import ObsType, ActType
import json
from typing import Any, Dict, List, Optional
import csv
from dataclasses import dataclass, asdict

try:
    from ..metrics import EpisodeRecord, MetricsRecorder
except ImportError:
    from metrics import EpisodeRecord, MetricsRecorder


class MetricsWrapper(gym.Wrapper):
    """
    Gym wrapper that logs episode metrics to JSONL/CSV files.
    Compatible with the existing MetricsRecorder interface.
    """
    
    def __init__(self, env: gym.Env, output_path: Optional[str] = None):
        super().__init__(env)
        self.recorder = MetricsRecorder()
        self.output_path = output_path
        self.episode_count = 0
        
        # Episode state tracking
        self.episode_start_info = None
        self.questions_asked = 0
        self.final_info = None
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Store episode start info
        self.episode_start_info = info.copy()
        self.questions_asked = 0
        self.final_info = None
        
        return obs, info
    
    def step(self, action: ActType):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track questions asked
        if info.get("action_type") == "ask":
            self.questions_asked += 1
        
        # If episode ended, log metrics
        if terminated or truncated:
            self.final_info = info.copy()
            self._log_episode()
        
        return obs, reward, terminated, truncated, info
    
    def _log_episode(self):
        """Log episode metrics to recorder."""
        if not self.episode_start_info or not self.final_info:
            return
        
        # Extract metrics from episode info
        category = self.episode_start_info.get("category", "unknown")
        chosen_id = self.final_info.get("chosen_product_id", -1)
        best_id = self.final_info.get("best_product_id", -1)
        chosen_score = self.final_info.get("chosen_score", 0.0)
        best_score = self.final_info.get("best_score", 0.0)
        regret = self.final_info.get("regret", 0.0)
        top1 = self.final_info.get("top1", False)
        top3 = self.final_info.get("top3", False)
        
        # Create episode record
        record = EpisodeRecord(
            episode=self.episode_count,
            category=category,
            chosen_id=chosen_id,
            best_id=best_id,
            chosen_score=chosen_score,
            best_score=best_score,
            regret=regret,
            top1=top1,
            top3=top3,
            num_questions=self.questions_asked,
            rationale="",  # Could extract from final_info if available
            agent_model="gym_agent"  # Could be passed as parameter
        )
        
        self.recorder.log(record)
        self.episode_count += 1
    
    def save_metrics(self, jsonl_path: str, csv_path: str):
        """Save accumulated metrics to files."""
        self.recorder.save_jsonl(jsonl_path)
        self.recorder.save_csv(csv_path)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged episodes."""
        records = self.recorder.to_list()
        if not records:
            return {}
        
        total_episodes = len(records)
        avg_regret = sum(r["regret"] for r in records) / total_episodes
        top1_accuracy = sum(r["top1"] for r in records) / total_episodes
        top3_accuracy = sum(r["top3"] for r in records) / total_episodes
        avg_questions = sum(r["num_questions"] for r in records) / total_episodes
        
        return {
            "total_episodes": total_episodes,
            "avg_regret": avg_regret,
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "avg_questions": avg_questions
        }


class EpisodeLogger(gym.Wrapper):
    """
    Simple wrapper that logs each episode transition for debugging.
    """
    
    def __init__(self, env: gym.Env, log_file: Optional[str] = None):
        super().__init__(env)
        self.log_file = log_file
        self.episode_log = []
        self.current_episode = []
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if self.current_episode:
            self.episode_log.append(self.current_episode)
            self.current_episode = []
        
        obs, info = self.env.reset(seed=seed, options=options)
        self.current_episode.append({
            "step": 0,
            "action": None,
            "reward": 0.0,
            "info": info
        })
        return obs, info
    
    def step(self, action: ActType):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_episode.append({
            "step": len(self.current_episode),
            "action": int(action),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        })
        
        if terminated or truncated:
            self.episode_log.append(self.current_episode)
            self.current_episode = []
        
        return obs, reward, terminated, truncated, info
    
    def save_log(self, path: str):
        """Save episode log to JSON file."""
        with open(path, "w") as f:
            json.dump(self.episode_log, f, indent=2)
