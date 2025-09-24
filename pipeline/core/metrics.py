import csv
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeRecord:
    episode: int
    category: str
    chosen_id: int
    best_id: int
    chosen_score: float
    best_score: float
    regret: float
    top1: bool
    top3: bool
    num_questions: int
    rationale: str
    agent_model: str
    confidence_favorite_prob: Optional[float] = None
    confidence_top5_prob: Optional[float] = None
    confidence_expected_score: Optional[float] = None
    confidence_expected_regret: Optional[float] = None


class MetricsRecorder:
    def __init__(self) -> None:
        self._records: List[EpisodeRecord] = []

    def log(self, record: EpisodeRecord) -> None:
        self._records.append(record)

    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self._records]

    def save_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in self.to_list():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_csv(self, path: str) -> None:
        rows = self.to_list()
        if not rows:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("")
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


