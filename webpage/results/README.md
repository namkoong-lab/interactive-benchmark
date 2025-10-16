# Results Directory

This directory contains evaluation results for different agent models.

## Structure

Each model should have its own subdirectory with the following files:

```
results/
└── your-model-name/
    ├── results.json       # Required: Evaluation results
    ├── metadata.json      # Required: Model metadata
    └── README.md          # Optional: Additional information
```

## File Formats

### results.json

```json
{
  "model_name": "Your Model Name",
  "version": "1.0.0",
  "date": "2024-10-14",
  "overall_score": 85.5,
  "metrics": {
    "accuracy": 87.2,
    "success_rate": 83.8,
    "average_steps": 12.5,
    "avg_time": 45.3
  },
  "task_breakdown": {
    "task_1": { "score": 90.0, "success": true },
    "task_2": { "score": 88.5, "success": true }
  }
}
```

### metadata.json

```json
{
  "model_name": "Your Model Name",
  "description": "Brief description",
  "authors": ["Author 1", "Author 2"],
  "institution": "Your Institution",
  "paper_url": "https://arxiv.org/abs/XXXX.XXXXX",
  "code_url": "https://github.com/username/repo",
  "contact": "email@example.com"
}
```

## How to Submit

1. Create a new folder with your model name
2. Add the required JSON files
3. Submit a pull request
4. Wait for review

See the [Submit page](../app/submit/page.tsx) for detailed instructions.

## Automatic Leaderboard Updates

The leaderboard will automatically read all JSON files in this directory and display them on the website. When you submit a PR with your results, the leaderboard will update automatically once merged.

