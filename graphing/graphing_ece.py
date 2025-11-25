import argparse
import re
import math
import json
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- Parsing -----------------------------

ROW_MAP = {
    "regret": ["regret"],
    "actual_rank": ["actual rank"],
    "prob_fav": ["probability favourite"],
    "prob_top5": ["probability top-5", "probablity top-5"], 
    "conf_within_5": ["confidence within 5"],
    "conf_within_10": ["confidence within 10"],
    "conf_within_20": ["confidence within 20"],
    "conf_within_30": ["confidence within 30"],
}

def _match_row(name: str, targets: List[str]) -> bool:
    s = str(name).lower().strip()
    return any(s == t for t in targets)

def load_buckets_from_json(json_path: str) -> Dict[Tuple[str, int], Dict[str, List[float]]]:
    """
    Parse the JSON into buckets keyed by (Episode, QuestionIndex 1..10).
    Each bucket holds lists for the various rows (prob_fav, actual_rank, etc.).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    buckets: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    
    # Filter episodes 1, 5, and 10
    target_episodes = {1, 5, 10}
    
    for result in data.get("results", []):
        episode_num = result.get("episode")
        if episode_num not in target_episodes:
            continue
        
        ep_key = f"EP{episode_num}"
        
        # Get progression arrays
        confidence_prog = result.get("confidence_progression", [])
        regret_prog = result.get("regret_progression", [])
        rank_prog = result.get("rank_progression", [])
        
        # Process questions 1-10 (indices 0-9 in arrays)
        for qi in range(1, 11):
            idx = qi - 1  # 0-indexed
            
            # Confidence data
            if idx < len(confidence_prog):
                conf_data = confidence_prog[idx]
                if "confidence_favorite_prob" in conf_data:
                    buckets.setdefault((ep_key, qi), {}).setdefault("prob_fav", []).append(float(conf_data["confidence_favorite_prob"]))
                if "confidence_top5_prob" in conf_data:
                    buckets.setdefault((ep_key, qi), {}).setdefault("prob_top5", []).append(float(conf_data["confidence_top5_prob"]))
                if "confidence_regret_within_5" in conf_data:
                    buckets.setdefault((ep_key, qi), {}).setdefault("conf_within_5", []).append(float(conf_data["confidence_regret_within_5"]))
                if "confidence_regret_within_10" in conf_data:
                    buckets.setdefault((ep_key, qi), {}).setdefault("conf_within_10", []).append(float(conf_data["confidence_regret_within_10"]))
                if "confidence_regret_within_20" in conf_data:
                    buckets.setdefault((ep_key, qi), {}).setdefault("conf_within_20", []).append(float(conf_data["confidence_regret_within_20"]))
                if "confidence_regret_within_30" in conf_data:
                    buckets.setdefault((ep_key, qi), {}).setdefault("conf_within_30", []).append(float(conf_data["confidence_regret_within_30"]))
            
            # Regret data
            if idx < len(regret_prog):
                regret_data = regret_prog[idx]
                if "regret" in regret_data:
                    buckets.setdefault((ep_key, qi), {}).setdefault("regret", []).append(float(regret_data["regret"]))
            
            # Rank data
            if idx < len(rank_prog):
                rank_data = rank_prog[idx]
                if "actual_rank" in rank_data:
                    buckets.setdefault((ep_key, qi), {}).setdefault("actual_rank", []).append(float(rank_data["actual_rank"]))
    
    return buckets


def load_buckets(csv_path: str) -> Dict[Tuple[str, int], Dict[str, List[float]]]:
    """
    Parse the CSV into buckets keyed by (Episode, QuestionIndex 1..10).
    Each bucket holds lists for the various rows (prob_fav, actual_rank, etc.).
    """
    df = pd.read_csv(csv_path)
    label_col = df.columns[0]
    qcols = list(df.columns[1:11]) 

    buckets: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    current_ep = None

    for _, row in df.iterrows():
        name = str(row[label_col]).strip()
        if re.fullmatch(r"EP(\d+):", name):
            current_ep = name.replace(":", "") 
            continue
        if current_ep is None:
            continue

        matched_key = None
        for k, opts in ROW_MAP.items():
            if _match_row(name, opts):
                matched_key = k
                break
        if matched_key is None:
            continue

        for qi, qc in enumerate(qcols, start=1):
            val = row.get(qc, np.nan)
            if pd.isna(val):
                continue
            buckets.setdefault((current_ep, qi), {}).setdefault(matched_key, []).append(float(val))
    return buckets


# ----------------------------- Calibration -----------------------------

def _fixed_bins(M: int):
    edges = np.linspace(0.0, 1.0, M + 1)
    return edges

def _quantile_bins(probs: np.ndarray, M: int):
    quantiles = np.linspace(0.0, 1.0, M + 1)
    edges = np.quantile(probs, quantiles)
    edges = np.maximum.accumulate(edges)
    edges[0] = 0.0
    edges[-1] = 1.0
    return edges

def reliability_stats(probs: np.ndarray, labels: np.ndarray, M: int = 10, binning: str = "fixed"):
    """
    Return (bin_centers, bin_acc, bin_conf, bin_counts, ece).
    """
    mask = ~np.isnan(probs) & ~np.isnan(labels)
    probs, labels = probs[mask], labels[mask]
    if len(probs) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.nan

    if binning == "quantile":
        edges = _quantile_bins(probs, M)
    else:
        edges = _fixed_bins(M)

    acc = np.zeros(M)
    conf = np.zeros(M)
    counts = np.zeros(M, dtype=int)
    centers = (edges[:-1] + edges[1:]) / 2.0

    for m in range(M):
        lo, hi = edges[m], edges[m + 1]
        last = (m == M - 1)
        sel = (probs >= lo) & ((probs <= hi) if last else (probs < hi))
        if sel.any():
            acc[m] = labels[sel].mean()
            # Use bin center for conf, so bar tops meet diagonal at center
            conf[m] = centers[m]
            counts[m] = sel.sum()
        else:
            acc[m] = np.nan
            conf[m] = np.nan
            counts[m] = 0

    N = counts.sum()
    gaps = np.abs(acc - conf)
    gaps[np.isnan(gaps)] = 0.0
    ece = float(np.nansum((counts / max(N, 1)) * gaps)) if N > 0 else np.nan
    return centers, acc, conf, counts, ece


def plot_reliability_panel(ax, probs, labels, title, M=10, binning="fixed", font_size=10):
    centers, acc, conf, counts, ece = reliability_stats(probs, labels, M=M, binning=binning)
    width = (1.0 / M) * 0.8
    
    # Handle NaN values - for empty buckets, use center as confidence (diagonal value)
    empty_mask = (counts == 0)
    acc = np.where(np.isnan(acc), 0.0, acc)
    # For empty buckets, confidence should be the center value (diagonal at that x)
    conf = np.where(np.isnan(conf), centers, conf)
    
    # Blue bar: accuracy (capped at confidence to never exceed diagonal)
    # If accuracy > confidence, cap it at confidence so blue bar stays below diagonal
    acc_bar = np.minimum(acc, conf)
    
    # Orange bar: gap from accuracy to confidence (diagonal line)
    # Orange bar top should reach the diagonal
    gap = conf - acc_bar
    
    # For empty buckets, show full orange bar up to diagonal
    acc_bar[empty_mask] = 0.0
    gap[empty_mask] = conf[empty_mask]  # Full orange up to diagonal
    
    ax.bar(centers, acc_bar, width=width, align='center', label="Outputs", color='steelblue')
    ax.bar(
        centers,
        gap,
        width=width,
        bottom=acc_bar,
        align='center',
        color='#F28C28',
        alpha=0.7,
        hatch='//',
        edgecolor='black',
        label="Gap"
    )
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='lightblue', linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    tick_fs = max(6, font_size - 1)
    title_fs = font_size + 2
    ax.set_xlabel("Confidence", fontsize=font_size)
    ax.set_ylabel("Accuracy", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.set_title(f"{title}\nECE = {ece:.4f}" if not math.isnan(ece) else f"{title}\nECE = n/a", fontsize=title_fs)
    ax.legend(loc='upper left', fontsize=tick_fs)


# ----------------------------- Driver -----------------------------

def extract_probs_labels(buckets, episode: str, question: int, metric: str, k: int = 10):
    """
    For a given episode and question, return (probs, labels) for:
      metric ∈ {"top1", "top5", "regk"} with k ∈ {5,10,20,30} for regk.
    """
    data = buckets.get((episode, question), {})
    if metric == "top1":
        probs = np.array(data.get("prob_fav", []), dtype=float)
        labels = (np.array(data.get("actual_rank", []), dtype=float) == 1).astype(float)
        return probs, labels

    if metric == "top5":
        probs = np.array(data.get("prob_top5", []), dtype=float)
        labels = (np.array(data.get("actual_rank", []), dtype=float) <= 5).astype(float)
        return probs, labels

    if metric == "regk":
        if k == 5:
            probs = np.array(data.get("conf_within_5", []), dtype=float)
            labels = (np.array(data.get("regret", []), dtype=float) <= 5).astype(float)
        elif k == 10:
            probs = np.array(data.get("conf_within_10", []), dtype=float)
            labels = (np.array(data.get("regret", []), dtype=float) <= 10).astype(float)
        elif k == 20:
            probs = np.array(data.get("conf_within_20", []), dtype=float)
            labels = (np.array(data.get("regret", []), dtype=float) <= 20).astype(float)
        elif k == 30:
            probs = np.array(data.get("conf_within_30", []), dtype=float)
            labels = (np.array(data.get("regret", []), dtype=float) <= 30).astype(float)
        else:
            raise ValueError("k must be one of {5, 10, 20, 30}")
        return probs, labels

    raise ValueError("metric must be one of {'top1','top5','regk'}")


def extract_pooled_probs_labels(buckets, episode: str, metric: str, k: int = 10):
    """
    Concatenate data across all questions (1..10) for a given episode and metric.
    Returns (probs, labels).
    """
    probs_list = []
    labels_list = []
    for q in range(1, 11):
        p, y = extract_probs_labels(buckets, episode, q, metric, k=k)
        if p.size and y.size:
            probs_list.append(p)
            labels_list.append(y)
    if len(probs_list) == 0:
        return np.array([]), np.array([])
    return np.concatenate(probs_list), np.concatenate(labels_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to input CSV file")
    ap.add_argument("--json", default=None, help="Path to input JSON file")
    ap.add_argument("--question", type=int, required=False, default=None, help="Question index (1..10). Not required with --collapse_questions")
    ap.add_argument("--metric", choices=["top1", "top5", "regk", "all"], required=True, help="Which metric to plot or 'all' for all panels")
    ap.add_argument("--k", type=int, default=10, help="Threshold for regk (one of 5,10,20,30)")
    ap.add_argument("--bins", type=int, default=10, help="Number of bins")
    ap.add_argument("--binning", choices=["fixed","quantile"], default="fixed", help="Binning strategy")
    ap.add_argument("--collapse_questions", action="store_true", help="Pool all 10 questions together per episode")
    ap.add_argument("--font_size", type=int, default=10, help="Base font size for axes labels and ticks")
    ap.add_argument("--summary_one_line", action="store_true", help="Print one-line ECE summary per episode and exit")
    args = ap.parse_args()

    if (not args.collapse_questions) and (not args.summary_one_line) and (args.question is None):
        raise SystemExit("error: --question is required unless --collapse_questions or --summary_one_line is provided")

    # Determine input file
    if args.json:
        input_file = args.json
        buckets = load_buckets_from_json(args.json)
    elif args.csv:
        input_file = args.csv
        buckets = load_buckets(args.csv)
    else:
        # Try default CSV, then look for JSON in current directory
        import os
        if os.path.exists("ece.csv"):
            input_file = "ece.csv"
            buckets = load_buckets("ece.csv")
        else:
            # Look for JSON files
            json_files = [f for f in os.listdir(".") if f.endswith(".json")]
            if json_files:
                input_file = json_files[0]
                buckets = load_buckets_from_json(json_files[0])
            else:
                raise SystemExit("error: No input file specified. Use --csv or --json, or place ece.csv in current directory")

    episodes = ["EP1", "EP5", "EP10"]
    
    if args.summary_one_line:
        results = []
        for ep in episodes:
            if args.metric == "regk":
                k_use = args.k
            else:
                k_use = None
            probs, labels = extract_pooled_probs_labels(buckets, ep, args.metric, k=args.k) if args.collapse_questions or True else extract_probs_labels(buckets, ep, args.question, args.metric, k=args.k)
            _, _, _, counts, ece = reliability_stats(probs, labels, M=args.bins, binning=args.binning)
            N = int(counts.sum())
            results.append((ep, N, ece))
        parts = [f"{ep}: N={N}, ECE={ece:.4f}" if not math.isnan(ece) else f"{ep}: N={N}, ECE=nan" for ep, N, ece in results]
        header = f"Summary | file={input_file} | metric={args.metric}{(' k='+str(args.k)) if args.metric=='regk' else ''} | pooled=AllQuestions"
        print(header + " | " + " | ".join(parts))
        return

    if args.metric != "all":
        plt.figure(figsize=(11, 3.3))
        for i, ep in enumerate(episodes, 1):
            if args.collapse_questions:
                probs, labels = extract_pooled_probs_labels(buckets, ep, args.metric, k=args.k)
            else:
                probs, labels = extract_probs_labels(buckets, ep, args.question, args.metric, k=args.k)
            ax = plt.subplot(1, 3, i)
            # Construct the title as originally
            title = f"{ep} | All Questions" if args.collapse_questions else f"{ep} | Q{args.question}"
            if args.metric == "regk":
                title += f" | Regret ≤ {args.k}"
            elif args.metric == "top1":
                title += " | Top-1"
            else:
                title += " | Top-5"
            print(f"Plot: {title} | Metric: {args.metric}{(' (k=' + str(args.k) + ')') if args.metric=='regk' else ''} | N={labels.size}")

            if probs.size == 0 or labels.size == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_axis_off()
                continue

            plot_reliability_panel(ax, probs, labels, title, M=args.bins, binning=args.binning, font_size=args.font_size)
        plt.tight_layout()
        plt.show()
        return

    metric_cols = [
        ("top1", None, "Top-1"),
        ("top5", None, "Top-5"),
        ("regk", 5,  "Regret ≤ 5"),
        ("regk", 10, "Regret ≤ 10"),
        ("regk", 20, "Regret ≤ 20"),
        ("regk", 30, "Regret ≤ 30"),
    ]

    n_rows, n_cols = len(episodes), len(metric_cols)
    plt.figure(figsize=(3.4 * n_cols, 3.0 * n_rows))
    plot_idx = 1
    for r, ep in enumerate(episodes):
        for c, (metric, k_val, col_title) in enumerate(metric_cols):
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            plot_idx += 1
            k_use = k_val if metric == "regk" else args.k
            if args.collapse_questions:
                probs, labels = extract_pooled_probs_labels(buckets, ep, metric, k=k_use)
                title = f"{ep} | {col_title}"
            else:
                probs, labels = extract_probs_labels(buckets, ep, args.question, metric, k=k_use)
                title = f"{ep} | Q{args.question}\n{col_title}"

            metric_tag = f"{metric}{(' (k=' + str(k_use) + ')') if metric=='regk' else ''}"
            context = "All Questions" if args.collapse_questions else f"Q{args.question}"
            print(f"Plot: {ep} | {context} | {col_title} [{metric_tag}] | N={labels.size}")

            if probs.size == 0 or labels.size == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_axis_off()
                continue

            plot_reliability_panel(ax, probs, labels, title, M=args.bins, binning=args.binning, font_size=args.font_size)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()