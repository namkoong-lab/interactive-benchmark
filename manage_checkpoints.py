#!/usr/bin/env python3
"""
Utility script to manage experiment checkpoints.
"""

import json
import os
import glob
from datetime import datetime
import argparse

def list_checkpoints(output_dir: str = "experiment1_results_with_checkpoints"):
    """List all available checkpoints."""
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return
    
    checkpoint_files = glob.glob(os.path.join(output_dir, "checkpoint_episode_*.json"))
    
    if not checkpoint_files:
        print(f"No checkpoints found in {output_dir}")
        return
    
    print(f"Available checkpoints in {output_dir}:")
    print("-" * 80)
    
    for checkpoint_file in sorted(checkpoint_files):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            episode_num = data.get('episode_num', 'Unknown')
            timestamp = data.get('timestamp', 'Unknown')
            model = data.get('model', 'Unknown')
            feedback_type = data.get('feedback_type', 'Unknown')
            total_episodes = data.get('episodes_completed', 0)
            categories = data.get('summary', {}).get('categories_tested', [])
            
            print(f"Episode {episode_num:3d}: {os.path.basename(checkpoint_file)}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Model: {model}, Feedback: {feedback_type}")
            print(f"  Episodes completed: {total_episodes}")
            print(f"  Categories: {', '.join(categories[:3])}{'...' if len(categories) > 3 else ''}")
            print()
            
        except Exception as e:
            print(f"Error reading {checkpoint_file}: {e}")

def inspect_checkpoint(checkpoint_file: str):
    """Inspect a specific checkpoint file."""
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file {checkpoint_file} does not exist")
        return
    
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        print(f"Checkpoint: {os.path.basename(checkpoint_file)}")
        print("=" * 60)
        
        # Basic info
        print(f"Episode: {data.get('episode_num', 'Unknown')}")
        print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
        print(f"Model: {data.get('model', 'Unknown')}")
        print(f"Feedback Type: {data.get('feedback_type', 'Unknown')}")
        print(f"Episodes Completed: {data.get('episodes_completed', 0)}")
        
        # Summary
        summary = data.get('summary', {})
        print(f"\nSummary:")
        print(f"  Categories tested: {summary.get('categories_tested', [])}")
        print(f"  Total episodes: {summary.get('total_episodes', 0)}")
        print(f"  Episodes by category: {summary.get('episodes_by_category', {})}")
        
        # Agent state
        agent_state = data.get('agent_state', {})
        print(f"\nAgent State:")
        print(f"  Episode count: {agent_state.get('episode_count', 0)}")
        print(f"  Learned preferences: {len(agent_state.get('learned_preferences', {}))} categories")
        print(f"  Feedback history: {len(agent_state.get('feedback_history', []))} entries")
        print(f"  Price sensitivity: {agent_state.get('price_sensitivity', {})}")
        print(f"  Quality preferences: {agent_state.get('quality_preferences', {})}")
        
        # Recent episodes
        all_results = data.get('all_results', [])
        if all_results:
            print(f"\nRecent Episodes (last 3):")
            for result in all_results[-3:]:
                episode = result.get('episode', 'Unknown')
                category = result.get('category', 'Unknown')
                score = result.get('final_info', {}).get('chosen_score', 'N/A')
                top1 = result.get('final_info', {}).get('top1', False)
                print(f"  Episode {episode}: {category} - Score: {score}, Top1: {top1}")
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

def clean_old_checkpoints(output_dir: str = "experiment1_results_with_checkpoints", keep_last: int = 1000):
    """Clean old checkpoints, keeping only the most recent ones."""
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return
    
    checkpoint_files = glob.glob(os.path.join(output_dir, "checkpoint_episode_*.json"))
    
    if len(checkpoint_files) <= keep_last:
        print(f"Only {len(checkpoint_files)} checkpoints found, keeping all")
        return
    
    # Sort by episode number
    def get_episode_num(filename):
        try:
            basename = os.path.basename(filename)
            return int(basename.split('_')[2])
        except:
            return 0
    
    checkpoint_files.sort(key=get_episode_num)
    
    # Keep the last N checkpoints
    files_to_delete = checkpoint_files[:-keep_last]
    
    print(f"Found {len(checkpoint_files)} checkpoints, keeping last {keep_last}")
    print(f"Will delete {len(files_to_delete)} old checkpoints:")
    
    for file_path in files_to_delete:
        print(f"  {os.path.basename(file_path)}")
    
    confirm = input("\nProceed with deletion? (y/N): ")
    if confirm.lower() == 'y':
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print("Deletion cancelled")

def resume_from_latest(output_dir: str = "experiment1_results_with_checkpoints"):
    """Find and display the latest checkpoint for resuming."""
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return
    
    checkpoint_files = glob.glob(os.path.join(output_dir, "checkpoint_episode_*.json"))
    
    if not checkpoint_files:
        print(f"No checkpoints found in {output_dir}")
        return
    
    # Find the latest checkpoint
    def get_episode_num(filename):
        try:
            basename = os.path.basename(filename)
            return int(basename.split('_')[2])
        except:
            return 0
    
    latest_checkpoint = max(checkpoint_files, key=get_episode_num)
    
    print(f"Latest checkpoint: {os.path.basename(latest_checkpoint)}")
    print(f"Full path: {latest_checkpoint}")
    print(f"\nTo resume from this checkpoint, use:")
    print(f"python run_experiment1_with_checkpoints.py --resume_from '{latest_checkpoint}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage experiment checkpoints")
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument("--inspect", help="Inspect a specific checkpoint file")
    parser.add_argument("--clean", type=int, metavar="N", help="Clean old checkpoints, keeping last N")
    parser.add_argument("--latest", action="store_true", help="Show latest checkpoint for resuming")
    parser.add_argument("--output_dir", default="experiment1_results_with_checkpoints", help="Output directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args.output_dir)
    elif args.inspect:
        inspect_checkpoint(args.inspect)
    elif args.clean is not None:
        clean_old_checkpoints(args.output_dir, args.clean)
    elif args.latest:
        resume_from_latest(args.output_dir)
    else:
        print("Use --help to see available options")
        print("\nQuick examples:")
        print("  python manage_checkpoints.py --list")
        print("  python manage_checkpoints.py --latest")
        print("  python manage_checkpoints.py --inspect checkpoint_file.json")
        print("  python manage_checkpoints.py --clean 5")
