"""
Evaluation: metrics, comparison across agents, and visualization.
"""
import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("[WARN] pandas/matplotlib not installed. Plotting disabled.")


class EpisodeResult:
    """Result of a single episode."""
    
    def __init__(self, agent_name: str, task_id: str, episode: int,
                 steps: int, total_reward: float, success: bool,
                 extra: dict = None):
        self.agent_name = agent_name
        self.task_id = task_id
        self.episode = episode
        self.steps = steps
        self.total_reward = total_reward
        self.success = success
        self.extra = extra or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "task": self.task_id,
            "episode": self.episode,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "success": self.success,
            "timestamp": self.timestamp,
            **self.extra,
        }


class Evaluator:
    """Collects results across agents/tasks and produces comparison metrics."""
    
    def __init__(self, results_dir: str = "logs"):
        self.results: List[EpisodeResult] = []
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def add_result(self, result: EpisodeResult):
        self.results.append(result)
    
    def save_results(self, filename: str = "results.jsonl"):
        """Save all results as JSONL."""
        path = self.results_dir / filename
        with open(path, "a") as f:
            for r in self.results:
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"[Eval] Results saved to {path}")
    
    def load_results(self, filename: str = "results.jsonl"):
        """Load results from JSONL."""
        path = self.results_dir / filename
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                self.results.append(EpisodeResult(
                    agent_name=data["agent"],
                    task_id=data["task"],
                    episode=data.get("episode", 0),
                    steps=data["steps"],
                    total_reward=data["total_reward"],
                    success=data["success"],
                    extra={k: v for k, v in data.items()
                           if k not in {"agent", "task", "episode", "steps",
                                       "total_reward", "success", "timestamp"}},
                ))
    
    def summary(self) -> Dict:
        """Compute summary statistics per agent per task."""
        if not HAS_PLOTTING:
            return self._basic_summary()
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        if df.empty:
            return {}
        
        summary = {}
        for agent in df["agent"].unique():
            agent_df = df[df["agent"] == agent]
            summary[agent] = {}
            for task in agent_df["task"].unique():
                task_df = agent_df[agent_df["task"] == task]
                summary[agent][task] = {
                    "episodes": len(task_df),
                    "success_rate": task_df["success"].mean(),
                    "avg_reward": task_df["total_reward"].mean(),
                    "avg_steps": task_df["steps"].mean(),
                    "std_reward": task_df["total_reward"].std(),
                }
        return summary
    
    def _basic_summary(self) -> Dict:
        """Fallback summary without pandas."""
        from collections import defaultdict
        groups = defaultdict(list)
        for r in self.results:
            groups[(r.agent_name, r.task_id)].append(r)
        
        summary = {}
        for (agent, task), results in groups.items():
            if agent not in summary:
                summary[agent] = {}
            rewards = [r.total_reward for r in results]
            successes = [r.success for r in results]
            steps = [r.steps for r in results]
            summary[agent][task] = {
                "episodes": len(results),
                "success_rate": sum(successes) / len(successes),
                "avg_reward": sum(rewards) / len(rewards),
                "avg_steps": sum(steps) / len(steps),
            }
        return summary
    
    def print_comparison(self):
        """Print a nice comparison table."""
        summary = self.summary()
        if not summary:
            print("No results to compare.")
            return
        
        print("\n" + "=" * 80)
        print("AGENT COMPARISON")
        print("=" * 80)
        
        all_tasks = set()
        for agent_data in summary.values():
            all_tasks.update(agent_data.keys())
        
        for task in sorted(all_tasks):
            print(f"\n{'─' * 60}")
            print(f"Task: {task}")
            print(f"{'Agent':<20} {'Success%':<12} {'AvgReward':<12} {'AvgSteps':<12} {'Episodes':<10}")
            print(f"{'─' * 60}")
            for agent in sorted(summary.keys()):
                if task in summary[agent]:
                    s = summary[agent][task]
                    print(
                        f"{agent:<20} "
                        f"{s['success_rate']*100:>8.1f}%   "
                        f"{s['avg_reward']:>9.2f}   "
                        f"{s['avg_steps']:>9.1f}   "
                        f"{s['episodes']:>6}"
                    )
        print(f"\n{'=' * 80}")
    
    def plot_comparison(self, save_path: str = None):
        """Generate comparison bar charts."""
        if not HAS_PLOTTING:
            print("[Eval] Plotting not available (install matplotlib, seaborn, pandas)")
            return
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        if df.empty:
            print("[Eval] No results to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Success rate
        success_data = df.groupby(["agent", "task"])["success"].mean().reset_index()
        sns.barplot(data=success_data, x="task", y="success", hue="agent", ax=axes[0])
        axes[0].set_title("Success Rate")
        axes[0].set_ylabel("Success Rate")
        axes[0].tick_params(axis='x', rotation=45)
        
        # Average reward
        reward_data = df.groupby(["agent", "task"])["total_reward"].mean().reset_index()
        sns.barplot(data=reward_data, x="task", y="total_reward", hue="agent", ax=axes[1])
        axes[1].set_title("Average Reward")
        axes[1].set_ylabel("Total Reward")
        axes[1].tick_params(axis='x', rotation=45)
        
        # Average steps
        steps_data = df.groupby(["agent", "task"])["steps"].mean().reset_index()
        sns.barplot(data=steps_data, x="task", y="steps", hue="agent", ax=axes[2])
        axes[2].set_title("Average Steps to Completion")
        axes[2].set_ylabel("Steps")
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        path = save_path or str(self.results_dir / "comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Eval] Comparison plot saved to {path}")
        plt.close()
