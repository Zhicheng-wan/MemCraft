#!/usr/bin/env python3
"""
evaluate.py - Run evaluation experiments comparing all three agents.

Runs each agent variant on the same tasks and produces a comparison report.

Usage:
    python evaluate.py --tasks mine_dirt,collect_wood --episodes 3
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from agent.brain import Brain
from agent.agent import NoMemoryAgent, NaiveMemoryAgent, MemAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval")

# ─── Evaluation Tasks ───
TASKS = {
    "mine_dirt": {
        "goal": "Mine 5 dirt blocks. Use find_and_mine_block with block_name='dirt'. "
                "When you have 5 dirt in your inventory, respond with DONE.",
        "success_check": lambda inv: inv.get("dirt", 0) >= 5,
        "max_steps": 30,
    },
    "collect_wood": {
        "goal": "Collect 3 oak logs. Use find_and_mine_block with block_name='oak_log'. "
                "When you have 3 oak_log in inventory, respond with DONE.",
        "success_check": lambda inv: inv.get("oak_log", 0) >= 3,
        "max_steps": 40,
    },
    "craft_planks": {
        "goal": "Collect 1 oak log then craft 4 oak planks. First mine oak_log, "
                "then craft oak_planks. When you have oak_planks, respond with DONE.",
        "success_check": lambda inv: inv.get("oak_planks", 0) >= 4,
        "max_steps": 50,
    },
    "craft_sticks": {
        "goal": "Collect wood, craft planks, then craft sticks. "
                "Mine oak_log → craft oak_planks → craft stick. "
                "When you have sticks, respond with DONE.",
        "success_check": lambda inv: inv.get("stick", 0) >= 1,
        "max_steps": 60,
    },
}

AGENT_TYPES = ["no_memory", "naive_memory", "memagent"]


def create_agent(agent_type: str, brain: Brain, bot_url: str,
                 max_steps: int, config: dict):
    if agent_type == "no_memory":
        return NoMemoryAgent(brain, bot_url=bot_url, max_steps=max_steps)
    elif agent_type == "naive_memory":
        return NaiveMemoryAgent(brain, bot_url=bot_url, max_steps=max_steps,
                                 history_length=10)
    elif agent_type == "memagent":
        return MemAgent(brain, bot_url=bot_url, max_steps=max_steps,
                        config=config, brain=brain)


def run_evaluation(task_names: list, episodes: int, bot_url: str,
                   api_key: str, model: str, config: dict):
    results = {}

    for task_name in task_names:
        if task_name not in TASKS:
            logger.warning(f"Unknown task: {task_name}, skipping")
            continue

        task = TASKS[task_name]
        results[task_name] = {}

        for agent_type in AGENT_TYPES:
            logger.info(f"\n{'='*50}")
            logger.info(f"Task: {task_name} | Agent: {agent_type}")
            logger.info(f"{'='*50}")

            episode_results = []

            for ep in range(episodes):
                logger.info(f"  Episode {ep+1}/{episodes}")

                brain = Brain(
                    api_key=api_key,
                    api_url="https://tritonai-api.ucsd.edu/v1/chat/completions",
                    model=model,
                    max_tokens=512,
                    temperature=0.3,
                )

                agent = create_agent(
                    agent_type, brain, bot_url,
                    task["max_steps"], config
                )

                try:
                    if agent_type == "memagent":
                        result = agent.run(
                            task["goal"],
                            persist_memory=f"memories/eval_{task_name}_rules.json"
                        )
                    else:
                        result = agent.run(task["goal"])
                except Exception as e:
                    logger.error(f"  Episode failed: {e}")
                    result = {
                        "success": False,
                        "total_steps": 0,
                        "brain_stats": brain.get_stats(),
                        "steps": [],
                    }

                episode_results.append({
                    "episode": ep,
                    "success": result.get("success", False),
                    "steps": result.get("total_steps", 0),
                    "tokens": result.get("brain_stats", {}).get("total_tokens", 0),
                    "api_calls": result.get("brain_stats", {}).get("total_calls", 0),
                    "action_success_rate": (
                        sum(1 for s in result.get("steps", []) if s.get("success"))
                        / max(len(result.get("steps", [])), 1)
                    ),
                })

                time.sleep(2)  # Brief pause between episodes

            # Aggregate
            successes = sum(1 for r in episode_results if r["success"])
            avg_steps = (sum(r["steps"] for r in episode_results)
                        / max(len(episode_results), 1))
            avg_tokens = (sum(r["tokens"] for r in episode_results)
                         / max(len(episode_results), 1))
            avg_action_sr = (sum(r["action_success_rate"] for r in episode_results)
                            / max(len(episode_results), 1))

            results[task_name][agent_type] = {
                "episodes": episode_results,
                "success_rate": successes / max(episodes, 1),
                "avg_steps": avg_steps,
                "avg_tokens": avg_tokens,
                "avg_action_success_rate": avg_action_sr,
            }

    return results


def print_report(results: dict):
    """Print evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    for task_name, task_results in results.items():
        print(f"\n{'─'*50}")
        print(f"Task: {task_name}")
        print(f"{'─'*50}")
        print(f"{'Agent':<20} {'Success%':<12} {'Avg Steps':<12} "
              f"{'Avg Tokens':<14} {'Action SR':<12}")
        print(f"{'─'*20} {'─'*10} {'─'*10} {'─'*12} {'─'*10}")

        for agent_type in AGENT_TYPES:
            r = task_results.get(agent_type, {})
            print(
                f"{agent_type:<20} "
                f"{r.get('success_rate', 0)*100:>8.0f}%   "
                f"{r.get('avg_steps', 0):>8.1f}   "
                f"{r.get('avg_tokens', 0):>10.0f}   "
                f"{r.get('avg_action_success_rate', 0)*100:>8.0f}%"
            )

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MemCraft Evaluation")
    parser.add_argument("--tasks", type=str, default="mine_dirt",
                        help="Comma-separated task names")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per agent per task")
    parser.add_argument("--bot-url", type=str, default="http://localhost:3001")
    parser.add_argument("--model", type=str, default="api-llama-4-scout")
    parser.add_argument("--config", type=str, default="configs/default.json")

    args = parser.parse_args()

    api_key = os.environ.get("TRITONAI_API_KEY")
    if not api_key:
        print("Set TRITONAI_API_KEY first")
        sys.exit(1)

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)

    Path("memories").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    task_names = [t.strip() for t in args.tasks.split(",")]

    results = run_evaluation(
        task_names, args.episodes, args.bot_url,
        api_key, args.model, config
    )

    print_report(results)

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"logs/eval_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
