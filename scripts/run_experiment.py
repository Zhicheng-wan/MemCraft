#!/usr/bin/env python3
"""
run_experiment.py - Main experiment runner for The Arena.

Runs all three agents (No-Memory, Naive Memory, MemAgent) on specified
MineDojo tasks and collects comparative results.

Usage:
    python scripts/run_experiment.py --task harvest_wool_with_shears_and_sheep --episodes 5
    python scripts/run_experiment.py --tasks all --episodes 3
    python scripts/run_experiment.py --agents memagent --task harvest_milk --episodes 10
"""
import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    DEFAULT_TASKS, NUM_EPISODES_PER_TASK, MAX_STEPS_PER_EPISODE,
    TokenTracker,
)
from env.minecraft_env import make_env, parse_observation, map_action_to_minedojo, TextObservation
from agents.no_memory_agent import NoMemoryAgent
from agents.naive_memory_agent import NaiveMemoryAgent
from agents.mem_agent import MemAgent
from evaluation.evaluator import Evaluator, EpisodeResult


AGENT_CLASSES = {
    "no_memory": NoMemoryAgent,
    "naive_memory": NaiveMemoryAgent,
    "memagent": MemAgent,
}


def run_episode(agent, env, task_id: str, episode_num: int,
                max_steps: int = MAX_STEPS_PER_EPISODE) -> EpisodeResult:
    """
    Run a single episode with the given agent and environment.
    
    Loop:
    1. Get observation from env
    2. Parse to TextObservation
    3. Agent selects action
    4. Map action to MineDojo format
    5. Step environment
    6. Report result to agent
    7. Repeat until done or max steps
    """
    print(f"\n  Episode {episode_num} | Agent: {agent.name} | Task: {task_id}")
    
    agent.reset()
    raw_obs = env.reset()
    text_obs = parse_observation(raw_obs)
    
    total_reward = 0.0
    done = False
    
    for step in range(max_steps):
        # Budget check
        if not TokenTracker.check_budget():
            print("  [BUDGET] Budget exhausted! Ending episode.")
            break
        
        # Agent decides
        action_str = agent.act(text_obs)
        
        # Map to MineDojo action
        action_arr = map_action_to_minedojo(action_str, env)
        
        # Step environment
        raw_obs, reward, done, info = env.step(action_arr)
        total_reward += reward
        
        # Parse new observation
        text_obs = parse_observation(raw_obs)
        
        # Report to agent
        agent.on_step_result(action_str, text_obs, reward, done, info)
        
        # Progress log (every 25 steps)
        if (step + 1) % 25 == 0:
            budget = TokenTracker.get_stats()
            print(
                f"    Step {step+1}/{max_steps} | "
                f"Reward: {total_reward:.2f} | "
                f"Action: {action_str} | "
                f"Tokens: {budget['total_input_tokens']+budget['total_output_tokens']}"
            )
        
        if done:
            print(f"    ✓ Done at step {step+1} with reward {total_reward:.2f}")
            break
    
    # Episode end
    agent.on_episode_end()
    
    success = total_reward > 0
    agent_stats = agent.get_stats()
    
    result = EpisodeResult(
        agent_name=agent.name,
        task_id=task_id,
        episode=episode_num,
        steps=step + 1,
        total_reward=total_reward,
        success=success,
        extra=agent_stats,
    )
    
    print(f"    Result: {'SUCCESS' if success else 'FAIL'} | "
          f"Steps: {step+1} | Reward: {total_reward:.2f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="MemAgent Arena Experiment")
    parser.add_argument("--task", type=str, default=None,
                        help="Single task ID to run")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task IDs, or 'all' for defaults")
    parser.add_argument("--agents", type=str, default="all",
                        help="Comma-separated agent names, or 'all'")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_PER_TASK,
                        help="Episodes per agent per task")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE,
                        help="Max steps per episode")
    parser.add_argument("--results-dir", type=str, default="logs",
                        help="Directory to save results")
    args = parser.parse_args()
    
    # Resolve tasks
    if args.task:
        tasks = [args.task]
    elif args.tasks == "all" or args.tasks is None:
        tasks = DEFAULT_TASKS
    else:
        tasks = args.tasks.split(",")
    
    # Resolve agents
    if args.agents == "all":
        agent_names = list(AGENT_CLASSES.keys())
    else:
        agent_names = args.agents.split(",")
    
    print("=" * 70)
    print("MemAgent Arena Experiment")
    print("=" * 70)
    print(f"Tasks: {tasks}")
    print(f"Agents: {agent_names}")
    print(f"Episodes per combo: {args.episodes}")
    print(f"Max steps/episode: {args.max_steps}")
    print("=" * 70)
    
    evaluator = Evaluator(results_dir=args.results_dir)
    
    for task_id in tasks:
        print(f"\n{'━' * 70}")
        print(f"TASK: {task_id}")
        print(f"{'━' * 70}")
        
        for agent_name in agent_names:
            if agent_name not in AGENT_CLASSES:
                print(f"  [WARN] Unknown agent: {agent_name}, skipping")
                continue
            
            # Create environment
            env = make_env(task_id)
            goal = getattr(env, "task_prompt", f"Complete: {task_id}")
            
            # Create agent
            AgentClass = AGENT_CLASSES[agent_name]
            agent = AgentClass(goal=goal, task_name=task_id)
            
            for ep in range(args.episodes):
                if not TokenTracker.check_budget():
                    print("[BUDGET] Budget exhausted! Stopping experiment.")
                    break
                
                result = run_episode(
                    agent, env, task_id, ep + 1,
                    max_steps=args.max_steps
                )
                evaluator.add_result(result)
            
            env.close()
            
            # Budget status
            stats = TokenTracker.get_stats()
            print(f"\n  Budget: ${stats['total_cost_usd']:.4f} / "
                  f"${stats['budget_remaining']:.2f} remaining")
    
    # Save and display results
    evaluator.save_results()
    evaluator.print_comparison()
    evaluator.plot_comparison()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Final token usage: {TokenTracker.get_stats()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
