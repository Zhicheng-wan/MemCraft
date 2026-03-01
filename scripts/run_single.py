#!/usr/bin/env python3
"""
run_single.py - Run a single agent on a single task for quick testing.

Usage:
    python scripts/run_single.py --agent memagent --task harvest_wool_with_shears_and_sheep
    python scripts/run_single.py --agent no_memory --task harvest_milk --episodes 3
    python scripts/run_single.py --agent memagent --mock  # use mock env (no MineDojo needed)
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import MAX_STEPS_PER_EPISODE, TokenTracker
from env.minecraft_env import make_env, MockEnv, parse_observation, map_action_to_minedojo
from agents.no_memory_agent import NoMemoryAgent
from agents.naive_memory_agent import NaiveMemoryAgent
from agents.mem_agent import MemAgent
from evaluation.evaluator import Evaluator, EpisodeResult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="memagent",
                        choices=["no_memory", "naive_memory", "memagent"])
    parser.add_argument("--task", type=str,
                        default="harvest_wool_with_shears_and_sheep")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--mock", action="store_true",
                        help="Use mock environment (no MineDojo)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Create env
    if args.mock:
        env = MockEnv(args.task)
    else:
        env = make_env(args.task)
    
    goal = getattr(env, "task_prompt", f"Complete: {args.task}")
    
    # Create agent
    agents_map = {
        "no_memory": NoMemoryAgent,
        "naive_memory": NaiveMemoryAgent,
        "memagent": MemAgent,
    }
    agent = agents_map[args.agent](goal=goal, task_name=args.task)
    
    print(f"Agent: {args.agent} | Task: {args.task} | Goal: {goal}")
    print(f"Episodes: {args.episodes} | Max steps: {args.max_steps}")
    print("-" * 50)
    
    evaluator = Evaluator()
    
    for ep in range(args.episodes):
        print(f"\n--- Episode {ep+1} ---")
        agent.reset()
        raw_obs = env.reset()
        text_obs = parse_observation(raw_obs)
        total_reward = 0.0
        
        for step in range(args.max_steps):
            if not TokenTracker.check_budget():
                print("[BUDGET] Exhausted!")
                break
            
            action_str = agent.act(text_obs)
            action_arr = map_action_to_minedojo(action_str, env)
            raw_obs, reward, done, info = env.step(action_arr)
            text_obs = parse_observation(raw_obs)
            total_reward += reward
            
            agent.on_step_result(action_str, text_obs, reward, done, info)
            
            if args.verbose:
                print(f"  Step {step+1}: action={action_str}, reward={reward}")
            
            if done:
                break
        
        agent.on_episode_end()
        
        result = EpisodeResult(
            agent_name=agent.name, task_id=args.task,
            episode=ep+1, steps=step+1,
            total_reward=total_reward,
            success=total_reward > 0,
            extra=agent.get_stats(),
        )
        evaluator.add_result(result)
        print(f"  Done: steps={step+1}, reward={total_reward:.2f}, "
              f"success={total_reward > 0}")
    
    env.close()
    evaluator.save_results(f"single_{args.agent}_{args.task}.jsonl")
    
    print(f"\nToken usage: {TokenTracker.get_stats()}")


if __name__ == "__main__":
    main()
