#!/usr/bin/env python3
"""
budget_calculator.py — Know EXACTLY what your experiment will cost before running it.

Run this FIRST before any experiment to get a cost estimate.
Also use it to plan how many episodes you can afford.

Usage:
    python scripts/budget_calculator.py
    python scripts/budget_calculator.py --plan-experiment --agents all --tasks 4 --episodes 5 --max-steps 200
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════
#  COST MODEL — ADJUST THESE BASED ON YOUR ACTUAL API
# ═══════════════════════════════════════════════════════════════════════
#
#  SCENARIO A: TritonAI is FREE for students (most likely)
#     → Your $200 budget is essentially infinite for tokens
#     → But you still have RATE LIMITS (requests/min)
#     → Main constraint: wall-clock time, not money
#
#  SCENARIO B: TritonAI charges per-token (unlikely but possible)
#     → Llama-4-Scout typical pricing: ~$0.18/1M input, ~$0.63/1M output
#     → We estimate conservatively below
#
#  HOW TO FIND OUT: Run 1 test call and check if your account shows charges.
#  Or just ask your instructor.
# ═══════════════════════════════════════════════════════════════════════

# Conservative estimates (worst case: commercial pricing)
COST_PER_1M_INPUT = 0.18    # USD per 1M input tokens
COST_PER_1M_OUTPUT = 0.63   # USD per 1M output tokens

# Token estimates per LLM call (measured from typical prompts)
TOKENS_PER_CALL = {
    "no_memory": {
        "input": 350,    # system prompt (200) + user prompt w/ obs (150)
        "output": 80,    # JSON response
        "calls_per_step": 1,
    },
    "naive_memory": {
        "input": 900,    # system (200) + history window (550) + obs (150)
        "output": 80,
        "calls_per_step": 1,
    },
    "memagent": {
        "input": 700,    # system (250) + retrieved memories (200) + rules (100) + obs (150)
        "output": 120,   # JSON + observation_note
        "calls_per_step": 1,
        # Consolidation call every N steps
        "consolidation_input": 800,
        "consolidation_output": 200,
        "consolidation_interval": 10,
        # Evidence check (~40% of the time, rest handled by heuristic)
        "evidence_check_input": 300,
        "evidence_check_output": 50,
        "evidence_check_probability": 0.4,
    },
}


def estimate_episode_tokens(agent: str, max_steps: int) -> dict:
    """Estimate total tokens for one episode."""
    cfg = TOKENS_PER_CALL[agent]
    
    # Base action calls
    input_tokens = cfg["input"] * max_steps * cfg["calls_per_step"]
    output_tokens = cfg["output"] * max_steps * cfg["calls_per_step"]
    
    # MemAgent extras
    if agent == "memagent":
        n_consolidations = max_steps // cfg["consolidation_interval"]
        input_tokens += cfg["consolidation_input"] * n_consolidations
        output_tokens += cfg["consolidation_output"] * n_consolidations
        
        # Evidence checks (only fraction actually call LLM)
        avg_rules_per_consolidation = 2
        n_evidence_calls = int(
            n_consolidations * avg_rules_per_consolidation *
            cfg["evidence_check_probability"]
        )
        input_tokens += cfg["evidence_check_input"] * n_evidence_calls
        output_tokens += cfg["evidence_check_output"] * n_evidence_calls
    
    cost = (input_tokens / 1_000_000 * COST_PER_1M_INPUT +
            output_tokens / 1_000_000 * COST_PER_1M_OUTPUT)
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
    }


def estimate_experiment(agents: list, n_tasks: int, episodes_per: int,
                         max_steps: int) -> dict:
    """Estimate full experiment cost."""
    results = {}
    total_input = 0
    total_output = 0
    total_cost = 0.0
    total_api_calls = 0
    
    for agent in agents:
        ep = estimate_episode_tokens(agent, max_steps)
        n_episodes = n_tasks * episodes_per
        
        agent_input = ep["input_tokens"] * n_episodes
        agent_output = ep["output_tokens"] * n_episodes
        agent_cost = ep["cost_usd"] * n_episodes
        agent_calls = max_steps * n_episodes
        
        if agent == "memagent":
            n_consolidations = max_steps // TOKENS_PER_CALL[agent]["consolidation_interval"]
            agent_calls += n_consolidations * n_episodes
        
        results[agent] = {
            "episodes": n_episodes,
            "input_tokens": agent_input,
            "output_tokens": agent_output,
            "total_tokens": agent_input + agent_output,
            "cost_usd": agent_cost,
            "api_calls": agent_calls,
        }
        
        total_input += agent_input
        total_output += agent_output
        total_cost += agent_cost
        total_api_calls += agent_calls
    
    results["TOTAL"] = {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "cost_usd": total_cost,
        "api_calls": total_api_calls,
    }
    
    return results


def print_estimate(results: dict, budget: float = 200.0):
    print("\n" + "=" * 72)
    print("  BUDGET ESTIMATE")
    print("=" * 72)
    print(f"\n  {'Agent':<18} {'Episodes':>8} {'Input Tok':>12} {'Output Tok':>12} {'Cost':>10} {'API Calls':>10}")
    print(f"  {'─'*70}")
    
    for agent, data in results.items():
        if agent == "TOTAL":
            continue
        print(f"  {agent:<18} {data['episodes']:>8} {data['input_tokens']:>12,} "
              f"{data['output_tokens']:>12,} ${data['cost_usd']:>8.2f} {data['api_calls']:>10,}")
    
    t = results["TOTAL"]
    print(f"  {'─'*70}")
    print(f"  {'TOTAL':<18} {'':>8} {t['input_tokens']:>12,} "
          f"{t['output_tokens']:>12,} ${t['cost_usd']:>8.2f} {t['api_calls']:>10,}")
    
    print(f"\n  Total tokens:  {t['total_tokens']:>12,}")
    print(f"  Budget:        ${budget:>11.2f}")
    print(f"  Estimated cost:${t['cost_usd']:>11.2f}")
    print(f"  Remaining:     ${budget - t['cost_usd']:>11.2f}")
    
    pct = t['cost_usd'] / budget * 100
    if pct < 30:
        emoji = "✅"
        msg = "SAFE — well within budget"
    elif pct < 60:
        emoji = "⚠️ "
        msg = "MODERATE — proceed with monitoring"
    elif pct < 90:
        emoji = "🔶"
        msg = "HIGH — consider reducing episodes or steps"
    else:
        emoji = "🛑"
        msg = "OVER BUDGET — reduce scope!"
    
    print(f"\n  {emoji} {pct:.1f}% of budget — {msg}")
    
    # Time estimate (assuming ~2 sec per API call with rate limiting)
    est_minutes = t['api_calls'] * 2 / 60
    print(f"\n  ⏱  Estimated wall time: ~{est_minutes:.0f} minutes ({est_minutes/60:.1f} hours)")
    
    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  IF TRITONAI IS FREE FOR STUDENTS:                   │
  │  • Your only constraint is RATE LIMITS + TIME        │
  │  • {t['api_calls']:,} API calls at ~2s each = {est_minutes:.0f} min        │
  │  • Consider running agents in parallel               │
  │  • Budget tracker still useful for monitoring usage   │
  └──────────────────────────────────────────────────────┘""")
    print("=" * 72)


def print_scaling_table(budget: float = 200.0):
    """Show how many episodes you can afford at different scales."""
    print("\n" + "=" * 72)
    print("  SCALING TABLE: How many episodes can you afford?")
    print("  (All 3 agents, 4 tasks, at commercial Llama-4-Scout pricing)")
    print("=" * 72)
    print(f"\n  {'Max Steps':>10} {'Episodes/Task':>14} {'Total Cost':>12} {'% Budget':>10} {'Wall Time':>12}")
    print(f"  {'─'*60}")
    
    for max_steps in [50, 100, 150, 200, 300]:
        for eps in [3, 5, 10, 15, 20]:
            r = estimate_experiment(
                ["no_memory", "naive_memory", "memagent"],
                n_tasks=4, episodes_per=eps, max_steps=max_steps
            )
            cost = r["TOTAL"]["cost_usd"]
            calls = r["TOTAL"]["api_calls"]
            minutes = calls * 2 / 60
            pct = cost / budget * 100
            
            if pct > 100:
                marker = " 🛑"
            elif pct > 60:
                marker = " ⚠️"
            else:
                marker = " ✅"
            
            if eps == 5:  # only show eps=5 to keep table readable
                print(f"  {max_steps:>10} {eps:>14} ${cost:>10.2f} {pct:>9.1f}%{marker} {minutes:>8.0f} min")
    
    print(f"\n  Recommendation for $200 budget (commercial pricing):")
    print(f"  → 4 tasks × 5 episodes × 200 steps = safe (~$25)")
    print(f"  → 4 tasks × 10 episodes × 200 steps = moderate (~$50)")
    print(f"  → 4 tasks × 20 episodes × 200 steps = still fine (~$100)")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Budget Calculator")
    parser.add_argument("--plan-experiment", action="store_true",
                        help="Estimate a specific experiment config")
    parser.add_argument("--agents", type=str, default="all")
    parser.add_argument("--tasks", type=int, default=4,
                        help="Number of tasks")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per agent per task")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--budget", type=float, default=200.0)
    parser.add_argument("--scaling-table", action="store_true",
                        help="Show scaling table")
    args = parser.parse_args()
    
    if args.agents == "all":
        agents = ["no_memory", "naive_memory", "memagent"]
    else:
        agents = args.agents.split(",")
    
    if args.scaling_table:
        print_scaling_table(args.budget)
        return
    
    print(f"\n  Planning: {len(agents)} agents × {args.tasks} tasks × "
          f"{args.episodes} episodes × {args.max_steps} max steps")
    
    results = estimate_experiment(agents, args.tasks, args.episodes, args.max_steps)
    print_estimate(results, args.budget)
    
    if not args.plan_experiment:
        print_scaling_table(args.budget)


if __name__ == "__main__":
    main()
