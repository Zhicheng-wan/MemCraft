#!/usr/bin/env python3
"""
run_agent.py - Main entry point for MemCraft agent.

Usage:
    python run_agent.py --task "mine 5 dirt" --agent memagent
    python run_agent.py --task "mine 5 dirt" --agent no_memory
    python run_agent.py --task "mine 5 dirt" --agent naive_memory

    # Compare all three agents:
    python run_agent.py --task "mine 5 dirt" --agent compare
"""

import argparse
import json
import logging
import os
import requests
import subprocess
import sys
import time
from pathlib import Path

from agent.brain import Brain
from agent.agent import NoMemoryAgent, NaiveMemoryAgent, MemAgent

MINECRAFT_SEED = "0"  # Known forest biome seed for 1.19.2
MINECRAFT_CONTAINER = "mc-server"


def reset_minecraft_world(container=MINECRAFT_CONTAINER, seed=MINECRAFT_SEED, wait=90):
    """
    Full world reset via Docker: delete world, restart server, wait for generation.
    This guarantees a completely clean state with no leftover items or blocks.
    """
    logger = logging.getLogger("memcraft")
    logger.info(f"Resetting Minecraft world (seed={seed}, wait={wait}s)...")
    
    try:
        # Set seed
        subprocess.run(
            ["sudo", "docker", "exec", container, "sh", "-c",
             f'sed -i "s/level-seed=.*/level-seed={seed}/" /data/server.properties'],
            check=True, capture_output=True, timeout=10
        )
        # Set peaceful difficulty (no hostile mobs)
        subprocess.run(
            ["sudo", "docker", "exec", container, "sh", "-c",
             'sed -i "s/difficulty=.*/difficulty=peaceful/" /data/server.properties'],
            check=True, capture_output=True, timeout=10
        )
        # Delete world
        subprocess.run(
            ["sudo", "docker", "exec", container, "rm", "-rf",
             "/data/world", "/data/world_nether", "/data/world_the_end"],
            check=True, capture_output=True, timeout=10
        )
        # Restart server
        subprocess.run(
            ["sudo", "docker", "restart", container],
            check=True, capture_output=True, timeout=30
        )
        # Wait for world generation
        time.sleep(wait)
        
        # Verify server is up
        import socket
        for i in range(12):
            try:
                s = socket.socket()
                s.settimeout(2)
                s.connect(("localhost", 25565))
                s.close()
                logger.info("✓ Minecraft server is ready")
                time.sleep(5)  # Extra buffer for full load
                return True
            except:
                if i < 11:
                    time.sleep(5)
        logger.error("✗ Server not ready after wait")
        return False
    except Exception as e:
        logger.error(f"World reset failed: {e}")
        return False


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def start_mineflayer_bridge(host: str, port: int, username: str,
                             version: str, http_port: int) -> subprocess.Popen:
    """Start the Node.js Mineflayer bridge."""
    bridge_dir = Path(__file__).parent / "mineflayer_bridge"

    # Check if node_modules exists
    if not (bridge_dir / "node_modules").exists():
        logging.info("Installing Mineflayer dependencies...")
        subprocess.run(["npm", "install"], cwd=bridge_dir, check=True)

    cmd = [
        "node", "bot.js",
        f"--host={host}",
        f"--port={port}",
        f"--username={username}",
        f"--version={version}",
        f"--http_port={http_port}",
    ]
    logging.info(f"Starting Mineflayer bridge: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=bridge_dir,
        stdout=None, stderr=None,  # Let output flow to console for debugging
    )
    return proc


def run_single_agent(agent_type: str, brain: Brain, goal: str,
                     bot_url: str, max_steps: int, config: dict,
                     memory_file: str = "memories/semantic_rules.json") -> dict:
    """Run a single agent variant."""
    if agent_type == "no_memory":
        agent = NoMemoryAgent(brain, bot_url=bot_url, max_steps=max_steps)
    elif agent_type == "naive_memory":
        agent = NaiveMemoryAgent(brain, bot_url=bot_url, max_steps=max_steps,
                                  history_length=10)
    elif agent_type == "memagent":
        agent = MemAgent(brain, bot_url=bot_url, max_steps=max_steps,
                         config=config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if agent_type == "memagent":
        consolidation_file = memory_file.replace("_semantic.json", "_consolidation.json")
        if consolidation_file == memory_file:
            consolidation_file = memory_file.replace(".json", "_consolidation.json")
        return agent.run(goal, persist_memory=memory_file,
                         persist_consolidation=consolidation_file)
    else:
        return agent.run(goal)


def print_results(results: dict):
    """Pretty-print run results."""
    print("\n" + "=" * 60)
    print(f"Agent: {results['agent_type']}")
    print(f"Goal: {results['goal']}")
    print(f"Success: {'✓ YES' if results['success'] else '✗ NO'}")
    print(f"Total Steps: {results['total_steps']}")

    stats = results.get("brain_stats", {})
    print(f"\nLLM Stats:")
    print(f"  API Calls: {stats.get('total_calls', 0)}")
    print(f"  Total Tokens: {stats.get('total_tokens', 0)}")
    print(f"  Avg Latency: {stats.get('avg_latency', 0):.2f}s")

    if results.get("semantic_rules"):
        print(f"\nLearned Rules:")
        for rule in results["semantic_rules"]:
            print(f"  • {rule}")

    # Success rate
    steps = results.get("steps", [])
    if steps:
        successes = sum(1 for s in steps if s.get("success"))
        print(f"\nAction Success Rate: {successes}/{len(steps)} "
              f"({100*successes/len(steps):.0f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MemCraft Agent Runner")
    parser.add_argument("--task", type=str, required=True,
                        help="Task/goal description (e.g., 'mine 5 dirt blocks')")
    parser.add_argument("--agent", type=str, default="memagent",
                        choices=["memagent", "no_memory", "naive_memory", "compare"],
                        help="Agent variant to run")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Minecraft server host")
    parser.add_argument("--port", type=int, default=25565,
                        help="Minecraft server port")
    parser.add_argument("--username", type=str, default="MemAgent",
                        help="Bot username")
    parser.add_argument("--version", type=str, default="1.20.4",
                        help="Minecraft version")
    parser.add_argument("--http-port", type=int, default=3001,
                        help="Mineflayer bridge HTTP port")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximum steps per run")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes per agent (for compare mode)")
    parser.add_argument("--model", type=str, default="api-llama-4-scout",
                        help="LLM model name on TritonAI")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--no-bridge", action="store_true",
                        help="Skip starting Mineflayer bridge (if already running)")
    parser.add_argument("--config", type=str, default="configs/default.json",
                        help="Config file path")
    parser.add_argument("--no-recipes", action="store_true",
                        help="Disable recipe hints (agents must discover crafting sequences)")
    parser.add_argument("--skip-agents", type=str, default="",
                        help="Comma-separated agents to skip (e.g. no_memory,memagent)")
    parser.add_argument("--memory-file", type=str, default="memories/semantic_rules.json",
                        help="Path to semantic memory JSON file for memagent")

    args = parser.parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger("memcraft")

    # Set recipe mode
    from agent.agent import set_recipe_mode
    set_recipe_mode(not args.no_recipes)
    if args.no_recipes:
        logger.info("Recipe hints DISABLED - agents must discover crafting sequences")

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Get API key
    api_key = os.environ.get("TRITONAI_API_KEY")
    if not api_key:
        print("ERROR: Set TRITONAI_API_KEY environment variable")
        print("  export TRITONAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Ensure memories directory exists
    Path("memories").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Start Mineflayer bridge
    bridge_proc = None
    if not args.no_bridge:
        bridge_proc = start_mineflayer_bridge(
            args.host, args.port, args.username, args.version, args.http_port
        )
        # Give it time to connect
        logger.info("Waiting for bot to connect to Minecraft...")
        time.sleep(5)

    bot_url = f"http://localhost:{args.http_port}"
    api_url = config.get("llm", {}).get(
        "api_url", "https://tritonai-api.ucsd.edu/v1/chat/completions"
    )

    try:
        if args.agent == "compare":
            # Run all three agents for comparison across multiple episodes
            all_results = {}
            num_episodes = args.episodes

            skip_agents = [a.strip() for a in args.skip_agents.split(",") if a.strip()]

            for agent_type in ["no_memory", "naive_memory", "memagent"]:
                if agent_type in skip_agents:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Skipping agent: {agent_type}")
                    logger.info(f"{'='*50}")
                    continue

                logger.info(f"\n{'='*50}")
                logger.info(f"Agent: {agent_type} | {num_episodes} episodes")
                logger.info(f"{'='*50}")

                all_results[agent_type] = []

                for ep in range(num_episodes):
                    logger.info(f"\n--- Episode {ep+1}/{num_episodes} ---")

                    # Full world reset per episode to guarantee clean state
                    # (no leftover items, blocks, or entities from previous runs)
                    if bridge_proc:
                        try:
                            requests.post(f"{bot_url}/disconnect", timeout=3)
                        except:
                            pass
                        bridge_proc.terminate()
                        try:
                            bridge_proc.wait(timeout=5)
                        except:
                            bridge_proc.kill()
                        bridge_proc = None
                    
                    # Kill any orphaned node processes on our port
                    try:
                        subprocess.run(
                            ["pkill", "-f", "node bot.js"],
                            capture_output=True, timeout=5
                        )
                    except:
                        pass
                    time.sleep(3)

                    reset_minecraft_world(wait=90)

                    bridge_proc = start_mineflayer_bridge(
                        args.host, args.port, args.username,
                        args.version, args.http_port
                    )
                    time.sleep(20)  # Extra wait after fresh world reset

                    # Fresh brain per episode (for token tracking)
                    # But MemAgent keeps semantic memory across episodes!
                    brain = Brain(
                        api_key=api_key,
                        api_url=api_url,
                        model=args.model,
                        max_tokens=config.get("llm", {}).get("max_tokens", 512),
                        temperature=config.get("llm", {}).get("temperature", 0.3),
                    )

                    try:
                        results = run_single_agent(
                            agent_type, brain, args.task, bot_url,
                            args.max_steps, config,
                            memory_file=args.memory_file,
                        )
                    except Exception as e:
                        logger.error(f"Episode failed: {e}")
                        results = {
                            "agent_type": agent_type,
                            "goal": args.task,
                            "success": False,
                            "total_steps": 0,
                            "brain_stats": brain.get_stats(),
                            "steps": [],
                        }

                    results["episode"] = ep + 1
                    all_results[agent_type].append(results)
                    print_results(results)

            # ─── Summary Table ───
            print("\n" + "=" * 75)
            print("COMPARISON SUMMARY")
            print("=" * 75)
            print(f"Task: {args.task} | Episodes: {num_episodes}")
            print(f"{'Agent':<18} {'Success':<12} {'Avg Steps':<12} "
                  f"{'Avg Tokens':<14} {'Avg Latency':<12}")
            print("-" * 75)

            for agent_type in ["no_memory", "naive_memory", "memagent"]:
                if agent_type not in all_results:
                    continue
                episodes = all_results[agent_type]
                successes = sum(1 for r in episodes if r.get("success"))
                avg_steps = sum(r.get("total_steps", 0) for r in episodes) / max(len(episodes), 1)
                avg_tokens = sum(r.get("brain_stats", {}).get("total_tokens", 0) for r in episodes) / max(len(episodes), 1)
                avg_latency = sum(r.get("brain_stats", {}).get("avg_latency", 0) for r in episodes) / max(len(episodes), 1)

                print(f"{agent_type:<18} "
                      f"{successes}/{len(episodes):<10} "
                      f"{avg_steps:<12.1f} "
                      f"{avg_tokens:<14.0f} "
                      f"{avg_latency:<12.2f}s")

                # Show per-episode breakdown
                for r in episodes:
                    status = "✓" if r.get("success") else "✗"
                    steps = r.get("total_steps", 0)
                    tokens = r.get("brain_stats", {}).get("total_tokens", 0)
                    print(f"  Ep{r.get('episode', '?')}: {status} "
                          f"steps={steps} tokens={tokens}")

            print("=" * 75)

            # Save comparison
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            with open(f"logs/comparison_{timestamp}.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Comparison saved to logs/comparison_{timestamp}.json")

        else:
            # Run single agent
            brain = Brain(
                api_key=api_key,
                api_url=api_url,
                model=args.model,
                max_tokens=config.get("llm", {}).get("max_tokens", 512),
                temperature=config.get("llm", {}).get("temperature", 0.3),
            )
            results = run_single_agent(
                args.agent, brain, args.task, bot_url,
                args.max_steps, config,
                memory_file=args.memory_file,
            )
            print_results(results)

            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"logs/run_{args.agent}_{timestamp}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {out_path}")

    finally:
        if bridge_proc:
            logger.info("Shutting down Mineflayer bridge...")
            try:
                import requests as req
                req.post(f"{bot_url}/disconnect", timeout=3)
            except:
                pass
            bridge_proc.terminate()
            bridge_proc.wait(timeout=5)


if __name__ == "__main__":
    main()