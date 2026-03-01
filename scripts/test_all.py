#!/usr/bin/env python3
"""
test_all.py — Validate every component of MemAgent before burning API credits.

RUN THIS FIRST. It tests in 3 phases:

  Phase 1: Offline unit tests (no API, no MineDojo)
           → memory systems, BM25 retrieval, observation parsing, delta computation
           
  Phase 2: API smoke test (1 single API call)
           → validates your TritonAI key works and response parses correctly
           
  Phase 3: Integration test (mock env + real API, ~10 API calls)
           → runs each agent for 3 steps on MockEnv to verify the full loop

Usage:
    python scripts/test_all.py                      # all phases
    python scripts/test_all.py --offline-only        # phase 1 only (free)
    python scripts/test_all.py --smoke-only          # phase 2 only (1 API call)
    python scripts/test_all.py --integration-only    # phase 3 only (~10 calls)
"""
import argparse
import sys
import os
import json
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "✅"
FAIL = "❌"
SKIP = "⏭️ "
results = []


def test(name, fn):
    """Run a test function, catch exceptions, track results."""
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL} {name}")
        print(f"     → {e}")
        traceback.print_exc()
        print()


# ═══════════════════════════════════════════════════════════════════
#  PHASE 1: OFFLINE TESTS (no API, no MineDojo, totally free)
# ═══════════════════════════════════════════════════════════════════

def run_offline_tests():
    print("\n" + "=" * 60)
    print("  PHASE 1: Offline Unit Tests (no API calls)")
    print("=" * 60 + "\n")
    
    # --- Test 1: Imports ---
    def test_imports():
        from memory.step_memory import StepMemory, StepEntry
        from memory.semantic_memory import SemanticMemory, SemanticRule
        from memory.bm25_retriever import BM25Retriever
        from memory.consolidation import SemanticConsolidator
        from env.minecraft_env import (
            TextObservation, parse_observation, compute_delta,
            map_action_to_minedojo, MockEnv
        )
        from agents.base_agent import BaseAgent
        from agents.no_memory_agent import NoMemoryAgent
        from agents.naive_memory_agent import NaiveMemoryAgent
        from agents.mem_agent import MemAgent
        from evaluation.evaluator import Evaluator, EpisodeResult
        from config.settings import TokenTracker
        from config.prompts import MEMAGENT_SYSTEM_PROMPT
    test("All imports succeed", test_imports)
    
    # --- Test 2: TextObservation parsing ---
    def test_observation_parsing():
        from env.minecraft_env import parse_observation, TextObservation
        
        raw = {
            "inventory": {
                "name": ["shears", "wool", "air", "air"],
                "quantity": [1, 3, 0, 0],
            },
            "location_stats": {
                "pos": [100.5, 65.0, -200.3],
                "pitch": [10.0],
                "yaw": [-45.0],
                "biome_id": 1,
                "world_time": [6000],
                "is_raining": [False],
            },
            "life_stats": {
                "life": [18.5],
                "food": [17.0],
                "oxygen": [300.0],
            },
            "equipment": {
                "name": ["shears", "air", "air", "air", "air", "air"],
            },
            "entity_info": {
                "entity_name": ["sheep", "cow"],
                "entity_distance": [5.2, 12.8],
            },
        }
        
        obs = parse_observation(raw)
        text = obs.to_text()
        
        assert "shears: 1" in text, f"Missing shears in: {text}"
        assert "wool: 3" in text, f"Missing wool in: {text}"
        assert "sheep" in text, f"Missing sheep in: {text}"
        assert "health=18" in text or "health=19" in text, f"Missing health in: {text}"
        assert obs.get_inventory_terms() == ["shears", "wool"], \
            f"Inventory terms wrong: {obs.get_inventory_terms()}"
        assert "sheep" in obs.get_entity_terms(), \
            f"Entity terms wrong: {obs.get_entity_terms()}"
    test("Observation parsing + text conversion", test_observation_parsing)
    
    # --- Test 3: Delta computation ---
    def test_delta():
        from env.minecraft_env import parse_observation, compute_delta
        
        obs1_raw = {
            "inventory": {"name": ["shears", "air"], "quantity": [1, 0]},
            "location_stats": {"pos": [100, 65, 100], "pitch": [0], "yaw": [0],
                               "biome_id": 1, "world_time": [6000], "is_raining": [False]},
            "life_stats": {"life": [20], "food": [20], "oxygen": [300]},
            "equipment": {"name": ["shears", "air", "air", "air", "air", "air"]},
            "entity_info": {"entity_name": [], "entity_distance": []},
        }
        obs2_raw = {
            "inventory": {"name": ["shears", "wool"], "quantity": [1, 1]},
            "location_stats": {"pos": [101, 65, 100], "pitch": [0], "yaw": [0],
                               "biome_id": 1, "world_time": [6020], "is_raining": [False]},
            "life_stats": {"life": [18], "food": [20], "oxygen": [300]},
            "equipment": {"name": ["shears", "air", "air", "air", "air", "air"]},
            "entity_info": {"entity_name": ["sheep"], "entity_distance": [3.0]},
        }
        
        obs1 = parse_observation(obs1_raw)
        obs2 = parse_observation(obs2_raw)
        delta = compute_delta(obs1, obs2)
        
        assert "inventory_changes" in delta, f"Should detect wool gained: {delta}"
        assert "wool" in delta["inventory_changes"], f"Should show wool: {delta}"
        assert delta["inventory_changes"]["wool"]["diff"] == 1
        assert "health_change" in delta, f"Should detect health loss: {delta}"
        assert delta["health_change"] == -2.0
        assert "entities_appeared" in delta, f"Should detect sheep: {delta}"
        assert "sheep" in delta["entities_appeared"]
    test("Delta computation (inventory + health + entities)", test_delta)
    
    # --- Test 4: Step Memory with delta filtering ---
    def test_step_memory():
        from memory.step_memory import StepMemory
        
        mem = StepMemory(max_entries=10)
        
        # Boring step (no delta) — should be filtered
        mem.add("forward", "obs1", delta={}, note="")
        assert len(mem) == 0, "Empty delta should be filtered"
        
        # Meaningful step (inventory change) — should be stored
        mem.add("attack", "obs2",
                delta={"inventory_changes": {"wool": {"old": 0, "new": 1, "diff": 1}}},
                note="Got wool!")
        assert len(mem) == 1, "Inventory change should be stored"
        
        # Force store
        mem.add("noop", "obs3", delta={}, note="forced", force=True)
        assert len(mem) == 2, "Forced entry should be stored"
        
        # Check text output
        texts = mem.get_all_texts()
        assert len(texts) == 2
        assert "wool" in texts[0].lower()
        
        # Check all_entries tracks everything
        assert len(mem.all_entries) == 3, "all_entries should have all 3"
        
        # Test clear
        mem.clear()
        assert len(mem) == 0
    test("Step Memory — delta filtering + forced store", test_step_memory)
    
    # --- Test 5: Semantic Memory ---
    def test_semantic_memory():
        from memory.semantic_memory import SemanticMemory
        
        mem = SemanticMemory(max_rules=5)
        
        mem.add_rule("Sheep must be sheared with shears to get wool")
        mem.add_rule("Spiders are hostile at night")
        assert len(mem) == 2
        
        # Near-duplicate should boost existing, not add new
        mem.add_rule("You need shears to shear sheep for wool")
        assert len(mem) == 2, f"Duplicate should merge, got {len(mem)}"
        assert mem.rules[0].times_verified == 1, "Should have boosted confidence"
        
        # Fill to max
        for i in range(5):
            mem.add_rule(f"Unique rule number {i} about topic {i*7}")
        assert len(mem) <= 5, f"Should cap at 5, got {len(mem)}"
        
        # Test text output
        text = mem.to_text()
        assert "shear" in text.lower() or "rule" in text.lower()
    test("Semantic Memory — add/dedup/eviction", test_semantic_memory)
    
    # --- Test 6: BM25 Retriever ---
    def test_bm25():
        from memory.step_memory import StepMemory
        from memory.semantic_memory import SemanticMemory
        from memory.bm25_retriever import BM25Retriever
        
        step_mem = StepMemory()
        step_mem.add("attack", "inventory: {shears: 1}\nnearby_entities: [sheep(3.0m)]",
                     delta={"inventory_changes": {"wool": {"old": 0, "new": 1, "diff": 1}}},
                     note="Sheared a sheep", force=True)
        step_mem.add("forward", "inventory: {shears: 1, wool: 1}\nnearby_entities: [cow(8.0m)]",
                     delta={"entities_appeared": ["cow"]},
                     note="Found a cow", force=True)
        step_mem.add("craft wooden_pickaxe", "inventory: {wooden_pickaxe: 1}",
                     delta={"inventory_changes": {"wooden_pickaxe": {"old": 0, "new": 1, "diff": 1}}},
                     note="Crafted pickaxe", force=True)
        
        sem_mem = SemanticMemory()
        sem_mem.add_rule("Sheep must be sheared with shears to obtain wool")
        sem_mem.add_rule("Cows can be milked with an empty bucket")
        sem_mem.add_rule("Stone requires a pickaxe to mine")
        
        retriever = BM25Retriever(top_k=2)
        
        # Query about sheep and wool
        step_results, sem_results = retriever.retrieve(
            goal="obtain wool from sheep",
            inventory_terms=["shears"],
            entity_terms=["sheep"],
            step_memory=step_mem,
            semantic_memory=sem_mem,
        )
        
        assert len(step_results) > 0, "Should retrieve sheep-related step"
        assert any("sheep" in r.lower() or "wool" in r.lower() or "shear" in r.lower()
                    for r in step_results), \
            f"Top result should mention sheep/wool: {step_results}"
        assert any("sheep" in r.lower() or "wool" in r.lower() or "shear" in r.lower()
                    for r in sem_results), \
            f"Should retrieve sheep rule: {sem_results}"
    test("BM25 Retrieval — query construction + ranking", test_bm25)
    
    # --- Test 7: MockEnv works ---
    def test_mock_env():
        from env.minecraft_env import MockEnv, parse_observation, map_action_to_minedojo
        import numpy as np
        
        env = MockEnv("harvest_wool_with_shears_and_sheep")
        assert env.task_prompt is not None
        
        raw_obs = env.reset()
        obs = parse_observation(raw_obs)
        text = obs.to_text()
        assert "inventory:" in text
        assert "position:" in text
        
        # Test action mapping
        act = map_action_to_minedojo("forward", env)
        assert act[0] == 1, f"forward should set act[0]=1, got {act}"
        
        act = map_action_to_minedojo("attack", env)
        assert act[5] == 3, f"attack should set act[5]=3, got {act}"
        
        # Step
        raw_obs2, reward, done, info = env.step(act)
        obs2 = parse_observation(raw_obs2)
        assert "inventory:" in obs2.to_text()
        
        env.close()
    test("MockEnv — reset/step/parse/action_map", test_mock_env)
    
    # --- Test 8: Evaluator ---
    def test_evaluator():
        from evaluation.evaluator import Evaluator, EpisodeResult
        import tempfile, os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = Evaluator(results_dir=tmpdir)
            ev.add_result(EpisodeResult("memagent", "harvest_wool", 1, 50, 1.0, True))
            ev.add_result(EpisodeResult("memagent", "harvest_wool", 2, 80, 0.0, False))
            ev.add_result(EpisodeResult("no_memory", "harvest_wool", 1, 100, 0.0, False))
            
            summary = ev.summary()
            assert "memagent" in summary
            assert summary["memagent"]["harvest_wool"]["success_rate"] == 0.5
            
            ev.save_results("test.jsonl")
            assert os.path.exists(os.path.join(tmpdir, "test.jsonl"))
    test("Evaluator — results collection + summary", test_evaluator)
    
    # --- Test 9: Token Tracker ---
    def test_token_tracker():
        from config.settings import TokenTracker
        
        # Just verify the interface works (doesn't actually need real API)
        assert TokenTracker.check_budget() == True
        stats = TokenTracker.get_stats()
        assert "total_input_tokens" in stats
        assert "budget_remaining" in stats
    test("TokenTracker — budget checking", test_token_tracker)
    
    # --- Test 10: Prompt formatting ---
    def test_prompts():
        from config.prompts import (
            MEMAGENT_SYSTEM_PROMPT, MEMAGENT_USER_TEMPLATE,
            NO_MEMORY_USER_TEMPLATE, NAIVE_MEMORY_USER_TEMPLATE,
        )
        
        # Verify templates can be formatted without error
        p1 = NO_MEMORY_USER_TEMPLATE.format(
            goal="get wool", observation="inventory: {shears: 1}"
        )
        assert "get wool" in p1
        
        p2 = NAIVE_MEMORY_USER_TEMPLATE.format(
            goal="get wool", window_size=20,
            history="step 1: forward", observation="inventory: {}"
        )
        assert "20" in p2
        
        p3 = MEMAGENT_USER_TEMPLATE.format(
            goal="get wool",
            step_memories="[Step 1] attacked sheep",
            semantic_rules="• shears needed for wool",
            observation="inventory: {shears: 1}",
        )
        assert "shears needed" in p3
        assert len(MEMAGENT_SYSTEM_PROMPT) > 100
    test("Prompt templates — all format correctly", test_prompts)


# ═══════════════════════════════════════════════════════════════════
#  PHASE 2: API SMOKE TEST (exactly 1 API call)
# ═══════════════════════════════════════════════════════════════════

def run_smoke_test():
    print("\n" + "=" * 60)
    print("  PHASE 2: API Smoke Test (1 API call)")
    print("=" * 60 + "\n")
    
    def test_api_connection():
        from config.settings import API_KEY, API_BASE_URL, MODEL_NAME
        import requests
        
        if not API_KEY:
            raise Exception(
                "TRITONAI_API_KEY not set!\n"
                "     Fix: export TRITONAI_API_KEY='your-key'\n"
                "     Or:  echo 'TRITONAI_API_KEY=your-key' > .env"
            )
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "Respond with JSON only: {\"test\": \"ok\"}"},
                {"role": "user", "content": "Say hello."},
            ],
            "temperature": 0.1,
            "max_tokens": 50,
        }
        
        print(f"    Calling {API_BASE_URL}")
        print(f"    Model: {MODEL_NAME}")
        
        t0 = time.time()
        resp = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=30)
        latency = time.time() - t0
        
        print(f"    Status: {resp.status_code}")
        print(f"    Latency: {latency:.2f}s")
        
        assert resp.status_code == 200, \
            f"API returned {resp.status_code}: {resp.text[:300]}"
        
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"    Response: {content[:200]}")
        
        usage = data.get("usage", {})
        print(f"    Tokens — input: {usage.get('prompt_tokens', '?')}, "
              f"output: {usage.get('completion_tokens', '?')}")
        
        # Check we can parse JSON from it
        from config.llm_client import LLMClient
        client = LLMClient(agent_name="test", task_name="test")
        parsed = client._parse_json(content)
        print(f"    Parsed: {parsed}")
    
    test("API connection + JSON response", test_api_connection)


# ═══════════════════════════════════════════════════════════════════
#  PHASE 3: INTEGRATION TEST (mock env + real API, ~15 calls)
# ═══════════════════════════════════════════════════════════════════

def run_integration_test():
    print("\n" + "=" * 60)
    print("  PHASE 3: Integration Test (~15 API calls)")
    print("=" * 60 + "\n")
    
    from env.minecraft_env import MockEnv, parse_observation, map_action_to_minedojo
    from config.settings import TokenTracker
    
    INTEGRATION_STEPS = 5  # 5 steps × 3 agents = ~15 API calls
    
    def run_agent_integration(AgentClass, agent_name):
        env = MockEnv("harvest_wool_with_shears_and_sheep")
        goal = env.task_prompt
        agent = AgentClass(goal=goal, task_name="integration_test")
        agent.reset()
        
        raw_obs = env.reset()
        text_obs = parse_observation(raw_obs)
        
        actions_taken = []
        for step in range(INTEGRATION_STEPS):
            action = agent.act(text_obs)
            actions_taken.append(action)
            
            act_arr = map_action_to_minedojo(action, env)
            raw_obs, reward, done, info = env.step(act_arr)
            text_obs = parse_observation(raw_obs)
            agent.on_step_result(action, text_obs, reward, done, info)
            
            if done:
                break
        
        agent.on_episode_end()
        env.close()
        
        print(f"    {agent_name}: {INTEGRATION_STEPS} steps → actions: {actions_taken}")
        
        stats = agent.get_stats()
        assert stats["steps"] == INTEGRATION_STEPS, \
            f"Expected {INTEGRATION_STEPS} steps, got {stats['steps']}"
        
        # Verify actions are valid strings
        for a in actions_taken:
            assert isinstance(a, str), f"Action should be string, got {type(a)}"
            assert len(a) > 0, "Action should not be empty"
    
    def test_no_memory_integration():
        from agents.no_memory_agent import NoMemoryAgent
        run_agent_integration(NoMemoryAgent, "no_memory")
    
    def test_naive_memory_integration():
        from agents.naive_memory_agent import NaiveMemoryAgent
        run_agent_integration(NaiveMemoryAgent, "naive_memory")
    
    def test_memagent_integration():
        from agents.mem_agent import MemAgent
        run_agent_integration(MemAgent, "memagent")
    
    test("No-Memory agent (5 steps, mock env)", test_no_memory_integration)
    test("Naive Memory agent (5 steps, mock env)", test_naive_memory_integration)
    test("MemAgent full pipeline (5 steps, mock env)", test_memagent_integration)
    
    # Show token usage
    stats = TokenTracker.get_stats()
    print(f"\n    Total tokens used in integration test:")
    print(f"    Input:  {stats['total_input_tokens']:,}")
    print(f"    Output: {stats['total_output_tokens']:,}")
    print(f"    Cost:   ${stats['total_cost_usd']:.4f}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    
    for r in results:
        print(f"  {r[0]} {r[1]}")
    
    print(f"\n  {passed} passed, {failed} failed, {len(results)} total")
    
    if failed == 0:
        print(f"""
  ✅ ALL TESTS PASSED — You're ready to run experiments!
  
  Next steps:
  ┌────────────────────────────────────────────────────────┐
  │  1. Quick mock run (no MineDojo needed):               │
  │     python scripts/run_single.py \\                     │
  │       --agent memagent --mock \\                        │
  │       --episodes 2 --max-steps 30 --verbose            │
  │                                                        │
  │  2. Real MineDojo experiment:                          │
  │     MINEDOJO_HEADLESS=1 python scripts/run_experiment.py \\│
  │       --task harvest_wool_with_shears_and_sheep \\      │
  │       --agents all --episodes 5                        │
  │                                                        │
  │  3. Full benchmark:                                    │
  │     MINEDOJO_HEADLESS=1 python scripts/run_experiment.py \\│
  │       --tasks all --agents all --episodes 5            │
  └────────────────────────────────────────────────────────┘""")
    else:
        print("\n  ⚠️  Fix the failures above before running experiments.")
        for r in results:
            if r[0] == FAIL:
                print(f"     → {r[1]}: {r[2]}")
    
    print("=" * 60)
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="MemAgent Test Suite")
    parser.add_argument("--offline-only", action="store_true",
                        help="Only run offline tests (no API calls)")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Only run API smoke test (1 call)")
    parser.add_argument("--integration-only", action="store_true",
                        help="Only run integration test (~15 calls)")
    args = parser.parse_args()
    
    print("\n" + "🧪 " * 20)
    print("  MemAgent Test Suite")
    print("🧪 " * 20)
    
    run_specific = args.offline_only or args.smoke_only or args.integration_only
    
    if not run_specific or args.offline_only:
        run_offline_tests()
    
    if not run_specific or args.smoke_only:
        run_smoke_test()
    
    if not run_specific or args.integration_only:
        run_integration_test()
    
    success = print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
