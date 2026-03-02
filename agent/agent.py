"""
agent.py - Main agent loop with three variants:

1. NoMemoryAgent (Baseline): Current observation only
2. NaiveMemoryAgent (Baseline): FIFO history, no filtering
3. MemAgent (Ours): Structured delta + BM25 retrieval + semantic rules
"""

import json
import time
import requests
import logging
from typing import Optional

from .brain import Brain
from .memory import StepMemory, SemanticMemory
from .observer import Observer
from .retrieval import BM25Retriever, build_query_terms
from .consolidation import Consolidator

logger = logging.getLogger("memcraft")

# --- Available actions description for the LLM ---
ACTIONS_SCHEMA = """
Available actions (respond with ONLY a JSON object, no other text):

MINING (auto-equips best tool):
  {"name": "find_and_mine_block", "params": {"block_name": "<n>", "count": <1-5>}}
    Examples: dirt, oak_log, birch_log, cobblestone, coal_ore, iron_ore

CRAFTING (auto-places crafting table when needed if you have 4+ planks):
  {"name": "craft_item", "params": {"item_name": "<n>", "count": <int>}}

SMELTING (auto-crafts & places furnace if you have 8 cobblestone):
  {"name": "smelt_item", "params": {"item_name": "<n>", "count": <int>}}
    Example: smelt_item raw_iron 1 -> iron_ingot. Uses auto fuel (coal/planks/logs).

RECIPE CHAINS:
  Wooden pickaxe: mine 4 logs -> craft planks (count=4) -> craft stick -> craft wooden_pickaxe
  Stone pickaxe: [wooden pickaxe first] -> mine 3 stone (gives cobblestone) -> craft stone_pickaxe
  Iron ingot: [stone pickaxe first] -> mine iron_ore (gives raw_iron) -> mine coal_ore (fuel) -> smelt_item raw_iron 1
  IMPORTANT: Match plank type to log type! oak_log -> oak_planks, birch_log -> birch_planks
  NOTE: You can mine "stone" or "cobblestone" — both work. Same for "iron_ore" or "raw_iron".

MOVEMENT:
  {"name": "move_forward", "params": {"steps": <1-5>}}
  {"name": "move_to", "params": {"x": <int>, "y": <int>, "z": <int>}}

OTHER:
  {"name": "equip_item", "params": {"item_name": "<n>", "destination": "hand"}}
  {"name": "collect_nearby_items", "params": {}}
  {"name": "scan_surroundings", "params": {"radius": <4-16>}}
  {"name": "DONE", "params": {}}
  {"name": "STUCK", "params": {}}

RULES:
- SESSION INVENTORY shows only items collected THIS run. Use DONE when goal item appears.
- Use find_and_mine_block to mine. It auto-equips the best tool you have.
- If an action fails, try a DIFFERENT approach. NEVER repeat a failed action more than once.
- Convert ALL logs to planks before crafting other items.
- To mine stone/ores you NEED a pickaxe. Craft one first!
"""





class BaseAgent:
    """Base class for all agent variants."""

    def __init__(self, brain: Brain, bot_url: str = "http://localhost:3001",
                 max_steps: int = 100):
        self.brain = brain
        self.bot_url = bot_url
        self.max_steps = max_steps
        self.observer = Observer()
        # Consecutive failure tracking
        self._last_action_key = None
        self._consecutive_fails = 0
        self._max_consecutive_fails = 3  # Force different action after 3 same failures

    def get_observation(self) -> dict:
        """Get observation from Mineflayer bot."""
        try:
            resp = requests.get(f"{self.bot_url}/observe", timeout=5)
            return resp.json()
        except Exception as e:
            logger.error(f"Observation failed: {e}")
            return {"error": str(e)}

    def execute_action(self, action_name: str, params: dict) -> dict:
        """Send action to Mineflayer bot."""
        try:
            resp = requests.post(
                f"{self.bot_url}/action",
                json={"name": action_name, "params": params},
                timeout=60
            )
            return resp.json()
        except Exception as e:
            logger.error(f"Action failed: {e}")
            return {"success": False, "message": str(e)}

    def wait_for_bot(self, timeout: int = 60):
        """Wait for bot to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{self.bot_url}/health", timeout=2)
                data = resp.json()
                if data.get("ready"):
                    logger.info("Bot is ready!")
                    return True
            except:
                pass
            time.sleep(2)
        raise RuntimeError("Bot not ready after timeout")

    def reset_inventory(self):
        """Reset bot state via /kill (respawns with empty inventory)."""
        try:
            logger.info("Resetting bot via /kill...")
            resp = requests.post(f"{self.bot_url}/reset", timeout=15)
            data = resp.json()
            if data.get("success"):
                logger.info(f"Reset: {data.get('message')}")
                if not data.get("inventory_empty"):
                    logger.warning("Inventory may not be fully empty after reset")
            else:
                logger.warning(f"Reset issue: {data.get('message')}")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Reset failed: {e}")

    def parse_action(self, llm_response: str) -> Optional[dict]:
        """Parse LLM response into action dict."""
        parsed = self.brain.parse_json_response(llm_response)
        if parsed and "name" in parsed:
            return parsed
        return None

    def track_action_result(self, action_name: str, params: dict, success: bool):
        """Track consecutive identical actions (both failed AND successful repeats)."""
        action_key = f"{action_name}:{json.dumps(params, sort_keys=True)}"
        if action_key == self._last_action_key:
            self._consecutive_fails += 1
        else:
            self._consecutive_fails = 0
        self._last_action_key = action_key

    def get_failure_warning(self) -> str:
        """Return warning text if agent is stuck repeating an action."""
        if self._consecutive_fails >= self._max_consecutive_fails:
            return (
                f"\n⚠ WARNING: You have repeated the SAME action {self._consecutive_fails} times! "
                f"This is NOT making progress. You MUST try a COMPLETELY DIFFERENT action type. "
                f"If you were using move_to, try find_and_mine_block or craft_item instead. "
                f"If crafting fails, you may be missing sticks - craft stick first. "
                f"Think about what sub-step is actually needed next."
            )
        return ""

    def check_goal_complete(self, goal: str, raw_obs: dict) -> bool:
        """
        Programmatic check if the goal is met based on session inventory.
        This is a safety net - doesn't rely on the LLM saying DONE.
        """
        session_inv = self.observer._get_session_inventory(raw_obs)
        raw_inv = raw_obs.get("inventory", {})
        baseline = self.observer.baseline_inventory or {}
        logger.debug(
            f"Goal check - raw_inv: {raw_inv}, baseline: {baseline}, "
            f"session_inv: {session_inv}"
        )

        goal_lower = goal.lower()

        # Parse "mine/collect N <item>" patterns
        import re
        match = re.search(r'(\d+)\s+(\w+)', goal_lower)
        if match:
            count = int(match.group(1))
            item_keyword = match.group(2)
            for item_name, item_count in session_inv.items():
                if item_keyword in item_name:
                    if item_count >= count:
                        return True

        # Check "craft a/an <item>" patterns
        craft_match = re.search(r'(?:craft|smelt|make|get)\s+(?:a\s+|an\s+)?(.+?)(?:\s*$)', goal_lower)
        if craft_match:
            target = craft_match.group(1).strip().replace(' ', '_')
            # Exact match: "stone_pickaxe" must match "stone_pickaxe", not "stone"
            if target in session_inv and session_inv[target] >= 1:
                return True
            # Also check without underscores for partial (e.g. "iron ingot" -> "iron_ingot")
            for item_name, item_count in session_inv.items():
                if item_name == target and item_count >= 1:
                    return True

        return False

    def build_prompt(self, goal: str, observation: str) -> tuple:
        """Build system and user prompts. Override in subclasses."""
        raise NotImplementedError


class NoMemoryAgent(BaseAgent):
    """
    Baseline 1: No Memory
    Input: Current observation only. No history. No retrieval.
    """

    def build_prompt(self, goal: str, observation: str, **kwargs) -> tuple:
        system = (
            "You are a Minecraft bot. You are given a goal and the current "
            "game state. Choose ONE action to take next.\n"
            "Respond with ONLY a JSON action object.\n\n"
            f"{ACTIONS_SCHEMA}"
        )
        failure_warning = self.get_failure_warning()
        user = (
            f"GOAL: {goal}\n\n"
            f"CURRENT STATE:\n{observation}{failure_warning}\n\n"
            f"Choose your next action (JSON only):"
        )
        return system, user

    def run(self, goal: str) -> dict:
        """Run the agent loop."""
        logger.info(f"[NoMemory] Starting task: {goal}")
        self.wait_for_bot()
        self.reset_inventory()
        self.observer.reset()

        # Snapshot current inventory so LLM only sees items gained this session
        baseline_obs = self.get_observation()
        self.observer.set_baseline_inventory(baseline_obs)

        results = {
            "agent_type": "no_memory",
            "goal": goal,
            "steps": [],
            "success": False,
            "total_steps": 0,
        }

        for step in range(self.max_steps):
            # Observe
            raw_obs = self.get_observation()
            obs_text = self.observer.observe_full(raw_obs)

            # Think
            system, user = self.build_prompt(goal, obs_text)
            response = self.brain.query(system, user)

            if response["error"]:
                logger.error(f"Step {step}: LLM error: {response['error']}")
                continue

            # Act
            action = self.parse_action(response["content"])
            if not action:
                logger.warning(f"Step {step}: Failed to parse action")
                continue

            action_name = action["name"]
            action_params = action.get("params", {})

            if action_name == "DONE":
                logger.info(f"Step {step}: Agent reports task complete!")
                results["success"] = True
                break
            if action_name == "STUCK":
                logger.warning(f"Step {step}: Agent reports stuck!")
                break

            result = self.execute_action(action_name, action_params)
            success = result.get("success", False)
            self.track_action_result(action_name, action_params, success)
            logger.info(
                f"Step {step}: {action_name}({action_params}) → "
                f"{'✓' if success else '✗'} {result.get('message', '')}"
            )

            results["steps"].append({
                "step": step,
                "action": action_name,
                "params": action_params,
                "result": result.get("message", ""),
                "success": success,
                "tokens": response["tokens_used"],
            })

            # Programmatic goal check (don't rely solely on LLM saying DONE)
            post_obs = self.get_observation()
            if self.check_goal_complete(goal, post_obs):
                logger.info(f"Step {step}: Goal achieved (programmatic check)!")
                results["success"] = True
                break

            time.sleep(0.5)  # Brief pause between steps

        results["total_steps"] = len(results["steps"])
        results["brain_stats"] = self.brain.get_stats()
        return results


class NaiveMemoryAgent(BaseAgent):
    """
    Baseline 2: Naive Memory
    Input: FIFO History (Last L steps). No delta filtering. No pruning.
    """

    def __init__(self, *args, history_length: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_length = history_length
        self.step_memory = StepMemory(capacity=history_length)

    def build_prompt(self, goal: str, observation: str,
                     history: str = "", **kwargs) -> tuple:
        system = (
            "You are a Minecraft bot. You are given a goal, your recent "
            "action history, and the current game state. Choose ONE action.\n"
            "Respond with ONLY a JSON action object.\n\n"
            f"{ACTIONS_SCHEMA}"
        )
        failure_warning = self.get_failure_warning()
        user = (
            f"GOAL: {goal}\n\n"
            f"RECENT HISTORY:\n{history}\n\n"
            f"CURRENT STATE:\n{observation}{failure_warning}\n\n"
            f"Choose your next action (JSON only):"
        )
        return system, user

    def run(self, goal: str) -> dict:
        logger.info(f"[NaiveMemory] Starting task: {goal}")
        self.wait_for_bot()
        self.reset_inventory()
        self.observer.reset()
        self.step_memory.clear()

        # Snapshot current inventory so LLM only sees items gained this session
        baseline_obs = self.get_observation()
        self.observer.set_baseline_inventory(baseline_obs)

        results = {
            "agent_type": "naive_memory",
            "goal": goal,
            "steps": [],
            "success": False,
            "total_steps": 0,
        }

        for step in range(self.max_steps):
            raw_obs = self.get_observation()
            obs_text = self.observer.observe_full(raw_obs)  # Full obs every time
            history = self.step_memory.get_recent_text(self.history_length)

            system, user = self.build_prompt(goal, obs_text, history=history)
            response = self.brain.query(system, user)

            if response["error"]:
                logger.error(f"Step {step}: LLM error: {response['error']}")
                continue

            action = self.parse_action(response["content"])
            if not action:
                logger.warning(f"Step {step}: Parse failed")
                continue

            action_name = action["name"]
            action_params = action.get("params", {})

            if action_name == "DONE":
                results["success"] = True
                break
            if action_name == "STUCK":
                break

            result = self.execute_action(action_name, action_params)
            success = result.get("success", False)
            self.track_action_result(action_name, action_params, success)

            # Store in FIFO (full observation, no delta)
            self.step_memory.add(
                action=f"{action_name}({json.dumps(action_params)})",
                observation=obs_text,
                result=result.get("message", ""),
                success=success
            )

            logger.info(
                f"Step {step}: {action_name} → "
                f"{'✓' if success else '✗'} {result.get('message', '')}"
            )
            results["steps"].append({
                "step": step,
                "action": action_name,
                "params": action_params,
                "result": result.get("message", ""),
                "success": result.get("success", False),
                "tokens": response["tokens_used"],
            })

            # Programmatic goal check
            post_obs = self.get_observation()
            if self.check_goal_complete(goal, post_obs):
                logger.info(f"Step {step}: Goal achieved (programmatic check)!")
                results["success"] = True
                break

            time.sleep(0.5)

        results["total_steps"] = len(results["steps"])
        results["brain_stats"] = self.brain.get_stats()
        return results


class MemAgent(BaseAgent):
    """
    MemAgent (Ours): Full hierarchical memory system.
    Input: Structured Delta + BM25 Retrieval + Semantic Rules.

    Key optimizations:
    - Delta encoding: only send what changed (saves tokens)
    - BM25 retrieval: fetch relevant memories, not all (saves tokens)
    - Semantic consolidation: compress patterns into rules (saves tokens)
    """

    def __init__(self, brain: Brain, bot_url: str = "http://localhost:3001",
                 max_steps: int = 100, config: dict = None):
        super().__init__(brain, bot_url=bot_url, max_steps=max_steps)
        cfg = config or {}
        mem_cfg = cfg.get("memory", {})

        self.step_memory = StepMemory(
            capacity=mem_cfg.get("step_memory_capacity", 50)
        )
        self.semantic_memory = SemanticMemory(
            capacity=mem_cfg.get("semantic_memory_capacity", 30)
        )
        self.retriever = BM25Retriever()
        self.consolidator = Consolidator(
            brain=brain,
            interval=mem_cfg.get("consolidation_interval", 10),
            evidence_window=mem_cfg.get("consolidation_evidence_window", 5),
        )
        self.retrieval_top_k = mem_cfg.get("retrieval_top_k", 5)

    def build_prompt(self, goal: str, observation: str,
                     retrieved_memories: str = "",
                     semantic_rules: str = "", **kwargs) -> tuple:
        system = (
            "You are a Minecraft bot with memory. You have:\n"
            "1. Current observation (delta: what changed)\n"
            "2. Retrieved relevant memories from past steps\n"
            "3. Learned rules from experience\n\n"
            "Use all information to choose the BEST next action.\n"
            "Respond with ONLY a JSON action object.\n\n"
            f"{ACTIONS_SCHEMA}"
        )

        # Build memory block (injected into prompt)
        memory_block = ""
        if retrieved_memories:
            memory_block += f"RETRIEVED MEMORIES:\n{retrieved_memories}\n\n"
        if semantic_rules:
            memory_block += f"LEARNED RULES:\n{semantic_rules}\n\n"

        failure_warning = self.get_failure_warning()
        user = (
            f"GOAL: {goal}\n\n"
            f"{memory_block}"
            f"CURRENT STATE (changes since last step):\n{observation}{failure_warning}\n\n"
            f"Choose your next action (JSON only):"
        )
        return system, user

    def run(self, goal: str, persist_memory: str = None) -> dict:
        """
        Run the MemAgent loop.

        Args:
            goal: Task description (e.g., "mine 10 dirt blocks")
            persist_memory: Optional filepath to load/save semantic memory
        """
        logger.info(f"[MemAgent] Starting task: {goal}")
        self.wait_for_bot()
        self.reset_inventory()
        self.observer.reset()
        self.step_memory.clear()

        # Snapshot current inventory so LLM only sees items gained this session
        baseline_obs = self.get_observation()
        self.observer.set_baseline_inventory(baseline_obs)

        # Load persisted semantic memory if available
        if persist_memory:
            self.semantic_memory.load(persist_memory)
            logger.info(
                f"Loaded {len(self.semantic_memory)} semantic rules from {persist_memory}"
            )

        results = {
            "agent_type": "memagent",
            "goal": goal,
            "steps": [],
            "success": False,
            "total_steps": 0,
            "consolidation_events": [],
        }

        for step in range(self.max_steps):
            # ── Observe (delta encoded) ──
            raw_obs = self.get_observation()
            if step == 0:
                obs_text = self.observer.observe_full(raw_obs)
            else:
                obs_text = self.observer.observe_delta(raw_obs)

            # ── BM25 Retrieval ──
            query_terms = build_query_terms(
                goal=goal,
                inventory_terms=self.observer.get_inventory_terms(raw_obs),
                entity_terms=self.observer.get_entity_terms(raw_obs)
            )

            # Retrieve from step memory
            step_results = self.retriever.retrieve(
                query_terms, self.step_memory.get_all(),
                top_k=self.retrieval_top_k
            )
            # Retrieve from semantic memory
            sem_results = self.retriever.retrieve(
                query_terms, self.semantic_memory.get_all(),
                top_k=3
            )

            retrieved_text = ""
            if step_results:
                lines = [f"  - {e['text']}" for e, score in step_results if score > 0]
                if lines:
                    retrieved_text = "\n".join(lines)

            rules_text = ""
            if sem_results:
                lines = [f"  - {e['text']}" for e, score in sem_results if score > 0]
                if lines:
                    rules_text = "\n".join(lines)
            elif len(self.semantic_memory) > 0:
                # If BM25 didn't match, still include top rules
                rules_text = self.semantic_memory.get_rules_text()

            # ── Think (LLM call) ──
            system, user = self.build_prompt(
                goal, obs_text,
                retrieved_memories=retrieved_text,
                semantic_rules=rules_text
            )
            response = self.brain.query(system, user)

            if response["error"]:
                logger.error(f"Step {step}: LLM error: {response['error']}")
                continue

            # ── Parse & Act ──
            action = self.parse_action(response["content"])
            if not action:
                logger.warning(f"Step {step}: Parse failed: {response['content'][:100]}")
                continue

            action_name = action["name"]
            action_params = action.get("params", {})

            if action_name == "DONE":
                logger.info(f"Step {step}: Task complete!")
                results["success"] = True
                break
            if action_name == "STUCK":
                logger.warning(f"Step {step}: Agent stuck!")
                break

            result = self.execute_action(action_name, action_params)
            success = result.get("success", False)
            self.track_action_result(action_name, action_params, success)

            # ── Store in step memory ──
            self.step_memory.add(
                action=f"{action_name}({json.dumps(action_params)})",
                observation=obs_text,
                result=result.get("message", ""),
                success=success
            )

            logger.info(
                f"Step {step}: {action_name}({action_params}) → "
                f"{'✓' if success else '✗'} {result.get('message', '')}"
            )

            results["steps"].append({
                "step": step,
                "action": action_name,
                "params": action_params,
                "result": result.get("message", ""),
                "success": success,
                "tokens": response["tokens_used"],
            })

            # Programmatic goal check
            post_obs = self.get_observation()
            if self.check_goal_complete(goal, post_obs):
                logger.info(f"Step {step}: Goal achieved (programmatic check)!")
                results["success"] = True
                break

            # ── Consolidation check ──
            if self.consolidator.should_consolidate(self.step_memory.step_counter):
                logger.info(f"Step {step}: Running consolidation...")
                new_rules = self.consolidator.consolidate(
                    self.step_memory, self.semantic_memory, goal
                )
                if new_rules:
                    logger.info(f"  New rules: {new_rules}")
                    results["consolidation_events"].append({
                        "step": step,
                        "new_rules": new_rules
                    })

            time.sleep(0.5)

        # ── Save semantic memory ──
        if persist_memory:
            self.semantic_memory.save(persist_memory)
            logger.info(f"Saved {len(self.semantic_memory)} rules to {persist_memory}")

        results["total_steps"] = len(results["steps"])
        results["brain_stats"] = self.brain.get_stats()
        results["semantic_rules"] = [r["text"] for r in self.semantic_memory.get_all()]
        return results