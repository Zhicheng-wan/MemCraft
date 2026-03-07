"""
observer.py - Structured delta observation encoder.

Key optimization: Only encode what CHANGED since the last observation.
This dramatically reduces token usage per step.
"""

import json
from typing import Optional

# Benchmark-relevant recipes: item → {material: qty_needed}
# Quantities are the minimum needed for one craft.
_CRAFTABLE_RECIPES = {
    "oak_planks":     {"oak_log": 1},
    "birch_planks":   {"birch_log": 1},
    "stick":          {"oak_planks": 2},
    "crafting_table": {"oak_planks": 4},
    "wooden_pickaxe": {"oak_planks": 3, "stick": 2},
    "stone_pickaxe":  {"cobblestone": 3, "stick": 2},
    "furnace":        {"cobblestone": 8},
    "iron_pickaxe":   {"iron_ingot": 3, "stick": 2},
}

# Tool prerequisites: to gather the primary material, you need this tool first.
_TOOL_PREREQUISITES = {
    "stone_pickaxe": "wooden_pickaxe",   # need wooden_pickaxe to mine cobblestone
    "iron_pickaxe":  "stone_pickaxe",    # need stone_pickaxe to mine iron ore
}

def _check_craftable(inv: dict) -> list:
    """Return list of items craftable from the given inventory."""
    return [
        item for item, needed in _CRAFTABLE_RECIPES.items()
        if all(inv.get(mat, 0) >= qty for mat, qty in needed.items())
    ]

def _missing_for_goal(inv: dict, goal: str) -> str:
    """Return missing materials for the goal item, or empty string if goal item not in recipes."""
    goal_lower = goal.lower()
    target = next(
        (item for item in _CRAFTABLE_RECIPES if item in goal_lower),
        None
    )
    if target is None:
        return ""

    # Check tool prerequisite first — must have the right tool before gathering materials
    prereq_tool = _TOOL_PREREQUISITES.get(target)
    if prereq_tool and inv.get(prereq_tool, 0) < 1:
        prereq_recipe = _CRAFTABLE_RECIPES.get(prereq_tool, {})
        prereq_missing = {mat: qty - inv.get(mat, 0) for mat, qty in prereq_recipe.items() if inv.get(mat, 0) < qty}
        if prereq_missing:
            parts = ", ".join(f"{mat} x{qty}" for mat, qty in prereq_missing.items())
            return f"Step 1: craft {prereq_tool} first. Missing: {parts}"
        else:
            return f"READY TO CRAFT: {prereq_tool} (needed before mining for {target})"

    needed = _CRAFTABLE_RECIPES[target]
    missing = {mat: qty - inv.get(mat, 0) for mat, qty in needed.items() if inv.get(mat, 0) < qty}
    if not missing:
        return f"READY TO CRAFT: {target}"
    parts = ", ".join(f"{mat} x{qty}" for mat, qty in missing.items())
    return f"Missing for {target}: {parts}"


class Observer:
    """Converts raw observations into compact text, with delta encoding."""

    def __init__(self):
        self.prev_observation: Optional[dict] = None
        self.baseline_inventory: Optional[dict] = None  # Inventory at session start

    def reset(self):
        """Reset for new episode."""
        self.prev_observation = None
        self.baseline_inventory = None

    def set_baseline_inventory(self, raw_obs: dict):
        """Snapshot the inventory at session start. 
        Only items gained AFTER this point will be shown to the LLM."""
        self.baseline_inventory = dict(raw_obs.get("inventory", {}))

    def _get_session_inventory(self, raw_obs: dict) -> dict:
        """Return only items gained since session start."""
        current = raw_obs.get("inventory", {})
        if self.baseline_inventory is None:
            return current
        session_inv = {}
        for item, count in current.items():
            baseline_count = self.baseline_inventory.get(item, 0)
            gained = count - baseline_count
            if gained > 0:
                session_inv[item] = gained
        return session_inv

    def observe_full(self, raw_obs: dict, goal: str = "") -> str:
        """Convert raw observation to full text representation."""
        parts = []

        # Position
        pos = raw_obs.get("position", {})
        parts.append(
            f"Position: ({pos.get('x', '?')}, {pos.get('y', '?')}, {pos.get('z', '?')})"
        )

        # Stats
        stats = raw_obs.get("stats", {})
        parts.append(
            f"Health: {stats.get('health', '?')}/20 | "
            f"Food: {stats.get('food', '?')}/20"
        )

        # Inventory (only items gained this session)
        inv = self._get_session_inventory(raw_obs)
        if inv:
            inv_str = ", ".join(f"{k}: {v}" for k, v in inv.items())
            parts.append(f"Inventory (this session): [{inv_str}]")
        else:
            parts.append("Inventory (this session): [empty]")

        # Craftable items + goal progress
        craftable = _check_craftable(inv)
        if craftable:
            parts.append(f"Craftable now: {', '.join(craftable)}")
        if goal:
            missing = _missing_for_goal(inv, goal)
            if missing:
                parts.append(missing)

        # Equipment
        eq = raw_obs.get("equipment", {})
        hand = eq.get("mainhand", "empty")
        if hand != "empty":
            parts.append(f"Holding: {hand}")

        # Environment
        env = raw_obs.get("environment", {})
        time_str = "day" if env.get("is_day", True) else "night"
        parts.append(f"Time: {time_str}")
        if env.get("is_raining"):
            parts.append("Weather: raining")

        # Nearby entities (compact)
        entities = raw_obs.get("nearby_entities", [])
        if entities:
            ent_strs = [f"{e['type']}(d={e['distance']})" for e in entities[:5]]
            parts.append(f"Nearby: {', '.join(ent_strs)}")

        # Nearby blocks (top 8, skip truly common noise)
        blocks = raw_obs.get("nearby_blocks", {})
        skip_blocks = {"dirt", "grass_block", "air", "bedrock"}
        interesting = {k: v for k, v in blocks.items() if k not in skip_blocks}
        if interesting:
            top = sorted(interesting.items(), key=lambda x: -x[1])[:8]
            block_strs = [f"{k}: {v}" for k, v in top]
            parts.append(f"Notable blocks: {', '.join(block_strs)}")

        # Error
        if raw_obs.get("last_error"):
            parts.append(f"⚠ Error: {raw_obs['last_error']}")

        return " | ".join(parts)

    def observe_delta(self, raw_obs: dict, goal: str = "") -> str:
        """
        Compute delta from previous observation.
        Only includes fields that changed.
        """
        if self.prev_observation is None:
            self.prev_observation = raw_obs
            return self.observe_full(raw_obs, goal=goal)

        prev = self.prev_observation
        changes = []

        # Position change
        pos = raw_obs.get("position", {})
        prev_pos = prev.get("position", {})
        if (pos.get("x") != prev_pos.get("x") or
            pos.get("y") != prev_pos.get("y") or
            pos.get("z") != prev_pos.get("z")):
            changes.append(
                f"Moved to ({pos.get('x')}, {pos.get('y')}, {pos.get('z')})"
            )

        # Stats change
        stats = raw_obs.get("stats", {})
        prev_stats = prev.get("stats", {})
        if stats.get("health") != prev_stats.get("health"):
            changes.append(f"Health: {stats.get('health')}/20")
        if stats.get("food") != prev_stats.get("food"):
            changes.append(f"Food: {stats.get('food')}/20")

        # Inventory diff (session-aware)
        inv = self._get_session_inventory(raw_obs)
        prev_inv = self._get_session_inventory(prev) if self.baseline_inventory else prev.get("inventory", {})
        inv_changes = []
        all_items = set(list(inv.keys()) + list(prev_inv.keys()))
        for item in all_items:
            curr = inv.get(item, 0)
            old = prev_inv.get(item, 0)
            if curr != old:
                diff = curr - old
                sign = "+" if diff > 0 else ""
                inv_changes.append(f"{item}: {sign}{diff} (session total: {curr})")
        if inv_changes:
            changes.append(f"Inventory: {', '.join(inv_changes)}")

        # Equipment change
        eq = raw_obs.get("equipment", {})
        prev_eq = prev.get("equipment", {})
        if eq.get("mainhand") != prev_eq.get("mainhand"):
            changes.append(f"Now holding: {eq.get('mainhand', 'empty')}")

        # New entities
        entities = raw_obs.get("nearby_entities", [])
        prev_types = {e["type"] for e in prev.get("nearby_entities", [])}
        new_entities = [e for e in entities if e["type"] not in prev_types]
        if new_entities:
            ent_strs = [f"{e['type']}(d={e['distance']})" for e in new_entities[:3]]
            changes.append(f"New nearby: {', '.join(ent_strs)}")

        # Error
        if raw_obs.get("last_error") and raw_obs["last_error"] != prev.get("last_error"):
            changes.append(f"⚠ Error: {raw_obs['last_error']}")

        self.prev_observation = raw_obs

        # Always show goal progress (even in delta — so agent never forgets what it needs)
        if goal:
            inv = self._get_session_inventory(raw_obs)
            missing = _missing_for_goal(inv, goal)
            if missing:
                changes.append(missing)

        if not changes:
            return "[No significant changes]"

        return " | ".join(changes)

    def get_inventory_terms(self, raw_obs: dict) -> list:
        """Extract inventory item names for BM25 query."""
        return list(raw_obs.get("inventory", {}).keys())

    def get_entity_terms(self, raw_obs: dict) -> list:
        """Extract nearby entity types for BM25 query."""
        return list(set(e["type"] for e in raw_obs.get("nearby_entities", [])))