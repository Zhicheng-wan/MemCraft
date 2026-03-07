"""
memory.py - Four-layer Hierarchical Memory

1. Working Memory (短期记忆):
   Current observation state — handled by Observer, injected into every prompt.
   No persistent class needed; the Observer generates it on each step.

2. Procedural Memory (程序性记忆 / 技能库):
   Learned skill sequences for reuse (craft_wooden_pickaxe, smelt_raw_iron, …).
   LLM extracts skills after successful multi-step completions.
   Stored in JSON; retrieved per-step by BM25.

3. Episodic Memory (情景/经验记忆):
   Failure reflections and key lessons.
   LLM writes one lesson per failed action.
   Stored in JSON; retrieved per-step by BM25.

4. Spatial Memory (空间记忆 / 地图):
   Important coordinates (resource veins, crafting spots, …).
   Auto-updated after successful mining — no LLM call needed.
   Stored in JSON; retrieved per-step by BM25.

FifoHistory: Simple FIFO buffer used by the NaiveMemoryAgent baseline.
"""

import json
import time
from typing import List
from collections import deque


# ---------------------------------------------------------------------------
# Baseline helper (NaiveMemoryAgent)
# ---------------------------------------------------------------------------

class FifoHistory:
    """Simple FIFO action history for the NaiveMemoryAgent baseline."""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.entries: deque = deque(maxlen=capacity)

    def add(self, action: str, result: str, success: bool = True):
        """Append an action-result pair."""
        self.entries.append({
            "action": action,
            "result": result,
            "success": success,
        })

    def get_recent_text(self, n: int = 5) -> str:
        recent = list(self.entries)[-n:]
        if not recent:
            return "[No history yet]"
        lines = []
        for e in recent:
            status = "✓" if e["success"] else "✗"
            lines.append(f"  {status} {e['action']} → {e['result']}")
        return "\n".join(lines)

    def clear(self):
        self.entries.clear()

    def __len__(self):
        return len(self.entries)


# ---------------------------------------------------------------------------
# Layer 2 — Procedural Memory
# ---------------------------------------------------------------------------

class ProceduralMemory:
    """
    Procedural Memory (程序性记忆) — Learned skill sequences.

    Stores reusable action procedures extracted by the LLM after successful
    multi-step completions (e.g. "craft_wooden_pickaxe", "smelt_raw_iron").
    Retrieved by BM25 to give the agent proven step-by-step knowledge.
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.skills: List[dict] = []

    def add_skill(self, skill_name: str, description: str):
        """Add or update a named skill."""
        for s in self.skills:
            if s["skill_name"] == skill_name:
                s["description"] = description
                s["text"] = f"{skill_name}: {description}"
                s["timestamp"] = time.time()
                return
        self.skills.append({
            "skill_name": skill_name,
            "description": description,
            "text": f"{skill_name}: {description}",
            "timestamp": time.time(),
        })
        if len(self.skills) > self.capacity:
            self.skills = self.skills[-self.capacity:]

    def get_all(self) -> List[dict]:
        return self.skills

    def get_text(self) -> str:
        if not self.skills:
            return "[No skills learned yet]"
        return "\n".join(f"  - {s['skill_name']}: {s['description']}"
                         for s in self.skills)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.skills, f, indent=2)

    def load(self, filepath: str):
        try:
            with open(filepath, "r") as f:
                self.skills = json.load(f)
        except FileNotFoundError:
            pass

    def clear(self):
        self.skills.clear()

    def __len__(self):
        return len(self.skills)


# ---------------------------------------------------------------------------
# Layer 3 — Episodic Memory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Episodic Memory (情景/经验记忆) — Failure reflections & key lessons.

    The LLM writes one actionable lesson per failed action.
    Stored as text entries; retrieved by BM25 to warn the agent before
    it repeats a known mistake.
    """

    def __init__(self, capacity: int = 30):
        self.capacity = capacity
        self.episodes: List[dict] = []

    def add_lesson(self, context: str, lesson: str):
        """Record a lesson learned from an episode."""
        self.episodes.append({
            "context": context,
            "lesson": lesson,
            "text": f"Lesson [{context}]: {lesson}",
            "timestamp": time.time(),
        })
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity:]

    def get_all(self) -> List[dict]:
        return self.episodes

    def get_text(self, n: int = None) -> str:
        entries = self.episodes if n is None else self.episodes[-n:]
        if not entries:
            return "[No lessons learned yet]"
        return "\n".join(f"  - {e['lesson']}" for e in entries)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.episodes, f, indent=2)

    def load(self, filepath: str):
        try:
            with open(filepath, "r") as f:
                self.episodes = json.load(f)
        except FileNotFoundError:
            pass

    def clear(self):
        self.episodes.clear()

    def __len__(self):
        return len(self.episodes)

