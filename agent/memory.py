"""
memory.py - Hierarchical Local Memory

Two-tier memory system from the slides:
  1. Step Memory (Mstep): High-fidelity trajectory of recent actions & observations
  2. Semantic Memory (Msem): Consolidated rules, constraints, and failure modes
"""

import json
import time
from typing import List, Optional
from collections import deque


class StepMemory:
    """
    Step Memory (Mstep) - Short-term, high-fidelity trajectory.
    
    Stores recent actions, observations, and outcomes.
    Uses a fixed-capacity FIFO with structured entries.
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.entries: deque = deque(maxlen=capacity)
        self.step_counter = 0

    def add(self, action: str, observation: str, result: str,
            success: bool = True):
        """Add a step to memory."""
        self.step_counter += 1
        entry = {
            "step": self.step_counter,
            "timestamp": time.time(),
            "action": action,
            "observation": observation,
            "result": result,
            "success": success,
            "text": f"Step {self.step_counter}: Action={action} | "
                    f"Result={result} | Obs={observation}"
        }
        self.entries.append(entry)

    def get_recent(self, n: int = 5) -> List[dict]:
        """Get the N most recent entries."""
        return list(self.entries)[-n:]

    def get_all(self) -> List[dict]:
        """Get all entries."""
        return list(self.entries)

    def get_failures(self) -> List[dict]:
        """Get all failed actions (useful for learning)."""
        return [e for e in self.entries if not e["success"]]

    def get_recent_text(self, n: int = 5) -> str:
        """Get recent entries as compact text."""
        recent = self.get_recent(n)
        if not recent:
            return "[No history yet]"
        lines = []
        for e in recent:
            status = "✓" if e["success"] else "✗"
            lines.append(f"  {status} {e['action']} → {e['result']}")
        return "\n".join(lines)

    def clear(self):
        """Clear all entries."""
        self.entries.clear()
        self.step_counter = 0

    def __len__(self):
        return len(self.entries)


class SemanticMemory:
    """
    Semantic Memory (Msem) - Long-term consolidated knowledge.

    Stores rules, constraints, preconditions, and failure modes
    that have been verified against evidence.
    """

    def __init__(self, capacity: int = 30):
        self.capacity = capacity
        self.rules: List[dict] = []

    def add_rule(self, rule_text: str, evidence_steps: List[int],
                 confidence: float = 1.0):
        """Add a verified rule."""
        if len(self.rules) >= self.capacity:
            # Remove lowest confidence rule
            self.rules.sort(key=lambda r: r["confidence"])
            self.rules.pop(0)

        self.rules.append({
            "text": rule_text,
            "evidence_steps": evidence_steps,
            "confidence": confidence,
            "created_at": time.time(),
            "use_count": 0,
        })

    def get_all(self) -> List[dict]:
        """Get all rules."""
        return self.rules

    def get_rules_text(self) -> str:
        """Get all rules as text."""
        if not self.rules:
            return "[No rules learned yet]"
        lines = []
        for i, r in enumerate(self.rules):
            lines.append(f"  Rule {i+1} (conf={r['confidence']:.1f}): {r['text']}")
        return "\n".join(lines)

    def mark_used(self, rule_text: str):
        """Increment use count for a rule."""
        for r in self.rules:
            if r["text"] == rule_text:
                r["use_count"] += 1
                break

    def has_similar_rule(self, rule_text: str) -> bool:
        """Check if a similar rule already exists (simple substring check)."""
        rule_lower = rule_text.lower()
        for r in self.rules:
            existing = r["text"].lower()
            # Check for significant overlap
            if (rule_lower in existing or existing in rule_lower):
                return True
        return False

    def clear(self):
        self.rules.clear()

    def __len__(self):
        return len(self.rules)

    def save(self, filepath: str):
        """Save semantic memory to file."""
        with open(filepath, "w") as f:
            json.dump(self.rules, f, indent=2)

    def load(self, filepath: str):
        """Load semantic memory from file."""
        try:
            with open(filepath, "r") as f:
                self.rules = json.load(f)
        except FileNotFoundError:
            pass
