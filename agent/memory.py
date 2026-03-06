"""
memory.py - Hierarchical Local Memory

Two-tier memory system:
  1. Step Memory (Mstep): High-fidelity trajectory of recent actions & observations
  2. Semantic Memory (Msem): Consolidated rules, constraints, and failure modes
     - Stored as JSON with IDs, supports INSERT / UPDATE / DELETE operations
"""

import json
import time
import uuid
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

    Stores rules, constraints, preconditions, and failure modes.
    Each rule has a unique ID for targeted UPDATE and DELETE operations.
    The LLM actively decides whether to insert, update, or delete rules.
    """

    def __init__(self, capacity: int = 30):
        self.capacity = capacity
        self.rules: List[dict] = []  # Each rule: {id, text, confidence, created_at, use_count}

    def _new_id(self) -> str:
        """Generate a short unique rule ID."""
        return uuid.uuid4().hex[:8]

    def apply_operations(self, operations: List[dict]) -> dict:
        """
        Apply a list of memory operations decided by the LLM.

        Each operation is one of:
          {"op": "insert", "text": "...", "confidence": 0.9}
          {"op": "update", "id": "abc123", "text": "...", "confidence": 0.95}
          {"op": "delete", "id": "abc123"}

        Returns a summary dict with counts of each op applied.
        """
        summary = {"inserted": 0, "updated": 0, "deleted": 0, "skipped": 0}

        for op_item in operations:
            op = op_item.get("op", "").lower()

            if op == "insert":
                text = op_item.get("text", "").strip()
                if not text or len(text) < 10:
                    summary["skipped"] += 1
                    continue
                # Enforce capacity: evict lowest-confidence rule
                if len(self.rules) >= self.capacity:
                    self.rules.sort(key=lambda r: r["confidence"])
                    self.rules.pop(0)
                self.rules.append({
                    "id": self._new_id(),
                    "text": text,
                    "confidence": float(op_item.get("confidence", 0.8)),
                    "created_at": time.time(),
                    "use_count": 0,
                })
                summary["inserted"] += 1

            elif op == "update":
                rule_id = op_item.get("id", "")
                text = op_item.get("text", "").strip()
                for rule in self.rules:
                    if rule["id"] == rule_id:
                        if text:
                            rule["text"] = text
                        if "confidence" in op_item:
                            rule["confidence"] = float(op_item["confidence"])
                        summary["updated"] += 1
                        break
                else:
                    summary["skipped"] += 1

            elif op == "delete":
                rule_id = op_item.get("id", "")
                before = len(self.rules)
                self.rules = [r for r in self.rules if r["id"] != rule_id]
                if len(self.rules) < before:
                    summary["deleted"] += 1
                else:
                    summary["skipped"] += 1

            else:
                summary["skipped"] += 1

        return summary

    # --- Legacy helpers (used by older code paths / retrieval) ---

    def add_rule(self, rule_text: str, evidence_steps: List[int] = None,
                 confidence: float = 1.0):
        """Add a rule directly (legacy / fallback)."""
        self.apply_operations([{"op": "insert", "text": rule_text,
                                "confidence": confidence}])

    def get_all(self) -> List[dict]:
        """Get all rules."""
        return self.rules

    def get_rules_text(self) -> str:
        """Get all rules as text (for prompts)."""
        if not self.rules:
            return "[No rules learned yet]"
        lines = []
        for r in self.rules:
            lines.append(
                f"  [{r['id']}] (conf={r['confidence']:.1f}): {r['text']}"
            )
        return "\n".join(lines)

    def get_rules_for_prompt(self) -> str:
        """Format rules with IDs so the LLM can reference them for UPDATE/DELETE."""
        return self.get_rules_text()

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
            if rule_lower in existing or existing in rule_lower:
                return True
        return False

    def clear(self):
        self.rules.clear()

    def __len__(self):
        return len(self.rules)

    def save(self, filepath: str):
        """Save semantic memory to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.rules, f, indent=2)

    def load(self, filepath: str):
        """Load semantic memory from JSON file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            # Migrate old format (rules without 'id') if needed
            for rule in data:
                if "id" not in rule:
                    rule["id"] = self._new_id()
            self.rules = data
        except FileNotFoundError:
            pass
