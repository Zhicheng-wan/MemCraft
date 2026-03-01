"""
Step Memory (Mstep): High-fidelity trajectory store.

Stores recent (action, observation, delta) tuples with delta-based filtering.
Only stores steps where something meaningful changed (inventory, health, entities, etc.)
to avoid wasting memory on repetitive "walked forward" entries.
"""
import json
from collections import deque
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import MSTEP_MAX_ENTRIES


class StepEntry:
    """A single entry in Step Memory."""
    
    __slots__ = ["step_num", "action", "observation_text", "delta", "note"]
    
    def __init__(self, step_num: int, action: str, observation_text: str,
                 delta: dict, note: str = ""):
        self.step_num = step_num
        self.action = action
        self.observation_text = observation_text
        self.delta = delta
        self.note = note  # LLM's own observation note
    
    def to_text(self) -> str:
        """Convert to searchable text for BM25."""
        parts = [f"[Step {self.step_num}] Action: {self.action}"]
        
        if self.delta:
            delta_strs = []
            if "inventory_changes" in self.delta:
                for item, info in self.delta["inventory_changes"].items():
                    diff = info["diff"]
                    sign = "+" if diff > 0 else ""
                    delta_strs.append(f"{item} {sign}{diff}")
            if "health_change" in self.delta:
                hc = self.delta["health_change"]
                delta_strs.append(f"health {'+'if hc>0 else ''}{hc:.0f}")
            if "entities_appeared" in self.delta:
                delta_strs.append(f"saw: {', '.join(self.delta['entities_appeared'])}")
            if "entities_disappeared" in self.delta:
                delta_strs.append(f"lost sight: {', '.join(self.delta['entities_disappeared'])}")
            if "mainhand_changed" in self.delta:
                mc = self.delta["mainhand_changed"]
                delta_strs.append(f"equipped {mc['to']} (was {mc['from']})")
            if "moved" in self.delta:
                delta_strs.append(f"moved {self.delta['moved']}")
            if delta_strs:
                parts.append(f"Changes: {'; '.join(delta_strs)}")
        
        if self.note:
            parts.append(f"Note: {self.note}")
        
        # Include condensed observation
        obs_lines = self.observation_text.split("\n")
        # Only include inventory and entities (most relevant for retrieval)
        for line in obs_lines:
            if line.startswith("inventory:") or line.startswith("nearby_entities:"):
                parts.append(line)
        
        return " | ".join(parts)
    
    def to_tokenized(self) -> List[str]:
        """Tokenize for BM25 indexing."""
        return self.to_text().lower().split()


class StepMemory:
    """
    Mstep: Bounded deque of StepEntry objects with delta filtering.
    
    Delta filtering: Only stores a step if something meaningful changed
    compared to the previous observation. This prevents the memory from
    being filled with "walked forward, nothing happened" entries.
    """
    
    def __init__(self, max_entries: int = MSTEP_MAX_ENTRIES):
        self.entries: deque = deque(maxlen=max_entries)
        self.all_entries: List[StepEntry] = []  # unfiltered for consolidation
        self._step_count = 0
    
    def should_store(self, delta: dict) -> bool:
        """
        Delta filter: returns True if this step is worth remembering.
        A step is stored if any meaningful change occurred.
        """
        if not delta:
            return False
        # Always store if inventory, health, or entities changed
        important_keys = {
            "inventory_changes", "health_change",
            "entities_appeared", "entities_disappeared",
            "mainhand_changed"
        }
        return bool(important_keys & set(delta.keys()))
    
    def add(self, action: str, observation_text: str, delta: dict,
            note: str = "", force: bool = False):
        """
        Add a step to memory.
        
        Args:
            action: The action taken
            observation_text: Current observation as text
            delta: Dict of changes from previous step
            note: LLM's observation note
            force: If True, skip delta filtering (e.g., first step, episode end)
        """
        self._step_count += 1
        entry = StepEntry(self._step_count, action, observation_text, delta, note)
        
        # Always keep in all_entries for consolidation
        self.all_entries.append(entry)
        
        # Delta filter for the bounded memory
        if force or self.should_store(delta):
            self.entries.append(entry)
    
    def get_recent(self, n: int = 5) -> List[StepEntry]:
        """Get the N most recent stored entries."""
        return list(self.entries)[-n:]
    
    def get_all_recent(self, n: int = 10) -> List[StepEntry]:
        """Get N most recent entries (unfiltered) for consolidation."""
        return self.all_entries[-n:]
    
    def get_all_texts(self) -> List[str]:
        """Get all entry texts for BM25 corpus."""
        return [e.to_text() for e in self.entries]
    
    def get_all_tokenized(self) -> List[List[str]]:
        """Get all tokenized entries for BM25."""
        return [e.to_tokenized() for e in self.entries]
    
    def clear(self):
        """Clear memory for new episode."""
        self.entries.clear()
        self.all_entries.clear()
        self._step_count = 0
    
    def __len__(self):
        return len(self.entries)
    
    def __repr__(self):
        return f"StepMemory({len(self.entries)} stored / {len(self.all_entries)} total)"
