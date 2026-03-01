"""
Semantic Memory (Msem): Consolidated rules store.

Stores generalizable rules/constraints the agent has learned, e.g.:
- "Sheep must be sheared with shears equipped in mainhand"
- "Spiders are hostile at night but neutral during day"
- "Cannot mine stone without a pickaxe"

Rules are extracted by the LLM from trajectory data and verified
against recent evidence before being kept.
"""
from typing import List, Optional
from collections import deque

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import MSEM_MAX_RULES


class SemanticRule:
    """A single consolidated rule in semantic memory."""
    
    __slots__ = ["text", "confidence", "times_verified", "source_step"]
    
    def __init__(self, text: str, confidence: float = 1.0, source_step: int = 0):
        self.text = text
        self.confidence = confidence
        self.times_verified = 0
        self.source_step = source_step
    
    def to_tokenized(self) -> List[str]:
        """Tokenize for BM25."""
        return self.text.lower().split()
    
    def __repr__(self):
        return f"Rule(conf={self.confidence:.1f}, verified={self.times_verified}): {self.text}"


class SemanticMemory:
    """
    Msem: Bounded store of consolidated rules.
    
    New rules are added via the consolidation process (triggered every N steps).
    Rules are verified against evidence and discarded if unsupported.
    When full, lowest-confidence rules are evicted.
    """
    
    def __init__(self, max_rules: int = MSEM_MAX_RULES):
        self.rules: List[SemanticRule] = []
        self.max_rules = max_rules
    
    def add_rule(self, text: str, confidence: float = 1.0, source_step: int = 0):
        """Add a new rule. Checks for near-duplicates first."""
        # Simple dedup: skip if very similar rule exists
        text_lower = text.lower().strip()
        for existing in self.rules:
            if self._is_similar(text_lower, existing.text.lower()):
                # Boost confidence of existing rule instead
                existing.confidence = min(existing.confidence + 0.1, 2.0)
                existing.times_verified += 1
                return
        
        rule = SemanticRule(text, confidence, source_step)
        
        if len(self.rules) < self.max_rules:
            self.rules.append(rule)
        else:
            # Evict lowest confidence rule
            min_idx = min(range(len(self.rules)),
                         key=lambda i: self.rules[i].confidence)
            if self.rules[min_idx].confidence < confidence:
                self.rules[min_idx] = rule
    
    def remove_rule(self, index: int):
        """Remove a rule by index."""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)
    
    def decay_unsupported(self, supported_indices: set):
        """Decay confidence of rules not supported by evidence."""
        for i, rule in enumerate(self.rules):
            if i not in supported_indices:
                rule.confidence *= 0.9  # decay
        # Prune rules below threshold
        self.rules = [r for r in self.rules if r.confidence > 0.3]
    
    def get_all_texts(self) -> List[str]:
        """Get all rule texts."""
        return [r.text for r in self.rules]
    
    def get_all_tokenized(self) -> List[List[str]]:
        """Get all tokenized rules for BM25."""
        return [r.to_tokenized() for r in self.rules]
    
    def to_text(self) -> str:
        """Format all rules as numbered list for LLM prompt."""
        if not self.rules:
            return "(no rules learned yet)"
        return "\n".join(
            f"{i+1}. {r.text}" for i, r in enumerate(self.rules)
        )
    
    def clear(self):
        """Clear all rules (new task)."""
        self.rules.clear()
    
    @staticmethod
    def _is_similar(a: str, b: str) -> bool:
        """Simple word-overlap similarity check."""
        # Remove common stopwords for better comparison
        stopwords = {"a", "an", "the", "to", "is", "in", "of", "for", "with",
                     "and", "or", "be", "you", "must", "can", "that", "this"}
        words_a = set(a.split()) - stopwords
        words_b = set(b.split()) - stopwords
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        smaller = min(len(words_a), len(words_b))
        # If >50% of the smaller set is shared, treat as duplicate
        return (overlap / smaller) > 0.5 if smaller > 0 else False
    
    def __len__(self):
        return len(self.rules)
    
    def __repr__(self):
        return f"SemanticMemory({len(self.rules)} rules)"
