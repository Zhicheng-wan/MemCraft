"""
Semantic Consolidation Module.

Triggered every N steps or at episode end.
1. Extracts rule texts (1-2 sentences) from recent trajectory via LLM
2. Evidence check: verifies each rule against last K steps
3. Discards unsupported rules

This is the "learning" mechanism - the agent builds up knowledge
about game mechanics, preconditions, and failure modes.
"""
import json
from typing import List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import CONSOLIDATION_INTERVAL, EVIDENCE_WINDOW
from config.prompts import (
    CONSOLIDATION_SYSTEM, CONSOLIDATION_USER_TEMPLATE,
    EVIDENCE_CHECK_SYSTEM, EVIDENCE_CHECK_USER_TEMPLATE,
)
from config.llm_client import LLMClient
from memory.step_memory import StepMemory
from memory.semantic_memory import SemanticMemory


class SemanticConsolidator:
    """
    Handles the periodic consolidation of trajectory data into semantic rules.
    
    From the slides:
    - Trigger: Every N steps or Episode End
    - Action: Extract Rule Texts (1-2 sentences)
    - Evidence Check: Verify rule against last K steps. Discard if unsupported.
    """
    
    def __init__(self, llm: LLMClient,
                 interval: int = CONSOLIDATION_INTERVAL,
                 evidence_window: int = EVIDENCE_WINDOW):
        self.llm = llm
        self.interval = interval
        self.evidence_window = evidence_window
        self._steps_since_last = 0
    
    def should_consolidate(self) -> bool:
        """Check if it's time to consolidate."""
        return self._steps_since_last >= self.interval
    
    def step(self):
        """Call this every agent step to track consolidation timing."""
        self._steps_since_last += 1
    
    def consolidate(
        self,
        goal: str,
        step_memory: StepMemory,
        semantic_memory: SemanticMemory,
    ) -> List[str]:
        """
        Run the full consolidation pipeline.
        
        1. Get recent trajectory
        2. Ask LLM to extract rules
        3. Verify each rule against evidence
        4. Add verified rules to semantic memory
        
        Returns list of newly added rules.
        """
        self._steps_since_last = 0
        
        # Get recent trajectory for rule extraction
        recent = step_memory.get_all_recent(n=self.interval)
        if len(recent) < 3:
            return []  # not enough data
        
        trajectory_text = "\n".join(e.to_text() for e in recent)
        existing_rules = semantic_memory.to_text()
        
        # ── Step 1: Extract candidate rules ──
        user_prompt = CONSOLIDATION_USER_TEMPLATE.format(
            goal=goal,
            window=len(recent),
            trajectory=trajectory_text,
            existing_rules=existing_rules,
        )
        
        result = self.llm.query(
            CONSOLIDATION_SYSTEM, user_prompt,
            temperature=0.2, max_tokens=300
        )
        
        candidate_rules = result.get("rules", [])
        if not candidate_rules:
            return []
        
        # ── Step 2: Evidence check each rule ──
        evidence_entries = step_memory.get_all_recent(n=self.evidence_window)
        evidence_text = "\n".join(e.to_text() for e in evidence_entries)
        
        verified_rules = []
        for rule_text in candidate_rules:
            if not isinstance(rule_text, str) or len(rule_text) < 10:
                continue
            
            # Verify against evidence
            if self._verify_rule(rule_text, evidence_text):
                current_step = step_memory._step_count
                semantic_memory.add_rule(
                    rule_text, confidence=1.0, source_step=current_step
                )
                verified_rules.append(rule_text)
        
        return verified_rules
    
    def _verify_rule(self, rule: str, evidence_text: str) -> bool:
        """
        Evidence check: Ask LLM if the rule is supported by recent observations.
        
        To save API calls, we do a simple heuristic check first,
        only calling the LLM for ambiguous cases.
        """
        # Quick heuristic: check if key terms from rule appear in evidence
        rule_words = set(rule.lower().split())
        evidence_words = set(evidence_text.lower().split())
        overlap = len(rule_words & evidence_words)
        
        # If very high overlap, accept without LLM call (saves budget)
        if overlap > len(rule_words) * 0.4:
            return True
        
        # If zero overlap, reject without LLM call
        if overlap == 0:
            return False
        
        # Ambiguous case: ask LLM
        user_prompt = EVIDENCE_CHECK_USER_TEMPLATE.format(
            rule=rule,
            window=self.evidence_window,
            evidence=evidence_text,
        )
        
        result = self.llm.query(
            EVIDENCE_CHECK_SYSTEM, user_prompt,
            temperature=0.1, max_tokens=100
        )
        
        return result.get("supported", False)
    
    def consolidate_at_episode_end(
        self,
        goal: str,
        step_memory: StepMemory,
        semantic_memory: SemanticMemory,
    ) -> List[str]:
        """Force consolidation at episode end regardless of step count."""
        self._steps_since_last = self.interval  # force trigger
        return self.consolidate(goal, step_memory, semantic_memory)
