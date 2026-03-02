"""
consolidation.py - Semantic Consolidation Loop

From the slides:
  Trigger: Every N steps or Episode End
  Action: Extract Rule Texts (1-2 sentences)
  Evidence Check: Verify rule against last K steps. Discard if unsupported.

This uses an LLM call to extract patterns, then verifies them.
"""

from typing import List, Optional
from .memory import StepMemory, SemanticMemory


class Consolidator:
    """
    Consolidates step memory into semantic rules using the LLM.
    Budget-aware: only runs every N steps.
    """

    def __init__(self, brain, interval: int = 10, evidence_window: int = 5):
        """
        Args:
            brain: Brain instance for LLM calls
            interval: Run consolidation every N steps
            evidence_window: Check rules against last K steps
        """
        self.brain = brain
        self.interval = interval
        self.evidence_window = evidence_window
        self.last_consolidation_step = 0

    def should_consolidate(self, current_step: int) -> bool:
        """Check if it's time to consolidate."""
        return (current_step - self.last_consolidation_step) >= self.interval

    def consolidate(self, step_memory: StepMemory,
                    semantic_memory: SemanticMemory,
                    goal: str) -> List[str]:
        """
        Extract rules from recent step memory and add verified ones
        to semantic memory.

        Returns list of new rules added.
        """
        self.last_consolidation_step = step_memory.step_counter

        recent = step_memory.get_recent(self.interval)
        if len(recent) < 3:
            return []

        # Build the consolidation prompt
        history_text = "\n".join(
            f"Step {e['step']}: {'✓' if e['success'] else '✗'} {e['action']} → {e['result']}"
            for e in recent
        )

        existing_rules = semantic_memory.get_rules_text()

        system_prompt = (
            "You are a Minecraft gameplay analyst. Extract useful rules/lessons "
            "from the agent's recent action history. Rules should be practical, "
            "specific, and help the agent avoid repeating mistakes or leverage "
            "successful strategies.\n\n"
            "Output ONLY a JSON array of rule strings. Each rule should be 1-2 sentences.\n"
            "Example: [\"Need a wooden pickaxe before mining stone blocks.\", "
            "\"Dirt can be mined by hand without any tools.\"]\n"
            "If no useful rules can be extracted, output: []"
        )

        user_prompt = (
            f"Goal: {goal}\n\n"
            f"Recent history ({len(recent)} steps):\n{history_text}\n\n"
            f"Existing rules (don't repeat these):\n{existing_rules}\n\n"
            "Extract 1-3 new rules from this history:"
        )

        result = self.brain.query(system_prompt, user_prompt)
        if result["error"]:
            return []

        parsed = self.brain.parse_json_response(result["content"])
        if not isinstance(parsed, list):
            return []

        # Evidence check: verify each rule against recent steps
        new_rules = []
        evidence_entries = step_memory.get_recent(self.evidence_window)
        evidence_text = " ".join(e["text"] for e in evidence_entries).lower()

        for rule in parsed:
            if not isinstance(rule, str) or len(rule) < 10:
                continue

            # Skip if similar rule already exists
            if semantic_memory.has_similar_rule(rule):
                continue

            # Simple evidence check: at least some keywords from the rule
            # should appear in recent evidence
            rule_words = set(rule.lower().split())
            evidence_words = set(evidence_text.split())
            overlap = rule_words & evidence_words
            # Need at least 3 word overlap for evidence support
            if len(overlap) >= 3:
                evidence_steps = [e["step"] for e in evidence_entries]
                semantic_memory.add_rule(rule, evidence_steps, confidence=0.8)
                new_rules.append(rule)

        return new_rules
