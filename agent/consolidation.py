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
            f"Step {e['step']}: {'OK' if e['success'] else 'FAIL'} {e['action']} -> {e['result']}"
            for e in recent
        )

        existing_rules = semantic_memory.get_rules_text()

        # Count failures by type to highlight patterns
        failures = [e for e in recent if not e['success']]
        failure_summary = ""
        if failures:
            failure_types = {}
            for f in failures:
                result = f.get('result', '')
                if 'failed to place' in result.lower():
                    failure_types['placement_failure'] = failure_types.get('placement_failure', 0) + 1
                elif 'no ' in result.lower() and 'found' in result.lower():
                    failure_types['resource_not_found'] = failure_types.get('resource_not_found', 0) + 1
                elif 'need' in result.lower() and ('planks' in result.lower() or 'table' in result.lower()):
                    failure_types['missing_crafting_materials'] = failure_types.get('missing_crafting_materials', 0) + 1
                elif 'timeout' in result.lower():
                    failure_types['action_timeout'] = failure_types.get('action_timeout', 0) + 1
                elif 'unknown item' in result.lower():
                    failure_types['wrong_item_name'] = failure_types.get('wrong_item_name', 0) + 1
                else:
                    failure_types['other'] = failure_types.get('other', 0) + 1
            failure_summary = f"\nFailure breakdown: {failure_types}\n"

        system_prompt = (
            "You are a Minecraft gameplay analyst. Extract ACTIONABLE rules "
            "from the agent's recent history. Analyze THREE categories:\n\n"
            "1. CRAFTING SEQUENCES: What order of crafting works? What prerequisites?\n"
            "   Example: 'Craft sticks from planks BEFORE attempting to craft any pickaxe.'\n\n"
            "2. RESOURCE PLANNING: Did the agent run out of materials at a critical moment?\n"
            "   Did it go underground without enough supplies? Did it fail to carry essentials?\n"
            "   Example: 'Mine 6+ logs and craft extra planks BEFORE going underground for iron.'\n"
            "   Example: 'Craft a furnace on the surface BEFORE mining iron ore underground.'\n"
            "   Example: 'Always carry at least 4 planks as emergency crafting material.'\n\n"
            "3. NAVIGATION & RECOVERY: Did the agent get stuck? Fail to find resources?\n"
            "   Example: 'Mine stone blocks (not cobblestone) - stone drops cobblestone.'\n"
            "   Example: 'If crafting_table placement fails, move to flat open ground first.'\n"
            "   Example: 'Use scan_surroundings to find iron_ore before digging randomly.'\n\n"
            "Rules must be SPECIFIC and PRACTICAL. Include exact item names.\n"
            "Do NOT output vague advice like 'be efficient' or 'plan ahead'.\n"
            "Do NOT repeat existing rules.\n"
            "Output ONLY a JSON array of 1-3 rule strings. If no useful rules, output: []"
        )

        user_prompt = (
            f"Goal: {goal}\n\n"
            f"Recent history ({len(recent)} steps):\n{history_text}\n"
            f"{failure_summary}\n"
            f"Existing rules (do NOT repeat these):\n{existing_rules}\n\n"
            "What SPECIFIC lessons should the agent learn from these steps? "
            "Focus especially on any REPEATED FAILURES - what should the agent do DIFFERENTLY?"
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

            # Evidence check: meaningful keywords from the rule should appear in evidence
            rule_words = set(rule.lower().split())
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for', 'with',
                         'it', 'that', 'this', 'from', 'by', 'as', 'if', 'not', 'but',
                         'you', 'your', 'do', 'does', 'did', 'have', 'has', 'had',
                         'will', 'would', 'should', 'can', 'could', 'may', 'might',
                         'before', 'after', 'than', 'then', 'when', 'while', 'so',
                         'any', 'all', 'each', 'every', 'no', 'more', 'most', 'other',
                         'into', 'through', 'during', 'about', 'between', 'same'}
            meaningful_words = rule_words - stop_words
            evidence_words = set(evidence_text.split())
            overlap = meaningful_words & evidence_words

            # Need at least 2 meaningful word overlap for evidence support
            if len(overlap) >= 2:
                evidence_steps = [e["step"] for e in evidence_entries]
                # Higher confidence for rules derived from repeated failures
                confidence = 0.9 if len(failures) >= 3 else 0.8
                semantic_memory.add_rule(rule, evidence_steps, confidence=confidence)
                new_rules.append(rule)

        return new_rules