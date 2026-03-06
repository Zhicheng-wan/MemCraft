"""
consolidation.py - Agentic Per-Step Memory Update

Flow:
  - Before each action: LLM reads all semantic memories (handled in agent.py)
  - After each action:  LLM sees the action result + current memory and
                        decides which memory operations to apply:

    [
      {"op": "insert", "text": "new rule", "confidence": 0.9},
      {"op": "update", "id": "abc123", "text": "revised rule", "confidence": 0.95},
      {"op": "delete", "id": "def456"}
    ]
"""

from .memory import StepMemory, SemanticMemory

_SYSTEM_PROMPT = (
    "You are a Minecraft gameplay analyst managing a structured memory of learned rules.\n\n"
    "You will receive:\n"
    "  1. The goal the agent is trying to achieve\n"
    "  2. The action just executed and its result (success or failure)\n"
    "  3. Recent action history for context\n"
    "  4. Current memory rules, each with a unique ID in brackets\n\n"
    "Decide how to update the memory. Output ONLY a JSON array of operations:\n"
    '  {"op": "insert", "text": "<new rule>", "confidence": <0.0-1.0>}\n'
    '  {"op": "update", "id": "<rule_id>", "text": "<revised rule>", "confidence": <0.0-1.0>}\n'
    '  {"op": "delete", "id": "<rule_id>"}\n\n'
    "Guidelines:\n"
    "- INSERT when this result reveals new actionable knowledge not already in memory.\n"
    "- UPDATE when this result shows an existing rule is incomplete or needs correction.\n"
    "- DELETE when this result proves an existing rule is wrong or misleading.\n"
    "- Most steps will need no change — output [] if nothing meaningful was learned.\n"
    "- Rules must be SPECIFIC (exact item names, counts, action sequences).\n"
    "- Never insert a rule already covered by an existing one.\n"
    "- Output ONLY valid JSON. No explanations."
)


def _apply_and_summarize(semantic_memory: SemanticMemory, parsed: list) -> dict:
    """Apply parsed operations and return a summary with new rule texts."""
    ids_before = {r["id"] for r in semantic_memory.get_all()}
    summary = semantic_memory.apply_operations(parsed)
    ids_after = {r["id"] for r in semantic_memory.get_all()}

    new_ids = ids_after - ids_before
    summary["new_rules"] = [
        r["text"] for r in semantic_memory.get_all() if r["id"] in new_ids
    ]
    return summary


def _empty_summary() -> dict:
    return {"inserted": 0, "updated": 0, "deleted": 0, "skipped": 0, "new_rules": []}


class Consolidator:
    """
    Per-step agentic memory updater.

    After every action the agent calls `update_after_action()`.
    The LLM decides which INSERT / UPDATE / DELETE operations to apply
    based on the just-executed action and its result.
    """

    def __init__(self, brain, evidence_window: int = 5):
        """
        Args:
            brain:           Brain instance for LLM calls
            evidence_window: Number of recent steps shown as context
        """
        self.brain = brain
        self.evidence_window = evidence_window

    def update_after_action(
        self,
        action_name: str,
        action_params: dict,
        action_result: str,
        action_success: bool,
        step_memory: StepMemory,
        semantic_memory: SemanticMemory,
        goal: str,
    ) -> dict:
        """
        Called after every action. The LLM reviews:
          - The action just executed + result
          - Recent history for context
          - Current semantic memory

        Returns a summary dict: {inserted, updated, deleted, skipped, new_rules}
        """
        import json

        recent = step_memory.get_recent(self.evidence_window)
        history_text = "\n".join(
            f"  {'OK' if e['success'] else 'FAIL'} {e['action']} -> {e['result']}"
            for e in recent
        ) or "  (no history yet)"

        existing_rules = semantic_memory.get_rules_for_prompt()

        user_prompt = (
            f"Goal: {goal}\n\n"
            f"Action just executed: {action_name}({json.dumps(action_params)})\n"
            f"Result: {'SUCCESS' if action_success else 'FAILURE'} — {action_result}\n\n"
            f"Recent history (last {len(recent)} steps):\n{history_text}\n\n"
            f"Current memory rules (use IDs for UPDATE/DELETE):\n{existing_rules}\n\n"
            "What memory operations should be applied? Output a JSON array of operations:"
        )

        result = self.brain.query(_SYSTEM_PROMPT, user_prompt)
        if result["error"]:
            return _empty_summary()

        parsed = self.brain.parse_json_response(result["content"])
        if not isinstance(parsed, list):
            return _empty_summary()

        return _apply_and_summarize(semantic_memory, parsed)

    # ------------------------------------------------------------------
    # Legacy batch consolidation (kept for episode-end summary if needed)
    # ------------------------------------------------------------------

    def consolidate(self, step_memory: StepMemory,
                    semantic_memory: SemanticMemory,
                    goal: str) -> dict:
        """
        Batch consolidation over recent N steps.
        Can be called at episode end for a final memory pass.
        """
        recent = step_memory.get_recent(self.evidence_window * 2)
        if len(recent) < 3:
            return _empty_summary()

        history_text = "\n".join(
            f"  Step {e['step']}: {'OK' if e['success'] else 'FAIL'} "
            f"{e['action']} -> {e['result']}"
            for e in recent
        )
        existing_rules = semantic_memory.get_rules_for_prompt()

        user_prompt = (
            f"Goal: {goal}\n\n"
            f"Recent history ({len(recent)} steps):\n{history_text}\n\n"
            f"Current memory rules (use IDs for UPDATE/DELETE):\n{existing_rules}\n\n"
            "Review the full history and decide what memory operations to apply. "
            "Output a JSON array of operations:"
        )

        result = self.brain.query(_SYSTEM_PROMPT, user_prompt)
        if result["error"]:
            return _empty_summary()

        parsed = self.brain.parse_json_response(result["content"])
        if not isinstance(parsed, list):
            return _empty_summary()

        return _apply_and_summarize(semantic_memory, parsed)
