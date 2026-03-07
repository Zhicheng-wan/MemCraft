"""
consolidation.py - Event-driven memory consolidation for the three-layer system.

Two consolidation triggers:

1. reflect_on_failure()  — called after every failed action
   LLM writes ONE actionable lesson → EpisodicMemory.

2. extract_skills()      — called every N steps
   LLM extracts reusable procedures from recent successes → ProceduralMemory.
"""

import json
import logging
from .memory import ProceduralMemory, EpisodicMemory

logger = logging.getLogger("memcraft")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_REFLECT_SYSTEM_PROMPT = (
    "You are helping a Minecraft agent learn from failures.\n\n"
    "The agent just failed to execute an action. Write ONE concrete, "
    "actionable lesson so the agent avoids this mistake in the future.\n\n"
    "Output ONLY a JSON object:\n"
    '  {"lesson": "<one-sentence lesson>", "context": "<brief failure context>"}\n\n'
    "Guidelines:\n"
    "- The lesson MUST say BOTH what went wrong AND the exact next action to take.\n"
    "- CRITICAL: The lesson MUST keep the agent on track toward its current goal. "
    "NEVER suggest abandoning the current goal step or retreating to easier actions like mining wood.\n"
    "- BAD:  'stone mining failed' (no next action specified)\n"
    "- BAD:  'mine wood logs instead' (retreats from goal)\n"
    "- GOOD: 'stone items fell into holes — immediately use collect_nearby_items to recover drops, then retry find_and_mine_block(stone) at a different spot'\n"
    "- GOOD: 'crafting table placement failed here — use move_forward(steps=3) to find flat ground, then retry craft_item'\n"
    "- GOOD: 'move_to timed out — use move_forward(steps=3) instead for short-distance navigation'\n"
    "- If items were lost to terrain: next step is collect_nearby_items, NOT switching to a different resource.\n"
    "- If placement failed: next step is move_forward to find flat ground, NOT changing goals.\n"
    "- If a resource is missing: name the exact item and the action to obtain it.\n"
    "- If wrong tool tier: name the required tier and how to craft it.\n"
    "- Output ONLY valid JSON. No explanations."
)

_SKILL_SYSTEM_PROMPT = (
    "You are helping a Minecraft agent build a procedural skill library.\n\n"
    "You will receive the agent's recent successful actions toward a goal. "
    "Extract complete, reusable named skills from these sequences.\n\n"
    "Output ONLY a JSON array:\n"
    '  [{"skill_name": "<verb_noun>", "description": "<steps and prerequisites>"}]\n\n'
    "Guidelines:\n"
    "- Only extract COMPLETE procedures (e.g. craft_wooden_pickaxe, smelt_raw_iron).\n"
    "- Include ALL prerequisites in the description "
    "(e.g. 'requires 4 oak_log: mine logs → craft planks → craft sticks → craft pickaxe').\n"
    "- Output [] if no complete reusable skill is apparent.\n"
    "- Output ONLY valid JSON. No explanations."
)


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------

class Consolidator:
    """
    Event-driven memory consolidator for the four-layer memory architecture.

    reflect_on_failure() → EpisodicMemory   (1 LLM call per failure)
    extract_skills()     → ProceduralMemory  (1 LLM call every N steps)
    update_spatial()     → SpatialMemory     (heuristic, no LLM call)
    """

    def __init__(self, brain, skill_extraction_window: int = 5):
        self.brain = brain
        self.skill_extraction_window = skill_extraction_window

    def reflect_on_failure(
        self,
        action_name: str,
        action_params: dict,
        action_result: str,
        goal: str,
        episodic_memory: EpisodicMemory,
    ) -> bool:
        """
        Called after a failed action.
        LLM reflects on the failure and stores one lesson in EpisodicMemory.
        Returns True if a lesson was stored.
        """
        user_prompt = (
            f"Goal: {goal}\n"
            f"Failed action: {action_name}({json.dumps(action_params)})\n"
            f"Failure message: {action_result}\n\n"
            "What lesson should the agent remember to avoid this mistake?\n"
            "Output a JSON object with 'lesson' and 'context' fields:"
        )
        result = self.brain.query(_REFLECT_SYSTEM_PROMPT, user_prompt)
        if result["error"]:
            return False

        parsed = self.brain.parse_json_response(result["content"])
        if not isinstance(parsed, dict):
            return False

        lesson = parsed.get("lesson", "").strip()
        context = parsed.get("context", action_name).strip()
        if lesson:
            episodic_memory.add_lesson(context, lesson)
            logger.info(f"  Lesson: {lesson}")
            return True
        return False

    def extract_skills(
        self,
        recent_steps: list,
        goal: str,
        procedural_memory: ProceduralMemory,
    ) -> int:
        """
        Called every N steps.
        LLM extracts reusable skills from recent successful steps.
        Returns the number of skills added or updated.
        """
        successful = [s for s in recent_steps if s.get("success")]
        if len(successful) < 2:
            return 0

        steps_text = "\n".join(
            f"  + {s['action']}({json.dumps(s.get('params', {}))}) -> {s['result']}"
            for s in successful[-self.skill_extraction_window:]
        )
        user_prompt = (
            f"Goal: {goal}\n\n"
            f"Recent successful steps:\n{steps_text}\n\n"
            "Extract reusable skills as a JSON array:"
        )
        result = self.brain.query(_SKILL_SYSTEM_PROMPT, user_prompt)
        if result["error"]:
            return 0

        parsed = self.brain.parse_json_response(result["content"])
        if not isinstance(parsed, list):
            return 0

        count = 0
        for skill in parsed:
            if isinstance(skill, dict):
                name = skill.get("skill_name", "").strip()
                desc = skill.get("description", "").strip()
                if name and desc:
                    procedural_memory.add_skill(name, desc)
                    count += 1
        return count

