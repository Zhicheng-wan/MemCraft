"""
No-Memory Baseline Agent.

Input: Current Observation Only. No History. No Retrieval.
The simplest baseline - purely reactive.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent
from env.minecraft_env import TextObservation
from config.llm_client import LLMClient
from config.prompts import SYSTEM_PROMPT_BASE, NO_MEMORY_USER_TEMPLATE


class NoMemoryAgent(BaseAgent):
    """Baseline: acts on current observation only."""
    
    name = "no_memory"
    
    def __init__(self, goal: str, task_name: str = "unknown"):
        super().__init__(goal)
        self.llm = LLMClient(agent_name=self.name, task_name=task_name)
    
    def act(self, obs: TextObservation) -> str:
        self.llm.set_step(self.step_count)
        
        user_prompt = NO_MEMORY_USER_TEMPLATE.format(
            goal=self.goal,
            observation=obs.to_text(),
        )
        
        result = self.llm.query(SYSTEM_PROMPT_BASE, user_prompt)
        return result.get("action", "noop")
