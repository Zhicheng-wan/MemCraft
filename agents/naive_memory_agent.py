"""
Naive Memory Baseline Agent.

Input: FIFO History (Last L steps). No Delta filtering. No Pruning.
Stores raw (action, observation) tuples in a sliding window.
"""
from collections import deque

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent
from env.minecraft_env import TextObservation
from config.llm_client import LLMClient
from config.prompts import SYSTEM_PROMPT_BASE, NAIVE_MEMORY_USER_TEMPLATE
from config.settings import NAIVE_MEMORY_WINDOW


class NaiveMemoryAgent(BaseAgent):
    """Baseline: FIFO sliding window of raw history."""
    
    name = "naive_memory"
    
    def __init__(self, goal: str, task_name: str = "unknown",
                 window_size: int = NAIVE_MEMORY_WINDOW):
        super().__init__(goal)
        self.llm = LLMClient(agent_name=self.name, task_name=task_name)
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
    
    def act(self, obs: TextObservation) -> str:
        self.llm.set_step(self.step_count)
        
        # Format history
        if self.history:
            history_text = "\n".join(
                f"[Step {h['step']}] Action: {h['action']}\n{h['obs']}"
                for h in self.history
            )
        else:
            history_text = "(no history yet)"
        
        user_prompt = NAIVE_MEMORY_USER_TEMPLATE.format(
            goal=self.goal,
            window_size=self.window_size,
            history=history_text,
            observation=obs.to_text(),
        )
        
        result = self.llm.query(SYSTEM_PROMPT_BASE, user_prompt)
        return result.get("action", "noop")
    
    def on_step_result(self, action: str, obs: TextObservation,
                       reward: float, done: bool, info: dict):
        super().on_step_result(action, obs, reward, done, info)
        # Store in FIFO (no filtering, no pruning - as per baseline spec)
        self.history.append({
            "step": self.step_count,
            "action": action,
            "obs": obs.to_text(),
        })
    
    def reset(self):
        super().reset()
        self.history.clear()
