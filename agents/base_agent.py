"""
Base agent class. All three agents inherit from this.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.minecraft_env import TextObservation, parse_observation, map_action_to_minedojo


class BaseAgent(ABC):
    """Abstract base for all agents."""
    
    name: str = "base"
    
    def __init__(self, goal: str):
        self.goal = goal
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_log: list = []
    
    @abstractmethod
    def act(self, obs: TextObservation) -> str:
        """
        Given current text observation, return action string.
        This is the core decision method.
        """
        pass
    
    def on_step_result(self, action: str, obs: TextObservation,
                       reward: float, done: bool, info: dict):
        """Called after environment step with results."""
        self.step_count += 1
        self.total_reward += reward
        self.episode_log.append({
            "step": self.step_count,
            "action": action,
            "reward": reward,
            "done": done,
        })
    
    def on_episode_end(self):
        """Called at the end of each episode."""
        pass
    
    def reset(self):
        """Reset for new episode."""
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_log = []
    
    def get_stats(self) -> dict:
        """Get episode statistics."""
        return {
            "agent": self.name,
            "goal": self.goal,
            "steps": self.step_count,
            "total_reward": self.total_reward,
        }
