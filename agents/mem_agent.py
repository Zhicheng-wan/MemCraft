"""
MemAgent (Ours): Structured Delta + BM25 Retrieval + Semantic Rules.

The full agent with:
- The Brain: LLM (Llama-4-Scout via TritonAI)
- Step Memory (Mstep): Delta-filtered trajectory
- Semantic Memory (Msem): Consolidated rules
- BM25 Retrieval: Query = Goal + Inventory + Entity terms → Top-K from Mstep & Msem
- Semantic Consolidation: Every N steps, extract & verify rules
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent
from env.minecraft_env import TextObservation, compute_delta
from config.llm_client import LLMClient
from config.prompts import MEMAGENT_SYSTEM_PROMPT, MEMAGENT_USER_TEMPLATE
from memory.step_memory import StepMemory
from memory.semantic_memory import SemanticMemory
from memory.bm25_retriever import BM25Retriever
from memory.consolidation import SemanticConsolidator
from config.settings import (
    MSTEP_MAX_ENTRIES, MSEM_MAX_RULES, BM25_TOP_K,
    CONSOLIDATION_INTERVAL, EVIDENCE_WINDOW,
)


class MemAgent(BaseAgent):
    """
    Full MemAgent with hierarchical local memory.
    
    Architecture (from slides):
    ┌─────────────────┐
    │   The Brain      │  ← LLM reasoning via JSON schema
    ├─────────────────┤
    │  Step Memory     │  ← High-fidelity trajectory (delta-filtered)
    │  (Mstep)         │
    ├─────────────────┤
    │  Semantic Memory │  ← Consolidated rules, constraints, failure modes
    │  (Msem)          │
    └─────────────────┘
    
    The Loop (Retrieval & Consolidation):
    - BM25 Retrieval: query=Goal+Inventory+Entity → Top-K from Mstep & Msem
    - Semantic Consolidation: every N steps, extract rules, verify, store
    """
    
    name = "memagent"
    
    def __init__(self, goal: str, task_name: str = "unknown",
                 persist_semantic: bool = True):
        super().__init__(goal)
        self.llm = LLMClient(agent_name=self.name, task_name=task_name)
        
        # Hierarchical Memory
        self.step_memory = StepMemory(max_entries=MSTEP_MAX_ENTRIES)
        self.semantic_memory = SemanticMemory(max_rules=MSEM_MAX_RULES)
        
        # Retrieval
        self.retriever = BM25Retriever(top_k=BM25_TOP_K)
        
        # Consolidation
        self.consolidator = SemanticConsolidator(
            llm=self.llm,
            interval=CONSOLIDATION_INTERVAL,
            evidence_window=EVIDENCE_WINDOW,
        )
        
        # Track previous observation for delta computation
        self._prev_obs: TextObservation = None
        self._persist_semantic = persist_semantic  # keep Msem across episodes
    
    def act(self, obs: TextObservation) -> str:
        """
        Main decision loop:
        1. Build BM25 query from goal + current obs
        2. Retrieve relevant memories from Mstep & Msem
        3. Inject retrieved memories into prompt
        4. LLM reasons and selects action
        """
        self.llm.set_step(self.step_count)
        
        # ── BM25 Retrieval ──
        inventory_terms = obs.get_inventory_terms()
        entity_terms = obs.get_entity_terms()
        
        step_results, semantic_results = self.retriever.retrieve(
            goal=self.goal,
            inventory_terms=inventory_terms,
            entity_terms=entity_terms,
            step_memory=self.step_memory,
            semantic_memory=self.semantic_memory,
        )
        
        # Format retrieved memories
        step_mem_text = "\n".join(step_results) if step_results else "(none yet)"
        sem_rules_text = "\n".join(
            f"• {r}" for r in semantic_results
        ) if semantic_results else self.semantic_memory.to_text()
        
        # ── LLM Reasoning ──
        user_prompt = MEMAGENT_USER_TEMPLATE.format(
            goal=self.goal,
            step_memories=step_mem_text,
            semantic_rules=sem_rules_text,
            observation=obs.to_text(),
        )
        
        result = self.llm.query(
            MEMAGENT_SYSTEM_PROMPT, user_prompt,
            temperature=0.3, max_tokens=512
        )
        
        action = result.get("action", "noop")
        note = result.get("observation_note", "")
        
        # ── Store in Step Memory (with delta filtering) ──
        if self._prev_obs is not None:
            delta = compute_delta(self._prev_obs, obs)
        else:
            delta = {"first_step": True}
        
        self.step_memory.add(
            action=action,
            observation_text=obs.to_text(),
            delta=delta,
            note=note,
            force=(self.step_count == 0),  # always store first step
        )
        
        # ── Semantic Consolidation ──
        self.consolidator.step()
        if self.consolidator.should_consolidate():
            new_rules = self.consolidator.consolidate(
                self.goal, self.step_memory, self.semantic_memory
            )
            if new_rules:
                print(f"  [MemAgent] Consolidated {len(new_rules)} new rules")
                for r in new_rules:
                    print(f"    → {r}")
        
        self._prev_obs = obs
        return action
    
    def on_step_result(self, action: str, obs: TextObservation,
                       reward: float, done: bool, info: dict):
        super().on_step_result(action, obs, reward, done, info)
        
        # If reward received, add a high-signal entry
        if reward > 0:
            self.step_memory.add(
                action=f"REWARD: {action}",
                observation_text=obs.to_text(),
                delta={"reward": reward},
                note=f"Received reward {reward} - goal progress!",
                force=True,
            )
    
    def on_episode_end(self):
        """Consolidate at episode end."""
        new_rules = self.consolidator.consolidate_at_episode_end(
            self.goal, self.step_memory, self.semantic_memory
        )
        if new_rules:
            print(f"  [MemAgent] End-of-episode consolidation: {len(new_rules)} rules")
    
    def reset(self):
        super().reset()
        self.step_memory.clear()
        self._prev_obs = None
        # Optionally preserve semantic memory across episodes
        if not self._persist_semantic:
            self.semantic_memory.clear()
    
    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats.update({
            "step_memory_size": len(self.step_memory),
            "step_memory_total_logged": len(self.step_memory.all_entries),
            "semantic_rules_count": len(self.semantic_memory),
            "semantic_rules": self.semantic_memory.get_all_texts(),
        })
        return stats
