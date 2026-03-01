"""
Global settings for MemAgent project.
Budget-aware configuration for TritonAI API.
"""
import os
import csv
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── API Configuration ───────────────────────────────────────────────
API_KEY = os.getenv("TRITONAI_API_KEY", "")
API_BASE_URL = "https://tritonai-api.ucsd.edu/v1/chat/completions"
MODEL_NAME = "api-llama-4-scout"  # primary model

# ─── Budget Tracking ─────────────────────────────────────────────────
BUDGET_CAP_USD = 200.0
# Rough estimates for Llama-4-Scout via TritonAI (adjust if you get actual pricing)
# If the school API is free/flat-rate, set these to 0
COST_PER_INPUT_TOKEN = 0.0  # TritonAI may be free for students
COST_PER_OUTPUT_TOKEN = 0.0
TOKEN_LOG_PATH = Path(__file__).parent.parent / "logs" / "token_usage.csv"

# ─── Agent Hyperparameters ────────────────────────────────────────────

# Step Memory (Mstep)
MSTEP_MAX_ENTRIES = 50          # max steps to keep in step memory
MSTEP_DELTA_THRESHOLD = 0.3     # min change ratio to store a step (delta filtering)

# Semantic Memory (Msem)
MSEM_MAX_RULES = 30             # max consolidated rules
CONSOLIDATION_INTERVAL = 10     # consolidate every N steps
EVIDENCE_WINDOW = 5             # verify rules against last K steps

# BM25 Retrieval
BM25_TOP_K = 5                  # retrieve top-K entries from memory

# Naive Memory Baseline
NAIVE_MEMORY_WINDOW = 20        # FIFO last L steps

# ─── Environment Settings ────────────────────────────────────────────
MAX_STEPS_PER_EPISODE = 300     # prevent runaway episodes
IMAGE_SIZE = (160, 256)         # MineDojo image size (not used for vision)
NEARBY_ENTITIES_TOP_N = 5       # top-N nearest entities to report

# ─── Experiment Settings ─────────────────────────────────────────────
DEFAULT_TASKS = [
    "harvest_wool_with_shears_and_sheep",
    "harvest_milk",
    "combat_spider_plains_leather_armors_diamond_sword_shield",
    "harvest_log_in_plains",
]
NUM_EPISODES_PER_TASK = 5


class TokenTracker:
    """Thread-safe token usage tracker with CSV logging."""
    
    _lock = threading.Lock()
    _total_input = 0
    _total_output = 0
    _total_cost = 0.0
    _initialized = False
    
    @classmethod
    def _ensure_csv(cls):
        if not cls._initialized:
            TOKEN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not TOKEN_LOG_PATH.exists():
                with open(TOKEN_LOG_PATH, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "agent", "task", "step",
                        "input_tokens", "output_tokens", "cost_usd"
                    ])
            cls._initialized = True
    
    @classmethod
    def log(cls, agent: str, task: str, step: int,
            input_tokens: int, output_tokens: int):
        cost = (input_tokens * COST_PER_INPUT_TOKEN +
                output_tokens * COST_PER_OUTPUT_TOKEN)
        with cls._lock:
            cls._total_input += input_tokens
            cls._total_output += output_tokens
            cls._total_cost += cost
            cls._ensure_csv()
            with open(TOKEN_LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    agent, task, step,
                    input_tokens, output_tokens, f"{cost:.6f}"
                ])
        return cost
    
    @classmethod
    def get_total_cost(cls) -> float:
        with cls._lock:
            return cls._total_cost
    
    @classmethod
    def get_stats(cls) -> dict:
        with cls._lock:
            return {
                "total_input_tokens": cls._total_input,
                "total_output_tokens": cls._total_output,
                "total_cost_usd": cls._total_cost,
                "budget_remaining": BUDGET_CAP_USD - cls._total_cost,
            }
    
    @classmethod
    def check_budget(cls) -> bool:
        """Returns True if we're still within budget."""
        return cls.get_total_cost() < BUDGET_CAP_USD
