#!/bin/bash
# reset_and_run.sh - Run agent comparison with per-episode world reset
#
# Usage:
#   ./reset_and_run.sh "craft a stone pickaxe" 5 30
#   ./reset_and_run.sh "craft a stone pickaxe" 5 30 api-gpt-oss-120b
#   ./reset_and_run.sh "craft a stone pickaxe" 5 30 api-gpt-oss-120b "--no-recipes"
#   ./reset_and_run.sh "craft a stone pickaxe" 10 40 api-gpt-oss-120b "--no-recipes --skip-agents no_memory"
#
# Args:
#   $1 = task (default: "craft a wooden pickaxe")
#   $2 = episodes per agent (default: 3)
#   $3 = max steps per episode (default: 30)
#   $4 = model name (default: api-llama-4-scout)
#   $5 = extra flags (e.g., "--no-recipes --skip-agents no_memory")

TASK="${1:-craft a wooden pickaxe}"
EPISODES="${2:-3}"
MAX_STEPS="${3:-30}"
MODEL="${4:-api-llama-4-scout}"
EXTRA_FLAGS="${5:-}"

echo "=========================================="
echo "MemCraft Comparison Runner"
echo "=========================================="
echo "Task: $TASK"
echo "Episodes: $EPISODES per agent"
echo "Max steps: $MAX_STEPS"
echo "Model: $MODEL"
echo "Extra flags: $EXTRA_FLAGS"
echo "World reset: per episode (clean state)"
echo ""

# Kill any existing bridge
pkill -f "node bot.js" 2>/dev/null
fuser -k 3001/tcp 2>/dev/null
sleep 2

# Clear semantic memory (so MemAgent starts fresh)
rm -f memories/semantic_rules.json
echo '[]' > memories/semantic_rules.json
echo "✓ Cleared semantic memory"

echo ""
echo "Starting comparison (each episode resets the Minecraft world)..."
echo "=========================================="

python run_agent.py \
    --task "$TASK" \
    --agent compare \
    --episodes $EPISODES \
    --port 25565 \
    --version 1.19.2 \
    --max-steps $MAX_STEPS \
    --model "$MODEL" \
    $EXTRA_FLAGS

echo ""
echo "Done! Check logs/ for results."