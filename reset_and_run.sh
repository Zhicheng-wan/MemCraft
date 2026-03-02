#!/bin/bash
# reset_and_run.sh - Run agent comparison with per-episode world reset
#
# Usage:
#   ./reset_and_run.sh "craft a wooden pickaxe" 3 30
#   ./reset_and_run.sh "craft a stone pickaxe" 3 40
#
# Args:
#   $1 = task (default: "craft a wooden pickaxe")
#   $2 = episodes per agent (default: 3)
#   $3 = max steps per episode (default: 30)

TASK="${1:-craft a wooden pickaxe}"
EPISODES="${2:-3}"
MAX_STEPS="${3:-30}"

echo "=========================================="
echo "MemCraft Comparison Runner"
echo "=========================================="
echo "Task: $TASK"
echo "Episodes: $EPISODES per agent"
echo "Max steps: $MAX_STEPS"
echo "World reset: per episode (clean state)"
echo ""

# Kill any existing bridge
pkill -f "node bot.js" 2>/dev/null
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
    --max-steps $MAX_STEPS

echo ""
echo "Done! Check logs/ for results."