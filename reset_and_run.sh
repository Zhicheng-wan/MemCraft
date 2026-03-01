#!/bin/bash
# reset_and_run.sh - Reset Minecraft world and run agent comparison
#
# Usage:
#   ./reset_and_run.sh "craft a wooden pickaxe" 3 30
#   ./reset_and_run.sh "mine 5 dirt blocks" 3 20
#
# Args:
#   $1 = task (default: "craft a wooden pickaxe")
#   $2 = episodes per agent (default: 3)
#   $3 = max steps per episode (default: 30)

TASK="${1:-craft a wooden pickaxe}"
EPISODES="${2:-3}"
MAX_STEPS="${3:-30}"
CONTAINER="mc-server"
SEED="0"  # Known good seed - spawns in forest biome for 1.19.2

echo "=========================================="
echo "MemCraft World Reset & Comparison Runner"
echo "=========================================="
echo "Task: $TASK"
echo "Episodes: $EPISODES per agent"
echo "Max steps: $MAX_STEPS"
echo "Seed: $SEED"
echo ""

# Kill any existing bridge
pkill -f "node bot.js" 2>/dev/null
sleep 2

# Clear semantic memory from previous runs (so MemAgent starts fresh)
rm -f memories/semantic_rules.json
echo '[]' > memories/semantic_rules.json
echo "✓ Cleared semantic memory"

# Set seed and reset Minecraft world
echo "Setting seed and resetting Minecraft world..."
sudo docker exec $CONTAINER sh -c "sed -i 's/level-seed=.*/level-seed=$SEED/' /data/server.properties"
sudo docker exec $CONTAINER rm -rf /data/world /data/world_nether /data/world_the_end
sudo docker restart $CONTAINER

echo "Waiting for server to generate new world (this takes ~45s)..."
sleep 45

# Verify server is up by trying TCP connection
for i in $(seq 1 12); do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost', 25565)); s.close(); print('OK')" 2>/dev/null; then
        echo "✓ Minecraft server is ready"
        break
    fi
    if [ $i -eq 12 ]; then
        echo "✗ Server not ready after 60s. Aborting."
        exit 1
    fi
    echo "  Waiting... ($i/12)"
    sleep 5
done

# Brief extra wait for world to fully load
sleep 5

# Run comparison
echo ""
echo "Starting comparison..."
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