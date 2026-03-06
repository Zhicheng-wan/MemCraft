#!/bin/bash
# MemCraft 快速启动脚本

# ── API 配置 ──────────────────────────────────────────────

# ── 任务配置 ──────────────────────────────────────────────
TASK="mine 5 dirt blocks"
AGENT="memagent"        # memagent | no_memory | naive_memory
MAX_STEPS=50
HOST="localhost"
PORT=25565
HTTP_PORT=3001

# ─────────────────────────────────────────────────────────
python run_agent.py \
    --task "$TASK" \
    --agent "$AGENT" \
    --host "$HOST" \
    --port "$PORT" \
    --http-port "$HTTP_PORT" \
    --max-steps "$MAX_STEPS"
