#!/bin/bash
# reset_and_run.sh - Run agent comparison with per-episode world reset
#
# Usage:
#   ./reset_and_run.sh                          # 运行全部20个任务
#   ./reset_and_run.sh --from 1 --to 5          # 运行任务 1~5
#   ./reset_and_run.sh --from 16                # 从任务16运行到末尾
#   ./reset_and_run.sh --task-id 3              # 只运行单个任务
#   ./reset_and_run.sh --list                   # 列出所有任务
#   ./reset_and_run.sh --from 1 --to 5 --episodes 5 --max-steps 30 --model api-llama-4-scout
#
# Args:
#   --from        起始任务编号 (default: 1)
#   --to          终止任务编号 (default: 20)
#   --task-id     只运行单个任务编号 (覆盖 --from/--to)
#   --episodes    每个 agent 的轮数 (default: 3)
#   --max-steps   每轮最大步数 (default: 30)
#   --model       模型名称 (default: api-llama-4-scout)
#   --extra-flags 附加参数 (e.g., "--no-recipes --skip-agents no_memory")
#   --list        列出所有任务并退出

export TRITONAI_API_KEY="${TRITONAI_API_KEY:-sk-s3xOCi2YPqR68mmQK67_HQ}"

# ── 读取任务列表 ──────────────────────────────────────────────────────────
TASKS_FILE="$(dirname "$0")/tasks.json"
if [ ! -f "$TASKS_FILE" ]; then
    echo "错误：找不到任务文件 $TASKS_FILE"
    exit 1
fi

# 用 Python 解析 JSON，输出 "task|max_steps" 每行一条
TASKS=()
while IFS= read -r line; do
    TASKS+=("$line")
done < <(python3 -c "
import json, sys
data = json.load(open('$TASKS_FILE'))
for t in data['tasks']:
    print(f\"{t['task']}|{t['max_steps']}\")
")

TASK_COUNT=${#TASKS[@]}

# ── 解析参数 ──────────────────────────────────────────────────────────────
FROM=1
TO=$TASK_COUNT
SINGLE=""
EPISODES=3
MAX_STEPS_OVERRIDE=""
MODEL="api-llama-4-scout"
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)
            echo "=========================================="
            echo "MemCraft 任务列表 (共 ${TASK_COUNT} 个)"
            echo "=========================================="
            for i in "${!TASKS[@]}"; do
                IFS='|' read -r task steps <<< "${TASKS[$i]}"
                num=$((i + 1))
                if [ $num -le 15 ]; then
                    tag="[简单]"
                else
                    tag="[复杂]"
                fi
                printf "%3d. %s %-40s (max %s steps)\n" "$num" "$tag" "$task" "$steps"
            done
            exit 0
            ;;
        --from)      FROM="$2";           shift 2 ;;
        --to)        TO="$2";             shift 2 ;;
        --task-id)   SINGLE="$2";         shift 2 ;;
        --episodes)  EPISODES="$2";       shift 2 ;;
        --max-steps) MAX_STEPS_OVERRIDE="$2"; shift 2 ;;
        --model)     MODEL="$2";          shift 2 ;;
        --extra-flags) EXTRA_FLAGS="$2";  shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# --task-id 覆盖 --from/--to
if [ -n "$SINGLE" ]; then
    FROM=$SINGLE
    TO=$SINGLE
fi

# 边界检查
if [ "$FROM" -lt 1 ] || [ "$TO" -gt "$TASK_COUNT" ] || [ "$FROM" -gt "$TO" ]; then
    echo "错误：任务编号范围无效 (有效范围: 1~${TASK_COUNT})"
    exit 1
fi

# ── 开始执行 ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "MemCraft Benchmark Runner"
echo "=========================================="
echo "任务范围: ${FROM} ~ ${TO}  (共 $((TO - FROM + 1)) 个任务)"
echo "Episodes: $EPISODES per agent"
echo "Model:    $MODEL"
echo "Extra:    ${EXTRA_FLAGS:-无}"
echo "=========================================="

# Kill any existing bridge
pkill -f "node bot.js" 2>/dev/null
fuser -k 3001/tcp 2>/dev/null
sleep 2

TOTAL_PASS=0
TOTAL_FAIL=0
FAILED_TASKS=()

for ((i = FROM; i <= TO; i++)); do
    IFS='|' read -r task default_steps <<< "${TASKS[$((i - 1))]}"
    MAX_STEPS="${MAX_STEPS_OVERRIDE:-$default_steps}"

    if [ $i -le 15 ]; then tag="简单"; else tag="复杂"; fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  任务 ${i}/${TASK_COUNT} [${tag}]"
    echo "  $task  (max ${MAX_STEPS} steps)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 清空语义记忆（每个任务重新开始）
    rm -f memories/semantic_rules.json
    echo '[]' > memories/semantic_rules.json

    python run_agent.py \
        --task "$task" \
        --agent compare \
        --episodes $EPISODES \
        --port 25565 \
        --version 1.19.2 \
        --max-steps $MAX_STEPS \
        --model "$MODEL" \
        $EXTRA_FLAGS

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        FAILED_TASKS+=("$i: $task")
    fi
done

# ── 汇总 ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Benchmark 完成"
echo "=========================================="
echo "完成: $((TOTAL_PASS + TOTAL_FAIL)) 个任务"
echo "成功: $TOTAL_PASS  失败: $TOTAL_FAIL"
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "失败任务:"
    for t in "${FAILED_TASKS[@]}"; do echo "  - $t"; done
fi
echo "详细结果: logs/ 目录"
