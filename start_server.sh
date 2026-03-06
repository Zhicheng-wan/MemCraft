#!/bin/bash
# 启动 Minecraft 服务器（Docker）
# 与 reset_and_run.sh 配套使用

CONTAINER_NAME="mc-server"   # 必须与 run_agent.py 中 MINECRAFT_CONTAINER 一致
MC_VERSION="1.19.2"          # 与 reset_and_run.sh 保持一致
PORT=25565

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "错误：Docker 未运行，请先启动 Docker Desktop"
    exit 1
fi

# 如果容器已存在则直接启动，否则创建
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "容器已存在，启动中..."
    docker start $CONTAINER_NAME
else
    echo "创建并启动 Minecraft 服务器 v${MC_VERSION}..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:25565 \
        -e EULA=TRUE \
        -e VERSION=$MC_VERSION \
        -e DIFFICULTY=peaceful \
        -e ONLINE_MODE=false \
        -e GAMEMODE=survival \
        -e LEVEL_SEED=0 \
        -e MEMORY=2G \
        itzg/minecraft-server
fi

echo "等待服务器启动（约 90 秒）..."
for i in $(seq 1 30); do
    sleep 5
    if nc -z localhost $PORT 2>/dev/null; then
        echo "Minecraft 服务器已就绪！端口：$PORT"
        echo ""
        echo "现在可以运行："
        echo "  bash reset_and_run.sh \"mine 5 dirt blocks\" 3 30"
        exit 0
    fi
    echo "  等待中... ($((i*5))s)"
done

echo "警告：150 秒后服务器仍未响应，请检查："
echo "  docker logs $CONTAINER_NAME"
