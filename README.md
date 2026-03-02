# MemCraft: Memory-Augmented Minecraft Agent

A hierarchical memory system for LLM-based Minecraft agents using Mineflayer (text-only, no vision).

## Architecture (from slides)

```
┌─────────────────────────────────────────────┐
│              The Brain (LLM)                │
│   UCSD TritonAI API (Llama-4-Scout)         │
│   Reasoning & Action via JSON Schema        │
├──────────────┬──────────────────────────────┤
│  Step Memory │    Semantic Memory           │
│  (Mstep)     │    (Msem)                    │
│  Recent acts,│    Consolidated rules,       │
│  observations│    constraints, failure modes │
├──────────────┴──────────────────────────────┤
│         BM25 Retrieval + Consolidation      │
│  Query = Goal + Inventory + Entity Terms    │
│  Retrieve Top-K from Mstep & Msem          │
├─────────────────────────────────────────────┤
│         Mineflayer Bot (Node.js)            │
│  Text observations: inventory, position,    │
│  stats, equipment, environment, entities    │
└─────────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for Mineflayer)
- **Minecraft Java Edition** (1.20.4 recommended)
- **UCSD TritonAI API key**

## Quick Start

### 1. Environment Setup

```bash
# Clone/copy this project
cd memcraft

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd mineflayer_bridge
npm install
cd ..

# Set your API key
export TRITONAI_API_KEY="your-key-here"
```

### 2. Start Minecraft Server

Option A: Use a local server
```bash
# Download and run a Minecraft server (1.20.4)
# In server.properties, set:
#   online-mode=false
#   gamemode=survival
#   difficulty=easy
```

Option B: Open a singleplayer world to LAN
- Open Minecraft → Singleplayer → Create World → Open to LAN
- Note the port number shown in chat

### 3. Run the Agent

```bash
# Basic dirt mining task
python run_agent.py --task "mine_dirt" --host localhost --port 25565

# Custom task
python run_agent.py --task "collect 10 wood logs" --host localhost --port 25565

# With debug logging
python run_agent.py --task "mine_dirt" --host localhost --port 25565 --debug
```

## Project Structure

```
memcraft/
├── README.md
├── requirements.txt
├── run_agent.py              # Main entry point
├── configs/
│   └── default.json          # Configuration
├── mineflayer_bridge/
│   ├── package.json
│   ├── bot.js                # Mineflayer bot (Node.js)
│   └── actions.js            # Available bot actions
├── agent/
│   ├── __init__.py
│   ├── brain.py              # LLM interface (TritonAI API)
│   ├── memory.py             # Hierarchical memory (Mstep + Msem)
│   ├── retrieval.py          # BM25 retrieval
│   ├── consolidation.py      # Semantic consolidation
│   ├── observer.py           # Structured delta observations
│   └── agent.py              # Main agent loop
├── memories/                 # Persisted memory files
└── logs/                     # Run logs
```

## Budget Considerations ($200 cap)

- Uses `api-llama-4-scout` (cheapest/fastest on TritonAI)
- Text-only observations (no vision = massive token savings)
- Delta encoding: only send what CHANGED since last step
- BM25 retrieval: no embedding API calls needed
- Semantic consolidation: compress repetitive memories into rules
- Estimated cost: ~$0.01-0.05 per episode (100 steps)

## Three Agent Configurations (Experiment Design)

1. **No-Memory Baseline**: Current observation only, no history
2. **Naive Memory Baseline**: FIFO history (last L steps), no filtering
3. **MemAgent (Ours)**: Structured delta + BM25 retrieval + semantic rules
