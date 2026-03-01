# MemCraft: Hierarchical Memory for LLM-Based Minecraft Agents

A hierarchical memory system for LLM-based Minecraft agents that enables learning across episodes. Built with Mineflayer (text-only, no vision) and evaluated on multi-step crafting tasks.

## Architecture

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

## Three Agent Variants (Experiment Design)

| Agent | Memory | Description |
|-------|--------|-------------|
| **NoMemory** | None | Current observation only. No history, no retrieval. |
| **NaiveMemory** | FIFO buffer | Last L steps as raw text. No filtering or consolidation. |
| **MemAgent** (Ours) | Hierarchical | Step memory + BM25 retrieval + semantic consolidation. Learns rules across episodes. |

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Minecraft Java Edition server** (1.19.2, running via Docker)
- **UCSD TritonAI API key**

## Quick Start

### 1. Setup

```bash
cd MemCraft

# Python dependencies
pip install -r requirements.txt

# Node.js dependencies
cd mineflayer_bridge && npm install && cd ..

# API key
export TRITONAI_API_KEY="your-key-here"
```

### 2. Verify Setup

```bash
# Test API connection
python test_api.py

# Full diagnostics (checks server, dependencies, etc.)
python diagnose.py --host localhost --port 25565 --version 1.19.2
```

### 3. Run a Single Agent

```bash
# Simple task
python run_agent.py --task "mine 5 dirt blocks" --agent memagent \
    --port 25565 --version 1.19.2 --max-steps 15

# Crafting task
python run_agent.py --task "craft a wooden pickaxe" --agent memagent \
    --port 25565 --version 1.19.2 --max-steps 20

# Complex multi-step task
python run_agent.py --task "craft a stone pickaxe" --agent memagent \
    --port 25565 --version 1.19.2 --max-steps 40

# With debug logging
python run_agent.py --task "mine 5 dirt blocks" --agent memagent --debug
```

### 4. Run Comparison (All 3 Agents)

The `reset_and_run.sh` script resets the Minecraft world to a fresh state and runs all three agents:

```bash
chmod +x reset_and_run.sh

# Craft a wooden pickaxe (3 episodes each, max 30 steps)
./reset_and_run.sh "craft a wooden pickaxe" 3 30

# Craft a stone pickaxe (harder — needs wooden pickaxe first)
./reset_and_run.sh "craft a stone pickaxe" 3 40

# Craft an iron ingot (hardest — full tool progression + smelting)
./reset_and_run.sh "craft an iron ingot" 3 60
```

The script:
1. Clears MemAgent's learned rules (fair start)
2. Deletes the Minecraft world and regenerates with seed `0` (forest biome)
3. Restarts the Docker server
4. Runs all 3 agents × N episodes on the same world
5. Prints a comparison table and saves results to `logs/`

### 5. Use a Different LLM Model

```bash
python run_agent.py --task "mine 5 dirt" --agent memagent \
    --model api-gpt-oss-120b --max-steps 15
```

## Available Actions

| Action | Description |
|--------|-------------|
| `find_and_mine_block` | Find and mine a block type (auto-equips best tool) |
| `craft_item` | Craft an item (auto-places crafting table if needed) |
| `smelt_item` | Smelt ore in a furnace (auto-crafts/places furnace, auto-selects fuel) |
| `scan_surroundings` | Scan nearby blocks in a radius |
| `move_forward` | Move forward N steps |
| `move_to` | Pathfind to coordinates |
| `equip_item` | Equip an item to hand/armor |
| `collect_nearby_items` | Walk to and pick up nearby dropped items |

## Task Difficulty Progression

| Task | Steps | Key Challenge |
|------|-------|--------------|
| Mine 5 dirt | ~2-3 | Basic action execution |
| Craft wooden pickaxe | ~4-6 | Multi-step crafting chain |
| Craft stone pickaxe | ~10-15 | Tool dependency (need wooden pickaxe first) |
| Craft iron ingot | ~20-30 | Full progression: tools → mining → smelting |

## Project Structure

```
MemCraft/
├── run_agent.py              # Main entry point (single + compare mode)
├── evaluate.py               # Batch evaluation across tasks
├── reset_and_run.sh          # World reset + comparison runner
├── diagnose.py               # Setup diagnostics
├── test_api.py               # API connection test
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.json          # LLM, memory, agent configuration
├── agent/
│   ├── __init__.py
│   ├── agent.py              # Agent variants (NoMemory, NaiveMemory, MemAgent)
│   ├── brain.py              # LLM interface (TritonAI API)
│   ├── memory.py             # Hierarchical memory (Mstep + Msem)
│   ├── retrieval.py          # BM25 retrieval
│   ├── consolidation.py      # Semantic rule consolidation
│   └── observer.py           # Structured delta observations + session inventory
├── mineflayer_bridge/
│   ├── package.json
│   ├── bot.js                # Mineflayer bot + HTTP bridge + reset/teleport
│   └── actions.js            # All bot actions (mine, craft, smelt, move, etc.)
├── memories/                 # Persisted semantic rules (gitignored)
└── logs/                     # Run logs and comparison results (gitignored)
```

## How Memory Works

1. **Step Memory (Mstep)**: Stores recent action-observation pairs with timestamps
2. **BM25 Retrieval**: Queries step memory using goal + inventory keywords to find relevant past experiences
3. **Semantic Consolidation**: Every N steps, the LLM extracts general rules from recent experiences (e.g., "Need a pickaxe to mine stone") and stores them in semantic memory
4. **Cross-Episode Learning**: Semantic rules persist across episodes, so MemAgent improves over time while baselines start fresh each episode

## Minecraft Server (Docker)

The server runs via the `itzg/minecraft-server` Docker image:

```bash
# Check server status
sudo docker ps

# View server logs
sudo docker logs mc-server --tail 20

# Reset world manually
sudo docker exec mc-server rm -rf /data/world /data/world_nether /data/world_the_end
sudo docker restart mc-server

# Change seed
sudo docker exec mc-server sh -c 'sed -i "s/level-seed=.*/level-seed=0/" /data/server.properties'
```

## Budget

- Uses `api-llama-4-scout` on TritonAI (cheapest available)
- Text-only observations (no vision = massive token savings)
- Delta encoding: only sends what changed since last step
- BM25 retrieval: no embedding API calls needed
- Estimated cost: ~$0.01-0.05 per episode