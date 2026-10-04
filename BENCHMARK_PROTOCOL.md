# Benchmark Protocol: Four Learning Methods on SMB1 World 1-1

This document specifies the exact evaluation commands for each of the four learning methods.
Run these on your MacBook with FCEUX emulator available and the SMB1 ROM accessible.

## Prerequisites

- FCEUX with Lua support: `/opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux`
- SMB1 ROM (SHA-256: `0b3d9e1f01ed1668205bab34d6c82b0e281456e137352e4f36a9b2cfa3b66dea`)
- Python 3.10+ with PyTorch and NumPy installed
- All four methods have model checkpoints already in the repository

## Evaluation Conditions (Fixed Across All Methods)

- **World**: 1
- **Level**: 1 (clean selected-world start via SMB1 title-screen world selector)
- **Episodes**: 20 deterministic episodes (required per acceptance criteria)
- **Evaluation Seed**: 2026 (ensures reproducibility)
- **Evaluation Mode**: Greedy, no learning, no exploration
- **Action Repeat**: 12 frames (fixed for all, embedded in evaluation code)
- **Start Protocol**: "SMB1 clean selected-world start with course-start retries"

## Method 1: Python Ape-X Rainbow DQN

**Checkpoint Location**: `runs/full-rainbow-input-fixed/model.pt`

**Evaluation Command**:
```bash
cd /workspace/super-mario-ai
PYTHONPATH=python python3 -m mario_ai_fceux.evaluate \
  --rom "/path/to/SuperMarioBros.nes" \
  --fceux "/opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux" \
  --run-dir runs/full-rainbow-input-fixed \
  --world 1 \
  --episodes 20 \
  --evaluation-seed 2026
```

**Output Files**:
- `runs/full-rainbow-input-fixed/evaluations/YYYYMMDD-HHMMSS/results.json` — benchmark-ready report with:
  - Algorithm: "Ape-X Rainbow (C51 + NoisyNet + Double DQN + Dueling + PER + N-step)"
  - Victories count
  - Completion rate (victories / 20)
  - Episodes array with: max_x, action_decisions, elapsed_seconds per episode

**Metrics to Extract**:
- Best X position reached: `max(episode['max_x'] for episode in results['episodes'])`
- Completion rate: `results['completion_rate']` or `results['victories'] / 20`
- Mean decisions per episode: `mean(episode['action_decisions'] ...)`
- Mean wall-clock seconds: `mean(episode['elapsed_seconds'] ...)`

---

## Method 2: Basic DDQN (Double DQN with Uniform Replay)

**Checkpoint Location**: No checkpoint exists yet—must be trained.

**Training Command** (only needed if checkpoint missing):
```bash
cd /workspace/super-mario-ai
PYTHONPATH=python python3 -m mario_ai_fceux.baseline_train \
  --algorithm ddqn \
  --rom "/path/to/SuperMarioBros.nes" \
  --fceux "/opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux" \
  --workers 1 \
  --worlds 1 \
  --steps 100000 \
  --run-dir runs/basic-ddqn
```

**Evaluation Command**:
```bash
cd /workspace/super-mario-ai
PYTHONPATH=python python3 -m mario_ai_fceux.evaluate \
  --rom "/path/to/SuperMarioBros.nes" \
  --fceux "/opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux" \
  --run-dir runs/basic-ddqn \
  --world 1 \
  --episodes 20 \
  --evaluation-seed 2026
```

**Output**: Same structure as Rainbow (results.json with episode details).

---

## Method 3: PPO (Clipped Categorical Policy with GAE)

**Checkpoint Location**: No checkpoint exists yet—must be trained.

**Training Command** (only needed if checkpoint missing):
```bash
cd /workspace/super-mario-ai
PYTHONPATH=python python3 -m mario_ai_fceux.baseline_train \
  --algorithm ppo \
  --rom "/path/to/SuperMarioBros.nes" \
  --fceux "/opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux" \
  --workers 1 \
  --worlds 1 \
  --steps 100000 \
  --run-dir runs/ppo-baseline
```

**Evaluation Command**:
```bash
cd /workspace/super-mario-ai
PYTHONPATH=python python3 -m mario_ai_fceux.evaluate \
  --rom "/path/to/SuperMarioBros.nes" \
  --fceux "/opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux" \
  --run-dir runs/ppo-baseline \
  --world 1 \
  --episodes 20 \
  --evaluation-seed 2026
```

**Output**: Same structure as Rainbow (results.json with episode details).

---

## Method 4: Lua NEAT + Contextual Q-Learning

**Checkpoint Location**: `mario_ai_neat.db` (persistent population and Q-memory)

**Champion Evaluation Setup**:
1. Edit `mario_ai_neat.lua` and set `PLAY_CHAMPION_ONLY = true` at the top
2. Run FCEUX with the Lua script to record at least 10 completed episodes
3. Lua will log to `mario_ai_neat.log` (auto-saved in same directory as script)

**FCEUX Launch Command** (manual in FCEUX GUI or CLI):
```bash
/opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux \
  -lua mario_ai_neat.lua \
  -nothrottle \
  /path/to/SuperMarioBros.nes
```

Then let it play ~10-20 complete episodes to generate the log.

**Import to Benchmark Format**:
```bash
cd /workspace/super-mario-ai
PYTHONPATH=python python3 -m mario_ai_fceux.import_lua_evaluation \
  --log mario_ai_neat.log \
  --database mario_ai_neat.db \
  --rom "/path/to/SuperMarioBros.nes" \
  --fceux "/opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux" \
  --world 1 \
  --level 1 \
  --episodes 20 \
  --evaluation-seed 2026 \
  --output runs/lua-neat/results.json
```

**Output**: `runs/lua-neat/results.json` with structure matching Python evaluations.

---

## Metrics Available from All Methods

Each `results.json` contains:

```json
{
  "algorithm": "Method name string",
  "checkpoint": "Path to model.pt or evaluated checkpoint",
  "world": 1,
  "level": 1,
  "evaluation_mode": "greedy_no_learning",
  "evaluation_seed": 2026,
  "rom_sha256": "0b3d9e1f01ed1668205bab34d6c82b0e281456e137352e4f36a9b2cfa3b66dea",
  "fceux_sha256": "...",
  "episodes_requested": 20,
  "episodes_finished": 20,
  "victories": N,
  "completion_rate": N/20,
  "episodes": [
    {
      "episode": 1,
      "reason": "victory" or "death" or "timeout",
      "max_x": 3456,
      "terminal_x": 3456,
      "action_decisions": 142,
      "elapsed_seconds": 31.5
    },
    ...
  ]
}
```

### Extracted Metrics

For the README benchmark table, extract:

1. **Best X position reached**: `max(e['max_x'] for e in episodes)`
2. **Completion rate**: `f"{victories / 20:.0%}"` or `f"{completion_rate:.0%}"`
3. **Mean decisions/episode**: `sum(e['action_decisions'] ...) / 20`
4. **Mean wall-clock seconds/episode**: `sum(e['elapsed_seconds'] ...) / 20`

---

## Creating the Benchmark Table

Once all four results.json files exist:

```bash
cd /workspace/super-mario-ai
PYTHONPATH=python python3 -m mario_ai_fceux.benchmark \
  --result 'Lua NEAT + Q-Memory=runs/lua-neat/results.json' \
  --result 'Ape-X Rainbow DQN=runs/full-rainbow-input-fixed/results.json' \
  --result 'Basic DDQN=runs/basic-ddqn/results.json' \
  --result 'PPO=runs/ppo-baseline/results.json' \
  --output BENCHMARK_RESULTS.md
```

This will generate a Markdown table with all metrics for each method.

---

## Execution Order (For MacBook)

1. **Rainbow** (already trained): Run evaluation command, capture results.json
2. **DDQN** (need to train): Run training command (~few hours), then evaluation, capture results.json
3. **PPO** (need to train): Run training command (~few hours), then evaluation, capture results.json
4. **Lua NEAT** (existing champion): Run FCEUX with `PLAY_CHAMPION_ONLY = true`, record 20 episodes, import to JSON
5. **Benchmark table**: Run benchmark command to merge all four results
6. **GIF**: Capture best 1-1 clear run from the evaluations above
7. **Update README**: Add table and GIF above the fold (in "See it in action" or similar section)
8. **Commit and push** with message: "Add benchmark results table and best 1-1 clear clip"

---

## Notes on Metrics

- **Wall-clock time**: Includes FCEUX emulation time only; does not include model load/inference overhead in this measurement.
- **Decisions per frame**: Action decisions ÷ frame count. With 12-frame action repeat, estimate frame count from episode duration.
- **ROM and FCEUX SHA-256**: Must match across all methods for the benchmark tool to accept the comparison.
- **Training time**: Only available for DDQN and PPO (baseline_train logs duration). Rainbow training was done previously; note the latest checkpoint's timestamp.
