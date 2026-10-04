# Evaluation Handoff: Four Methods Benchmark

**Status**: Cloud-side prep complete. Ready for MacBook evaluation.

**Branch**: `feature/benchmark-results` (committed and pushed via heaven-hm)

## What's Done on Cloud

1. ✅ **BENCHMARK_PROTOCOL.md**: Exact evaluation commands for all four methods
2. ✅ **README.md**: Benchmark table stub with headers (rows empty, waiting for numbers)
3. ✅ **Branch**: `feature/benchmark-results` created and committed with structure in place

## What Needs MacBook Execution

Run all evaluations on your MacBook with FCEUX emulator.

### Execution Checklist

#### 1. Ape-X Rainbow DQN (Fastest — already trained)
**Checkpoint exists**: `runs/full-rainbow-input-fixed/model.pt` (1.2 GB trained, 500k transitions)

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

**Expected output**: `runs/full-rainbow-input-fixed/evaluations/YYYYMMDD-HHMMSS/results.json`

**Metrics to extract**:
- Best X: `max(e['max_x'] for e in results['episodes'])`
- Completion: `results['victories']` (count) and `results['completion_rate']` (percent)
- Mean X: `sum(e['max_x'] ...) / 20`
- Mean decisions: `sum(e['action_decisions'] ...) / 20`
- Mean time: `sum(e['elapsed_seconds'] ...) / 20`

---

#### 2. Basic DDQN (Requires Training — ~2-4 hours)

**No checkpoint exists yet.** Train first:

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

Then evaluate:

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

**Expected output**: `runs/basic-ddqn/evaluations/YYYYMMDD-HHMMSS/results.json`

**Note**: Record the training wall-clock time from the baseline_train output (will be printed to console or run.json).

---

#### 3. PPO (Requires Training — ~2-4 hours)

**No checkpoint exists yet.** Train first:

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

Then evaluate:

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

**Expected output**: `runs/ppo-baseline/evaluations/YYYYMMDD-HHMMSS/results.json`

**Note**: Record the training wall-clock time.

---

#### 4. Lua NEAT + Contextual Q-Learning (Manual, ~30 minutes)

**Database exists**: `mario_ai_neat.db` (Q-learning memory + evolved champion genome)

**Steps:**
1. Edit `mario_ai_neat.lua` and find the line with `PLAY_CHAMPION_ONLY`. Set it to `true`.
2. Launch FCEUX with Lua:
   ```bash
   /opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux \
     -lua mario_ai_neat.lua \
     -nothrottle \
     /path/to/SuperMarioBros.nes
   ```
3. Let it run until the champion completes at least 20 episodes. Each episode will show the world, level, distance, and decision count in the log.
4. Once done, the Lua script auto-saves to `mario_ai_neat.log` (in the directory where mario_ai_neat.lua is).
5. Convert the log to benchmark JSON:
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

**Expected output**: `runs/lua-neat/results.json`

**Note**: Lua NEAT runs natively in FCEUX; no Python benchmark tool needed, but importer converts the log format to JSON.

---

## After All Four Results Exist

Once you have results.json files from all four methods in:
- `runs/full-rainbow-input-fixed/evaluations/...results.json`
- `runs/basic-ddqn/evaluations/...results.json`
- `runs/ppo-baseline/evaluations/...results.json`
- `runs/lua-neat/results.json`

### 5. Generate Benchmark Table

```bash
cd /workspace/super-mario-ai
PYTHONPATH=python python3 -m mario_ai_fceux.benchmark \
  --result 'Lua NEAT + Q-Memory=runs/lua-neat/results.json' \
  --result 'Ape-X Rainbow DQN=runs/full-rainbow-input-fixed/evaluations/YYYYMMDD-HHMMSS/results.json' \
  --result 'Basic DDQN=runs/basic-ddqn/evaluations/YYYYMMDD-HHMMSS/results.json' \
  --result 'PPO=runs/ppo-baseline/evaluations/YYYYMMDD-HHMMSS/results.json' \
  --output BENCHMARK_RESULTS.md
```

This generates a Markdown table with all metrics side-by-side.

---

### 6. Capture Best 1-1 Clear GIF

From the 20 evaluation episodes above, find the one where a method cleared 1-1 (if any).
Capture a screen recording of the best run showing Mario completing World 1-1.

**If no method clears 1-1**: Document the best X position reached instead, explaining the difficulty.

---

### 7. Update README Table with Numbers

Hand-fill the table in README.md with extracted numbers from the four results.json files:

| Learning Method | Best X | 1-1 Completion | Mean X | Mean Decisions | Mean Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lua NEAT + Q-Memory | [best_x] | [victories]/20 | [mean_x] | [mean_decisions] | [mean_time] |
| Ape-X Rainbow DQN | [best_x] | [victories]/20 | [mean_x] | [mean_decisions] | [mean_time] |
| Basic DDQN | [best_x] | [victories]/20 | [mean_x] | [mean_decisions] | [mean_time] |
| PPO | [best_x] | [victories]/20 | [mean_x] | [mean_decisions] | [mean_time] |

---

## Commit & Push

Once the table and GIF are in README:

```bash
cd /workspace/super-mario-ai
# Make sure you're on feature/benchmark-results branch
git add README.md BENCHMARK_RESULTS.md best-1-1-clear.gif  # or .mp4
git commit -m "Add benchmark results table and best 1-1 clear clip

- Measured 20 deterministic episodes per method (World 1-1, seed=2026, greedy)
- Lua NEAT + Q-Memory: X positions and completion rates from champion evaluator
- Ape-X Rainbow DQN: Results from trained model checkpoint
- Basic DDQN: Results from 100k-step baseline training
- PPO: Results from 100k-step baseline training
- Best 1-1 clear run included as media clip
- All metrics verified from actual measured runs (not fabricated)
- Benchmark protocol and evaluation commands documented in BENCHMARK_PROTOCOL.md"

# Push via heaven-hm profile (MacBook only)
git push origin feature/benchmark-results
```

---

## Key Validation Points (Crew Rule: Evidence Before Claims)

- ✅ ROM SHA-256: `0b3d9e1f01ed1668205bab34d6c82b0e281456e137352e4f36a9b2cfa3b66dea`
- ✅ FCEUX SHA-256: `2d84a8310d530745263cfeadf00de9a631162a8dd76c71e1bed659c73753f2f6`
- ✅ World 1, Level 1 clean start (title-screen world selector)
- ✅ 20 episodes per method (not abbreviated to fewer)
- ✅ Greedy evaluation (no exploration, no learning)
- ✅ Seed 2026 for reproducibility
- ✅ All numbers from results.json files (not guesses or theoretical claims)
- ✅ Training times from actual baseline_train logs if applicable
- ✅ No force-push, no history rewrite (crew rule #2)
- ✅ All commits from heaven-hm profile on MacBook (crew rule #8)

---

## Notes

- **Rainbow DQN** is the fastest to evaluate (already trained; ~5-10 min for 20 episodes)
- **DDQN and PPO** each require training (~2-4 hours) + evaluation (~5-10 min)
- **Lua NEAT** requires manual FCEUX play (~30 min for 20 episodes) + 1-2 min conversion
- Total time estimate: 6-12 hours depending on training speed
- All four can run in parallel on separate FCEUX windows if your Mac can handle it

---

## Files Created on Cloud

- `BENCHMARK_PROTOCOL.md` — This evaluation protocol document
- `EVALUATION_HANDOFF.md` — This handoff summary
- `feature/benchmark-results` branch with README table stub

Ready to hand off to MacBook for execution.
