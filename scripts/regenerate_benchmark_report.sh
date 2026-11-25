#!/usr/bin/env bash
# Regenerate BENCHMARK_RESULTS.md from checked-in measurement artifacts.
# No network, ROM, emulator or trained model is needed: this only reads JSON
# that is already committed to the repository.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/publish_benchmark.py --root . \
  --blocker 'Lua NEAT + contextual Q-memory=No champion evaluation log and no mario_ai_neat.db are committed here (the database is a GitHub Release asset). BENCHMARK_PROTOCOL.md requires setting PLAY_CHAMPION_ONLY = true, running FCEUX with the matching 2.6.6_12 build, then importing the log with import_lua_evaluation. None of those artifacts exist in the repository, so no clear rate can be measured.' \
  --blocker 'Basic DDQN=No trained DDQN checkpoint (runs/basic-ddqn/model.pt) and no DDQN evaluation report exist. BENCHMARK_PROTOCOL.md requires 100000 baseline training steps (estimated 2-4 hours on the target MacBook) before evaluation; that training was not run.' \
  --blocker 'PPO=No trained FCEUX-bridge PPO checkpoint (runs/ppo-baseline/model.pt) and no PPO evaluation report exist. BENCHMARK_PROTOCOL.md requires 100000 baseline training steps (estimated 2-4 hours) before evaluation. The separate smb1_ppo (gym-super-mario-bros) path also has no committed 20-episode result.' \
  --legacy-result 'Ape-X Rainbow, full-rainbow-input-fixed best policy=runs/full-rainbow-input-fixed/best_policy_eval.json' \
  --legacy-result 'Ape-X Rainbow, w1-bootcamp best policy=runs/w1-bootcamp/best_policy_eval.json' \
  --aggregate 'Rainbow pin-004, raw-greedy Mac launches=runs/drift-20261003/drift-study/pooling-pin-004.json' \
  --aggregate 'Rainbow pin-004, momentum-twin rule-off launches=runs/probe-momentum/momentum-probe.json' \
  --aggregate 'COMBINED (union of the two rows above, not independent)=benchmark/pin-004-raw-greedy-pool.json' \
  --note 'Rainbow pin-004=Training wall-clock was not recorded with this checkpoint; the drift study records optimizer step 4592778 but no hours.' \
  --command 'Verify ROM (ROM bytes are never committed)=shasum -a 256 /path/to/SuperMarioBros.nes  # expected 0b3d9e1f01ed1668205bab34d6c82b0e281456e137352e4f36a9b2cfa3b66dea' \
  --command 'Documented Rainbow evaluation (requires FCEUX 2.6.6_12)=PYTHONPATH=python python3 -m mario_ai_fceux.evaluate --rom /path/to/SuperMarioBros.nes --fceux /opt/homebrew/Cellar/fceux/2.6.6_12/bin/fceux --run-dir runs/drift-20261003/drift-study/pin-004 --world 1 --episodes 20 --evaluation-seed 2026 --device cpu' \
  --command 'Regenerate this report (offline, no emulator)=bash scripts/regenerate_benchmark_report.sh' \
  --command 'Probe local FCEUX Lua liveness=FCEUX_LUA_PROBE_OUTPUT=/tmp/fceux-lua-probe.txt fceux --loadlua "$PWD/scripts/probe_fceux_lua.lua" /path/to/SuperMarioBros.nes ; cat /tmp/fceux-lua-probe.txt' \
  --environment 'ROM SHA-256=0b3d9e1f01ed1668205bab34d6c82b0e281456e137352e4f36a9b2cfa3b66dea (verified locally; matches BENCHMARK_PROTOCOL.md)' \
  --environment 'Local FCEUX=/opt/homebrew/bin/fceux 2.7.0, SHA-256 e09717a3b02ee5ba3387c88abcec0814638d16ca0396527c24cf63b98870ac09' \
  --environment 'Protocol FCEUX=2.6.6_12, SHA-256 2d84a8310d530745263cfeadf00de9a631162a8dd76c71e1bed659c73753f2f6 (not installed here)' \
  --environment 'Fresh-run blocker=scripts/probe_fceux_lua.lua wrote only "start paused=false emulating=true" and no frames=30 line in 8 s, so emu.frameadvance() never resumes and the Python/FCEUX bridge produces no observation.json' \
  --environment 'Checkpoints present=runs/drift-20261003/drift-study/pin-004/model.pt (Rainbow, optimizer step 4592778); no DDQN or PPO checkpoint' \
  --output BENCHMARK_RESULTS.md

printf '%s\n' 'Wrote BENCHMARK_RESULTS.md'
