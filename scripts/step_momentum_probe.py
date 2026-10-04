#!/usr/bin/env python3
"""Zero-training twin probe for step_edge_momentum.

Frozen weights, one selection change: rule-on vs rule-off, same pin, same
seed, matched episode counts. The protocol that produced pit_edge_commit's
+224px and killed CUSE-1's stall-breaker (+3px). Judge depth deltas >50px at
n=10+ per the project rule; a win-count difference at n=20 is a draw, not a
verdict.

Usage (quiet machine, 0 fceux+trainer CPU):
  python scripts/step_momentum_probe.py --run-dir runs/probe-momentum       --pin-dir ~/.capy-work/mario-watch/pins/20261003-pre-stall-lever       --launches 2 --episodes 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from mario_ai_fceux import evaluate as evaluate_module  # noqa: E402
from mario_ai_fceux.actions import step_edge_momentum  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def run_arm(arm: str, run_dir: Path, episodes: int, world: int, rom: Path) -> dict:
    """Run one eval launch with the momentum rule ON or OFF, return results."""
    eval_root = run_dir / "evaluations"
    before = {d.name for d in eval_root.glob("*")} if eval_root.exists() else set()
    argv = [
        sys.executable, "-m", "mario_ai_fceux.evaluate",
        "--rom", str(rom), "--fceux", "fceux",
        "--run-dir", str(run_dir), "--world", str(world),
        "--episodes", str(episodes), "--device", "cpu", "--trace-actions",
        "--max-seconds", "3600",
    ]
    env = {"ARM": arm, "PROBE_RUN_DIR": str(run_dir)}
    if arm == "on":
        # Inject the rule by monkeypatching through a sitecustomize-style shim:
        # simplest honest approach - run a wrapper that patches then delegates.
        shim = run_dir / f"probe_shim_{arm}.py"
        shim.write_text('''
import sys, runpy
sys.path.insert(0, %r)
from mario_ai_fceux import evaluate as ev
from mario_ai_fceux.actions import step_edge_momentum
_orig = ev.agent.select_actions if hasattr(ev, "agent") else None
# Patch at the agent class level so evaluate's episode loop picks it up.
from mario_ai_fceux.agent import RainbowAgent
_orig_select = RainbowAgent.select_actions
def patched(self, states, explore=True):
    import numpy as np
    actions = _orig_select(self, states, explore=explore)
    if explore:
        return actions
    out = list(actions)
    for i, st in enumerate(np.atleast_2d(states)):
        q, _ = self.inspect(st.reshape(1, -1))
        out[i] = step_edge_momentum(st, q[0], int(actions[i]))
    return np.asarray(out)
RainbowAgent.select_actions = patched
sys.argv = ["mario_ai_fceux.evaluate"] + %r
runpy.run_module("mario_ai_fceux.evaluate", run_name="__main__")
''' % (str(REPO / "python"), argv[3:]))
        cmd = [sys.executable, str(shim)]
    else:
        cmd = argv
    res = subprocess.run(cmd, cwd=REPO, env={**dict(**__import__("os").environ), **env},
                         capture_output=True, text=True)
    evals = sorted(eval_root.glob("*/results.json")) if eval_root.exists() else []
    fresh = [e for e in evals if e.parent.name not in before]
    chosen = fresh[-1] if fresh else (evals[-1] if evals else None)
    r = json.loads(chosen.read_text()) if chosen else {}
    xs = [e.get("max_x", 0) for e in r.get("episodes", [])]
    return {
        "arm": arm, "eval_dir": chosen.parent.name if chosen else None,
        "wins": r.get("victories"), "episodes": r.get("episodes_finished"),
        "max_x_mean": round(sum(xs) / len(xs), 1) if xs else None,
        "max_x_median": sorted(xs)[len(xs) // 2] if xs else None,
        "returncode": res.returncode,
        "stderr_tail": res.stderr.strip().splitlines()[-4:] if res.returncode else [],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--launches", type=int, default=2)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--rom", type=Path, default=Path("SuperMarioBros.nes"),
                   help="ROM path; must be absolute or resolve from --cwd. The probe "
                        "runs the evaluator with the REPO of this script as cwd, which "
                        "is a worktree - pass the ROM absolutely.")
    args = p.parse_args()
    rom = args.rom if args.rom.is_absolute() else (REPO / args.rom).resolve()
    points = []
    for launch in range(1, args.launches + 1):
        for arm in ("on", "off"):
            point = run_arm(arm, args.run_dir, args.episodes, args.world, rom)
            point["launch"] = launch
            points.append(point)
            print(json.dumps(point), flush=True)
    out = args.run_dir / "momentum-probe.json"
    out.write_text(json.dumps({"protocol": "twin, frozen weights, rule-on vs rule-off", "points": points}, indent=2))
    print(json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
