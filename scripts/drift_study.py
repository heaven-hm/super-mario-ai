#!/usr/bin/env python3
"""Drift study: turn "continued training degrades clean performance" into a curve.

Method (agreed with CUSE-1, 2026-10-03):
  1. Start from an exact pin (model.pt + replay.npz pair, sha recorded).
  2. Train in short windows (--window-steps transitions each). The trainer's
     --steps bound is an ABSOLUTE cumulative transitions target: the learner
     carries its step counter across --resume (verified 2026-10-03, the live
     run stopped at 5,005,543 against the default 5,000,000 bound), so each
     window passes steps_at_window_start + window_steps and the trainer exits
     gracefully at its own boundary. Passing --steps 25000 on a resumed
     4.5M-step pin would exit after a single 10-second status tick.
  3. Pin model.pt + replay.npz at every window boundary (sha256 each).
  4. When the machine is quiet: one eval pass over every pin (t0 included),
     20 episodes each, deterministic, deployed selection path, CPU context
     recorded. The pass REFUSES to start while any training process, evaluator,
     or fceux emulator is alive, or fceux+trainer CPU exceeds 2% -
     contamination fakes wins and losses alike (project lesson).
  5. Write curve.json: {step, wins, episodes, max_x stats} per point. Judge
     every post-change window against the PIN's own measurement (twin rule).

Phases:
  windows - train and pin only; run this first. Relaunch each window with
            --checkpoint-every 1000 so machine kills cost at most one window
            (CUSE-1's lesson: their VM restarted 7+ times and only dense
            checkpointing kept windows recoverable).
  evals   - the quiet pass over the pins the windows phase wrote, plus any
            --extra-pin directories (e.g. the live checkpoint), no training.
  all     - windows then evals in one run.

Labels per point: stack mix and config delta, per the project protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_step(model: Path) -> int | None:
    try:
        import torch
        d = torch.load(model, map_location="cpu", weights_only=False)
        return int(d.get("steps"))
    except Exception:
        return None


def pin_pair(run_dir: Path, pin_dir: Path, label: str) -> dict:
    pin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "replay.npz"):
        (pin_dir / name).write_bytes((run_dir / name).read_bytes())
    meta = {
        "label": label,
        "dir": pin_dir.name,
        "step": read_step(pin_dir / "model.pt"),
        "model_sha": sha16(pin_dir / "model.pt"),
        "replay_sha": sha16(pin_dir / "replay.npz"),
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (pin_dir / "pin.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def launch_window(repo: Path, run_dir: Path, flags: list[str], abs_steps: int) -> None:
    cmd = [
        os.path.join(repo, ".venv-fceux/bin/python"), "-m", "mario_ai_fceux.apex_train",
        "--rom", "SuperMarioBros.nes", "--fceux", "fceux",
        "--run-dir", str(run_dir),
        "--steps", str(abs_steps),
        "--checkpoint-every", "1000",
        "--window-layout", "config/fceux-window-layout.ini",
        "--cheats-enabled-workers", "",
        "--disable-eval",
        *flags,
    ]
    env = dict(os.environ, PYTHONPATH="python", LC_ALL="C")
    log = open("/tmp/drift-study.log", "a")
    proc = subprocess.Popen(cmd, cwd=str(repo), env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # The trainer exits gracefully once its cumulative transitions reach abs_steps.
    proc.wait()
    log.close()


def cpu_context() -> tuple[str, float]:
    try:
        out = subprocess.run(["ps", "-Ao", "%cpu=,command="], capture_output=True, text=True).stdout
        busy = 0.0
        third = 0.0
        for line in out.splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2 or "ps -Ao" in parts[1]:
                continue
            cpu = float(parts[0])
            cmdtext = parts[1]
            if "drift_study" in cmdtext:
                continue
            if "fceux --gamegenie" in cmdtext or "mario_ai_fceux.apex_train" in cmdtext:
                busy += cpu
            elif "multiprocessing.spawn" in cmdtext or "resource_tracker" in cmdtext:
                third += cpu
        ctx = f"{busy:.0f}% fceux+trainer CPU at eval start (0 = uncontaminated)"
        if third > 0.5:
            ctx += f"; note {third:.0f}% CPU in an unrelated multiprocessing learner tree"
        return ctx, busy
    except Exception:
        return "cpu context unavailable", -1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run dir for the study. Use one no other trainer writes: "
                        "shared run dirs corrupt both metrics and pins (verified "
                        "2026-10-03, smoke-test collision).")
    p.add_argument("--pin", type=Path, required=True,
                   help="Directory holding the t0 pin (model.pt + replay.npz).")
    p.add_argument("--phase", choices=["all", "windows", "evals"], default="all")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--window-steps", type=int, default=25000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-world", type=int, default=1)
    p.add_argument("--extra-pin", type=Path, action="append", default=[],
                   help="Additional pin dir (with pin.json) to include in the eval pass.")
    p.add_argument("--flags", type=str,
                   default="--worlds 1,1,1,1,2,2,2,2 --resume --frontier-spacing 128 "
                           "--frontier-retries 6 --learn-per-batch 4",
                   help="Training flags; identical across all windows (one change at a time).")
    args = p.parse_args()

    study = args.run_dir / "drift-study"
    study.mkdir(parents=True, exist_ok=True)
    flags = args.flags.split()
    points: list[dict] = []

    if args.phase in ("all", "windows"):
        for name in ("model.pt", "replay.npz"):
            (args.run_dir / name).write_bytes((args.pin / name).read_bytes())
        points.append(pin_pair(args.run_dir, study / "pin-000-t0", "t0"))
        base = points[0]["step"] or 0
        for w in range(1, args.windows + 1):
            launch_window(args.repo, args.run_dir, flags, base + w * args.window_steps)
            points.append(pin_pair(args.run_dir, study / f"pin-{w:03d}", f"window-{w}"))
        (study / "points.json").write_text(json.dumps(points, indent=2), encoding="utf-8")
        if args.phase == "windows":
            print(json.dumps({
                "phase": "windows", "pins": len(points),
                "next": "run --phase evals at a quiet moment (0 fceux, 0 apex_train)",
                "points_file": str(study / "points.json"),
            }, indent=2))
            return

    if args.phase == "evals":
        for pin_json in sorted(study.glob("pin-*/pin.json")):
            points.append(json.loads(pin_json.read_text(encoding="utf-8")))
        for extra in args.extra_pin:
            meta = json.loads((extra / "pin.json").read_text(encoding="utf-8"))
            meta["dir"] = str(extra)
            points.append(meta)
        if not points:
            raise SystemExit("no pins found - run --phase windows first")
        points.sort(key=lambda m: (m.get("step") is None, m.get("step") or 0))

    procs = subprocess.run(["pgrep", "-f", "mario_ai_fceux[.]apex_train|mario_ai_fceux[.]evaluate"],
                           capture_output=True, text=True).stdout.split()
    emus = subprocess.run(["pgrep", "-x", "fceux"], capture_output=True, text=True).stdout.split()
    if procs or emus:
        raise SystemExit(f"refusing to eval while {len(procs) + len(emus)} training/eval/emulator processes run: {procs + emus}")
    ctx, busy = cpu_context()
    if busy > 2.0:
        raise SystemExit(f"refusing to eval: {ctx} (need ~0 for an uncontaminated pass)")

    for point in points:
        pin_dir = Path(point["dir"])
        if not pin_dir.is_absolute():
            pin_dir = study / point["dir"]
        eval_root = args.run_dir / "evaluations"
        before = {d.name for d in eval_root.glob("*")} if eval_root.exists() else set()
        for name in ("model.pt", "replay.npz"):
            (args.run_dir / name).write_bytes((pin_dir / name).read_bytes())
        env = dict(os.environ, PYTHONPATH="python", LC_ALL="C")
        res = subprocess.run([
            os.path.join(args.repo, ".venv-fceux/bin/python"), "-m", "mario_ai_fceux.evaluate",
            "--rom", "SuperMarioBros.nes", "--fceux", "fceux",
            "--run-dir", str(args.run_dir), "--world", str(args.eval_world),
            "--episodes", str(args.eval_episodes), "--device", "cpu", "--trace-actions",
        ], cwd=str(args.repo), env=env, capture_output=True, text=True)
        evals = sorted(eval_root.glob("*/results.json")) if eval_root.exists() else []
        fresh = [e for e in evals if e.parent.name not in before]
        chosen = fresh[-1] if fresh else (evals[-1] if evals else None)
        r = json.loads(chosen.read_text()) if chosen else {}
        xs = [e.get("max_x", 0) for e in r.get("episodes", [])]
        point.update({
            "wins": r.get("victories"),
            "episodes": r.get("episodes_finished"),
            "max_x_mean": round(sum(xs) / len(xs), 1) if xs else None,
            "max_x_median": sorted(xs)[len(xs) // 2] if xs else None,
            "eval_dir": chosen.parent.name if chosen else None,
            "cpu_context": ctx,
            "eval_log_sha": hashlib.sha256(res.stdout.encode()).hexdigest()[:16],
        })

    (study / "curve.json").write_text(json.dumps({
        "protocol": f"windows pinned, one quiet pass, n={args.eval_episodes} per point",
        "flags": args.flags,
        "cpu_context": ctx,
        "points": points,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "evals", "points": len(points), "out": str(study / "curve.json")}, indent=2))


if __name__ == "__main__":
    main()
