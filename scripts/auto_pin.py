#!/usr/bin/env python3
"""Pin model+replay at every N-transition boundary of a live run dir.

The drift curve depends on boundary pins, and manual pinning missed the
50k/75k/125k boundaries on 2026-10-04. This watcher reads the run dir's
learner_status.json transitions counter and pins the current pair whenever it
crosses the next multiple of --every.

Pair-order convention (from the two-writers crash lesson): copy replay.npz
first, then model.pt. The trainer saves model-first, so this bounds skew to
model-ahead-by-one-save-cycle -- the same relationship as the trainer's own
.bak pair. Each pin records its step read from model.pt (never from a log),
model/replay sha16, and the boundary it stands for.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_study import read_step, sha16  # noqa: E402

STATE_FILE = "auto_pin_state.json"


def boundaries_crossed(last_pinned: int, transitions: int, every: int) -> list[int]:
    out = []
    b = last_pinned + every
    while b <= transitions:
        out.append(b)
        b += every
    return out


def pin_boundary(run_dir: Path, pin_root: Path, boundary: int, transitions: int) -> dict:
    pin_dir = pin_root / f"auto-{time.strftime('%Y%m%d-%H%M%S')}-b{boundary}"
    pin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("replay.npz", "model.pt"):
        shutil.copyfile(run_dir / name, pin_dir / name)
    meta = {
        "label": f"auto boundary pin at {boundary} transitions (live counter {transitions})",
        "dir": pin_dir.name,
        "boundary": boundary,
        "transitions_at_pin": transitions,
        "step": read_step(pin_dir / "model.pt"),
        "model_sha": sha16(pin_dir / "model.pt"),
        "replay_sha": sha16(pin_dir / "replay.npz"),
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (pin_dir / "pin.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def transitions_now(run_dir: Path) -> int | None:
    try:
        return int(json.loads((run_dir / "learner_status.json").read_text())["transitions_received"])
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--pin-root", type=Path, required=True)
    p.add_argument("--every", type=int, default=25000)
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--once", action="store_true", help="one pass, then exit (for tests/cron)")
    args = p.parse_args()

    state_path = args.pin_root / STATE_FILE
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
    if "last_pinned_boundary" not in state:
        t = transitions_now(args.run_dir)
        state["last_pinned_boundary"] = t or 0
        args.pin_root.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")

    while True:
        t = transitions_now(args.run_dir)
        if t is not None:
            for b in boundaries_crossed(state["last_pinned_boundary"], t, args.every):
                meta = pin_boundary(args.run_dir, args.pin_root, b, t)
                state["last_pinned_boundary"] = b
                state_path.write_text(json.dumps(state), encoding="utf-8")
                print(json.dumps({"pinned": meta["dir"], "boundary": b, "step": meta["step"]}), flush=True)
        if args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
