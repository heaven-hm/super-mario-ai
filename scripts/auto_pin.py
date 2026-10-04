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


def free_gib() -> float:
    import shutil as _sh
    return _sh.disk_usage(str(Path.home())).free / 1024 / 1024 / 1024


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
    args.pin_root.mkdir(parents=True, exist_ok=True)
    if "last_pinned_boundary" not in state:
        t = transitions_now(args.run_dir)
        if t is None:
            # learner_status.json not written yet (trainer still booting): do
            # NOT base the boundary counter at 0 - that floods pins from
            # boundary 0 to the live counter (~73 copies, 2.5GB, ENOSPC killed
            # the trainer on 2026-10-04). Retry next poll instead.
            if args.once:
                return
            time.sleep(args.poll)
            return main_restart(p, args)
        state["last_pinned_boundary"] = t
        state_path.write_text(json.dumps(state), encoding="utf-8")

    while True:
        t = transitions_now(args.run_dir)
        if t is not None:
            pending = boundaries_crossed(state["last_pinned_boundary"], t, args.every)
            if len(pending) > 2:
                # Sanity guard: a live run never crosses >2 boundaries per poll
                # (25k steps take minutes). More means the baseline drifted -
                # re-baseline instead of pinning a backlog.
                state["last_pinned_boundary"] = t - (t % args.every)
                state_path.write_text(json.dumps(state), encoding="utf-8")
                print(json.dumps({"rebaselined": t, "skipped": len(pending)}), flush=True)
                pending = []
            for b in pending:
                if free_gib() < 2.0:
                    print(json.dumps({"skipped_boundary": b, "reason": "low disk"}), flush=True)
                    state["last_pinned_boundary"] = b
                    continue
                meta = pin_boundary(args.run_dir, args.pin_root, b, t)
                state["last_pinned_boundary"] = b
                state_path.write_text(json.dumps(state), encoding="utf-8")
                print(json.dumps({"pinned": meta["dir"], "boundary": b, "step": meta["step"]}), flush=True)
        if args.once:
            return
        time.sleep(args.poll)


def main_restart(p, args):
    """Retry the state init after a poll (trainer may still be booting)."""
    import argparse
    return main()


if __name__ == "__main__":
    main()
