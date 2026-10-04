"""Ape-X Rainbow coordinator: launches actors, learner, and eval worker.

Architecture:
    8 FCEUX Actors ──► multiprocessing.Queue (experience) ──► GPU Learner
                                                                    │
                                                         (weights broadcast)
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    ▼                                               ▼
             Actor 1..8 weight queues                   Eval worker weight queue

Key design decisions vs. old train.py:
    - Actors and learner run in *separate processes* (truly async).
    - Actors push *batches* (default 32) to reduce IPC overhead.
    - Learner owns ALL replay and optimizer state — no shared mutable objects.
    - SQLite is NOT touched in the training hot path.
    - Weights are broadcast every N optimizer steps, not every frame.
    - Queue has a hard capacity limit for backpressure (default 10,000 batches).
    - Actors use actor-specific epsilon-greedy; NoisyNet is disabled in actors.
    - Separate eval worker runs greedy episodes every 2 minutes.

Usage:
    python -m mario_ai_fceux.apex_train \\
        --rom SuperMarioBros.nes \\
        --workers 8 \\
        --device mps \\
        --run-dir runs/apex-rainbow \\
        [--resume] \\
        [--replay-capacity 500000]
"""

from __future__ import annotations

import argparse
import atexit
import configparser
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .actions import ACTION_COUNT, ACTION_DURATIONS
from .agent import AgentConfig
from .apex_actor import ActorConfig, actor_main, _apex_epsilon, training_epsilon
from .apex_eval import eval_worker_main
from .apex_learner import apex_learner_main
from .environment import START_PROTOCOL, FileWorker, launch_fceux_workers
from .protocol import atomic_write_json, read_json
from .train import read_lua_neat_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


HEALTH_CHECK_INTERVAL = 10 * 60   # seconds
WORKER_STALE_SECONDS = 3 * 60
# A healthy bridge publishes one observation per action (6/12/24 frames).
# This is deliberately much longer than a normal action so startup and macOS
# scheduling do not cause false positives, while still recovering a live but
# wedged FCEUX window instead of waiting forever.
OBSERVATION_STALE_SECONDS = 90


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ape-X Rainbow for Super Mario Bros — distributed async RL."
    )
    parser.add_argument("--rom", type=Path, required=True,
                        help="Path to a legally obtained SMB1 NES ROM.")
    parser.add_argument("--fceux", default="fceux",
                        help="FCEUX executable path or command.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/apex-rainbow"))
    parser.add_argument("--workers", type=int, default=8,
                        help="Training actor count (Ape-X recommendation: 8).")
    parser.add_argument("--worlds", default="1",
                        help="Comma-separated SMB1 worlds (one per worker).")
    parser.add_argument("--steps", type=int, default=5_000_000,
                        help="Total training steps target.")
    parser.add_argument("--resume", action="store_true",
                        help="Load model.pt and replay.npz if they exist.")
    parser.add_argument("--fresh-replay", action="store_true",
                        help="With --resume, keep the learned weights and optimizer state "
                             "but start an empty replay buffer (after changing reward shaping).")
    parser.add_argument("--device", default=None,
                        help="PyTorch device: mps, cuda, or cpu.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replay-capacity", type=int, default=500_000,
                        help="PER buffer size (250k–1M recommended).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Learner mini-batch size.")
    parser.add_argument("--actor-batch-size", type=int, default=32,
                        help="Transitions per actor queue push.")
    parser.add_argument("--weight-sync-every", type=int, default=500,
                        help="Broadcast weights every N learner optimizer steps.")
    parser.add_argument("--actor-weight-sync-every", type=int, default=400,
                        help="Actors reload weights every N collected steps.")
    parser.add_argument("--unsolved-epsilon-floor", type=float, default=0.10,
                        help="Minimum exploration probability before an actor wins a level.")
    parser.add_argument("--frontier-spacing", type=int, default=256,
                        help="Grounded progress pixels between retryable savestate frontiers.")
    parser.add_argument("--frontier-retries", type=int, default=3,
                        help="Attempts from a saved frontier before returning to level start.")
    parser.add_argument("--enemy-separation-bonus", type=float, default=0.0,
                        help="Reward for vertical separation from a close enemy (0 disables).")
    parser.add_argument("--stall-approach-penalty", type=float, default=0.0,
                        help="Charge for losing speed near a close enemy (0 disables).")
    parser.add_argument("--protect-contrast-pairs", action="store_true", default=False,
                        help="Protect pit-edge approach contexts (grounded, gap-ahead) from every "
                             "episode as a third quota-limited kind (default off).")
    parser.add_argument("--queue-capacity", type=int, default=10_000,
                        help="Max batches in experience queue (backpressure).")
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--learn-per-batch", type=int, default=4,
                        help="Optimizer updates per received experience batch. 16 was tried "
                             "on 2026-10-02 and reverted: conversion fell across two windows "
                             "(1.0%% -> 0.32%% -> 0%%) while training loss dropped to 0.92, "
                             "which is the buffer being fitted rather than the level learned.")
    parser.add_argument("--checkpoint-every", type=int, default=10_000,
                        help="Checkpoint every N optimizer steps.")
    parser.add_argument("--eval-every", type=float, default=120.0,
                        help="Seconds between greedy evaluation runs.")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Episodes per evaluation run.")
    parser.add_argument("--eval-world", type=int, default=1,
                        help="World for the greedy evaluation actor.")
    parser.add_argument("--disable-eval", action="store_true",
                        help="Do not launch the periodic greedy evaluator or its FCEUX window.")
    parser.add_argument("--window-layout", type=Path,
                        default=Path("config/fceux-window-layout.ini"),
                        help="INI file containing worker window rectangles.")
    parser.add_argument("--cheats-enabled-workers", default=None,
                        help="Override the layout INI: zero-based indexes loading the ROM .cht file.")
    parser.add_argument("--cheat-file", type=Path, default=Path("config/SuperMarioBros.cht"),
                        help="Optional FCEUX .cht file copied into enabled workers.")
    parser.add_argument("--alternate-cheat-campaigns", action="store_true",
                        help="For initially powered workers, alternate cheat mode after each World-N-4 win.")
    parser.add_argument("--repeat-level-on-victory", action="store_true",
                        help="Repeat the current level after each win for a mastery curriculum.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Health reporting (statistics only — no hot-path DB writes)
# ---------------------------------------------------------------------------

def write_health_report(
    run_dir: Path,
    repo_root: Path,
    learner_status: dict,
    actor_metrics: list[dict],
    eval_record: dict | None,
    expected_workers: int,
) -> None:
    """Write a health JSON + markdown snapshot every 10 minutes."""
    health_dir = run_dir / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "architecture": "Ape-X Rainbow (C51 + NoisyNet + Double + Dueling + PER + n-step)",
        "healthy": True,
        "repair_required": [],
        "learner": learner_status,
        "actors": actor_metrics,
        "expected_workers": expected_workers,
        "evaluation": eval_record,
        "disk": {
            "free_bytes": shutil.disk_usage(repo_root).free,
            "total_bytes": shutil.disk_usage(repo_root).total,
        },
    }
    lua_log = repo_root / "mario_ai_neat.log"
    lua_database = repo_root / "mario_ai_neat.db"
    lua_age = (round(max(0.0, time.time() - lua_log.stat().st_mtime), 1)
               if lua_log.exists() else None)
    lua_summary = read_lua_neat_summary(lua_database, lua_log)
    report["lua_neat"] = {
        "state": "active" if lua_age is not None and lua_age <= HEALTH_CHECK_INTERVAL else "stale_or_not_detected",
        "log_age_seconds": lua_age,
        **lua_summary,
    }
    actor_best_x = max((int(item.get("best_episode_x", 0)) for item in actor_metrics), default=0)
    actor_victories = sum(int(item.get("victories", 0)) for item in actor_metrics)
    report["measured_progress"] = {
        "python_transitions": learner_status.get("transitions_received", 0),
        "python_optimizer_updates": learner_status.get("optimizer_updates", 0),
        "python_replay_size": learner_status.get("replay_transitions", 0),
        "python_best_episode_x": actor_best_x,
        "python_victories": actor_victories,
    }
    atomic_write_json(health_dir / "latest.json", report)
    with (health_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report, separators=(",", ":")) + "\n")
    # Markdown summary.
    rows = [
        "# Ape-X Rainbow — Health Report",
        "",
        f"Generated: {report['checked_at']}",
        "",
        "## Learner",
        f"- Steps: {learner_status.get('steps', '?')}",
        f"- Optimizer updates: {learner_status.get('optimizer_updates', '?')}",
        f"- Replay size: {learner_status.get('replay_transitions', '?')}",
        f"- Latest loss: {learner_status.get('latest_loss', '?')}",
        "",
        "## Actors",
    ]
    for m in actor_metrics:
        rows.append(
            f"- Actor {m.get('actor', '?')}: ε={m.get('epsilon', '?')} "
            f"steps={m.get('steps', 0)} episodes={m.get('episodes', 0)} "
            f"victories={m.get('victories', 0)}"
        )
    if eval_record:
        rows += [
            "",
            "## Evaluation (greedy, frozen weights)",
            f"- Win rate: {eval_record.get('win_rate', '?')}",
            f"- Avg X: {eval_record.get('avg_max_x', '?')}",
            f"- Episodes: {eval_record.get('episodes', '?')}",
        ]
    rows += [
        "",
        "## Learning comparison",
        "| System | Latest evidence |",
        "| --- | --- |",
        (f"| Python Ape-X Rainbow | {learner_status.get('transitions_received', 0)} transitions; "
         f"{learner_status.get('optimizer_updates', 0)} updates; replay "
         f"{learner_status.get('replay_transitions', 0)}; best actor X {actor_best_x}; "
         f"victories {actor_victories} |"),
        (f"| Lua NEAT | generation {lua_summary.get('generation')}; "
         f"best fitness {lua_summary.get('best_fitness')}; "
         f"latest X {lua_summary.get('latest_max_x')}; log {report['lua_neat']['state']} |"),
        "",
        "Movement skill is not called learned until clean evaluation repeats it reliably.",
    ]
    (health_dir / "report.md").write_text("\n".join(rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

def worker_cheat_modes(layout_path: Path, worker_count: int,
                       override: str | None = None) -> tuple[bool, ...]:
    """Use the saved window layout's cheat assignment unless CLI overrides it."""
    if override is not None:
        indexes = {int(value.strip()) for value in override.split(",") if value.strip()}
        if any(index < 0 or index >= worker_count for index in indexes):
            raise ValueError("--cheats-enabled-workers indexes must be within the worker count")
        return tuple(index in indexes for index in range(worker_count))
    layout = configparser.ConfigParser()
    if layout.read(layout_path):
        modes = []
        for index in range(worker_count):
            section = f"worker-{index:02d}"
            if not layout.has_option(section, "cheats"):
                raise ValueError(f"{layout_path} lacks cheats= for {section}")
            value = layout.get(section, "cheats").strip().lower()
            if value not in ("enabled", "disabled"):
                raise ValueError(f"{layout_path} has invalid cheats= for {section}")
            modes.append(value == "enabled")
        return tuple(modes)
    return tuple(index < min(worker_count, 4) for index in range(worker_count))


def main() -> None:
    args = parse_arguments()
    if not 0.0 <= args.unsolved_epsilon_floor <= 1.0:
        raise ValueError("--unsolved-epsilon-floor must be in [0, 1]")
    if args.frontier_spacing < 1:
        raise ValueError("--frontier-spacing must be positive")
    if args.frontier_retries < 0:
        raise ValueError("--frontier-retries must be non-negative")
    worker_cheats = worker_cheat_modes(args.window_layout, args.workers,
                                      args.cheats_enabled_workers)
    if any(worker_cheats) and not args.cheat_file.is_file():
        raise FileNotFoundError(f"enabled workers need a cheat file: {args.cheat_file}")
    try:
        requested_worlds = tuple(int(v.strip()) for v in args.worlds.split(",") if v.strip())
    except ValueError as exc:
        raise ValueError("--worlds must be integers in 1..8") from exc
    if len(requested_worlds) == 1:
        requested_worlds *= args.workers
    if len(requested_worlds) != args.workers or any(w < 1 or w > 8 for w in requested_worlds):
        raise ValueError("--worlds must provide one value in 1..8 per training worker")

    repository_root = Path(__file__).resolve().parents[2]
    bridge_template = repository_root / "python" / "fceux_bridge" / "mario_ai_fceux_bridge.lua"
    args.run_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "unknown"

    # Metadata snapshot (statistics use — not training hot path).
    metadata = {
        "algorithm": "Ape-X Rainbow (C51 + NoisyNet + Double DQN + Dueling + PER + N-step)",
        "architecture": "distributed_async",
        "training_actors": args.workers,
        "eval_actor": 1,
        "evaluation_enabled": not args.disable_eval,
        "worlds": requested_worlds,
        "cheats_enabled_workers": [index for index, enabled in enumerate(worker_cheats) if enabled],
        "alternate_cheat_campaigns": bool(args.alternate_cheat_campaigns),
        "repeat_level_on_victory": bool(args.repeat_level_on_victory),
        "window_layout": str(args.window_layout),
        "replay_capacity": args.replay_capacity,
        "exploration": {
            "actors": "Ape-X epsilon-greedy only; actor networks run in eval mode",
            "learner": "NoisyNet remains active in Rainbow learner updates",
            "action_priors": "none",
        },
        "actor_epsilons": [round(training_epsilon(i, args.workers, 0,
                                                   args.unsolved_epsilon_floor), 5)
                           for i in range(args.workers)],
        "unsolved_epsilon_floor": args.unsolved_epsilon_floor,
        "exploration_schedule": "Per-level epsilon floor until that actor wins the level",
        "frontier_curriculum": {
            "strategy": "grounded savestate checkpoints with bounded retries",
            "spacing_pixels": args.frontier_spacing,
            "retries": args.frontier_retries,
        },
        "enemy_separation_bonus": args.enemy_separation_bonus,
        "stall_approach_penalty": args.stall_approach_penalty,
        "actor_seeds": [args.seed + i * 1000 for i in range(args.workers)],
        "experience_queue_max_batches": args.queue_capacity,
        "actor_batch_size": args.actor_batch_size,
        "weight_sync_every_optimizer_steps": args.weight_sync_every,
        "n_step": args.n_step,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "evaluation_seed": args.seed,
        "eval_world": args.eval_world,
        "eval_episodes": args.eval_episodes,
        "action_repeat_frames": list(ACTION_DURATIONS),
        "action_count": ACTION_COUNT,
        "action_policy": "learned movement x 6/12/24-frame horizon; includes backward jump",
        "start_protocol": START_PROTOCOL,
        "determinism": "seeded components; asynchronous queue interleaving is not bitwise reproducible",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "source_revision": source_revision,
        "rom_sha256": hashlib.sha256(args.rom.read_bytes()).hexdigest(),
    }
    fceux_executable = shutil.which(args.fceux) or args.fceux
    metadata["fceux_executable"] = str(Path(fceux_executable).resolve())
    try:
        metadata["fceux_sha256"] = hashlib.sha256(Path(fceux_executable).resolve().read_bytes()).hexdigest()
    except OSError:
        metadata["fceux_sha256"] = None
    (args.run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    atomic_write_json(args.run_dir / "learner_status.json", {
        "steps": 0, "transitions_received": 0, "optimizer_updates": 0,
        "replay_transitions": 0, "epsilon": 0.0, "latest_loss": None,
    })
    logger.info("Run config written: %s", args.run_dir / "run.json")

    context = multiprocessing.get_context("spawn")

    # One shared bounded experience queue; full queues pause actors instead
    # of dropping transitions. The default capacity is 10,000 batches.
    experience_queue: multiprocessing.Queue = context.Queue(maxsize=args.queue_capacity)

    # One weight queue per actor, plus an optional evaluator queue.
    total_weight_queues = args.workers + (0 if args.disable_eval else 1)
    weight_queues: list[multiprocessing.Queue] = [
        context.Queue(maxsize=2) for _ in range(total_weight_queues)
    ]
    eval_weight_queue = None if args.disable_eval else weight_queues[-1]

    # Status channel (learner health queries).
    status_inbox: multiprocessing.Queue = context.Queue(maxsize=10)
    status_outbox: multiprocessing.Queue = context.Queue(maxsize=10)

    # ---- Build AgentConfig (learner-side).
    agent_config = AgentConfig(
        observation_size=184,
        action_count=ACTION_COUNT,
        gamma=0.99,
        learning_rate=1.25e-4,
        batch_size=args.batch_size,
        learning_starts=10_000,
        target_sync_steps=2_000,
        n_step=args.n_step,
        atom_count=51,
        value_min=-250.0,
        value_max=250.0,
        per_beta_start=0.4,
        per_beta_steps=1_000_000,
        seed=args.seed,
    )
    learner_config = {**vars(agent_config), "replay_capacity": args.replay_capacity,
                      "fresh_replay": args.fresh_replay}

    # ---- Build ActorConfig (shared template; index injected at launch).
    actor_config = ActorConfig(
        observation_size=184,
        action_count=ACTION_COUNT,
        gamma=0.99,
        n_step=args.n_step,
        batch_size=args.actor_batch_size,
        atom_count=51,
        value_min=-250.0,
        value_max=250.0,
        weight_sync_every=args.actor_weight_sync_every,
        unsolved_epsilon_floor=args.unsolved_epsilon_floor,
        frontier_spacing=args.frontier_spacing,
        frontier_retries=args.frontier_retries,
        enemy_separation_bonus=args.enemy_separation_bonus,
        stall_approach_penalty=args.stall_approach_penalty,
        repeat_level_on_victory=args.repeat_level_on_victory,
        protect_contrast_pairs=args.protect_contrast_pairs,
        seed=args.seed,
    )

    # Track direct emulator processes immediately so a failed later startup
    # stage cannot leave orphan FCEUX windows behind.
    fceux_processes = []
    eval_fceux = []

    def _stop_emulators() -> None:
        for process in fceux_processes + eval_fceux:
            if process.poll() is None:
                process.terminate()
        for process in fceux_processes + eval_fceux:
            try:
                process.wait(timeout=5)
            except Exception:
                if process.poll() is None:
                    process.kill()

    atexit.register(_stop_emulators)

    # ---- Launch FCEUX processes (training actors).
    fceux_processes.extend(launch_fceux_workers(
        args.fceux, args.rom, bridge_template,
        args.run_dir, args.workers, requested_worlds, action_profile="rainbow",
        cheats_enabled=worker_cheats,
        window_layout=args.window_layout,
        cheat_file=args.cheat_file,
    ))
    training_workers = [
        FileWorker(f"actor-{i:02d}", args.run_dir / f"worker-{i:02d}",
                   action_profile="rainbow")
        for i in range(args.workers)
    ]

    # ---- Optionally launch a separate FCEUX process for greedy evaluation.
    eval_run_dir = args.run_dir / "eval"
    eval_file_worker: FileWorker | None = None
    if not args.disable_eval:
        eval_run_dir.mkdir(parents=True, exist_ok=True)
        eval_fceux.extend(launch_fceux_workers(
            args.fceux, args.rom, bridge_template,
            eval_run_dir, 1, (args.eval_world,), action_profile="rainbow",
            cheats_enabled=(False,), window_layout=None, cheat_file=None,
            extra_args=("--xscale", "1", "--yscale", "1", "-qwindowgeometry", "512x469+851+205"),
        ))
        eval_file_worker = FileWorker("eval", eval_run_dir / "worker-00",
                                      action_profile="rainbow")
    worker_launched_at = [time.time()] * args.workers
    eval_launched_at = time.time() if not args.disable_eval else 0.0

    # ---- Launch learner process.
    learner_process = context.Process(
        target=apex_learner_main,
        args=(experience_queue, weight_queues, status_inbox, status_outbox,
              str(args.run_dir), learner_config, args.resume, args.device,
              args.weight_sync_every, args.checkpoint_every, args.learn_per_batch),
        daemon=True,
    )
    learner_process.start()
    logger.info("Learner process started (PID %d)", learner_process.pid)

    # ---- Launch actor processes.
    actor_processes = []
    for i, worker in enumerate(training_workers):
        actor_config_dict = {**vars(actor_config), "alternate_cheat_campaigns": bool(
            args.alternate_cheat_campaigns and worker_cheats[i]
        )}
        p = context.Process(
            target=actor_main,
            args=(i, args.workers, worker, experience_queue, weight_queues[i],
                  str(args.run_dir), actor_config_dict, args.device),
            daemon=True,
        )
        p.start()
        actor_processes.append(p)
        logger.info("Actor %d started (PID %d, ε=%.4f)", i, p.pid,
                    _apex_epsilon(i, args.workers))

    # ---- Launch eval worker process only when periodic evaluation is enabled.
    eval_process: multiprocessing.Process | None = None
    if not args.disable_eval:
        assert eval_file_worker is not None and eval_weight_queue is not None
        eval_process = context.Process(
            target=eval_worker_main,
            args=(eval_file_worker, eval_weight_queue, str(args.run_dir),
                  agent_config.observation_size, agent_config.action_count,
                  agent_config.atom_count, agent_config.value_min, agent_config.value_max,
                  args.eval_every, args.eval_episodes, 300.0, args.device),
            daemon=True,
        )
        eval_process.start()
        logger.info("Eval worker started (PID %d)", eval_process.pid)
    else:
        logger.info("Periodic evaluator disabled; only training workers will launch")

    monitor_path = args.run_dir / "supervisor.json"

    def write_supervisor_heartbeat(running: bool) -> None:
        atomic_write_json(monitor_path, {
            "running": running,
            "heartbeat_unix": time.time(),
            "trainer_pid": os.getpid(),
            "actor_pids": [process.pid for process in actor_processes],
            "emulator_pids": [process.pid for process in fceux_processes],
            "eval_pid": eval_process.pid if eval_process is not None else -1,
            "eval_emulator_pids": [process.pid for process in eval_fceux],
        })

    write_supervisor_heartbeat(True)

    # ---- Coordinator loop (health monitoring only — no training logic here).
    active = True

    def stop(*_: object) -> None:
        nonlocal active
        active = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    next_health_at = time.monotonic()
    next_status_at = time.monotonic()
    next_heartbeat_at = time.monotonic() + 5.0
    learner_status: dict = {"steps": 0, "optimizer_updates": 0,
                            "replay_transitions": 0, "epsilon": 0.0, "latest_loss": None}
    active_cheat_modes = list(worker_cheats)

    def restart_requested_worker(index: int, cheats_enabled: bool) -> None:
        """Restart one campaign worker so FCEUX reloads its private cheat config."""
        previous = fceux_processes[index]
        if previous.poll() is None:
            previous.terminate()
            try:
                previous.wait(timeout=5)
            except subprocess.TimeoutExpired:
                previous.kill()
        replacement = launch_fceux_workers(
            args.fceux, args.rom, bridge_template, args.run_dir, 1,
            (requested_worlds[index],), action_profile="rainbow",
            cheats_enabled=(cheats_enabled,), window_layout=args.window_layout,
            cheat_file=args.cheat_file, worker_indexes=(index,),
        )[0]
        fceux_processes[index] = replacement
        worker_launched_at[index] = time.time()
        active_cheat_modes[index] = cheats_enabled
        logger.info("Worker %d restarted for campaign cheat mode: %s", index,
                    "enabled" if cheats_enabled else "disabled")

    def restart_stale_emulator(index: int) -> None:
        """Replace a live FCEUX process whose bridge stopped publishing."""
        previous = fceux_processes[index]
        if previous.poll() is None:
            previous.terminate()
            try:
                previous.wait(timeout=5)
            except subprocess.TimeoutExpired:
                previous.kill()
        replacement = launch_fceux_workers(
            args.fceux, args.rom, bridge_template, args.run_dir, 1,
            (requested_worlds[index],), action_profile="rainbow",
            cheats_enabled=(active_cheat_modes[index],),
            window_layout=args.window_layout, cheat_file=args.cheat_file,
            worker_indexes=(index,),
        )[0]
        fceux_processes[index] = replacement
        worker_launched_at[index] = time.time()
        logger.warning("Restarted stale Python/FCEUX worker %d after %.0fs without observations",
                       index, OBSERVATION_STALE_SECONDS)

    def restart_stale_eval_emulator() -> None:
        """Replace a wedged evaluator emulator without restarting training."""
        nonlocal eval_launched_at
        previous = eval_fceux[0]
        if previous.poll() is None:
            previous.terminate()
            try:
                previous.wait(timeout=5)
            except subprocess.TimeoutExpired:
                previous.kill()
        eval_fceux[0] = launch_fceux_workers(
            args.fceux, args.rom, bridge_template, eval_run_dir, 1,
            (args.eval_world,), action_profile="rainbow",
            cheats_enabled=(False,), window_layout=None, cheat_file=None,
            extra_args=("--xscale", "1", "--yscale", "1",
                         "-qwindowgeometry", "512x469+851+205"),
        )[0]
        eval_launched_at = time.time()
        logger.warning("Restarted stale greedy evaluator emulator")

    try:
        while active:
            now = time.monotonic()
            if args.alternate_cheat_campaigns:
                for index in range(args.workers):
                    request = read_json(args.run_dir / f"worker-{index:02d}" / "mode_request.json")
                    if not isinstance(request, dict) or "cheats_enabled" not in request:
                        continue
                    requested_mode = bool(request["cheats_enabled"])
                    if requested_mode != active_cheat_modes[index]:
                        restart_requested_worker(index, requested_mode)
            if now >= next_heartbeat_at:
                write_supervisor_heartbeat(True)
                next_heartbeat_at = now + 5.0
            if now >= next_status_at:
                if not learner_process.is_alive():
                    raise RuntimeError("Rainbow learner exited; see the learner log before resuming")
                dead_actors = [index for index, process in enumerate(actor_processes)
                               if not process.is_alive()]
                if dead_actors:
                    raise RuntimeError(f"Ape-X actor process(es) exited: {dead_actors}")
                dead_emulators = [index for index, process in enumerate(fceux_processes)
                               if process.poll() is not None]
                if dead_emulators:
                    raise RuntimeError(f"FCEUX training emulator(s) exited: {dead_emulators}")
                stale_workers = []
                for index in range(args.workers):
                    observation_path = args.run_dir / f"worker-{index:02d}" / "observation.json"
                    last_observation = (observation_path.stat().st_mtime
                                        if observation_path.exists() else worker_launched_at[index])
                    if time.time() - last_observation > OBSERVATION_STALE_SECONDS:
                        stale_workers.append(index)
                for index in stale_workers:
                    restart_stale_emulator(index)
                if eval_process is None:
                    pass
                elif not eval_process.is_alive():
                    logger.warning("Evaluation process exited; training continues without current eval data")
                else:
                    eval_observation = eval_run_dir / "worker-00" / "observation.json"
                    last_eval_observation = (eval_observation.stat().st_mtime
                                             if eval_observation.exists() else eval_launched_at)
                    if time.time() - last_eval_observation > OBSERVATION_STALE_SECONDS:
                        restart_stale_eval_emulator()
                try:
                    status_inbox.put_nowait(("status",))
                    _, learner_status = status_outbox.get(timeout=3)
                    atomic_write_json(args.run_dir / "learner_status.json", learner_status)
                except Exception as exc:
                    logger.warning("Learner status query failed: %s", exc)
                if learner_status.get("transitions_received", 0) >= args.steps:
                    logger.info("Reached requested experience target of %d transitions", args.steps)
                    active = False
                next_status_at = now + 10.0
            if now >= next_health_at:
                # Read actor metrics from per-actor JSON files.
                actor_metrics = []
                for i in range(args.workers):
                    metrics_path = args.run_dir / f"actor_{i:02d}_metrics.json"
                    try:
                        import json as _json
                        actor_metrics.append(_json.loads(metrics_path.read_text()))
                    except Exception:
                        actor_metrics.append({"actor": i, "steps": 0})

                # Read latest eval result.
                eval_record: dict | None = None
                eval_latest = args.run_dir / "eval_latest.json"
                try:
                    import json as _json
                    eval_record = _json.loads(eval_latest.read_text())
                except Exception:
                    pass

                write_health_report(
                    args.run_dir, repository_root,
                    learner_status, actor_metrics, eval_record, args.workers,
                )
                logger.info(
                    "Health | learner steps=%d optimizer=%d replay=%d loss=%s",
                    learner_status.get("steps", 0),
                    learner_status.get("optimizer_updates", 0),
                    learner_status.get("replay_transitions", 0),
                    f"{learner_status.get('latest_loss', 0):.5f}"
                    if learner_status.get("latest_loss") else "warming up",
                )
                next_health_at = now + HEALTH_CHECK_INTERVAL

            time.sleep(5.0)

    finally:
        logger.info("Shutting down...")
        status_inbox.put(("stop",))
        learner_process.join(timeout=90)
        if learner_process.is_alive():
            learner_process.kill()
        for p in actor_processes + ([eval_process] if eval_process is not None else []):
            if p.is_alive():
                p.terminate()
        for p in actor_processes + ([eval_process] if eval_process is not None else []):
            try:
                p.join(timeout=5)
            except Exception:
                p.kill()
        for proc in fceux_processes + eval_fceux:
            if proc.poll() is None:
                proc.terminate()
        for proc in fceux_processes + eval_fceux:
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        write_supervisor_heartbeat(False)
        logger.info("All processes stopped.")


if __name__ == "__main__":
    main()
