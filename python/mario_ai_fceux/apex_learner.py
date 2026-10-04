"""Ape-X GPU Learner process.

Responsibilities:
- Owns the in-memory SumTree PER buffer (250k–1M transitions).
- Receives transition *batches* from actors via experience_queue.
- Trains: Double DQN + Dueling + N-step + PER + C51.
- Broadcasts serialized weights every ``weight_sync_every`` optimizer updates.
- Writes checkpoints every ``checkpoint_every`` optimizer updates.
- Responds to status queries from the coordinator.

The learner NEVER blocks waiting for actors. Actors NEVER block waiting for
the learner beyond the backpressure of experience_queue.maxsize.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import signal
import time
from collections import deque
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, Deque

import numpy as np
import torch
from torch import nn

from .agent import AgentConfig, RainbowAgent
from .replay import PrioritizedReplayBuffer, Transition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight broadcast helpers
# ---------------------------------------------------------------------------

def _serialize_weights(network: nn.Module) -> bytes:
    """Serialize online network weights to bytes for multiprocessing broadcast."""
    buf = io.BytesIO()
    torch.save(network.state_dict(), buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Learner main entry point
# ---------------------------------------------------------------------------

def apex_learner_main(
    experience_queue: Queue,   # receives ("batch", actor_id, [Transition,...])
    weight_queues: list[Queue],  # one per actor + eval worker; push weight bytes
    status_inbox: Queue,         # receives ("status",) / ("stop",) / ("save",)
    status_outbox: Queue,        # sends ("status", {...})
    run_directory: str,
    config_dict: dict[str, Any],
    resume: bool,
    device_str: str | None,
    weight_sync_every: int = 500,
    checkpoint_every: int = 10_000,
    learn_per_batch: int = 4,    # optimizer steps per received batch; 16 overfit the buffer
) -> None:
    """Own replay, optimizer, and checkpoints. Broadcast weights asynchronously."""

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    directory = Path(run_directory)
    # replay_capacity is an Ape-X-level param; strip it before building AgentConfig.
    replay_capacity = int(config_dict.pop("replay_capacity", 500_000))
    fresh_replay = bool(config_dict.pop("fresh_replay", False))
    config = AgentConfig(**config_dict)
    replay_path = directory / "replay.npz"
    checkpoint_path = directory / "model.pt"

    if resume and checkpoint_path.exists():
        if fresh_replay:
            # Keep the weights, step count and optimizer state, but start a clean
            # buffer: transitions collected under earlier reward shaping would
            # otherwise keep teaching the policy the old value of a commitment.
            checkpoint_to_load = checkpoint_path
            replay = PrioritizedReplayBuffer(
                config.observation_size,
                capacity=max(replay_capacity, config.batch_size * 2),
                seed=config.seed,
            )
        else:
            checkpoint_to_load, replay = RainbowAgent.load_checkpoint_pair(
                checkpoint_path, replay_path, config.observation_size,
                config.seed, max(replay_capacity, config.batch_size * 2),
            )
    elif resume and replay_path.exists():
        checkpoint_to_load = checkpoint_path
        replay = PrioritizedReplayBuffer.load(replay_path, config.seed)
    else:
        checkpoint_to_load = checkpoint_path
        replay = PrioritizedReplayBuffer(
            config.observation_size,
            capacity=max(replay_capacity, config.batch_size * 2),
            seed=config.seed,
        )
    agent = RainbowAgent(replay, config=config, device=device_str)
    if resume and checkpoint_path.exists():
        try:
            agent.load(checkpoint_to_load, validate_replay=not fresh_replay)
            logger.info("Resumed from %s (step %d, opt %d)", checkpoint_to_load,
                        agent.steps, agent.optimizer_steps)
            if fresh_replay:
                logger.info("Fresh replay buffer: learned weights kept, buffer starts empty")
        except Exception as exc:
            raise RuntimeError(
                f"Could not safely resume {checkpoint_path}; refusing to discard learned state: {exc}"
            ) from exc
        if agent.action_space_migrated:
            # Persist the expanded action heads and remapped 12-frame replay
            # IDs immediately. A restart then loads the new schema directly.
            agent.save(checkpoint_path, replay_path)
            logger.info("Migrated legacy six-action checkpoint to %d movement-duration actions",
                        config.action_count)
        protected_from_previous = replay.protect_existing_successes()
        if protected_from_previous:
            logger.info("Protected %d surviving victory transitions from the prior replay",
                        protected_from_previous)

    last_loss: float | None = None
    last_weight_sync = 0
    last_checkpoint = agent.optimizer_steps
    active = True

    def _broadcast_weights() -> None:
        """Push current online weights to every actor and eval worker queue."""
        weight_bytes = _serialize_weights(agent.online)
        for wq in weight_queues:
            try:
                # Non-blocking: if the actor hasn't consumed the last update yet,
                # skip rather than stacking stale weights.
                while not wq.empty():
                    try:
                        wq.get_nowait()
                    except Exception:
                        break
                wq.put_nowait(("weights", weight_bytes))
            except Exception:
                pass  # Queue full or closed; actor will sync next cycle.

    # Broadcast initial weights immediately.
    _broadcast_weights()
    last_weight_sync = agent.optimizer_steps

    logger.info("Learner started on device %s", agent.device)

    while active:
        # --- Process one experience batch from any actor ---
        try:
            message = experience_queue.get(timeout=0.02)
        except queue.Empty:
            message = None

        if message is not None:
            kind = message[0]
            if kind == "batch":
                _, actor_id, transitions = message
                for t in transitions:
                    # n-step folding was done actor-side; add directly to replay.
                    replay.add(t)
                agent.steps += len(transitions)
                # Train on the newly added experience.
                for _ in range(learn_per_batch):
                    loss = agent.learn()
                    if loss is not None:
                        last_loss = loss
            elif kind == "success":
                _, actor_id, world, level, transitions = message
                for transition in transitions:
                    replay.add(transition, protect=True, level=(world, level),
                               kind=PrioritizedReplayBuffer.SUCCESS)
                logger.info("Protected %d successful transitions from actor %d course %d-%d",
                            len(transitions), actor_id, world + 1, level + 1)
            elif kind == "frontier":
                _, actor_id, world, level, transitions = message
                for transition in transitions:
                    replay.add(transition, protect=True, level=(world, level),
                               kind=PrioritizedReplayBuffer.FRONTIER)
                logger.info("Protected %d frontier recovery transitions from actor %d course %d-%d",
                            len(transitions), actor_id, world + 1, level + 1)
            elif kind == "contrast":
                _, actor_id, world, level, transitions = message
                for transition in transitions:
                    replay.add(transition, protect=True, level=(world, level),
                               kind=PrioritizedReplayBuffer.CONTRAST)
                logger.info("Protected %d contrast transitions from actor %d course %d-%d",
                            len(transitions), actor_id, world + 1, level + 1)

        # --- Check coordinator status messages (non-blocking) ---
        try:
            ctrl = status_inbox.get_nowait()
            ctrl_kind = ctrl[0]
            if ctrl_kind == "status":
                status_outbox.put(
                    (
                        "status",
                        {
                            "steps": agent.steps,
                            "optimizer_updates": agent.optimizer_steps,
                            "replay_transitions": len(replay),
                            "protected_success_transitions": replay.protected_count(
                                PrioritizedReplayBuffer.SUCCESS),
                            "protected_frontier_transitions": replay.protected_count(
                                PrioritizedReplayBuffer.FRONTIER),
                            "protected_contrast_transitions": replay.protected_count(
                                PrioritizedReplayBuffer.CONTRAST),
                            "transitions_received": agent.steps,
                            "epsilon": 0.0,
                            "latest_loss": last_loss,
                        },
                    )
                )
            elif ctrl_kind == "save":
                agent.save(checkpoint_path, replay_path)
                logger.info("Manual checkpoint saved at step %d", agent.steps)
            elif ctrl_kind == "stop":
                active = False
        except queue.Empty:
            pass

        # --- Periodic weight broadcast ---
        if agent.optimizer_steps - last_weight_sync >= weight_sync_every:
            _broadcast_weights()
            last_weight_sync = agent.optimizer_steps
            logger.debug("Weights broadcast at optimizer step %d", agent.optimizer_steps)

        # --- Periodic checkpoint ---
        if agent.optimizer_steps - last_checkpoint >= checkpoint_every:
            agent.save(checkpoint_path, replay_path)
            last_checkpoint = agent.optimizer_steps
            logger.info("Checkpoint at optimizer step %d (replay=%d)",
                        agent.optimizer_steps, len(replay))

    # Final save.
    agent.save(checkpoint_path, replay_path)
    logger.info("Learner stopped. Final checkpoint written.")
