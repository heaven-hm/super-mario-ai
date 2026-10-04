"""Ape-X actor: runs FCEUX, collects experience batches, sends to shared queue.

Each actor plays independently with its own epsilon (Ape-X exploration schedule).
N-step returns are computed per-actor so trajectories are never mixed.
Actors send transition *batches* (default 32) to reduce IPC overhead.
Model weights are received from the learner and loaded periodically.
"""

from __future__ import annotations

import io
import logging
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, replace
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Deque

import numpy as np
import torch

from .actions import (ACTION_COUNT, LEGACY_DURATION_FRAMES, decode_action,
                      greedy_action, pit_edge_commit, safe_start_action,
                      STATE_GAP_AHEAD, STATE_GROUNDED)
from .environment import FileWorker, NoProgressTracker, Observation
from .model import RainbowNetwork
from .protocol import atomic_write_json, read_json
from .replay import Transition

logger = logging.getLogger(__name__)

# A death must outweigh several ordinary progress rewards.  The old -5
# terminal penalty was only a few maximum-sized (+2) progress decisions, so
# replay taught the policy that repeatedly reaching a dangerous state was
# still worthwhile.  Keep victory separate and strongly positive.
# 60 was tried on 2026-10-02 and reverted the same day: with the frontier
# curriculum restoring deaths onto the frontier, the -60 shocks flooded replay
# with catastrophe transitions and the clean-start win rate collapsed to 0/20
# from 3-5/20.  20 is the measured-good value.
DEATH_REWARD_PENALTY = 20.0
VICTORY_REWARD_BONUS = 20.0

# Progress is capped just above the furthest a committed action can travel: 24
# frames at SMB1's ~2.5 px/frame top running speed is 60 px, or 3.75 reward.
# The old 2.0 cap paid a 24-frame commitment half the return per frame that a
# 6-frame tap earned, so replay rewarded tapping over building speed and the
# greedy policy learned to tap exactly where successful trajectories committed.
MAX_PROGRESS_REWARD = 4.0

# Temporally correlated exploration: while exploring, reuse the previous action
# with this probability instead of resampling.  Independent random actions
# cancel out in a momentum game like SMB1 (run, brake, jump-back, run), while
# short random *sequences* stay coherent enough to cross a gap or land a stomp,
# which is exactly what exploration has to discover.
EXPLORATION_STICKINESS = 0.25

# Ape-X exploration schedule: actor 0 explores most, actor 7 exploits most.
# Matches the Ape-X paper's per-actor epsilon annealing philosophy.
APEX_EPSILONS = (0.40, 0.20, 0.10, 0.05, 0.025, 0.012, 0.006, 0.003)
UNSOLVED_WORLD_EPSILON_FLOOR = 0.10
FRONTIER_SPACING_PIXELS = 256
FRONTIER_RETRIES = 3


def _apex_epsilon(actor_index: int, total_actors: int) -> float:
    """Return actor-specific epsilon following the Ape-X schedule."""
    if total_actors <= 1:
        return 0.10
    fraction = actor_index / max(1, total_actors - 1)
    # Interpolate log-linearly between max and min epsilon.
    log_max = np.log(APEX_EPSILONS[0])
    log_min = np.log(APEX_EPSILONS[-1])
    return float(np.exp(log_max + fraction * (log_min - log_max)))


def training_epsilon(actor_index: int, total_actors: int, victories: int,
                     unsolved_floor: float = UNSOLVED_WORLD_EPSILON_FLOOR) -> float:
    """Give an unsolved assigned level meaningful exploration in the exploring head.

    Ape-X only works while its population spans exploration to exploitation.  A
    floor applied to every actor while the assigned level is unsolved removes the
    exploit end entirely, so the learner never sees the state distribution its
    own greedy policy produces and the argmax drifts away from the trajectories
    that actually earned reward.  Keep the floor on the exploring head of the
    population and let the rest follow the Ape-X schedule.
    """
    base = _apex_epsilon(actor_index, total_actors)
    if victories:
        return base
    exploring_head = max(1, total_actors // 3)
    return max(base, unsolved_floor) if actor_index < exploring_head else base


@dataclass
class ActorConfig:
    observation_size: int = 184
    action_count: int = ACTION_COUNT
    gamma: float = 0.99
    n_step: int = 3
    batch_size: int = 32          # transitions per queue push
    atom_count: int = 51
    # Must match the learner's support exactly: the actor reads its Q-values off
    # the same atom layout, so a narrower support here would misprice actions.
    value_min: float = -250.0
    value_max: float = 250.0
    weight_sync_every: int = 400  # steps between weight pulls
    seed: int = 7
    alternate_cheat_campaigns: bool = False
    repeat_level_on_victory: bool = False
    unsolved_epsilon_floor: float = UNSOLVED_WORLD_EPSILON_FLOOR
    frontier_spacing: int = FRONTIER_SPACING_PIXELS
    frontier_retries: int = FRONTIER_RETRIES
    # At a frontier death, the episode tail holds the recovery decisions the
    # agent keeps getting wrong; protect them so PER cannot evict them as low
    # priority. The learner bounds these per level.
    frontier_tail_transitions: int = 64
    # Approach-discrimination contrast pairs (default-off lever): at episode
    # end, protect the pit-edge decision contexts (grounded, gap-ahead) from
    # BOTH winning and losing episodes as their own quota-limited kind. The
    # two enemy-approach reward levers failed by teaching one approach
    # behaviour generalized wrongly to pits; rehearsing pit-approach contexts
    # as a distinct class (with their own sub-quota, never into the general
    # protected set) is the training-signal version of that discrimination.
    protect_contrast_pairs: bool = False
    contrast_tail_transitions: int = 16
    # Enemy-clearance shaping, off by default so existing runs and checkpoints
    # are unaffected. When enabled it pays a small bonus for being vertically
    # separated from a nearby enemy ahead -- the stomp/jump-over mechanic the
    # x~1780 wall trace shows the policy failing (approach, stall, contact death).
    # Capped at the 6-frame time cost so parking above an enemy cannot farm it.
    enemy_separation_bonus: float = 0.0

    # Stall-on-approach discipline: the mirror experiment to enemy_separation_bonus.
    # That pays for the escape (vertical separation); this charges the failure mode
    # feeding the contact, which the x~1780 wall traces show as speed_x collapsing
    # toward 0 with an enemy in the contact zone before the death. Capped at the
    # 6-frame time cost for symmetry with the separation bonus.
    #
    # MEASURED HARM (2026-10-03): at 0.02 for 50,922 steps it REGRESSED the clean
    # wall - 0/10 median max_x 1454 versus 2/20 median 2757 for the identical
    # pre-change pin at matched exposure (eval 20261003-105938 vs 20261003-102858).
    # Hypothesis (unverified): it teaches never-lose-speed, so the policy stopped
    # braking for pit approaches and died mid-air at the early pit. Keep at 0.0
    # unless a redesign targets the approach *timing* rather than speed loss;
    # any future trial must be a twin (same pin, lever on/off) per the project
    # lesson on window attribution.
    stall_approach_penalty: float = 0.0


class NStepBuffer:
    """Per-worker n-step return accumulator. Trajectories are never mixed."""

    def __init__(self, gamma: float, n_step: int) -> None:
        self.gamma = gamma
        self.n_step = n_step
        self.pending: Deque[Transition] = deque()

    def push(self, transition: Transition) -> list[Transition]:
        """Add one step; return ready n-step transitions."""
        self.pending.append(transition)
        return self._drain(force=transition.terminated)

    def flush(self) -> list[Transition]:
        """Force-drain remaining steps at episode end."""
        return self._drain(force=True)

    def _drain(self, force: bool) -> list[Transition]:
        ready: list[Transition] = []
        while self.pending and (force or len(self.pending) >= self.n_step):
            reward, discount, terminal = 0.0, 1.0, False
            next_state = self.pending[0].next_state
            for step in list(self.pending)[: self.n_step]:
                reward += discount * step.reward
                # Each decision carries its own duration-adjusted discount.
                discount *= step.discount
                next_state, terminal = step.next_state, step.terminated
                if terminal:
                    break
            first = self.pending.popleft()
            ready.append(
                Transition(
                    first.state,
                    first.action,
                    reward,
                    next_state,
                    terminal,
                    0.0 if terminal else discount,
                    priority=1.0,  # learner assigns real priority after TD-error
                )
            )
            if not force and len(self.pending) < self.n_step:
                break
        return ready


def _select_action(
    network: RainbowNetwork,
    support: torch.Tensor,
    state: np.ndarray,
    epsilon: float,
    action_count: int,
    device: torch.device,
    previous_action: int | None = None,
    stickiness: float = EXPLORATION_STICKINESS,
) -> int:
    """Epsilon-greedy action selection using the actor's local network copy."""
    action, _, _ = _action_details(network, support, state, epsilon,
                                  action_count, device,
                                  previous_action=previous_action,
                                  stickiness=stickiness)
    return action


def _action_details(
    network: RainbowNetwork,
    support: torch.Tensor,
    state: np.ndarray,
    epsilon: float,
    action_count: int,
    device: torch.device,
    safe_start: bool = False,
    previous_action: int | None = None,
    stickiness: float = EXPLORATION_STICKINESS,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return action, Q values, and encoder summary for the live FCEUX HUD."""
    # Actor policy values use learned mean weights; exploration comes only
    # from the Ape-X epsilon schedule. NoisyNet stays active in the learner.
    network.eval()
    obs = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        encoded = network.encoder(obs)
        q_values = network(obs, support)
    greedy_index = (greedy_action(q_values[0].cpu().numpy())
                    if action_count == ACTION_COUNT
                    else int(q_values.argmax(dim=1).item()))
    if np.random.random() < epsilon:
        # Sticky exploration keeps random steps in short coherent sequences.
        action = (previous_action if previous_action is not None
                  and np.random.random() < stickiness
                  else int(np.random.randint(action_count)))
    else:
        action = greedy_index
    if safe_start and action_count == ACTION_COUNT:
        # Do not let either an unstable Q estimate or exploratory noise turn
        # Mario around before the level's first hazard.
        action = safe_start_action(q_values[0].cpu().numpy())
    if (action_count == ACTION_COUNT and epsilon == 0.0
            and bool(state[171] > 0) and bool(state[182])):
        # Greedy play at a gap edge: brake/roll ties park Mario on the edge
        # (measured 2370 median depth). Prefer a committed near-tied jump.
        action = pit_edge_commit(q_values[0].cpu().numpy(), action)
    hidden_summary = encoded.reshape(-1, 16, 16).mean(dim=2)
    return action, q_values[0].cpu().numpy(), hidden_summary[0].cpu().numpy()


def _load_weights_from_bytes(network: RainbowNetwork, weight_bytes: bytes,
                              device: torch.device, evaluation: bool = False) -> None:
    """Deserialise a weight snapshot broadcast by the learner."""
    buf = io.BytesIO(weight_bytes)
    state_dict = torch.load(buf, map_location=device, weights_only=True)
    network.load_state_dict(state_dict)
    network.train(mode=not evaluation)


def _enqueue_batch(experience_queue: Queue, actor_index: int,
                   transitions: list[Transition]) -> None:
    """Send a complete batch, waiting for learner capacity instead of dropping it."""
    if transitions:
        experience_queue.put(("batch", actor_index, list(transitions)))


def _terminal_transition(previous: Observation, current: Observation,
                         action: int,
                         enemy_separation_bonus: float = 0.0,
                         stall_approach_penalty: float = 0.0) -> Transition:
    """Create the final state-action-reward record for death or level completion."""
    return Transition(previous.state, action,
                      _shaped_reward(previous, current,
                                     enemy_separation_bonus=enemy_separation_bonus,
                                     stall_approach_penalty=stall_approach_penalty),
                      current.state, True, 0.0)


def _publish_hud(worker: FileWorker, run_directory: str, observation: Observation,
                 action: int, q_values: np.ndarray, hidden: np.ndarray,
                 steps: int, episodes: int, deaths: int, victories: int,
                 best_x: int, epsilon: float) -> None:
    """Publish the actor's real state/action values for the FCEUX overlay."""
    learner = read_json(Path(run_directory) / "learner_status.json") or {}
    atomic_write_json(worker.directory / "hud.json", {
        "sequence": observation.sequence,
        "steps": steps,
        "updates": int(learner.get("optimizer_updates", 0)),
        "replay": int(learner.get("replay_transitions", 0)),
        "epsilon": epsilon,
        "episodes": episodes,
        "deaths": deaths,
        "victories": victories,
        "best_x": best_x,
        "loss": float(learner.get("latest_loss") or 0.0),
        "action": action,
        "values": [float(value) for value in q_values],
        "grid": [float(value) for value in observation.state[:169]],
        "globals": [float(value) for value in observation.state[169:184]],
        "hidden": [float(value) for value in hidden],
    })


def contrast_exemplars(transitions):
    """Pit-edge approach contexts: grounded with a gap ahead.

    The contrast-pair classifier. These are the decision contexts where the
    policy must discriminate pit approach from ordinary movement - the class
    both enemy-approach reward levers generalized wrongly across. Returned in
    trajectory order; callers quota them as their own protected kind.
    """
    return [t for t in transitions
            if len(t.state) > STATE_GAP_AHEAD
            and float(t.state[STATE_GAP_AHEAD]) > 0
            and float(t.state[STATE_GROUNDED]) > 0]


def actor_main(
    actor_index: int,
    total_actors: int,
    worker: FileWorker,
    experience_queue: Queue,          # send batches of Transition to learner
    weight_queue: Queue,              # receive serialized weights from learner
    run_directory: str,
    config_dict: dict,
    device_str: str | None,
) -> None:
    """Entry point for one Ape-X actor subprocess.

    The actor:
    1. Plays Mario using a local copy of the network.
    2. Accumulates n-step returns without touching shared state.
    3. Pushes transition *batches* to experience_queue.
    4. Reloads weights from weight_queue every ``weight_sync_every`` steps.
    """
    # Ignore Ctrl-C; parent handles shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    config = ActorConfig(**config_dict)
    epsilon = training_epsilon(actor_index, total_actors, 0,
                               config.unsolved_epsilon_floor)
    device = torch.device(
        device_str or ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    )
    support = torch.linspace(config.value_min, config.value_max,
                             config.atom_count, device=device)
    actor_seed = config.seed + actor_index * 1000
    np.random.seed(actor_seed)
    torch.manual_seed(actor_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actor_seed)
    network = RainbowNetwork(config.observation_size, config.action_count,
                             config.atom_count).to(device)
    # Train mode activates factorised NoisyNet exploration.  This network has
    # no dropout or batch-normalisation layers, so only NoisyLinear changes.
    # Pure epsilon-greedy actors: NoisyLinear uses learned mean weights in
    # eval mode. The learner keeps its separate network in train mode.
    network.eval()

    n_step_buf = NStepBuffer(config.gamma, config.n_step)
    progress_tracker = NoProgressTracker()
    batch: list[Transition] = []
    episode_replay: Deque[Transition] = deque(maxlen=8192)

    steps = 0
    last_sync = 0
    weights_ready = False
    episode_max_x = 0
    best_episode_x = 0
    episodes = 0
    victories = 0
    won_levels: set[tuple[int, int]] = set()
    levels_completed = 0
    campaigns_completed = 0
    frontier_course: tuple[int, int] | None = None
    next_frontier_x = 0
    frontier_available = False
    frontier_retries = 0
    frontier_resets = 0
    start_resets = 0

    metrics_path = Path(run_directory) / f"actor_{actor_index:02d}_metrics.json"

    def _flush_batch() -> None:
        if batch:
            # Backpressure preserves experience until the learner can consume it.
            _enqueue_batch(experience_queue, actor_index, batch)
            batch.clear()

    def _sync_weights() -> None:
        nonlocal last_sync, weights_ready
        # Drain all pending weight updates; use the most recent one.
        latest: bytes | None = None
        while True:
            try:
                _, weight_bytes = weight_queue.get_nowait()
                latest = weight_bytes
            except Exception:
                break
        if latest is not None:
            try:
                _load_weights_from_bytes(network, latest, device, evaluation=True)
                weights_ready = True
            except Exception as exc:
                logger.warning("actor %d weight load failed: %s", actor_index, exc)
        if weights_ready:
            last_sync = steps

    logger.info("Actor %d started (epsilon=%.4f)", actor_index, epsilon)

    while True:
        if not weights_ready or steps - last_sync >= config.weight_sync_every:
            _sync_weights()
        if not weights_ready:
            time.sleep(0.002)
            continue
        observation = worker.next_observation()
        if observation is None:
            time.sleep(0.002)
            # Still check weights while idle.
            if not weights_ready or steps - last_sync >= config.weight_sync_every:
                _sync_weights()
            continue
        if worker.consume_bridge_restart():
            # The terminal win was already flushed before the supervisor
            # restarted FCEUX. Do not create a transition across cheat modes.
            worker.previous = None
            n_step_buf.flush()
            episode_replay.clear()
            progress_tracker.reset()
            frontier_course = None
            frontier_available = False
            frontier_retries = 0

        course = (observation.world, observation.level)
        if course != frontier_course:
            # Each level owns its savestate. Never restore a checkpoint made
            # in a preceding level or after a worker's cheat-mode restart.
            frontier_course = course
            next_frontier_x = observation.world_x + config.frontier_spacing
            frontier_available = False
            frontier_retries = 0

        episode_max_x = max(episode_max_x, observation.world_x)
        best_episode_x = max(best_episode_x, episode_max_x)
        epsilon = training_epsilon(
            actor_index, total_actors,
            int((observation.world, observation.level) in won_levels),
            config.unsolved_epsilon_floor,
        )

        previous_duration = getattr(worker, "previous_action_duration", LEGACY_DURATION_FRAMES)
        if not observation.terminal and progress_tracker.update(
            observation.world_x, previous_duration if worker.previous is not None else 0
        ):
            observation = replace(observation, terminal=True, reason="stuck")

        if observation.terminal:
            # Replay must contain terminal outcomes, not just episode metrics.
            if worker.previous is not None:
                ready = n_step_buf.push(_terminal_transition(
                    worker.previous, observation, getattr(worker, "previous_action", 0),
                    config.enemy_separation_bonus,
                    config.stall_approach_penalty,
                ))
                batch.extend(ready)
                episode_replay.extend(ready)
                steps += 1
            episodes += 1
            if observation.reason == "victory":
                victories += 1
                levels_completed += 1
                won_levels.add((observation.world, observation.level))
            # Flush n-step buffer at episode boundary.
            tail = n_step_buf.flush()
            batch.extend(tail)
            episode_replay.extend(tail)
            _flush_batch()
            if config.protect_contrast_pairs and episode_replay:
                contrast = contrast_exemplars(episode_replay)
                if contrast:
                    experience_queue.put(("contrast", actor_index, observation.world,
                                          observation.level,
                                          contrast[-config.contrast_tail_transitions:]))
            if observation.reason == "victory" and episode_replay:
                # Rehearse the complete n-step winning trajectory after its
                # ordinary batch. The learner stores a bounded protected copy.
                experience_queue.put(("success", actor_index, observation.world,
                                      observation.level, list(episode_replay)))
            elif (observation.reason != "victory" and episode_replay
                    and frontier_available):
                # Death at a checkpointed frontier: the tail is the recovery
                # behaviour this level keeps failing. Protect it per level so
                # it survives replay turnover instead of being forgotten.
                tail_window = list(episode_replay)[-config.frontier_tail_transitions:]
                experience_queue.put(("frontier", actor_index, observation.world,
                                      observation.level, tail_window))
            episode_replay.clear()
            if observation.reason == "victory":
                # Each actor owns one SMB1 world campaign.  The bridge keeps
                # its current-level checkpoint after 1-1/1-2/1-3 wins, then
                # restores that world's 1-1 state after 1-4.
                if config.repeat_level_on_victory:
                    # Mastery curriculum: retain the exact clean level start
                    # and collect additional complete demonstrations before
                    # permitting the campaign to advance.
                    worker.reset(observation)
                elif observation.level >= 3:
                    campaigns_completed += 1
                    if config.alternate_cheat_campaigns:
                        # First campaign is powered; each later completed
                        # World-N-1..N-4 cycle flips to normal, then powered.
                        next_cheat_mode = campaigns_completed % 2 == 0
                        worker.restart_world_with_cheat_mode(observation, next_cheat_mode)
                    else:
                        worker.restart_world(observation)
                else:
                    worker.advance_level(observation)
            else:
                # Death and no-progress retries stay on the current level.
                restore_frontier = (frontier_available
                                    and frontier_retries < config.frontier_retries)
                worker.reset(observation, restore_frontier=restore_frontier)
                if restore_frontier:
                    frontier_retries += 1
                    frontier_resets += 1
                else:
                    # A few attempts from a saved frontier are useful for
                    # credit assignment; repeated failures return to the
                    # clean start so the policy cannot overfit one bad state.
                    frontier_retries = 0
                    start_resets += 1
            worker.previous = None
            worker.previous_action = 0  # type: ignore[attr-defined]
            worker.previous_action_duration = LEGACY_DURATION_FRAMES  # type: ignore[attr-defined]
            progress_tracker.reset()
            atomic_write_json(
                metrics_path,
                {
                    "actor": actor_index,
                    "epsilon": round(epsilon, 5),
                    "steps": steps,
                    "episodes": episodes,
                    "victories": victories,
                    "levels_completed": levels_completed,
                    "campaigns_completed": campaigns_completed,
                    "world": observation.world + 1,
                    "level": observation.level + 1,
                    "episode_max_x": episode_max_x,
                    "best_episode_x": best_episode_x,
                    "frontier_x": next_frontier_x - config.frontier_spacing
                    if frontier_available else None,
                    "frontier_retries": frontier_retries,
                    "frontier_resets": frontier_resets,
                    "start_resets": start_resets,
                },
            )
            episode_max_x = 0
            continue

        # Select action.
        action, q_values, hidden = _action_details(
            network, support, observation.state, epsilon, config.action_count, device,
            safe_start=(observation.world_x <= 160
                        and bool(observation.state[171])
                        and not bool(observation.state[182])),
            previous_action=getattr(worker, "previous_action", None),
        )
        _publish_hud(worker, run_directory, observation, action, q_values, hidden,
                     steps, episodes, episodes - victories, victories,
                     best_episode_x, epsilon)

        # Save only stable, grounded advances. A spatial interval avoids
        # writing a savestate every decision while providing a retryable
        # frontier for the next obstacle.
        checkpoint_frontier = (
            bool(observation.state[171])
            and observation.world_x >= next_frontier_x
        )
        if checkpoint_frontier:
            frontier_available = True
            frontier_retries = 0
            next_frontier_x = observation.world_x + config.frontier_spacing
        worker.send_action(observation, action, checkpoint_frontier=checkpoint_frontier)

        # Record transition once we have a previous state.
        if worker.previous is not None:
            prev = worker.previous
            prev_action = getattr(worker, "previous_action", 0)
            raw_reward = _shaped_reward(prev, observation, previous_duration,
                                        config.enemy_separation_bonus,
                                        config.stall_approach_penalty)
            t = Transition(
                state=prev.state,
                action=prev_action,
                reward=raw_reward,
                next_state=observation.state,
                terminated=False,
                discount=config.gamma ** (
                    previous_duration
                    / LEGACY_DURATION_FRAMES
                ),
            )
            ready = n_step_buf.push(t)
            batch.extend(ready)
            episode_replay.extend(ready)
            steps += 1

            if len(batch) >= config.batch_size:
                _flush_batch()

        worker.previous = observation
        worker.previous_action = action  # type: ignore[attr-defined]
        worker.previous_action_duration = decode_action(action)[1]  # type: ignore[attr-defined]

        # Periodic weight sync.
        if steps - last_sync >= config.weight_sync_every:
            _sync_weights()


# ---------------------------------------------------------------------------
# Reward shaping (same formula as the old train.py, kept actor-local)
# ---------------------------------------------------------------------------

def _enemy_separation_reward(state: np.ndarray, bonus: float) -> float:
    """Reward vertical separation from a close enemy ahead (the stomp clearance).

    A nearby enemy is the contact-danger zone (0 < enemy_dx <= 0.25). Standing
    above it is what clears it; the x~1780 wall traces show the policy dying on
    contact instead. The bonus is capped at half the 6-frame time cost so that
    hovering above an enemy can never pay for itself.
    """
    if bonus <= 0.0:
        return 0.0
    enemy_dx = float(state[174])
    enemy_dy = float(state[175])
    if not (0.0 < enemy_dx <= 0.25):
        return 0.0
    return min(0.5 * bonus * max(0.0, enemy_dy), 0.01)


def _stall_approach_penalty(state: np.ndarray, penalty: float) -> float:
    """Charge momentum loss while an enemy is in the contact-danger zone.

    The mirror experiment to _enemy_separation_reward: that pays for the escape
    (stomp/jump-over), this charges the approach failure that feeds the contact -
    the x~1780 wall traces show speed_x collapsing toward 0 with an enemy close
    ahead before the death. Capped at the 6-frame time cost so it is a nudge and
    never dominates progress, victory, or the death penalty.
    """
    if penalty <= 0.0:
        return 0.0
    enemy_dx = float(state[174])
    if not (0.0 < enemy_dx <= 0.25):
        return 0.0
    speed_x = float(state[169])
    if speed_x >= 0.25:
        return 0.0
    return -min(penalty * (0.25 - speed_x) / 0.25, 0.01)


def _shaped_reward(previous: Observation, current: Observation,
                   duration_frames: int = LEGACY_DURATION_FRAMES,
                   enemy_separation_bonus: float = 0.0,
                   stall_approach_penalty: float = 0.0) -> float:
    """Reward progress and charge game time so standing still loses value."""
    reward = max(-MAX_PROGRESS_REWARD,
                 min(MAX_PROGRESS_REWARD, (current.world_x - previous.world_x) / 16.0))
    reward += max(-0.2, min(0.2, (current.power - previous.power) * 0.1))
    reward += _enemy_separation_reward(current.state, enemy_separation_bonus)
    reward += _stall_approach_penalty(current.state, stall_approach_penalty)
    if current.terminal:
        reward += VICTORY_REWARD_BONUS if current.reason == "victory" else -DEATH_REWARD_PENALTY
    else:
        reward -= 0.04 * duration_frames / LEGACY_DURATION_FRAMES
    return reward
