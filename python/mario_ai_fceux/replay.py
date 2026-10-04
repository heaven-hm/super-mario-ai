"""Fast in-memory, globally prioritized experience replay for Rainbow DQN.

The buffer deliberately stays out of SQLite's hot path. FCEUX collectors send
transitions to one learner, which keeps this array-backed buffer in RAM. A
compressed snapshot is written only with a training checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import os
from pathlib import Path
import uuid

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    discount: float
    priority: float = 1.0


class SumTree:
    """Binary tree supporting exact global proportional-priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # A padded power-of-two leaf layer keeps prefix search valid for any
        # configured capacity, not only power-of-two replay sizes.
        self.leaf_count = 1 << (capacity - 1).bit_length()
        self.values = np.zeros(2 * self.leaf_count, dtype=np.float64)
        self.minimums = np.full(2 * self.leaf_count, np.inf, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.values[1])

    @property
    def minimum(self) -> float:
        return float(self.minimums[1])

    def update(self, index: int, value: float) -> None:
        node = index + self.leaf_count
        self.values[node] = value
        self.minimums[node] = value
        node //= 2
        while node:
            self.values[node] = self.values[node * 2] + self.values[node * 2 + 1]
            self.minimums[node] = min(self.minimums[node * 2], self.minimums[node * 2 + 1])
            node //= 2

    def find_prefixsum(self, value: float) -> int:
        node = 1
        while node < self.leaf_count:
            left = node * 2
            if value < self.values[left]:
                node = left
            else:
                value -= self.values[left]
                node = left + 1
        return node - self.leaf_count


class PrioritizedReplayBuffer:
    """Array-backed replay with true global PER and no database transactions."""

    SUCCESS = "success"
    FRONTIER = "frontier"
    CONTRAST = "contrast"

    def __init__(self, observation_size: int, capacity: int = 100_000,
                 alpha: float = 0.6, priority_epsilon: float = 1e-5,
                 seed: int = 0,
                 success_quota_per_level: int | None = None,
                 frontier_window_per_level: int = 256,
                 contrast_quota_per_level: int | None = None) -> None:
        if observation_size < 1 or capacity < 1:
            raise ValueError("observation_size and replay capacity must be positive")
        if not 0.0 <= alpha <= 1.0 or priority_epsilon <= 0.0:
            raise ValueError("PER alpha must be in [0, 1] and epsilon must be positive")
        if frontier_window_per_level < 1:
            raise ValueError("frontier_window_per_level must be positive")
        self.observation_size = observation_size
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, observation_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int16)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.tree = SumTree(capacity)
        self.size = 0
        self.position = 0
        self.max_priority = 1.0
        self.snapshot_id: str | None = None
        # Keep a bounded rehearsal set of completed runs and frontier recovery
        # segments. Ordinary PER still samples globally; normal writes skip
        # these protected slots. Each (level, kind) bucket is quota-limited so
        # one level's evidence cannot evict another's.
        self.protected_limit = min(10_000, max(0, capacity // 20))
        self.protected = np.zeros(capacity, dtype=np.bool_)
        self.protected_order: deque[int] = deque()
        self.success_quota_per_level = (success_quota_per_level if success_quota_per_level is not None
                                        else max(1, self.protected_limit // 4))
        self.frontier_window_per_level = frontier_window_per_level
        self.contrast_quota_per_level = (contrast_quota_per_level if contrast_quota_per_level is not None
                                        else max(1, self.protected_limit // 8))
        self.protected_tags: dict[int, tuple[tuple[int, int] | None, str]] = {}
        self.protected_buckets: dict[tuple[tuple[int, int] | None, str], deque[int]] = {}

    def __len__(self) -> int:
        return self.size

    def _quota(self, level: tuple[int, int] | None, kind: str) -> int:
        if self.protected_limit == 0:
            return 0
        if level is None:
            # Untagged evidence (legacy snapshots, resume carries) has no level
            # to quota by; it keeps global FIFO within the overall limit.
            return self.protected_limit
        if kind == self.FRONTIER:
            return min(self.frontier_window_per_level, self.protected_limit)
        if kind == self.CONTRAST:
            return min(self.contrast_quota_per_level, self.protected_limit)
        return min(self.success_quota_per_level, self.protected_limit)

    def _unprotect(self, index: int) -> None:
        self.protected[index] = False
        tag = self.protected_tags.pop(index, None)
        if tag is not None:
            bucket = self.protected_buckets.get(tag)
            if bucket is not None:
                try:
                    bucket.remove(index)
                except ValueError:
                    pass
                if not bucket:
                    self.protected_buckets.pop(tag, None)
        try:
            self.protected_order.remove(index)
        except ValueError:
            pass

    def _protect(self, index: int, level: tuple[int, int] | None,
                 kind: str = SUCCESS) -> None:
        if self.protected_limit == 0:
            return
        while len(self.protected_order) >= self.protected_limit:
            self._unprotect(self.protected_order[0])
        tag = (level, kind)
        bucket = self.protected_buckets.setdefault(tag, deque())
        while len(bucket) >= self._quota(level, kind):
            self._unprotect(bucket[0])
        self.protected[index] = True
        self.protected_order.append(index)
        self.protected_tags[index] = tag
        bucket.append(index)

    def protected_count(self, kind: str) -> int:
        return sum(1 for _, tag_kind in self.protected_tags.values() if tag_kind == kind)

    def protect_existing_successes(self, minimum_reward: float = 10.0) -> int:
        """Carry surviving victory evidence forward from pre-memory snapshots."""
        if self.protected_order or self.protected_limit == 0:
            return 0
        indices = np.flatnonzero(self.terminated[:self.size]
                                  & (self.rewards[:self.size] >= minimum_reward))
        for index in indices[-self.protected_limit:]:
            self._protect(int(index), None, self.SUCCESS)
        return len(self.protected_order)

    def add(self, transition: Transition, protect: bool = False,
            level: tuple[int, int] | None = None,
            kind: str = SUCCESS) -> None:
        if protect and self.protected_limit == 0:
            protect = False
        index = self.position
        while self.protected[index]:
            index = (index + 1) % self.capacity
        self.states[index] = np.asarray(transition.state, dtype=np.float32).reshape(self.observation_size)
        self.next_states[index] = np.asarray(transition.next_state, dtype=np.float32).reshape(self.observation_size)
        self.actions[index] = int(transition.action)
        self.rewards[index] = float(transition.reward)
        self.discounts[index] = float(transition.discount)
        self.terminated[index] = bool(transition.terminated)
        priority = max(float(transition.priority), self.max_priority, self.priority_epsilon)
        self.tree.update(index, priority ** self.alpha)
        if protect:
            self._protect(index, level, kind)
        self.max_priority = max(self.max_priority, priority)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float,
               protected_fraction: float = 0.25) -> tuple[np.ndarray, list[Transition], np.ndarray]:
        """Sample PER plus a bounded rehearsal share of winning trajectories."""
        if self.size < batch_size:
            raise ValueError("not enough replay transitions")
        if not 0.0 <= protected_fraction <= 1.0:
            raise ValueError("protected_fraction must be in [0, 1]")
        total = self.tree.total
        if total <= 0:
            raise RuntimeError("replay priorities are empty")
        protected_indices = np.asarray(self.protected_order, dtype=np.int64)
        protected_count = min(int(batch_size * protected_fraction), len(protected_indices))
        if protected_count:
            selected_protected = self.rng.choice(protected_indices, size=protected_count,
                                                 replace=False).astype(np.int64)
        else:
            selected_protected = np.empty(0, dtype=np.int64)
        per_count = batch_size - protected_count
        if protected_count:
            # Demonstrations are intentionally sampled uniformly. Draw the
            # remaining PER part from non-protected transitions so the batch
            # actually contains the requested rehearsal proportion.
            candidates = np.flatnonzero(~self.protected[:self.size])
            priorities_for_candidates = self.tree.values[candidates + self.tree.leaf_count]
            probabilities_for_candidates = priorities_for_candidates / priorities_for_candidates.sum()
            per_indices = self.rng.choice(candidates, size=per_count, replace=True,
                                          p=probabilities_for_candidates).astype(np.int64)
        else:
            boundaries = np.linspace(0.0, total, per_count + 1)
            samples = self.rng.uniform(boundaries[:-1], boundaries[1:])
            per_indices = np.asarray([self.tree.find_prefixsum(float(sample)) for sample in samples], dtype=np.int64)
        priorities = self.tree.values[per_indices + self.tree.leaf_count]
        probabilities = priorities / total
        minimum_probability = self.tree.minimum / total
        maximum_weight = (self.size * minimum_probability) ** (-beta)
        per_weights = ((self.size * probabilities) ** (-beta) / maximum_weight).astype(np.float32)
        indices = np.concatenate((selected_protected, per_indices))
        # The protected slice is deliberate demonstration rehearsal rather
        # than unbiased PER, so it receives neutral importance weighting.
        weights = np.concatenate((np.ones(protected_count, dtype=np.float32), per_weights))
        sampled_priorities = self.tree.values[indices + self.tree.leaf_count]
        transitions = [Transition(self.states[index].copy(), int(self.actions[index]), float(self.rewards[index]),
                                  self.next_states[index].copy(), bool(self.terminated[index]),
                                  float(self.discounts[index]), float(sampled_priorities[row]))
                       for row, index in enumerate(indices)]
        return indices, transitions, weights

    def update_priorities(self, indices: np.ndarray | list[int], priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities):
            raw_priority = max(float(priority), self.priority_epsilon)
            self.tree.update(int(index), raw_priority ** self.alpha)
            self.max_priority = max(self.max_priority, raw_priority)

    def save(self, path: str | Path, snapshot_id: str | None = None) -> str:
        """Atomically persist all replay state needed to resume sampling exactly."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        self.snapshot_id = snapshot_id or uuid.uuid4().hex
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, observation_size=self.observation_size, capacity=self.capacity,
                                alpha=self.alpha, priority_epsilon=self.priority_epsilon, size=self.size,
                                position=self.position, max_priority=self.max_priority,
                                snapshot_id=self.snapshot_id, format_version=4,
                                protected_limit=self.protected_limit,
                                success_quota_per_level=self.success_quota_per_level,
                                frontier_window_per_level=self.frontier_window_per_level,
                                contrast_quota_per_level=self.contrast_quota_per_level,
                                protected_order=np.asarray(self.protected_order, dtype=np.int64),
                                protected_tags=np.asarray([self.protected_tags[index]
                                                           for index in self.protected_order],
                                                          dtype=object),
                                states=self.states[:self.size],
                                next_states=self.next_states[:self.size], actions=self.actions[:self.size],
                                rewards=self.rewards[:self.size], discounts=self.discounts[:self.size],
                                terminated=self.terminated[:self.size], tree=self.tree.values,
                                rng_state=np.asarray([self.rng.bit_generator.state], dtype=object))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        return self.snapshot_id

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "PrioritizedReplayBuffer":
        with np.load(Path(path), allow_pickle=True) as payload:
            buffer = cls(int(payload["observation_size"]), int(payload["capacity"]), float(payload["alpha"]),
                         float(payload["priority_epsilon"]), seed,
                         success_quota_per_level=int(payload["success_quota_per_level"])
                         if "success_quota_per_level" in payload else None,
                         frontier_window_per_level=int(payload["frontier_window_per_level"])
                         if "frontier_window_per_level" in payload else 256,
                         contrast_quota_per_level=int(payload["contrast_quota_per_level"])
                         if "contrast_quota_per_level" in payload else None)
            buffer.size = int(payload["size"])
            buffer.position = int(payload["position"])
            buffer.max_priority = float(payload["max_priority"])
            buffer.snapshot_id = str(payload["snapshot_id"]) if "snapshot_id" in payload else None
            if "protected_order" in payload:
                buffer.protected_limit = int(payload["protected_limit"])
                order = [int(index) for index in payload["protected_order"]]
                if (buffer.protected_limit < 0 or buffer.protected_limit >= buffer.capacity
                        or len(order) > buffer.protected_limit or len(set(order)) != len(order)
                        or any(index < 0 or index >= buffer.size for index in order)):
                    raise ValueError("invalid protected replay indices in snapshot")
                buffer.protected_order = deque(order)
                buffer.protected[order] = True
                if "protected_tags" in payload:
                    tags = payload["protected_tags"]
                    if len(tags) != len(order):
                        raise ValueError("protected tags do not match protected indices")
                    for index, raw in zip(order, tags, strict=True):
                        level = tuple(raw[0]) if raw[0] is not None else None
                        kind = str(raw[1])
                        buffer.protected_tags[index] = (level, kind)
                        buffer.protected_buckets.setdefault((level, kind), deque()).append(index)
                else:
                    # Pre-quota snapshots protected successes without tags.
                    for index in order:
                        buffer.protected_tags[index] = (None, buffer.SUCCESS)
                        buffer.protected_buckets.setdefault((None, buffer.SUCCESS), deque()).append(index)
            buffer.states[:buffer.size] = payload["states"]
            buffer.next_states[:buffer.size] = payload["next_states"]
            buffer.actions[:buffer.size] = payload["actions"]
            buffer.rewards[:buffer.size] = payload["rewards"]
            buffer.discounts[:buffer.size] = payload["discounts"]
            buffer.terminated[:buffer.size] = payload["terminated"]
            saved_tree = payload["tree"]
            if saved_tree.shape == buffer.tree.values.shape:
                buffer.tree.values[:] = saved_tree
                # Rebuild min-tree leaves from exact active sum-tree leaves.
                leaf_values = buffer.tree.values[buffer.tree.leaf_count:
                                                  buffer.tree.leaf_count + buffer.size]
                buffer.tree.minimums[buffer.tree.leaf_count:
                                     buffer.tree.leaf_count + buffer.size] = leaf_values
                for node in range(buffer.tree.leaf_count - 1, 0, -1):
                    buffer.tree.minimums[node] = min(buffer.tree.minimums[node * 2],
                                                     buffer.tree.minimums[node * 2 + 1])
            else:
                # Migrate snapshots made before power-of-two tree padding.
                old_leaf_offset = buffer.capacity
                for index in range(buffer.size):
                    buffer.tree.update(index, float(saved_tree[index + old_leaf_offset]))
            buffer.rng.bit_generator.state = payload["rng_state"][0]
        return buffer


# Compatibility alias for code that imported the old SQLite-backed class.
ReplayDatabase = PrioritizedReplayBuffer
