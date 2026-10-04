import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from mario_ai_fceux.apex_actor import contrast_exemplars
from mario_ai_fceux.replay import PrioritizedReplayBuffer, Transition
from mario_ai_fceux.actions import STATE_GAP_AHEAD, STATE_GROUNDED

GRID_PLUS_GLOBALS = 13 * 13 + 15


def state(grounded=1, gap=0):
    s = np.zeros(GRID_PLUS_GLOBALS, dtype=np.float32)
    s[STATE_GROUNDED] = grounded
    s[STATE_GAP_AHEAD] = gap
    return s


def transition(gap=0, grounded=1):
    return Transition(state=state(grounded=grounded, gap=gap),
                      next_state=state(grounded=grounded, gap=gap),
                      action=3, reward=0.0, discount=0.99, terminated=False, priority=1.0)


def test_classifier_keeps_only_pit_edge_contexts():
    seq = [transition(gap=0), transition(gap=1), transition(gap=1, grounded=-1), transition(gap=1)]
    got = contrast_exemplars(seq)
    assert len(got) == 2
    assert all(float(t.state[STATE_GAP_AHEAD]) > 0 and float(t.state[STATE_GROUNDED]) > 0 for t in got)


def test_classifier_empty_on_no_pit_states():
    assert contrast_exemplars([transition(gap=0), transition(gap=0, grounded=-1)]) == []


def test_contrast_kind_has_own_quota_per_level():
    # capacity=200 gives protected_limit=10 (capacity//20); per-kind quotas
    # below that stay distinct, which is what this asserts.
    buf = PrioritizedReplayBuffer(GRID_PLUS_GLOBALS, capacity=200,
                                  success_quota_per_level=8,
                                  frontier_window_per_level=6,
                                  contrast_quota_per_level=3)
    assert buf.protected_limit == 10
    assert buf._quota((0, 0), buf.CONTRAST) == 3
    assert buf._quota((0, 0), buf.SUCCESS) == 8
    assert buf._quota((0, 0), buf.FRONTIER) == 6


def test_kind_quota_clamps_to_the_global_protected_limit():
    # Pre-existing guarantee: a kind can never claim more than protected_limit.
    buf = PrioritizedReplayBuffer(GRID_PLUS_GLOBALS, capacity=200,
                                  success_quota_per_level=50, contrast_quota_per_level=50)
    assert buf._quota((0, 0), buf.CONTRAST) == buf.protected_limit
    assert buf._quota((0, 0), buf.SUCCESS) == buf.protected_limit


def test_contrast_protection_evicts_only_its_own_bucket():
    buf = PrioritizedReplayBuffer(GRID_PLUS_GLOBALS, capacity=200,
                                  success_quota_per_level=50, contrast_quota_per_level=2)
    for _ in range(4):
        buf.add(transition(gap=1), protect=True, level=(0, 0), kind=buf.CONTRAST)
    assert buf.protected_count(buf.CONTRAST) == 2
    assert buf.protected_count(buf.SUCCESS) == 0
    buf.add(transition(gap=0), protect=True, level=(0, 0), kind=buf.SUCCESS)
    assert buf.protected_count(buf.SUCCESS) == 1  # contrast evictions never touch success


def test_contrast_quotas_are_per_level():
    buf = PrioritizedReplayBuffer(GRID_PLUS_GLOBALS, capacity=200, contrast_quota_per_level=2)
    for _ in range(3):
        buf.add(transition(gap=1), protect=True, level=(0, 0), kind=buf.CONTRAST)
        buf.add(transition(gap=1), protect=True, level=(1, 1), kind=buf.CONTRAST)
    assert buf.protected_count(buf.CONTRAST) == 4  # 2 per level, not 2 total


def test_default_flag_off_sends_no_contrast_message():
    # The classifier is gated by ActorConfig.protect_contrast_pairs (default False);
    # the flag exists and defaults off.
    from mario_ai_fceux.apex_actor import ActorConfig
    assert ActorConfig().protect_contrast_pairs is False


def test_contrast_quota_survives_snapshot_roundtrip(tmp_path):
    buf = PrioritizedReplayBuffer(GRID_PLUS_GLOBALS, capacity=200,
                                  contrast_quota_per_level=7)
    buf.add(transition(gap=1), protect=True, level=(0, 0), kind=buf.CONTRAST)
    snap = tmp_path / "replay.npz"
    buf.save(snap)
    restored = PrioritizedReplayBuffer.load(snap)
    assert restored.contrast_quota_per_level == 7
    assert restored.protected_count(buf.CONTRAST) == 1
