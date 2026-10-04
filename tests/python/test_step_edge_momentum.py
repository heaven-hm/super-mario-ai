import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from mario_ai_fceux.actions import (ACTION_COUNT, encode_action, step_edge_momentum)

GRID = 13 * 13


def state(grounded=1, speed=0.4, gap=0, solid_cells=()):
    s = np.zeros(GRID + 15, dtype=np.float32)
    for cell in solid_cells:
        s[cell] = 1.0
    s[169] = speed
    s[171] = grounded
    s[182] = gap
    return s


def q(base_of_best=1):
    v = np.full(ACTION_COUNT, 0.0, dtype=np.float32)
    v[encode_action(base_of_best, 2)] = 1.0
    return v


def test_momentum_gap_ahead_defers_to_pit_edge_rule():
    s = state(gap=1, solid_cells=(6 * 13 + 7,))
    assert step_edge_momentum(s, q(3), encode_action(3, 1)) == encode_action(3, 1)


def test_momentum_airborne_defers():
    s = state(grounded=-1, solid_cells=(6 * 13 + 7,))
    assert step_edge_momentum(s, q(3), encode_action(3, 1)) == encode_action(3, 1)


def test_momentum_no_step_ahead_defers():
    s = state(speed=0.1)
    assert step_edge_momentum(s, q(3), encode_action(3, 1)) == encode_action(3, 1)


def test_momentum_slow_before_step_forces_locomotion():
    s = state(speed=0.1, solid_cells=(6 * 13 + 7,))
    action = step_edge_momentum(s, q(3), encode_action(3, 1))
    assert action // 3 in (0, 5)  # run or walk, never a jump from a dead stop


def test_momentum_slow_never_jumps_even_if_jump_is_best():
    s = state(speed=0.0, solid_cells=(6 * 13 + 8,))
    action = step_edge_momentum(s, q(1), encode_action(1, 2))
    assert action // 3 in (0, 5)


def test_momentum_fast_promotes_near_tied_jump():
    v = np.full(ACTION_COUNT, 0.0, dtype=np.float32)
    v[encode_action(3, 1)] = 1.0
    v[encode_action(1, 2)] = 0.9  # near-tied jump+run
    s = state(speed=0.6, solid_cells=(6 * 13 + 7,))
    assert step_edge_momentum(s, v, encode_action(3, 1)) == encode_action(1, 2)


def test_momentum_fast_leaves_genuinely_disliked_jump_alone():
    v = np.full(ACTION_COUNT, 0.0, dtype=np.float32)
    v[encode_action(3, 1)] = 1.0
    v[encode_action(1, 2)] = 0.1  # far below best
    s = state(speed=0.6, solid_cells=(6 * 13 + 7,))
    assert step_edge_momentum(s, v, encode_action(3, 1)) == encode_action(3, 1)


def test_momentum_rejects_short_state():
    try:
        step_edge_momentum(np.zeros(10), q(3), encode_action(3, 1))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a short state")
