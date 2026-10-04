"""SMB1 high-level actions and their semi-Markov frame durations."""

from __future__ import annotations


# Each base action has short, normal, and committed horizons.  The normal
# 12-frame choices preserve the meaning of actions in older checkpoints.
ACTION_BASES = ("run", "jump_run", "retreat", "brake", "jump_place", "walk", "jump_back")
ACTION_DURATIONS = (6, 12, 24)
ACTION_COUNT = len(ACTION_BASES) * len(ACTION_DURATIONS)
LEGACY_ACTION_COUNT = 6
LEGACY_DURATION_FRAMES = 12
# If the network values two jump horizons almost equally, use the longer
# learned macro. A six-frame tap often cannot clear SMB1 obstacles reliably.
JUMP_HORIZON_TIE_TOLERANCE = 0.05
# At a clean SMB1 spawn Mario faces right on stable ground. Backtracking,
# braking, or jumping backward there cannot be a competent policy response.
SAFE_START_BASES = (0, 1, 4, 5)  # run, jump-run, jump-in-place, walk


def encode_action(base_index: int, duration_index: int) -> int:
    if not 0 <= base_index < len(ACTION_BASES):
        raise ValueError("invalid SMB1 action base")
    if not 0 <= duration_index < len(ACTION_DURATIONS):
        raise ValueError("invalid SMB1 action duration")
    return base_index * len(ACTION_DURATIONS) + duration_index


def decode_action(action: int) -> tuple[str, int]:
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"invalid SMB1 action: {action}")
    base_index, duration_index = divmod(action, len(ACTION_DURATIONS))
    return ACTION_BASES[base_index], ACTION_DURATIONS[duration_index]


ACTION_NAMES = tuple(
    f"{base.replace('_', '+')}@{duration}"
    for base in ACTION_BASES
    for duration in ACTION_DURATIONS
)


def migrate_legacy_action(action: int) -> int:
    """Map a legacy fixed 12-frame action to its equivalent new action."""
    if not 0 <= action < LEGACY_ACTION_COUNT:
        raise ValueError(f"invalid legacy SMB1 action: {action}")
    return encode_action(action, ACTION_DURATIONS.index(LEGACY_DURATION_FRAMES))


def greedy_action(q_values) -> int:
    """Break near-ties toward a longer learned jump horizon.

    Old six-action checkpoints expand each learned value into three duration
    variants. Until training separates those values, ties favor the normal
    horizon for locomotion and the full horizon for jump actions.
    """
    values = [float(value) for value in q_values]
    if len(values) != ACTION_COUNT:
        raise ValueError(f"expected {ACTION_COUNT} Q values, got {len(values)}")
    best = max(values)
    tied = [index for index, value in enumerate(values) if best - value <= 1e-6]
    priorities = {}
    for base_index in range(len(ACTION_BASES)):
        preferred_duration = 2 if base_index in (1, 4, 6) else 1
        priorities[encode_action(base_index, preferred_duration)] = 1
    selected = max(tied, key=lambda index: (priorities.get(index, 0), -index))
    selected_base = selected // len(ACTION_DURATIONS)
    if selected_base in (1, 4, 6):
        candidates = [encode_action(selected_base, duration_index)
                      for duration_index in range(len(ACTION_DURATIONS))]
        near_tied = [index for index in candidates
                     if best - values[index] <= JUMP_HORIZON_TIE_TOLERANCE]
        if near_tied:
            selected = max(near_tied, key=lambda index: index % len(ACTION_DURATIONS))
    return selected


def safe_start_action(q_values) -> int:
    """Choose the best forward-capable action at a clear level start.

    This is a narrow safety constraint, not a replacement policy: it is used
    only before the first obstacle/gap, where moving left or braking has no
    valid SMB1 objective and a collapsed value estimate otherwise self-traps.
    """
    values = list(float(value) for value in q_values)
    if len(values) != ACTION_COUNT:
        raise ValueError(f"expected {ACTION_COUNT} Q values, got {len(values)}")
    masked = [value if index // len(ACTION_DURATIONS) in SAFE_START_BASES
              else float("-inf") for index, value in enumerate(values)]
    return greedy_action(masked)


# At a gap edge the policy's top two actions are typically within a fraction of
# a reward point of each other, and the tie repeatedly resolves to braking or a
# short forward roll, which parks Mario on the edge. Prefer a committed jump
# whenever one is already near-tied for best.
PIT_EDGE_JUMP_MARGIN = 0.35


def pit_edge_commit(q_values, selected: int) -> int:
    """Resolve near-tied values at a gap edge toward a committed jump.

    A narrow constraint, not a replacement policy: it only acts when the best
    jump is already within PIT_EDGE_JUMP_MARGIN of the highest Q value, so it
    cannot force a jump the policy genuinely dislikes. Measured effect of the
    same rule as an eval-time override: 2370 -> 2594 median depth on the
    lever-arm checkpoint, zero training.
    """
    values = [float(value) for value in q_values]
    if len(values) != ACTION_COUNT:
        raise ValueError(f"expected {ACTION_COUNT} Q values, got {len(values)}")
    if not 0 <= selected < ACTION_COUNT:
        raise ValueError(f"invalid SMB1 action: {selected}")
    best_jump = max((encode_action(1, duration_index)
                     for duration_index in range(len(ACTION_DURATIONS))),
                    key=lambda index: values[index])
    if max(values) - values[best_jump] > PIT_EDGE_JUMP_MARGIN:
        return selected
    return best_jump if selected // len(ACTION_DURATIONS) != 1 else selected


# The 2594-2597 step-edge wall: the policy arrives WITHOUT approach momentum,
# and forcing a jump from a dead stop wedges Mario against the step (CUSE-1's
# stall-breaker probe measured +3px, all deaths stuck, Mario airborne at
# spd=(0,0)). The fix is ordering, not bravery: carry speed into the step,
# then commit the jump. Gate is grounded AND step-ahead AND no gap at the
# PRE-JUMP state - gating on the decision state fires at grd=0 and recreates
# the dead-stop jump (CUSE-1's probe-design trap).
GRID_COLS = 13
STATE_SPEED_X = 169
STATE_GROUNDED = 171
STATE_GAP_AHEAD = 182
STEP_EDGE_SPEED_MIN = 0.35
STEP_EDGE_JUMP_MARGIN = 0.35
LOCOMOTION_BASES = (0, 5)  # run, walk


def _solid_tile_ahead(state) -> bool:
    """True when a solid tile sits 1-2 tiles ahead at chest/head height.

    The bridge packs a 13x13 grid first (rows vertical -96..96 by 16, cols
    horizontal -96..96 by 16, Mario at row 6 col 6). Cells are 1 solid, -1
    enemy, 0 empty - only 1 counts as a step/wall here.
    """
    if len(state) < GRID_COLS * GRID_COLS:
        raise ValueError("state too small for the solid grid")
    for index in (6 * GRID_COLS + 7, 6 * GRID_COLS + 8,
                  5 * GRID_COLS + 7, 5 * GRID_COLS + 8):
        if int(round(float(state[index]))) == 1:
            return True
    return False


def step_edge_momentum(state, q_values, selected: int) -> int:
    """Order a step approach: run into it, jump through it with speed.

    Two arms, both narrow constraints rather than a replacement policy:
    - speed below STEP_EDGE_SPEED_MIN: mask to forward locomotion so the
      policy cannot brake/roll/backtrack/jump-from-stop before the step;
    - speed at or above the minimum: promote a near-tied jump+run, exactly
      like pit_edge_commit's margin rule.
    At a gap edge this defers entirely to pit_edge_commit (gap gate).
    """
    values = [float(value) for value in q_values]
    if len(values) != ACTION_COUNT:
        raise ValueError(f"expected {ACTION_COUNT} Q values, got {len(values)}")
    if not 0 <= selected < ACTION_COUNT:
        raise ValueError(f"invalid SMB1 action: {selected}")
    if len(state) < GRID_COLS * GRID_COLS + 15:
        raise ValueError(f"expected {GRID_COLS * GRID_COLS + 15}-dim state, got {len(state)}")
    if int(round(float(state[STATE_GROUNDED]))) <= 0:
        return selected
    if int(round(float(state[STATE_GAP_AHEAD]))) > 0:
        return selected
    if not _solid_tile_ahead(state):
        return selected
    speed = float(state[STATE_SPEED_X])
    if speed < STEP_EDGE_SPEED_MIN:
        masked = [value if index // len(ACTION_DURATIONS) in LOCOMOTION_BASES
                  else float("-inf") for index, value in enumerate(values)]
        return greedy_action(masked)
    best_jump = max((encode_action(1, duration_index)
                     for duration_index in range(len(ACTION_DURATIONS))),
                    key=lambda index: values[index])
    if max(values) - values[best_jump] > STEP_EDGE_JUMP_MARGIN:
        return selected
    return best_jump if selected // len(ACTION_DURATIONS) != 1 else selected

