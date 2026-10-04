import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from auto_pin import boundaries_crossed, pin_boundary, transitions_now


def test_boundaries_crossed_advances_by_every():
    assert boundaries_crossed(0, 100_000, 25_000) == [25_000, 50_000, 75_000, 100_000]


def test_boundaries_crossed_skips_when_no_new_boundary():
    assert boundaries_crossed(50_000, 60_000, 25_000) == []
    assert boundaries_crossed(50_000, 75_000, 25_000) == [75_000]


def test_pin_boundary_copies_replay_then_model_and_records_shas(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model.pt").write_bytes(b"model-bytes")
    (run_dir / "replay.npz").write_bytes(b"replay-bytes")
    meta = pin_boundary(run_dir, tmp_path / "pins", 25_000, 25_100)
    pin_dir = tmp_path / "pins" / meta["dir"]
    assert (pin_dir / "model.pt").read_bytes() == b"model-bytes"
    assert (pin_dir / "replay.npz").read_bytes() == b"replay-bytes"
    recorded = json.loads((pin_dir / "pin.json").read_text())
    assert recorded["boundary"] == 25_000
    assert recorded["transitions_at_pin"] == 25_100
    assert recorded["model_sha"] == meta["model_sha"] != meta["replay_sha"]


def test_transitions_now_reads_counter(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "learner_status.json").write_text(json.dumps({"transitions_received": 4694088}))
    assert transitions_now(run_dir) == 4694088
    (run_dir / "learner_status.json").unlink()
    assert transitions_now(run_dir) is None
