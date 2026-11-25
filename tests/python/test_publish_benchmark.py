from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts" / "publish_benchmark.py"
spec = importlib.util.spec_from_file_location("publish_benchmark", MODULE_PATH)
assert spec is not None and spec.loader is not None
publisher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publisher
spec.loader.exec_module(publisher)


def sample_report(**overrides: object) -> dict:
    report = {
        "algorithm": "Rainbow DQN",
        "rom_sha256": "rom",
        "fceux_sha256": "fceux",
        "world": 1,
        "level": 1,
        "evaluation_seed": 2026,
        "episodes_requested": 4,
        "episodes_finished": 4,
        "action_repeat_frames": 12,
        "start_protocol": "clean-start-v1",
        "evaluation_mode": "greedy_no_learning",
        "victories": 2,
        "completion_rate": 0.5,
        "episodes": [
            {"episode": 1, "reason": "victory", "max_x": 3000, "action_decisions": 100,
             "elapsed_seconds": 20.0},
            {"episode": 2, "reason": "death", "max_x": 1000, "action_decisions": 40,
             "elapsed_seconds": 8.0},
            {"episode": 3, "reason": "victory", "max_x": 3100, "action_decisions": 110,
             "elapsed_seconds": 22.0},
            {"episode": 4, "reason": "death", "max_x": 900, "action_decisions": 35,
             "elapsed_seconds": 7.0},
        ],
    }
    report.update(overrides)
    return report


class WilsonIntervalTests(unittest.TestCase):
    def test_zero_and_full_samples_are_bounded(self) -> None:
        low, high = publisher.wilson_interval(0, 20)
        self.assertEqual(low, 0.0)
        self.assertLess(high, 1.0)
        low, high = publisher.wilson_interval(20, 20)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_ci_widens_for_smaller_samples(self) -> None:
        narrow = publisher.wilson_interval(16, 100)
        wide = publisher.wilson_interval(16, 20)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            publisher.wilson_interval(1, 0)
        with self.assertRaises(ValueError):
            publisher.wilson_interval(5, 4)


class ReportValidationTests(unittest.TestCase):
    def test_accepts_consistent_report_and_computes_metrics(self) -> None:
        report = publisher.validate_episode_report(sample_report(), "Rainbow")
        stats = publisher.summarize(report)
        self.assertEqual(stats["wins"], 2)
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["best_x"], 3100)
        self.assertEqual(stats["mean_decisions"], 71.25)
        self.assertAlmostEqual(stats["mean_seconds"], 14.25)

    def test_rejects_mismatched_victory_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "victories field"):
            publisher.validate_episode_report(sample_report(victories=3), "Rainbow")

    def test_rejects_exploratory_evaluation(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a greedy"):
            publisher.validate_episode_report(
                sample_report(evaluation_mode="epsilon_greedy"), "Rainbow")

    def test_rejects_incomplete_episode_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete or mismatched"):
            publisher.validate_episode_report(sample_report(episodes_finished=3), "Rainbow")

    def test_protocol_key_detects_seed_change(self) -> None:
        self.assertNotEqual(publisher.protocol_key(sample_report()),
                            publisher.protocol_key(sample_report(evaluation_seed=7)))


class AggregateParsingTests(unittest.TestCase):
    def test_parses_launches_and_points_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "pooling.json"
            first.write_text(json.dumps({"launches": [
                {"eval_dir": "a", "wins": 16, "episodes": 20},
                {"eval_dir": "b", "wins": 15, "episodes": 20},
            ]}), encoding="utf-8")
            second = Path(directory) / "probe.json"
            second.write_text(json.dumps({"points": [
                {"arm": "on", "wins": 0, "episodes": 20},
                {"arm": "off", "wins": 16, "episodes": 20, "launch": 1},
            ]}), encoding="utf-8")
            self.assertEqual(publisher.aggregate_launches(first),
                             [("a", 16, 20), ("b", 15, 20)])
            self.assertEqual(publisher.aggregate_launches(second),
                             [("rule-off launch 1", 16, 20)])

    def test_rejects_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.json"
            path.write_text('{"unexpected": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unrecognised aggregate schema"):
                publisher.aggregate_launches(path)


if __name__ == "__main__":
    unittest.main()
