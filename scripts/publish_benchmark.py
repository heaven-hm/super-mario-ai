#!/usr/bin/env python3
"""Publish an evidence-backed SMB1 World 1-1 benchmark report.

Design rules (enforced in code, not only in prose):

1. Only self-consistent episode-level JSON produced by
   ``mario_ai_fceux.evaluate`` is eligible for the main table. Every accepted
   report must match the same protocol tuple (ROM hash, FCEUX hash, world,
   level, seed, episode count, action repeat, start protocol, greedy mode).
2. A method with no eligible report is rendered as *not measured* with an
   explicit blocker. Zero wins are never invented for missing data.
3. Binomial clear rates always carry a Wilson 95% interval and a sample count.
4. Aggregate/summary files are rendered in a separate, labelled section. They
   are never pooled with episode-level reports or with a different selection
   path.

Usage::

    python3 scripts/publish_benchmark.py --root . \
      --result 'NAME=runs/.../results.json' \
      --blocker 'OTHER METHOD=explicit reason' \
      --aggregate 'ARCHIVE LABEL=runs/.../pooling.json' \
      --command 'Evaluate=PYTHONPATH=python python3 -m ...' \
      --output BENCHMARK_RESULTS.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

Z_95 = 1.959963984540054  # standard normal 0.975 quantile
PROTOCOL_KEYS = (
    "rom_sha256", "fceux_sha256", "world", "level", "evaluation_seed",
    "episodes_requested", "action_repeat_frames", "start_protocol",
    "evaluation_mode",
)


def wilson_interval(wins: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("sample count must be positive")
    if not 0 <= wins <= n:
        raise ValueError("wins must be within 0..n")
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _finite_nonnegative(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def validate_episode_report(report: object, label: str) -> dict:
    """Fail closed on anything that is not a self-consistent evaluate.py report."""
    if not isinstance(report, dict):
        raise ValueError(f"{label}: report is not a JSON object")
    required = PROTOCOL_KEYS + ("algorithm", "episodes", "victories", "completion_rate")
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"{label}: missing fields {missing}")
    if report["evaluation_mode"] != "greedy_no_learning":
        raise ValueError(f"{label}: not a greedy no-learning evaluation")
    episodes = report["episodes"]
    requested = report["episodes_requested"]
    if (not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0
            or not isinstance(episodes, list) or len(episodes) != requested
            or report.get("episodes_finished") != requested):
        raise ValueError(f"{label}: incomplete or mismatched episode records")
    for index, episode in enumerate(episodes, 1):
        if not isinstance(episode, dict) or episode.get("reason") not in {"victory", "death", "timeout"}:
            raise ValueError(f"{label}: bad episode {index} reason")
        if not _finite_nonnegative(episode.get("max_x")):
            raise ValueError(f"{label}: bad episode {index} max_x")
        decisions = episode.get("action_decisions")
        if not isinstance(decisions, int) or isinstance(decisions, bool) or decisions < 0:
            raise ValueError(f"{label}: bad episode {index} action_decisions")
        if not _finite_nonnegative(episode.get("elapsed_seconds")):
            raise ValueError(f"{label}: bad episode {index} elapsed_seconds")
    wins = sum(episode["reason"] == "victory" for episode in episodes)
    if wins != report["victories"]:
        raise ValueError(f"{label}: victories field disagrees with episode records")
    rate = float(report["completion_rate"])
    if not math.isfinite(rate) or abs(rate - wins / len(episodes)) > 1e-9:
        raise ValueError(f"{label}: completion_rate disagrees with episode records")
    return report


def protocol_key(report: dict) -> tuple:
    return tuple(json.dumps(report[name], sort_keys=True) for name in PROTOCOL_KEYS)


def summarize(report: dict) -> dict:
    episodes = report["episodes"]
    n = len(episodes)
    wins = int(report["victories"])
    low, high = wilson_interval(wins, n)
    return {
        "n": n, "wins": wins, "rate": wins / n, "ci_low": low, "ci_high": high,
        "best_x": max(int(episode["max_x"]) for episode in episodes),
        "mean_x": sum(float(episode["max_x"]) for episode in episodes) / n,
        "mean_decisions": sum(int(episode["action_decisions"]) for episode in episodes) / n,
        "mean_seconds": sum(float(episode["elapsed_seconds"]) for episode in episodes) / n,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_launches(path: Path) -> list[tuple[str, int, int]]:
    """Extract (label, wins, episodes) from a committed aggregate record.

    Only records with an explicit schema are accepted; unknown schemas are
    rejected instead of being guessed at.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    launches: list[tuple[str, int, int]] = []
    if isinstance(data, dict) and isinstance(data.get("launches"), list):
        for item in data["launches"]:
            launches.append((str(item.get("eval_dir", "launch")),
                             int(item["wins"]), int(item["episodes"])))
    elif isinstance(data, dict) and isinstance(data.get("points"), list):
        for item in data["points"]:
            if item.get("arm") == "off" and item.get("wins") is not None:
                launches.append((f"rule-off launch {item.get('launch')}",
                                 int(item["wins"]), int(item["episodes"])))
    else:
        raise ValueError(f"{path}: unrecognised aggregate schema")
    if not launches:
        raise ValueError(f"{path}: no countable launches")
    return launches


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def split_option(value: str, option: str) -> tuple[str, str]:
    name, sep, text = value.partition("=")
    if not sep or not name or not text:
        raise ValueError(f"bad {option}: {value}")
    return name, text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--result", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--legacy-result", action="append", default=[], metavar="LABEL=PATH",
                        help="Episode-level report that does not meet the 20-episode/seed-2026 "
                             "protocol; rendered in a clearly separated section.")
    parser.add_argument("--blocker", action="append", default=[], metavar="METHOD=REASON")
    parser.add_argument("--aggregate", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--note", action="append", default=[], metavar="METHOD=NOTE")
    parser.add_argument("--command", action="append", default=[], metavar="LABEL=COMMAND")
    parser.add_argument("--environment", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    results: list[tuple[str, Path, dict, dict]] = []
    for value in args.result:
        name, raw = split_option(value, "--result")
        path = (root / raw).resolve()
        report = validate_episode_report(json.loads(path.read_text(encoding="utf-8")), name)
        results.append((name, path.relative_to(root), report, summarize(report)))
    if results:
        reference = protocol_key(results[0][2])
        for name, _, report, _ in results[1:]:
            if protocol_key(report) != reference:
                raise ValueError(f"{name}: protocol differs from {results[0][0]}; refusing to pool")

    legacy: list[tuple[str, Path, dict, dict]] = []
    for value in args.legacy_result:
        label, raw = split_option(value, "--legacy-result")
        path = (root / raw).resolve()
        report = validate_episode_report(json.loads(path.read_text(encoding="utf-8")), label)
        legacy.append((label, path.relative_to(root), report, summarize(report)))
    if legacy:
        reference = protocol_key(legacy[0][2])
        for label, _, report, _ in legacy[1:]:
            if protocol_key(report) != reference:
                raise ValueError(f"{label}: legacy reports disagree with each other; refusing to pool")

    blockers = [split_option(value, "--blocker") for value in args.blocker]
    notes = dict(split_option(value, "--note") for value in args.note)
    commands = list(split_option(value, "--command") for value in args.command)
    environment = list(split_option(value, "--environment") for value in args.environment)

    aggregates = []
    for value in args.aggregate:
        label, raw = split_option(value, "--aggregate")
        path = (root / raw).resolve()
        launches = aggregate_launches(path)
        wins = sum(item[1] for item in launches)
        episodes = sum(item[2] for item in launches)
        low, high = wilson_interval(wins, episodes)
        aggregates.append((label, path.relative_to(root), launches, wins, episodes, low, high))

    lines: list[str] = []
    add = lines.append
    add("# SMB1 World 1-1 benchmark results")
    add("")
    add("Generated by `scripts/publish_benchmark.py`. Every number below comes from a "
        "checked-in artifact or a freshly measured `mario_ai_fceux.evaluate` report. "
        "Methods without eligible measurements are reported as **not measured**; a "
        "missing result is never replaced with a zero-win result.")
    add("")
    add("## Main comparison (episode-level, identical protocol)")
    add("")
    if results:
        ref = results[0][2]
        add(f"- ROM SHA-256: `{ref['rom_sha256']}`")
        add(f"- FCEUX SHA-256: `{ref['fceux_sha256']}` (`{ref['fceux_executable']}`)")
        add(f"- World {ref['world']}-{ref['level']}, seed {ref['evaluation_seed']}, "
            f"{ref['episodes_requested']} episodes per method, greedy/no-learning, "
            f"action repeat {ref['action_repeat_frames']}")
        add(f"- Start protocol: {ref['start_protocol']}")
    else:
        add("No checked-in episode-level report uses the issue's 20-episode seed-2026 "
            "protocol, so no rate is published in this section. Rainbow DQN's archived "
            "20-episode aggregate counts are listed in the archived section below.")
    add("")
    add("| Method | Clear rate (95% Wilson CI) | Wins / episodes | Best X | Mean X | "
        "Mean decisions | Mean seconds | Training wall-clock |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for name, _, report, stats in results:
        add(f"| {name} | {percent(stats['rate'])} "
            f"({percent(stats['ci_low'])}–{percent(stats['ci_high'])}) | "
            f"{stats['wins']}/{stats['n']} | {stats['best_x']} | {stats['mean_x']:.1f} | "
            f"{stats['mean_decisions']:.1f} | {stats['mean_seconds']:.1f} | "
            f"{notes.get(name, 'not recorded with the checkpoint')} |")
    for name, _ in blockers:
        add(f"| {name} | **not measured** | — | — | — | — | — | {notes.get(name, '—')} |")
    add("")
    if blockers:
        add("### Why some methods have no row")
        add("")
        for name, reason in blockers:
            add(f"- **{name}:** {reason}")
        add("")
    if results:
        add("### Artifact provenance")
        add("")
        add("| Method | Artifact | Artifact SHA-256 | Algorithm label |")
        add("| --- | --- | --- | --- |")
        for name, path, report, _ in results:
            add(f"| {name} | `{path}` | `{sha256(root / path)}` | {report['algorithm']} |")
        add("")
    if legacy:
        add("## Legacy episode-level evaluations (protocol does not meet the benchmark)")
        add("")
        add("These reports are genuine measurements, but they use **5 episodes, seed 7 and "
            "action-repeat set [6, 12, 24]** (and an older checkpoint), so they are **not** "
            "the issue's 20-episode seed-2026 World 1-1 comparison. They are shown for "
            "transparency, never substituted for a missing method rate.")
        add("")
        ref = legacy[0][2]
        add(f"- ROM SHA-256: `{ref['rom_sha256']}`; FCEUX SHA-256: `{ref['fceux_sha256']}`")
        add(f"- Seed {ref['evaluation_seed']}, {ref['episodes_requested']} episodes, "
            f"action repeat {ref['action_repeat_frames']}")
        add("")
        add("| Report | Clear rate (95% Wilson CI) | Wins / episodes | Best X | Mean X | "
            "Mean decisions | Mean seconds | Source |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for label, path, _, stats in legacy:
            add(f"| {label} | {percent(stats['rate'])} "
                f"({percent(stats['ci_low'])}–{percent(stats['ci_high'])}) | "
                f"{stats['wins']}/{stats['n']} | {stats['best_x']} | {stats['mean_x']:.1f} | "
                f"{stats['mean_decisions']:.1f} | {stats['mean_seconds']:.1f} | `{path}` |")
        add("")
    if aggregates:
        add("## Archived aggregate measurements (separate provenance)")
        add("")
        add("These counts are recorded in committed summary files. They are **not** pooled "
            "with the table above: the per-episode JSON behind them is not checked in, so "
            "best-X, decisions and seconds cannot be recomputed. Counts from a different "
            "selection path or platform are deliberately excluded from each pool. Rows "
            "marked COMBINED are the union of component rows and must not be added to them.")
        add("")
        add("| Record | Wins / episodes | Rate (95% Wilson CI) | Source |")
        add("| --- | ---: | ---: | --- |")
        for label, path, _, wins, episodes, low, high in aggregates:
            add(f"| {label} | {wins}/{episodes} | {percent(wins / episodes)} "
                f"({percent(low)}–{percent(high)}) | `{path}` |")
        add("")
        for label, path, launches, _, _, _, _ in aggregates:
            add(f"<details><summary>{label}: launch-level counts</summary>")
            add("")
            for launch, launch_wins, launch_episodes in launches:
                add(f"- `{launch}`: {launch_wins}/{launch_episodes}")
            add("")
            add("</details>")
            add("")
    add("## Exact commands")
    add("")
    if commands:
        for label, command in commands:
            add(f"**{label}:**")
            add("")
            add("```bash")
            add(command)
            add("```")
            add("")
    else:
        add("Commands used for each included artifact are recorded in its source directory.")
        add("")
    add("## Environment")
    add("")
    add(f"- OS: {platform.system()} {platform.release()} ({platform.machine()})")
    add(f"- Python: {platform.python_version()}")
    for key, value in environment:
        add(f"- {key}: {value}")
    try:
        import numpy
        import torch
        add(f"- PyTorch {torch.__version__}, NumPy {numpy.__version__}")
    except ImportError:
        add("- PyTorch / NumPy: not importable in the report-generation environment")
    add("")
    add("## Interpretation limits")
    add("")
    add("- A Wilson interval describes binomial sampling uncertainty only. It does not "
        "cover checkpoint selection, emulator differences, or non-deterministic replay "
        "interleaving.")
    add("- A rate is only comparable across methods when the protocol tuple above matches "
        "exactly; the validator refuses mixed reports.")
    add("- No full 1-1 clear clip or GIF is claimed here. A clip must come from a recorded "
        "winning run on a desktop session; none is present in the repository.")
    add("")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
