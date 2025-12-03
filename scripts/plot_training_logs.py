#!/usr/bin/env python3
"""Plot mean returns from JSONL training logs to compare sample efficiency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="+",
        help=(
            "Log specifications of the form label=path/to/log.jsonl. "
            "Use 'scratch' and 'transfer' labels to differentiate runs."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Smoothing window (in episodes) for moving average (default: 25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/sample_efficiency.png"),
        help="Where to save the generated plot (default: plots/sample_efficiency.png).",
    )
    parser.add_argument(
        "--title",
        default="Sample efficiency comparison",
        help="Title for the plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    parser.add_argument(
        "--plot-per-agent",
        action="store_true",
        help="Also plot per-agent returns using the `per_agent_return` field if present.",
    )
    parser.add_argument(
        "--per-agent-output",
        type=Path,
        help="Base path for per-agent plots (default: derive from --output).",
    )
    return parser.parse_args()


def parse_log_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Log spec '{spec}' must be in the format label=/path/to/log.jsonl")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path).expanduser().resolve()


def load_log(path: Path) -> Tuple[List[int], List[float], Dict[str, List[float]]]:
    episodes: List[int] = []
    returns: List[float] = []
    per_agent: Dict[str, List[float]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Log file '{path}' does not exist")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            episodes.append(int(record.get("episode", len(episodes) + 1)))
            returns.append(float(record.get("mean_return", 0.0)))
            per_agent_returns = record.get("per_agent_return", {}) or {}
            for agent, value in per_agent_returns.items():
                per_agent.setdefault(agent, []).append(float(value))
    if not episodes:
        raise ValueError(f"Log file '{path}' contained no entries")
    return episodes, returns, per_agent


def smooth(values: Sequence[float], window: int) -> np.ndarray:
    if window <= 1 or window > len(values):
        return np.asarray(values)
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def plot_series(
    series: Dict[str, Tuple[List[int], List[float], Dict[str, List[float]]]],
    window: int,
    title: str,
    output: Path,
    show: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for label, (episodes, returns, _) in series.items():
        episodes_arr = np.asarray(episodes)
        returns_arr = smooth(returns, window)
        plt.plot(episodes_arr[: len(returns_arr)], returns_arr, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Mean return")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved plot to {output}")
    if show:
        plt.show()
    plt.close()


def sanitize_agent_name(agent_name: str) -> str:
    return agent_name.replace("/", "_").replace(" ", "_")


def derive_per_agent_base(output: Path) -> Path:
    return output.with_name(f"{output.stem}_per_agent{output.suffix}")


def plot_per_agent_series(
    series: Dict[str, Tuple[List[int], List[float], Dict[str, List[float]]]],
    window: int,
    title: str,
    base_output: Path,
    show: bool,
) -> None:
    agent_names = sorted(
        {
            agent
            for _, (_, _, per_agent) in series.items()
            for agent in per_agent.keys()
        }
    )
    if not agent_names:
        print("No per-agent returns found in provided logs; skipping per-agent plots.")
        return
    for agent in agent_names:
        agent_safe = sanitize_agent_name(agent)
        agent_output = base_output.with_name(f"{base_output.stem}_{agent_safe}{base_output.suffix}")
        agent_output.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        for label, (episodes, _, per_agent_returns) in series.items():
            if agent not in per_agent_returns:
                continue
            agent_values = smooth(per_agent_returns[agent], window)
            episodes_arr = np.asarray(episodes)
            plt.plot(episodes_arr[: len(agent_values)], agent_values, label=label)
        plt.xlabel("Episode")
        plt.ylabel(f"Return ({agent})")
        plt.title(f"{title} — {agent}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(agent_output)
        print(f"Saved per-agent plot for '{agent}' to {agent_output}")
        if show:
            plt.show()
        plt.close()


def main() -> None:
    args = parse_args()
    series: Dict[str, Tuple[List[int], List[float], Dict[str, List[float]]]] = {}
    for spec in args.logs:
        label, path = parse_log_spec(spec)
        series[label] = load_log(path)
    plot_series(series, args.window, args.title, args.output, args.show)
    if args.plot_per_agent:
        base_output = args.per_agent_output or derive_per_agent_base(args.output)
        plot_per_agent_series(series, args.window, args.title, base_output, args.show)


if __name__ == "__main__":
    main()
