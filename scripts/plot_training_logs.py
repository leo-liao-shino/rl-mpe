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
    return parser.parse_args()


def parse_log_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Log spec '{spec}' must be in the format label=/path/to/log.jsonl")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path).expanduser().resolve()


def load_log(path: Path) -> Tuple[List[int], List[float]]:
    episodes: List[int] = []
    returns: List[float] = []
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
    if not episodes:
        raise ValueError(f"Log file '{path}' contained no entries")
    return episodes, returns


def smooth(values: Sequence[float], window: int) -> np.ndarray:
    if window <= 1 or window > len(values):
        return np.asarray(values)
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def plot_series(series: Dict[str, Tuple[List[int], List[float]]], window: int, title: str, output: Path, show: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for label, (episodes, returns) in series.items():
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


def main() -> None:
    args = parse_args()
    series: Dict[str, Tuple[List[int], List[float]]] = {}
    for spec in args.logs:
        label, path = parse_log_spec(spec)
        series[label] = load_log(path)
    plot_series(series, args.window, args.title, args.output, args.show)


if __name__ == "__main__":
    main()
