#!/usr/bin/env python3
"""Generate supervised datasets for communicative agents in MPE presets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from mpe_env_setup.env_factory import PRESET_SPECS  # noqa: E402  (import after path tweak)
from mpe_env_setup.supervised_dataset import collect_speaker_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", choices=sorted(PRESET_SPECS.keys()))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Optional subset of agents to include (defaults to all with targets).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination .npz file (defaults to datasets/<env>_speaker_<timestamp>.npz)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def _default_output_path(env_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("datasets") / f"{env_name}_speaker_{timestamp}.npz"


def main() -> None:
    args = parse_args()

    output_path = args.output or _default_output_path(args.env)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing file '{output_path}'. Use --overwrite to proceed."
        )

    datasets, counts = collect_speaker_dataset(
        args.env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        agents=args.agents,
        seed=args.seed,
    )

    payload = {}
    for agent_name, data in datasets.items():
        payload[f"{agent_name}_obs"] = data.observations
        payload[f"{agent_name}_labels"] = data.labels

    meta = {
        "env": args.env,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "agents": list(datasets.keys()),
        "counts": counts,
    }
    payload["meta_json"] = json.dumps(meta)

    np.savez_compressed(output_path, **payload)

    print(f"Wrote dataset to {output_path}")
    for agent_name in sorted(counts):
        print(f"  - {agent_name}: {counts[agent_name]} samples")


if __name__ == "__main__":
    main()
