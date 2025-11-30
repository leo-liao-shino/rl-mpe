"""Command line helpers for running quick PettingZoo environment checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from .env_factory import PRESET_SPECS
from .rollout import run_random_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick random-action rollouts for PettingZoo MPE environments."
    )
    parser.add_argument(
        "env",
        choices=[*PRESET_SPECS.keys(), "all"],
        help="Which environment preset to run (or 'all').",
    )
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=10,
        help="Number of agent turns to simulate per rollout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for env.reset().",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rollout_summary.json"),
        help="Destination file for the rollout summary (JSON).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output for easier reading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_names: Iterable[str] = PRESET_SPECS.keys() if args.env == "all" else [args.env]
    summaries: List[dict] = []

    for env_name in env_names:
        result = run_random_episode(
            env_name,
            max_agent_steps=args.max_agent_steps,
            seed=args.seed,
        )
        summaries.append(result)

    args.output.write_text(json.dumps(summaries, indent=2 if args.pretty else None))
    print(f"Wrote {len(summaries)} summary entries to {args.output.resolve()}")


if __name__ == "__main__":
    main()
