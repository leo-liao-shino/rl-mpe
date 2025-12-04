#!/usr/bin/env python3
"""Register and launch Weights & Biases sweeps for mpe_env_setup.train_cli."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mpe_env_setup.train_cli import parse_args as parse_train_args, run_training  # noqa: E402

try:  # optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is optional for JSON configs
    yaml = None

import wandb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="simple_reference_v3")
    parser.add_argument("--project", default="rl-mpe")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--sweep-method", choices=["grid", "random", "bayes"], default="bayes")
    parser.add_argument("--metric-name", default="summary/best_mean_return")
    parser.add_argument("--metric-goal", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--count", type=int, default=10, help="Number of runs for this agent.")
    parser.add_argument("--sweep-id", default=None, help="Existing sweep id to attach new agents to.")
    parser.add_argument("--only-create", action="store_true", help="Register the sweep and exit without launching agents.")
    parser.add_argument("--config-path", type=Path, default=None, help="Path to a JSON or YAML sweep config to load instead of the defaults.")
    parser.add_argument("--dump-config", type=Path, default=None, help="Write the resolved sweep config to disk for record keeping.")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/wandb_sweeps"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/wandb_sweeps"))
    parser.add_argument("--episodes-per-log", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--training-plan", choices=["default", "best"], default="best")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--algorithm", choices=["reinforce", "actor-critic"], default="reinforce")
    parser.add_argument("--language-arch", choices=["simple", "encdec"], default="simple")
    parser.add_argument("--eval-from-best", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the training args for each run without launching.")
    parser.add_argument("--hidden-options", nargs="+", default=["128,128", "256,256", "256,128,64"], help="Comma-separated hidden layer layouts to explore (e.g., '128,128').")
    parser.add_argument("--lr-options", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--entropy-options", type=float, nargs="+", default=[0.0025, 0.005, 0.01])
    parser.add_argument("--baseline-options", type=float, nargs="+", default=[0.9, 0.95, 0.99])
    parser.add_argument("--episodes-options", type=int, nargs="+", default=[600, 800, 1000])
    parser.add_argument("--grad-clip-options", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--seed-options", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to read .yaml configs. Install pyyaml or provide JSON.")
            return yaml.safe_load(fh)
        return json.load(fh)


def _build_default_config(args: argparse.Namespace) -> Dict[str, Any]:
    name = args.sweep_name or f"{args.env}-baseline-sweep"
    return {
        "name": name,
        "method": args.sweep_method,
        "metric": {"name": args.metric_name, "goal": args.metric_goal},
        "parameters": {
            "episodes": {"values": args.episodes_options},
            "lr": {"values": args.lr_options},
            "entropy_coef": {"values": args.entropy_options},
            "baseline_momentum": {"values": args.baseline_options},
            "grad_clip": {"values": args.grad_clip_options},
            "hidden_spec": {"values": args.hidden_options},
            "seed": {"values": args.seed_options},
        },
    }


def _parse_hidden_spec(spec: str) -> List[int]:
    values = [part.strip() for part in spec.split(",") if part.strip()]
    if not values:
        raise ValueError("Hidden spec must contain at least one integer, e.g. '128,128'.")
    return [int(value) for value in values]


def _ensure_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_training_argv(
    script_args: argparse.Namespace,
    run_name: str,
    config: Mapping[str, Any],
    *,
    log_dir: Path | None,
    checkpoint_root: Path | None,
) -> List[str]:
    argv: List[str] = [script_args.env]
    argv.extend(["--device", script_args.device])
    argv.extend(["--training-plan", script_args.training_plan])
    
    # Algorithm can come from config (sweep) or script args (fallback)
    algorithm = config.get("algorithm", script_args.algorithm)
    argv.extend(["--algorithm", str(algorithm)])
    argv.extend(["--language-arch", script_args.language_arch])
    argv.extend(["--episodes-per-log", str(script_args.episodes_per_log)])
    argv.extend(["--eval-episodes", str(script_args.eval_episodes)])
    argv.extend(["--lr", str(config.get("lr"))])
    argv.extend(["--entropy-coef", str(config.get("entropy_coef"))])
    argv.extend(["--baseline-momentum", str(config.get("baseline_momentum"))])
    argv.extend(["--grad-clip", str(config.get("grad_clip"))])
    argv.extend(["--episodes", str(config.get("episodes"))])
    argv.extend(["--seed", str(config.get("seed", 0))])
    
    # Actor-critic specific parameters from sweep config
    if config.get("gae_lambda") is not None:
        argv.extend(["--gae-lambda", str(config.get("gae_lambda"))])
    if config.get("value_loss_coef") is not None:
        argv.extend(["--value-loss-coef", str(config.get("value_loss_coef"))])
    if config.get("normalize_rewards"):
        argv.append("--normalize-rewards")
    if config.get("policy_lr") is not None:
        argv.extend(["--policy-lr", str(config.get("policy_lr"))])
    if config.get("value_lr") is not None:
        argv.extend(["--value-lr", str(config.get("value_lr"))])
    
    # Flat action space baseline flag
    if config.get("flat_action_space"):
        argv.append("--flat-action-space")
    
    hidden_spec = config.get("hidden_spec")
    if hidden_spec:
        hidden_layers = _parse_hidden_spec(str(hidden_spec))
        argv.append("--hidden")
        argv.extend(str(layer) for layer in hidden_layers)
    if script_args.eval_from_best:
        argv.append("--eval-from-best")
    if log_dir:
        argv.extend(["--log-dir", str(log_dir)])
        argv.extend(["--log-label", run_name])
    if checkpoint_root:
        run_ckpt = checkpoint_root / script_args.env / run_name
        run_ckpt.mkdir(parents=True, exist_ok=True)
        argv.extend(["--checkpoint-dir", str(run_ckpt)])
    return argv


def _run_single_job(script_args: argparse.Namespace, sweep_id: str) -> None:
    log_dir = _ensure_dir(script_args.log_dir)
    checkpoint_dir = _ensure_dir(script_args.checkpoint_dir)

    def _train() -> None:
        with wandb.init(project=script_args.project, entity=script_args.entity, group=script_args.group, tags=script_args.wandb_tags) as run:
            cfg = dict(wandb.config)
            run_name = run.name or run.id
            argv = _build_training_argv(
                script_args,
                run_name,
                cfg,
                log_dir=log_dir,
                checkpoint_root=checkpoint_dir,
            )
            train_args = parse_train_args(argv)
            if script_args.dry_run:
                print(f"[dry-run] train_cli args: {argv}")
                return
            run_training(train_args, wandb_run=run)

    wandb.agent(
        sweep_id,
        function=_train,
        project=script_args.project,
        entity=script_args.entity,
        count=script_args.count,
    )


def main() -> None:
    args = parse_args()

    if args.config_path:
        sweep_config = _load_config(args.config_path)
        if args.sweep_name:
            sweep_config.setdefault("name", args.sweep_name)
    else:
        sweep_config = _build_default_config(args)

    if args.dump_config:
        args.dump_config.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_config.open("w", encoding="utf-8") as fh:
            json.dump(sweep_config, fh, indent=2)
        print(f"Wrote sweep config to {args.dump_config}")

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Reusing existing sweep id: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        print(f"Registered sweep '{sweep_config.get('name')}' -> {sweep_id}")

    if args.only_create:
        print("Sweep registered; run this script again (or wandb agent) with --sweep-id to launch workers.")
        return

    if args.count <= 0:
        raise ValueError("--count must be positive when launching agents.")

    # wandb.agent expects an identifier string, not the full config dict.
    _run_single_job(args, sweep_id)


if __name__ == "__main__":
    main()
