"""CLI entry point for training PettingZoo MPE presets."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

from .env_factory import PRESET_SPECS, build_parallel_mpe_env
from .trainers import EpisodeStats, IndependentPolicyTrainer

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="pygame.pkgdata",
)


@dataclass(frozen=True)
class TrainingSettings:
    episodes: int
    lr: float
    gamma: float
    hidden: List[int]
    log_interval: int
    entropy_coef: float
    grad_clip: Optional[float]
    baseline_momentum: Optional[float]

    def as_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "lr": self.lr,
            "gamma": self.gamma,
            "hidden": list(self.hidden),
            "log_interval": self.log_interval,
            "entropy_coef": self.entropy_coef,
            "grad_clip": self.grad_clip,
            "baseline_momentum": self.baseline_momentum,
        }


DEFAULT_EPISODES = 200
DEFAULT_LR = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_HIDDEN = [128, 128]
DEFAULT_LOG_INTERVAL = 10
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_GRAD_CLIP = 0.5
DEFAULT_BASELINE = 0.95


BEST_TRAINING_RECIPES = {
    "simple_reference_v3": TrainingSettings(
        episodes=800,
        lr=5e-4,
        gamma=0.98,
        hidden=[256, 256],
        log_interval=20,
        entropy_coef=0.02,
        grad_clip=0.5,
        baseline_momentum=0.97,
    ),
    "simple_world_comm_v3": TrainingSettings(
        episodes=1200,
        lr=3e-4,
        gamma=0.995,
        hidden=[256, 256, 128],
        log_interval=25,
        entropy_coef=0.015,
        grad_clip=0.7,
        baseline_momentum=0.985,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train independent policy-gradient agents on PettingZoo MPE presets."
    )
    parser.add_argument(
        "env",
        choices=[*PRESET_SPECS.keys(), "all"],
        help="Which environment preset to train (or 'all').",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=f"Number of training episodes (default: {DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
    default=None,
    help=f"Learning rate for Adam optimizers (default: {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
    default=None,
    help=f"Discount factor for return computation (default: {DEFAULT_GAMMA}).",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="*",
    default=None,
    help=f"Hidden layer sizes for the policy networks (default: {DEFAULT_HIDDEN}).",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
    default=None,
    help=f"Entropy bonus coefficient applied to the policy loss (default: {DEFAULT_ENTROPY_COEF}).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
    default=None,
    help=f"Global gradient-norm clipping value; set <=0 to disable (default: {DEFAULT_GRAD_CLIP}).",
    )
    parser.add_argument(
        "--baseline-momentum",
        type=float,
    default=None,
    help=f"EMA factor for the running reward baseline; set <=0 to disable (default: {DEFAULT_BASELINE}).",
    )
    parser.add_argument(
        "--episodes-per-log",
        type=int,
    default=None,
    help=f"How frequently to print training metrics and save checkpoints (default: {DEFAULT_LOG_INTERVAL}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory to store per-agent checkpoints.",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Directory containing per-agent checkpoints to load before training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Select computation device; 'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Raise an error if CUDA cannot be used (helpful when you only want GPU training).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="If provided, write JSONL metrics to <log-dir>/<env>_train_log.jsonl.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional JSONL file to append final summaries (one object per env).",
    )
    parser.add_argument(
        "--training-plan",
        choices=["default", "best"],
        default="default",
        help="Use built-in tuned hyperparameters per environment.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of greedy evaluation rollouts to run after training (0 to skip).",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "online", "offline", "dryrun"],
        default="disabled",
        help="Weights & Biases logging mode (disabled by default).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="rl-mpe",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional Weights & Biases entity/organization.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Override the auto-generated W&B run name.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Group multiple runs together in W&B.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional list of tags for the W&B run.",
    )
    return parser.parse_args()


def _settings_from_args(args: argparse.Namespace) -> TrainingSettings:
    episodes = args.episodes if args.episodes is not None else DEFAULT_EPISODES
    lr = args.lr if args.lr is not None else DEFAULT_LR
    gamma = args.gamma if args.gamma is not None else DEFAULT_GAMMA
    hidden = list(args.hidden) if args.hidden else list(DEFAULT_HIDDEN)
    log_interval = args.episodes_per_log if args.episodes_per_log is not None else DEFAULT_LOG_INTERVAL

    entropy_coef = args.entropy_coef if args.entropy_coef is not None else DEFAULT_ENTROPY_COEF

    grad_clip_val = args.grad_clip if args.grad_clip is not None else DEFAULT_GRAD_CLIP
    grad_clip = None if grad_clip_val is not None and grad_clip_val <= 0 else grad_clip_val

    baseline_val = (
        args.baseline_momentum if args.baseline_momentum is not None else DEFAULT_BASELINE
    )
    baseline = None if baseline_val is not None and baseline_val <= 0 else baseline_val

    return TrainingSettings(
        episodes=episodes,
        lr=lr,
        gamma=gamma,
        hidden=hidden,
        log_interval=log_interval,
        entropy_coef=entropy_coef,
        grad_clip=grad_clip,
        baseline_momentum=baseline,
    )


def _resolve_settings(
    args: argparse.Namespace, env_name: str, base_settings: TrainingSettings
) -> TrainingSettings:
    settings = base_settings
    if args.training_plan == "best":
        recipe = BEST_TRAINING_RECIPES.get(env_name)
        if recipe:
            settings = recipe
    return _apply_overrides(settings, args)


def _apply_overrides(settings: TrainingSettings, args: argparse.Namespace) -> TrainingSettings:
    overrides = {}
    if args.episodes is not None:
        overrides["episodes"] = args.episodes
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    if args.hidden is not None:
        overrides["hidden"] = list(args.hidden)
    if args.episodes_per_log is not None:
        overrides["log_interval"] = args.episodes_per_log
    if args.entropy_coef is not None:
        overrides["entropy_coef"] = args.entropy_coef
    if args.grad_clip is not None:
        overrides["grad_clip"] = None if args.grad_clip <= 0 else args.grad_clip
    if args.baseline_momentum is not None:
        overrides["baseline_momentum"] = (
            None if args.baseline_momentum <= 0 else args.baseline_momentum
        )

    return replace(settings, **overrides) if overrides else settings


def _init_wandb_run(
    args: argparse.Namespace, env_name: str, settings: TrainingSettings
):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - surfaced at runtime
        raise RuntimeError(
            "wandb is not installed. Run `pip install wandb` or update the environment."
        ) from exc

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.wandb_run_name or f"{env_name}-{timestamp}"
    config = {
        "env": env_name,
        "device": args.device,
        "seed": args.seed,
        **settings.as_dict(),
    }

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        reinit=True,
        config=config,
    )


def _episode_logger(wandb_run, env_name: str):
    if wandb_run is None:
        return None

    def _callback(stats: EpisodeStats) -> None:
        payload = {
            "env": env_name,
            "episode": stats.episode,
            "train/mean_return": stats.mean_return,
            "train/steps": stats.steps,
        }
        for agent, value in stats.per_agent_return.items():
            payload[f"train/agent/{agent}_return"] = value
        wandb_run.log(payload, step=stats.episode)

    return _callback


def _log_final_metrics(wandb_run, env_name: str, summary: dict, eval_metrics: Optional[dict]) -> None:
    if wandb_run is None:
        return
    payload = {
        "env": env_name,
        "summary/final_mean_return": summary.get("final_mean_return"),
        "summary/best_mean_return": summary.get("best_mean_return"),
        "summary/best_episode": summary.get("best_episode"),
    }
    if eval_metrics:
        payload["eval/mean_return"] = eval_metrics.get("mean_return")
        for agent, value in eval_metrics.get("agent_mean_return", {}).items():
            payload[f"eval/agent/{agent}_return"] = value
    wandb_run.log(payload)


def _evaluate_policy(
    trainer: IndependentPolicyTrainer,
    env_name: str,
    episodes: int,
    seed: Optional[int],
) -> Optional[dict]:
    if episodes <= 0:
        return None

    env = build_parallel_mpe_env(env_name)
    rng = np.random.default_rng(seed)
    per_episode_returns: List[dict[str, float]] = []
    try:
        for _ in range(episodes):
            ep_seed = int(rng.integers(0, 1_000_000)) if seed is not None else None
            obs, _ = env.reset(seed=ep_seed)
            per_agent = {agent: 0.0 for agent in trainer.possible_agents}
            steps = 0
            while env.agents:
                actions = {}
                for agent in env.agents:
                    obs_tensor = torch.as_tensor(
                        obs[agent], dtype=torch.float32, device=trainer.device
                    )
                    dist = trainer.models[agent](obs_tensor)
                    action = int(dist.probs.argmax().item())
                    actions[agent] = action
                obs, rewards, terminations, truncations, _ = env.step(actions)
                for agent in actions:
                    per_agent[agent] += float(rewards.get(agent, 0.0))
                steps += 1
                if steps > trainer.spec.max_cycles * len(trainer.possible_agents):
                    break
            per_episode_returns.append(per_agent)
    finally:
        env.close()

    if not per_episode_returns:
        return None

    agent_totals: dict[str, List[float]] = {agent: [] for agent in trainer.possible_agents}
    mean_returns = []
    for returns in per_episode_returns:
        mean_returns.append(float(np.mean(list(returns.values()) or [0.0])))
        for agent, value in returns.items():
            agent_totals.setdefault(agent, []).append(value)

    agent_means = {agent: float(np.mean(values)) for agent, values in agent_totals.items()}

    return {
        "episodes": episodes,
        "mean_return": float(np.mean(mean_returns)),
        "agent_mean_return": agent_means,
    }


def main() -> None:
    args = parse_args()
    device = _resolve_device_choice(args.device)
    if args.require_cuda and device != "cuda":
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a GPU-capable PyTorch build (see README)."
        )
    env_names: Iterable[str] = PRESET_SPECS.keys() if args.env == "all" else [args.env]

    log_dir: Path | None = args.log_dir
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    base_settings = _settings_from_args(args)

    for env_name in env_names:
        settings = _resolve_settings(args, env_name, base_settings)
        if args.training_plan == "best":
            print(
                f"[{env_name}] Using tuned recipe: episodes={settings.episodes}, lr={settings.lr}, "
                f"hidden={settings.hidden}"
            )
        wandb_run = _init_wandb_run(args, env_name, settings)
        episode_cb = _episode_logger(wandb_run, env_name)
        trainer = IndependentPolicyTrainer(
            env_name,
            lr=settings.lr,
            gamma=settings.gamma,
            hidden_sizes=settings.hidden,
            device=device,
            entropy_coef=settings.entropy_coef,
            grad_clip=settings.grad_clip,
            baseline_momentum=settings.baseline_momentum,
        )
        if args.init_from:
            loaded_paths = trainer.load_checkpoints(args.init_from)
            if loaded_paths:
                loaded_summary = ", ".join(
                    f"{agent}:{path.name}" for agent, path in loaded_paths.items()
                )
                print(f"[{env_name}] Loaded checkpoints -> {loaded_summary}")
        try:
            history = trainer.train(
                episodes=settings.episodes,
                seed=args.seed,
                log_interval=settings.log_interval,
                checkpoint_dir=args.checkpoint_dir,
                episode_callback=episode_cb,
            )

            if log_dir:
                _write_history(log_dir / f"{env_name}_train_log.jsonl", history)

            summary = _summarize_history(env_name, history)
            eval_metrics = _evaluate_policy(
                trainer, env_name, args.eval_episodes, args.seed
            )
            if eval_metrics:
                summary["evaluation_mean_return"] = eval_metrics["mean_return"]

            summary_records.append(summary)
            _print_summary(summary)
            if eval_metrics:
                print(
                    f"[{env_name}] Evaluation mean return over {args.eval_episodes} episodes: "
                    f"{eval_metrics['mean_return']:.3f}"
                )

            _log_final_metrics(wandb_run, env_name, summary, eval_metrics)
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    if args.summary_file:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_file.open("a", encoding="utf-8") as fh:
            for record in summary_records:
                fh.write(json.dumps(record) + "\n")


def _resolve_device_choice(choice: str) -> str:
    if choice == "cpu":
        return "cpu"

    if choice == "cuda":
        if _cuda_is_usable():
            return "cuda"
        print("CUDA requested but unusable; falling back to CPU.")
        return "cpu"

    # auto
    return "cuda" if _cuda_is_usable() else "cpu"


def _cuda_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError as exc:
        print(f"CUDA is not usable ({exc}); using CPU instead.")
        return False
    return True


def _write_history(path: Path, history: List[EpisodeStats]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for entry in history:
            fh.write(json.dumps(entry.to_dict()) + "\n")


def _summarize_history(env_name: str, history: List["EpisodeStats"]) -> dict:
    if not history:
        return {"env": env_name, "episodes": 0}

    best = max(history, key=lambda s: s.mean_return)
    last = history[-1]
    return {
        "env": env_name,
        "episodes": len(history),
        "best_mean_return": best.mean_return,
        "best_episode": best.episode,
        "final_mean_return": last.mean_return,
    }


def _print_summary(summary: dict) -> None:
    env = summary.get("env", "unknown")
    final_ret = summary.get("final_mean_return")
    best_ret = summary.get("best_mean_return")
    best_ep = summary.get("best_episode")
    print(
        f"[{env}] Training complete. Final mean return: {final_ret:.3f} | "
        f"Best mean return: {best_ret:.3f} (episode {best_ep})"
    )


if __name__ == "__main__":
    main()
