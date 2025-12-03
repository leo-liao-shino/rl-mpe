#!/usr/bin/env python3
"""Utility to orchestrate baseline vs. transfer runs for the MPE presets."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Optional

try:  # optional progress bar support
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    _tqdm = None


@dataclass
class TrainingJob:
    label: str
    env_name: str
    args: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-env", default="simple_reference_v3")
    parser.add_argument("--target-env", default="simple_world_comm_v3")
    parser.add_argument("--source-episodes", type=int, default=400)
    parser.add_argument("--target-episodes", type=int, default=600)
    parser.add_argument("--episodes-per-log", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument(
        "--training-plan-source", choices=["default", "best"], default="best"
    )
    parser.add_argument(
        "--training-plan-target", choices=["default", "best"], default="best"
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs/logs"))
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("runs/summary_transfer.jsonl"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--freeze-comm-head", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--eval-from-best",
        action="store_true",
        help="Reload best checkpoints before evaluation for every job.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Also run the reverse direction (target -> source).",
    )
    parser.add_argument(
        "--language-arch",
        choices=["simple", "encdec"],
        default="simple",
        help="Communication head architecture passed to every training job.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "online", "offline", "dryrun"],
        default="disabled",
        help="Weights & Biases logging mode for all jobs (default: disabled).",
    )
    parser.add_argument(
        "--wandb-project",
        default="rl-mpe",
        help="Weights & Biases project name (default: rl-mpe).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional Weights & Biases entity/organization.",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Optional group to collect runs under in W&B.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional list of tags applied to every W&B run.",
    )
    parser.add_argument(
        "--wandb-run-prefix",
        default=None,
        help="Optional prefix prepended to the auto-generated W&B run names.",
    )
    return parser.parse_args()


def _ensure_dirs(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_job(
    *,
    label: str,
    env_name: str,
    checkpoint_dir: Path,
    log_dir: Path,
    summary_file: Path,
    episodes: int,
    episodes_per_log: int,
    training_plan: str,
    eval_episodes: int,
    device: str,
    log_label: Optional[str] = None,
    eval_from_best: bool = False,
    extra_args: Sequence[str] = (),
    language_arch: str = "simple",
    wandb_mode: str = "disabled",
    wandb_project: str = "rl-mpe",
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
    wandb_run_prefix: Optional[str] = None,
) -> TrainingJob:
    args = [
        f"--episodes={episodes}",
        f"--episodes-per-log={episodes_per_log}",
        f"--checkpoint-dir={checkpoint_dir}",
        f"--log-dir={log_dir}",
        f"--summary-file={summary_file}",
        f"--eval-episodes={eval_episodes}",
        f"--training-plan={training_plan}",
        f"--device={device}",
    ]
    if log_label:
        args.append(f"--log-label={log_label}")
    if eval_from_best:
        args.append("--eval-from-best")
    if language_arch:
        args.append(f"--language-arch={language_arch}")
    if wandb_mode != "disabled":
        args.append(f"--wandb-mode={wandb_mode}")
        if wandb_project:
            args.append(f"--wandb-project={wandb_project}")
        if wandb_entity:
            args.append(f"--wandb-entity={wandb_entity}")
        if wandb_group:
            args.append(f"--wandb-group={wandb_group}")
        if wandb_tags:
            args.append("--wandb-tags")
            args.extend(wandb_tags)
        run_name_parts = [wandb_run_prefix, label]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_name = "-".join(part for part in run_name_parts if part) or label
        run_name = f"{run_name}-{timestamp}"
        args.append(f"--wandb-run-name={run_name}")
    args.extend(extra_args)
    return TrainingJob(label=label, env_name=env_name, args=list(args))


def run_job(job: TrainingJob, *, dry_run: bool, env: dict) -> None:
    cmd = [sys.executable, "-m", "mpe_env_setup.train_cli", job.env_name, *job.args]
    cmd_display = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n==> [{job.label}] {cmd_display}")
    if dry_run:
        print("(dry-run) skipping execution")
        return
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()

    log_dir = _ensure_dirs(args.log_dir)
    summary_file = args.summary_file
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else src_path

    jobs: List[TrainingJob] = []

    wandb_kwargs = dict(
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
        wandb_run_prefix=args.wandb_run_prefix,
    )

    source_ckpt = _ensure_dirs(args.checkpoint_root / f"{args.source_env}_source")
    target_scratch_ckpt = _ensure_dirs(args.checkpoint_root / f"{args.target_env}_scratch")
    target_transfer_ckpt = _ensure_dirs(args.checkpoint_root / f"{args.target_env}_transfer")

    jobs.append(
        build_job(
            label=f"{args.source_env}-source",
            env_name=args.source_env,
            checkpoint_dir=source_ckpt,
            log_dir=log_dir,
            summary_file=summary_file,
            episodes=args.source_episodes,
            episodes_per_log=args.episodes_per_log,
            training_plan=args.training_plan_source,
            eval_episodes=args.eval_episodes,
            device=args.device,
            log_label=f"{args.source_env}-source",
            eval_from_best=args.eval_from_best,
            language_arch=args.language_arch,
            **wandb_kwargs,
        )
    )

    jobs.append(
        build_job(
            label=f"{args.target_env}-scratch",
            env_name=args.target_env,
            checkpoint_dir=target_scratch_ckpt,
            log_dir=log_dir,
            summary_file=summary_file,
            episodes=args.target_episodes,
            episodes_per_log=args.episodes_per_log,
            training_plan=args.training_plan_target,
            eval_episodes=args.eval_episodes,
            device=args.device,
            log_label=f"{args.target_env}-scratch",
            eval_from_best=args.eval_from_best,
            language_arch=args.language_arch,
            **wandb_kwargs,
        )
    )

    source_best_ckpt = source_ckpt / "best"
    transfer_args = [f"--init-comm-from={source_best_ckpt}"]
    if args.freeze_comm_head:
        transfer_args.append("--freeze-comm-head")
    jobs.append(
        build_job(
            label=f"{args.target_env}-transfer",
            env_name=args.target_env,
            checkpoint_dir=target_transfer_ckpt,
            log_dir=log_dir,
            summary_file=summary_file,
            episodes=args.target_episodes,
            episodes_per_log=args.episodes_per_log,
            training_plan=args.training_plan_target,
            eval_episodes=args.eval_episodes,
            device=args.device,
            log_label=f"{args.target_env}-transfer",
            eval_from_best=args.eval_from_best,
            extra_args=transfer_args,
            language_arch=args.language_arch,
            **wandb_kwargs,
        )
    )

    if args.reverse:
        reverse_ckpt = _ensure_dirs(args.checkpoint_root / f"{args.target_env}_source")
        reverse_baseline_ckpt = _ensure_dirs(args.checkpoint_root / f"{args.source_env}_scratch")
        reverse_transfer_ckpt = _ensure_dirs(args.checkpoint_root / f"{args.source_env}_transfer")

        jobs.append(
            build_job(
                label=f"{args.target_env}-source",
                env_name=args.target_env,
                checkpoint_dir=reverse_ckpt,
                log_dir=log_dir,
                summary_file=summary_file,
                episodes=args.target_episodes,
                episodes_per_log=args.episodes_per_log,
                training_plan=args.training_plan_target,
                eval_episodes=args.eval_episodes,
                device=args.device,
                log_label=f"{args.target_env}-source",
                eval_from_best=args.eval_from_best,
                language_arch=args.language_arch,
                **wandb_kwargs,
            )
        )
        jobs.append(
            build_job(
                label=f"{args.source_env}-scratch",
                env_name=args.source_env,
                checkpoint_dir=reverse_baseline_ckpt,
                log_dir=log_dir,
                summary_file=summary_file,
                episodes=args.source_episodes,
                episodes_per_log=args.episodes_per_log,
                training_plan=args.training_plan_source,
                eval_episodes=args.eval_episodes,
                device=args.device,
                log_label=f"{args.source_env}-scratch",
                eval_from_best=args.eval_from_best,
                language_arch=args.language_arch,
                **wandb_kwargs,
            )
        )
        reverse_best_ckpt = reverse_ckpt / "best"
        reverse_transfer_args = [f"--init-comm-from={reverse_best_ckpt}"]
        if args.freeze_comm_head:
            reverse_transfer_args.append("--freeze-comm-head")
        jobs.append(
            build_job(
                label=f"{args.source_env}-transfer",
                env_name=args.source_env,
                checkpoint_dir=reverse_transfer_ckpt,
                log_dir=log_dir,
                summary_file=summary_file,
                episodes=args.source_episodes,
                episodes_per_log=args.episodes_per_log,
                training_plan=args.training_plan_source,
                eval_episodes=args.eval_episodes,
                device=args.device,
                log_label=f"{args.source_env}-transfer",
                eval_from_best=args.eval_from_best,
                extra_args=reverse_transfer_args,
                language_arch=args.language_arch,
                **wandb_kwargs,
            )
        )

    if _tqdm is not None:
        progress = _tqdm(jobs, desc="Experiment runs", unit="job")
        for job in progress:
            progress.set_postfix_str(job.label)
            run_job(job, dry_run=args.dry_run, env=env)
    else:
        for job in jobs:
            run_job(job, dry_run=args.dry_run, env=env)


if __name__ == "__main__":
    main()
