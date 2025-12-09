#!/usr/bin/env python3
"""Orchestrate baseline vs. transfer experiments for MPE communication transfer.

This script runs a series of training jobs to evaluate communication transfer:
1. Train source environment (learn communication protocol)
2. Train target environment from scratch (baseline)
3. Train target environment with transferred comm_encoder (transfer)

Optionally, with --reverse, also runs the reverse direction.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@dataclass
class TrainingJob:
    """A single training job configuration."""
    label: str
    env_name: str
    args: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Environment settings
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--source-env", default="simple_reference_v3",
        help="Source environment for learning communication")
    env_group.add_argument("--target-env", default="simple_world_comm_v3",
        help="Target environment for transfer")
    
    # Training settings
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--source-episodes", type=int, default=None,
        help="Episodes for source env (default: use best recipe)")
    train_group.add_argument("--target-episodes", type=int, default=None,
        help="Episodes for target env (default: use best recipe)")
    train_group.add_argument("--algorithm", choices=["reinforce", "actor-critic"], 
        default="actor-critic", help="Training algorithm")
    train_group.add_argument("--training-plan", choices=["default", "best"], 
        default="best", help="Hyperparameter plan")
    train_group.add_argument("--episodes-per-log", type=int, default=25)
    train_group.add_argument("--eval-episodes", type=int, default=25)
    train_group.add_argument("--language-arch", choices=["simple", "encdec"], default="simple")
    
    # Transfer settings
    transfer_group = parser.add_argument_group("Transfer")
    transfer_group.add_argument("--encoder-only", action="store_true",
        help="Transfer only comm_encoder (required when comm_dim differs)")
    transfer_group.add_argument("--freeze-comm-encoder", action="store_true",
        help="Freeze comm_encoder after transfer")
    transfer_group.add_argument("--freeze-comm-head", action="store_true",
        help="Freeze entire comm head after transfer")
    
    # Output settings
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--checkpoint-root", type=Path, 
        default=Path("checkpoints/transfer_exp"))
    output_group.add_argument("--log-dir", type=Path, default=Path("runs/logs"))
    output_group.add_argument("--summary-file", type=Path, 
        default=Path("runs/summary_transfer.jsonl"))
    
    # Execution settings
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    exec_group.add_argument("--eval-from-best", action="store_true")
    exec_group.add_argument("--reverse", action="store_true",
        help="Also run reverse transfer (target -> source)")
    exec_group.add_argument("--dry-run", action="store_true")
    exec_group.add_argument("--seed", type=int, default=None)
    
    # W&B settings
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument("--wandb-mode", choices=["disabled", "online", "offline", "dryrun"],
        default="online")
    wandb_group.add_argument("--wandb-project", default="rl-mpe")
    wandb_group.add_argument("--wandb-entity", default=None)
    wandb_group.add_argument("--wandb-group", default=None)
    wandb_group.add_argument("--wandb-tags", nargs="*", default=None)
    wandb_group.add_argument("--wandb-run-prefix", default=None)
    
    return parser.parse_args()


def build_job(label, env_name, checkpoint_dir, args, extra_args=()):
    """Build a training job with consistent arguments."""
    job_args = [
        f"--checkpoint-dir={checkpoint_dir}",
        f"--log-dir={args.log_dir}",
        f"--summary-file={args.summary_file}",
        f"--training-plan={args.training_plan}",
        f"--algorithm={args.algorithm}",
        f"--device={args.device}",
        f"--episodes-per-log={args.episodes_per_log}",
        f"--eval-episodes={args.eval_episodes}",
        f"--language-arch={args.language_arch}",
        f"--log-label={label}",
    ]
    
    # Only pass episodes if explicitly overriding
    if env_name == args.source_env and args.source_episodes is not None:
        job_args.append(f"--episodes={args.source_episodes}")
    elif env_name == args.target_env and args.target_episodes is not None:
        job_args.append(f"--episodes={args.target_episodes}")
    
    if args.eval_from_best:
        job_args.append("--eval-from-best")
    if args.seed is not None:
        job_args.append(f"--seed={args.seed}")
    
    # W&B settings
    if args.wandb_mode != "disabled":
        job_args.append(f"--wandb-mode={args.wandb_mode}")
        job_args.append(f"--wandb-project={args.wandb_project}")
        if args.wandb_entity:
            job_args.append(f"--wandb-entity={args.wandb_entity}")
        if args.wandb_group:
            job_args.append(f"--wandb-group={args.wandb_group}")
        if args.wandb_tags:
            job_args.append("--wandb-tags")
            job_args.extend(args.wandb_tags)
    
    job_args.extend(extra_args)
    return TrainingJob(label=label, env_name=env_name, args=job_args)


def build_transfer_args(args, source_ckpt):
    """Build the transfer-specific arguments."""
    best_ckpt = source_ckpt / "best"
    
    if args.encoder_only:
        transfer_args = [f"--init-comm-encoder-from={best_ckpt}"]
        if args.freeze_comm_encoder:
            transfer_args.append("--freeze-comm-encoder")
    else:
        transfer_args = [f"--init-comm-from={best_ckpt}"]
        if args.freeze_comm_head:
            transfer_args.append("--freeze-comm-head")
    
    return transfer_args


def run_job(job, dry_run, env):
    """Execute a single training job."""
    cmd = [sys.executable, "-m", "mpe_env_setup.train_cli", job.env_name] + job.args
    cmd_display = " ".join(shlex.quote(part) for part in cmd)
    
    print(f"\n{'='*80}")
    print(f"[{job.label}]")
    print(f"{'='*80}")
    print(f"Command: {cmd_display}\n")
    
    if dry_run:
        print("(dry-run) Skipping execution\n")
        return
    
    subprocess.run(cmd, check=True, env=env)


def main():
    args = parse_args()
    
    # Setup directories
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup environment with PYTHONPATH
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = src_path + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path
    
    # Define checkpoint directories
    source_ckpt = args.checkpoint_root / f"{args.source_env}_source"
    target_scratch_ckpt = args.checkpoint_root / f"{args.target_env}_scratch"
    target_transfer_ckpt = args.checkpoint_root / f"{args.target_env}_transfer"
    
    for d in [source_ckpt, target_scratch_ckpt, target_transfer_ckpt]:
        d.mkdir(parents=True, exist_ok=True)
    
    jobs = []
    
    # Job 1: Train source environment
    jobs.append(build_job(
        label=f"{args.source_env}-source",
        env_name=args.source_env,
        checkpoint_dir=source_ckpt,
        args=args,
    ))
    
    # Job 2: Train target from scratch (baseline)
    jobs.append(build_job(
        label=f"{args.target_env}-scratch",
        env_name=args.target_env,
        checkpoint_dir=target_scratch_ckpt,
        args=args,
    ))
    
    # Job 3: Train target with transfer
    jobs.append(build_job(
        label=f"{args.target_env}-transfer",
        env_name=args.target_env,
        checkpoint_dir=target_transfer_ckpt,
        args=args,
        extra_args=build_transfer_args(args, source_ckpt),
    ))
    
    # Optional: Reverse transfer
    if args.reverse:
        rev_source = args.checkpoint_root / f"{args.target_env}_source"
        rev_scratch = args.checkpoint_root / f"{args.source_env}_scratch"
        rev_transfer = args.checkpoint_root / f"{args.source_env}_transfer"
        
        for d in [rev_source, rev_scratch, rev_transfer]:
            d.mkdir(parents=True, exist_ok=True)
        
        jobs.append(build_job(f"{args.target_env}-source", args.target_env, rev_source, args))
        jobs.append(build_job(f"{args.source_env}-scratch", args.source_env, rev_scratch, args))
        jobs.append(build_job(
            f"{args.source_env}-transfer", args.source_env, rev_transfer, args,
            extra_args=build_transfer_args(args, rev_source),
        ))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRANSFER EXPERIMENT PLAN")
    print("=" * 80)
    print(f"Source env: {args.source_env}")
    print(f"Target env: {args.target_env}")
    print(f"Transfer mode: {'encoder-only' if args.encoder_only else 'full comm head'}")
    freeze = "encoder" if args.freeze_comm_encoder else "head" if args.freeze_comm_head else "none"
    print(f"Freeze: {freeze}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Training plan: {args.training_plan}")
    print(f"Jobs: {len(jobs)}")
    for i, job in enumerate(jobs, 1):
        print(f"  {i}. {job.label}")
    print("=" * 80 + "\n")
    
    # Run jobs
    if tqdm is not None:
        progress = tqdm(jobs, desc="Experiments", unit="job")
        for job in progress:
            progress.set_postfix_str(job.label)
            run_job(job, args.dry_run, env)
    else:
        for job in jobs:
            run_job(job, args.dry_run, env)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print(f"Results: {args.summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
