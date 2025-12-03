# RL Environment Sandbox

This repository contains a lightweight harness for bootstrapping PettingZoo's Multi-Agent Particle Environments (MPE) that are relevant to the project proposal (`6_7920_project_proposal.pdf`). It focuses on two scenarios that match your request:

- `simple_reference_v3`
- `simple_world_comm_v3`

## Prerequisites

- Python 3.10+ (tested with 3.12)
- System packages are **not** required because we rely on the pre-built `pygame` wheel.

## Setting up with conda (preferred)

Run the helper script (customize the env name by passing it as the first argument):

```bash
./scripts/setup_conda_env.sh            # creates/updates env named "rl-mpe"
./scripts/setup_conda_env.sh my-env     # alternative env name
```

The script performs three steps:

1. `conda create -n <env> python=3.12 -y` (skipped if the env already exists)
2. `conda run -n <env> python -m pip install --upgrade pip`
3. `conda run -n <env> python -m pip install -r requirements.txt`

After it finishes, activate the environment with `conda activate <env>`.

### Using `environment.yml`

If you prefer declarative setup, the repo now includes `environment.yml`:

```bash
conda env create -f environment.yml          # first-time setup
conda activate rl-mpe
```

To update an existing environment, run `conda env update -f environment.yml`.

Once activated, conda supplies the interpreter (`python`) while `pip` installs everything listed in `requirements.txt` into that environment.

### Using `venv` instead (optional)

If you do not use conda, a standard virtualenv still works:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the demo CLI

The CLI lives inside `src/mpe_env_setup/cli.py`. Run it as a module so Python automatically discovers the package:

```bash
PYTHONPATH=src .venv/bin/python -m mpe_env_setup.cli simple_reference_v3 --max-agent-steps 2 --pretty
```

Key flags:

- `env`: `simple_reference_v3`, `simple_world_comm_v3`, or `all`
- `--max-agent-steps`: number of agent turns per environment
- `--seed`: deterministic resets
- `--output`: where to write the JSON summary (defaults to `rollout_summary.json`)
- `--pretty`: pretty-print JSON

The command creates a JSON blob similar to `simple_reference_summary.json` and `simple_world_summary.json`, which are already checked in as examples.

## Training the preset environments

### Quick start

Run either preset with the default REINFORCE baseline:

```bash
PYTHONPATH=src python -m mpe_env_setup.train_cli simple_reference_v3 --episodes 200 --episodes-per-log 20 --checkpoint-dir checkpoints/simple_reference

PYTHONPATH=src python -m mpe_env_setup.train_cli simple_world_comm_v3 --episodes 200 --hidden 256 256 128
```

To switch on the new actor-critic baseline (value head + Generalized Advantage Estimation), add `--algorithm actor-critic` and optionally tune its specific knobs:

```bash
PYTHONPATH=src python -m mpe_env_setup.train_cli simple_reference_v3 \
    --training-plan best --algorithm actor-critic \
    --policy-lr 3e-4 --value-lr 5e-4 \
    --gae-lambda 0.9 --value-loss-coef 1.0 --normalize-rewards \
    --auto-entropy-target 1.5 --auto-entropy-lr 5e-4 \
    --checkpoint-dir checkpoints/simple_reference_actor_critic
```

Important flags:

- `--hidden`, `--lr`, `--gamma`: control the shared policy networks and optimizer.
- `--episodes-per-log`: cadence for console logging and checkpoint dumps.
- `--entropy-coef`, `--grad-clip`, `--baseline-momentum`: stability knobs (entropy bonus, gradient clipping, EMA reward baseline).
- `--algorithm {reinforce,actor-critic}`: toggle between the original REINFORCE baseline and the new actor-critic variant (default: `reinforce`).
- `--gae-lambda`, `--value-loss-coef`: actor-critic–specific knobs (Generalized Advantage Estimation decay and critic loss weight, respectively).
- `--policy-lr`, `--value-lr`: optionally decouple the actor vs. critic learning rates (defaults fall back to `--lr`).
- `--normalize-rewards`: normalize per-episode rewards before computing returns/advantages to cut variance.
- `--auto-entropy-target`, `--auto-entropy-lr`, `--min-entropy-coef`: enable automatic entropy-coefficient tuning when running the actor-critic trainer.
- `--language-arch {simple,encdec}`: switch between the original single-projection communication head and a new encoder-decoder variant for language experiments.
- `--device` + `--require-cuda`: choose CPU/GPU execution and optionally enforce CUDA-only training.
- `--log-dir` + `--summary-file`: persist per-episode JSON lines and append coarse summaries for later analysis.
- `--eval-episodes`: number of greedy evaluation rollouts to run after training (set `0` to skip).

### Tuned recipes for better returns

To squeeze noticeably better returns out of both presets, enable the built-in recipe that was tuned for `simple_reference_v3` and `simple_world_comm_v3`:

```bash
PYTHONPATH=src python -m mpe_env_setup.train_cli all \
    --training-plan best \
    --device cuda --require-cuda \
    --log-dir runs/logs --summary-file runs/summary.jsonl \
    --eval-episodes 25
```

`--training-plan best` overrides the core hyperparameters per environment (episode count, learning rate, network width/depth, entropy bonus, gradient clipping, and baseline momentum). You will see a console line such as `simple_world_comm_v3 Using tuned recipe...` so you know which settings were applied.

### 🚀 Kicking off transfer-learning experiments

The trainer now keeps movement and communication heads separate for every agent, so you can reuse the "language" learned in one scenario while fine-tuning the movement policy in another. The CLI exposes three new flags to help:

| Flag | Purpose |
| --- | --- |
| `--init-comm-from <dir>` | Load only the communication-head weights from a checkpoint directory (movement weights stay at their current initialization). |
| `--freeze-comm-head` | Disable gradient updates to the communication head—useful when you want to keep the transferred language fixed. |
| `--freeze-move-head` | Freeze the movement head instead (handy for the reverse experiment). |

A typical workflow looks like this:

1. **Train the source environment** and save checkpoints:
   ```bash
   PYTHONPATH=src python -m mpe_env_setup.train_cli simple_reference_v3 \
       --training-plan best \
       --episodes 800 --episodes-per-log 20 \
       --checkpoint-dir checkpoints/simple_reference_src
   ```

2. **Seed just the communication head** when moving to the target environment, optionally freezing it to measure pure transfer:
   ```bash
   PYTHONPATH=src python -m mpe_env_setup.train_cli simple_world_comm_v3 \
       --training-plan best \
       --init-comm-from checkpoints/simple_reference_src \
       --freeze-comm-head \
       --checkpoint-dir checkpoints/world_from_reference \
       --log-dir runs/logs --summary-file runs/summary.jsonl
   ```

3. **Compare against baselines** by running the same command without `--init-comm-from` (scratch) or with `--init-from` to reuse the whole policy. Because movement and communication heads are separate, you can mix-and-match these options to test hypotheses such as "Does reusing language alone improve sample efficiency?".

Evaluation tips:

- Use `--eval-episodes` (e.g., 25) to record greedy rollout performance at the end of every run.
- Plot the JSONL logs in `runs/logs` to compare learning curves (episodes vs. mean return).
- When `--freeze-comm-head` is enabled, the console prints `Freezing communication heads` so you can trace the exact configuration later.
- Toggle `--language-arch encdec` (and the same flag on `scripts/run_transfer_experiments.py`) to benchmark the new encoder-decoder communication head against the original single projection without changing any other hyperparameters.

### Tracking with Weights & Biases

The CLI now speaks Weights & Biases (wandb) for experiment tracking. Set the desired logging mode (`online`, `offline`, or `dryrun`) and provide your project metadata:

```bash
PYTHONPATH=src python -m mpe_env_setup.train_cli simple_reference_v3 \
    --training-plan best \
    --device cuda --require-cuda \
    --wandb-mode online --wandb-project rl-mpe --wandb-group simple-mpe \
    --wandb-tags reference_v3 curriculum \
    --log-dir runs/logs --summary-file runs/summary.jsonl
```

Use `--wandb-mode offline` when you want to log locally and sync later (`wandb login` is still required once). Each environment run generates its own W&B run with config metadata plus streaming episode metrics (`train/mean_return`, per-agent returns, evaluation scores, etc.).

Before running in `online` or `offline` mode, authenticate once on the machine:

```bash
wandb login                  # opens browser flow
# or, for headless environments
WANDB_API_KEY=... wandb login --relogin
```

The transfer orchestrator (`scripts/run_transfer_experiments.py`) now forwards the same W&B settings to every job. Example dry-run showing a real online configuration:

```bash
python scripts/run_transfer_experiments.py \
    --checkpoint-root checkpoints/new_full_run \
    --log-dir runs/logs \
    --summary-file runs/summary_transfer.jsonl \
    --device cuda --freeze-comm-head --eval-from-best \
    --wandb-mode online --wandb-project rl-mpe --wandb-group transfer \
    --wandb-run-prefix nov2 \
    --wandb-tags reference transfer curriculum \
    --dry-run
```

Each generated training command now includes `--wandb-*` flags so the runs stream directly to the specified project (grouped under the optional prefix/tag metadata). Remove `--dry-run` once you are satisfied with the queue.

## GPU acceleration

1. Install a CUDA build of PyTorch that supports compute capability 12.0 (required for RTX 5080) by running:

```bash
./scripts/install_torch_cuda.sh            # installs pinned nightly cu124 wheels into rl-mpe

# Customize build date / env / CUDA suffix as needed
TORCH_BUILD=20250310 CUDA_SUFFIX=+cu125 PYTORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu125 \
    ./scripts/install_torch_cuda.sh my-env
```

The script activates the specified conda environment before running pip and pins `torch`, `torchvision`, and `torchaudio` to the same nightly build number (defaults to `TORCH_BUILD=20250226`).
Adjust the `TORCH_BUILD` environment variable if you need a newer build; just ensure all three packages have matching dates.

2. Verify the install:

```bash
conda activate rl-mpe
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

3. Run training with GPU enforcement:

```bash
PYTHONPATH=src python -m mpe_env_setup.train_cli simple_reference_v3 \
    --device cuda --require-cuda --episodes 200
```

Notes:

- `--require-cuda` stops immediately if CUDA kernels are unavailable, so you don’t accidentally train on CPU.
- `--device auto` remains useful when you’re iterating on machines with and without GPUs—it will fall back to CPU but print why.
- If pip ever complains about version conflicts, rerun `scripts/install_torch_cuda.sh` with a `TORCH_BUILD` value that exists for all three packages (check the [nightly wheel index](https://download.pytorch.org/whl/nightly/cu124) for available dates) or pin explicit versions via `TORCH_VERSION`, `TORCHVISION_VERSION`, and `TORCHAUDIO_VERSION` env vars.
- For long-term stability, monitor PyTorch’s [installation page](https://pytorch.org/get-started/locally/) for official releases that advertise compute capability ≥12.0.

## Python API

You can also import the helpers directly:

```python
from mpe_env_setup import build_mpe_env, run_random_episode

env = build_mpe_env("simple_reference_v3", max_cycles=150)
summary = run_random_episode("simple_world_comm_v3", max_agent_steps=5, seed=123)
```

`run_random_episode` returns a dictionary with:

- `env_name`
- `steps_taken`
- per-agent reward totals
- observation shapes per agent
- action-space type metadata

## Repository structure

```
.
├── README.md
├── requirements.txt
├── 6_7920_project_proposal.pdf
├── simple_reference_summary.json  # sample output for simple_reference_v3
├── simple_world_summary.json      # sample output for simple_world_comm_v3
└── src/
    └── mpe_env_setup/
        ├── __init__.py
        ├── cli.py
        ├── env_factory.py
        └── rollout.py
```

## Next steps

- Swap the random policies with your learning agents.
- Extend `MPEEnvironmentSpec` if you need to tweak interaction parameters (number of landmarks, speed, etc.).
- Wire the CLI into your training scripts or notebooks for quick smoke tests of new configurations.
