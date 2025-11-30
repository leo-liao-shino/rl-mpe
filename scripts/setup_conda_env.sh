#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/setup_conda_env.sh [ENV_NAME]
# Creates (if needed) a conda environment, upgrades pip, and installs requirements.

ENV_NAME=${1:-rl-mpe}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "Could not find requirements file at $REQUIREMENTS_FILE" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda executable not found in PATH. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

echo "Using conda environment: $ENV_NAME (Python $PYTHON_VERSION)"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' already exists; skipping creation."
else
  echo "Creating environment '$ENV_NAME'..."
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

echo "Upgrading pip inside '$ENV_NAME'..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip

echo "Installing project requirements..."
conda run -n "$ENV_NAME" python -m pip install -r "$REQUIREMENTS_FILE"

echo "All set! Activate the environment with:"
echo "  conda activate $ENV_NAME"
