#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/install_torch_cuda.sh [conda_env]
# Installs the latest PyTorch nightly build with CUDA 12.4 support
# suitable for compute capability 12.x GPUs (e.g., RTX 5080).

ENV_NAME=${1:-rl-mpe}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu124}
TORCH_BUILD=${TORCH_BUILD:-20250226}
CUDA_SUFFIX=${CUDA_SUFFIX:-+cu124}

TORCH_VERSION=${TORCH_VERSION:-2.7.0.dev${TORCH_BUILD}}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.22.0.dev${TORCH_BUILD}}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.6.0.dev${TORCH_BUILD}}

PKG_SPECS=(
  "torch==${TORCH_VERSION}${CUDA_SUFFIX}"
  "torchvision==${TORCHVISION_VERSION}${CUDA_SUFFIX}"
  "torchaudio==${TORCHAUDIO_VERSION}${CUDA_SUFFIX}"
)

if ! command -v conda >/dev/null 2>&1; then
  echo "conda executable not found. Please install Miniconda/Anaconda." >&2
  exit 1
fi

CONDA_BASE=$(conda info --base)
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Unable to locate conda activation script at $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' not found. Create it first (e.g., conda create -n $ENV_NAME python=3.12)." >&2
  exit 1
fi

conda activate "$ENV_NAME"
CURRENT_PREFIX=${CONDA_PREFIX:-}
echo "Using conda environment at $CURRENT_PREFIX"

echo "Installing nightly PyTorch build into Conda env '$ENV_NAME' using $PYTORCH_INDEX_URL"
echo "Torch build date: $TORCH_BUILD"

echo "Removing existing torch/vision/audio packages (if any)..."
python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

INSTALL_CMD=(python -m pip install --pre --upgrade --upgrade-strategy eager)
for spec in "${PKG_SPECS[@]}"; do
  INSTALL_CMD+=("$spec")
done
INSTALL_CMD+=(--index-url "$PYTORCH_INDEX_URL")

"${INSTALL_CMD[@]}"

python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```
