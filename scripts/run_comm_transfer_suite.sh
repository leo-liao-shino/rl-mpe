#!/usr/bin/env bash
set -euo pipefail

# Run a 4-part experiment suite for communication transfer:
# 1) reference (simple_reference_v3, tuned recipe)
# 2) scratch   (simple_world_comm_v3, best config from W&B screenshot)
# 3) transfer  (init comm_encoder from reference, freeze encoder)
# 4) finetune  (init comm_encoder from reference, allow updates)
#
# All runs log to W&B with clear run names and tags.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src"

# W&B settings
WANDB_MODE="online"
WANDB_PROJECT="rl-mpe"
WANDB_ENTITY=""            # optional; leave empty to use your default account
GROUP="comm-transfer-suite-$(date +%Y%m%d-%H%M%S)"

# Shared paths
REF_INIT_DIR="${ROOT_DIR}/checkpoints/transfer_exp/simple_reference_v3_source"  # source language
OUT_DIR_BASE="${ROOT_DIR}/checkpoints/transfer_suite_strach"

run_ref() {
  python -m mpe_env_setup.train_cli simple_reference_v3 \
    --training-plan best \
    --algorithm actor-critic \
    --checkpoint-dir "${OUT_DIR_BASE}/reference" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project "${WANDB_PROJECT}" \
    ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
    --wandb-group "${GROUP}" \
    --wandb-run-name "reference" \
    --wandb-tags reference simple_reference_v3 \
    --device cuda
}

run_scratch() {
  python -m mpe_env_setup.train_cli simple_world_comm_v3 \
    --episodes 1500 \
    --lr 1e-4 \
    --gamma 0.995 \
    --hidden 512 256 \
    --entropy-coef 0.005 \
    --grad-clip 1 \
    --baseline-momentum 0.95 \
    --algorithm actor-critic \
    --gae-lambda 0.95 \
    --value-loss-coef 0.5 \
    --normalize-rewards \
    --seed 2 \
    --episodes-per-log 25 \
    --device cuda \
    --checkpoint-dir "${OUT_DIR_BASE}/scratch" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project "${WANDB_PROJECT}" \
    ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
    --wandb-group "${GROUP}" \
    --wandb-run-name "scratch" \
    --wandb-tags scratch simple_world_comm_v3
}

run_transfer() {
  python -m mpe_env_setup.train_cli simple_world_comm_v3 \
    --episodes 1500 \
    --lr 1e-4 \
    --gamma 0.995 \
    --hidden 512 256 \
    --entropy-coef 0.005 \
    --grad-clip 1 \
    --baseline-momentum 0.95 \
    --algorithm actor-critic \
    --gae-lambda 0.95 \
    --value-loss-coef 0.5 \
    --normalize-rewards \
    --seed 2 \
    --episodes-per-log 25 \
    --device cuda \
    --init-comm-encoder-from "${REF_INIT_DIR}" \
    --average-comm-encoder \
    --freeze-comm-encoder \
    --checkpoint-dir "${OUT_DIR_BASE}/transfer" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project "${WANDB_PROJECT}" \
    ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
    --wandb-group "${GROUP}" \
    --wandb-run-name "transfer" \
    --wandb-tags transfer simple_world_comm_v3
}

run_finetune() {
  python -m mpe_env_setup.train_cli simple_world_comm_v3 \
    --episodes 1500 \
    --lr 1e-4 \
    --gamma 0.995 \
    --hidden 512 256 \
    --entropy-coef 0.005 \
    --grad-clip 1 \
    --baseline-momentum 0.95 \
    --algorithm actor-critic \
    --gae-lambda 0.95 \
    --value-loss-coef 0.5 \
    --normalize-rewards \
    --seed 2 \
    --episodes-per-log 25 \
    --device cuda \
    --init-comm-encoder-from "${REF_INIT_DIR}" \
    --average-comm-encoder \
    --checkpoint-dir "${OUT_DIR_BASE}/finetune" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project "${WANDB_PROJECT}" \
    ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
    --wandb-group "${GROUP}" \
    --wandb-run-name "finetune" \
    --wandb-tags finetune simple_world_comm_v3
}

main() {
  mkdir -p "${OUT_DIR_BASE}"
  echo "Starting reference run..." && run_ref
  echo "Starting scratch run..." && run_scratch
  echo "Starting transfer (frozen encoder) run..." && run_transfer
  echo "Starting finetune run..." && run_finetune
}

main "$@"
