#!/bin/bash
# EgoSchema subset (paper Fig. 3).
# Usage: DATA_DIR=... CKPT=... bash scripts/eval/egoschema.sh
set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the benchmark root}"
: "${CKPT:?set CKPT to a trained StreamTTT checkpoint directory}"
NPROC="${NPROC:-$(python3 -c 'import torch;print(torch.cuda.device_count())')}"
PROCESSOR="${PROCESSOR:-${CKPT}}"   # released checkpoints carry a complete processor
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node="${NPROC}" -m evaluation.egoschema.distributed_evaluate_egoschema \
  --model_type streamttt \
  --model_name_or_path "${CKPT}" \
  --benchmark_path "${DATA_DIR}/egoschema" \
  --processor_name "${PROCESSOR}" \
  --video_min_pixels 256 \
  --video_max_pixels 256 \
  --video_total_pixels 262144 \
  --max_inference_context_window 4096 \
  --fps 2 \
  --save_every_n_samples 0 \
  --subset \
  "$@"
