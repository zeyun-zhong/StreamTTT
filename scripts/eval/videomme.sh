#!/bin/bash
# Video-MME (long split used for the window-budget analysis, paper Fig. 3).
# Vary --max_inference_context_window over 4096..65536 to reproduce the sweep.
# Usage: DATA_DIR=... CKPT=... bash scripts/eval/videomme.sh
set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the benchmark root}"
: "${CKPT:?set CKPT to a trained StreamTTT checkpoint directory}"
NPROC="${NPROC:-$(python3 -c 'import torch;print(torch.cuda.device_count())')}"
PROCESSOR="${PROCESSOR:-${CKPT}}"   # released checkpoints carry a complete processor
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node="${NPROC}" -m evaluation.videomme.distributed_evaluate_videomme \
  --model_type streamttt \
  --model_name_or_path "${CKPT}" \
  --benchmark_path "${DATA_DIR}/Video-MME/videomme/videomme.jsonl" \
  --processor_name "${PROCESSOR}" \
  --multi_forward_training \
  --video_min_pixels 128 \
  --video_max_pixels 128 \
  --video_total_pixels 256000000 \
  --max_inference_context_window 4096 \
  --max_frames 2048 \
  --max_window_frames 384 \
  --fps 2 \
  --save_every_n_samples 256 \
  --resume \
  "$@"
