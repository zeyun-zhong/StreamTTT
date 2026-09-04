#!/bin/bash
# OVO-Bench (paper Table 1). Streaming multi-forward inference at 2 fps with a 4K sliding KV cache.
# Usage: DATA_DIR=... CKPT=... bash scripts/eval/ovo_bench.sh [--splits backward --tasks EPM]
set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the benchmark root}"
: "${CKPT:?set CKPT to a trained StreamTTT checkpoint directory}"
NPROC="${NPROC:-$(python3 -c 'import torch;print(torch.cuda.device_count())')}"
PROCESSOR="${PROCESSOR:-${CKPT}}"   # released checkpoints carry a complete processor
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node="${NPROC}" -m evaluation.ovo_bench.evaluate_ovo \
  --model_type streamttt \
  --model_name_or_path "${CKPT}" \
  --benchmark_path "${DATA_DIR}/OVO-Bench" \
  --processor_name "${PROCESSOR}" \
  --multi_forward_training \
  --video_min_pixels 256 \
  --video_max_pixels 256 \
  --video_total_pixels 256000000 \
  --max_inference_context_window 4096 \
  --max_frames 10240 \
  --max_window_duration 96 \
  --fps 2 \
  --save_every_n_samples 64 \
  "$@"
