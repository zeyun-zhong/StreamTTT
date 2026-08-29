#!/bin/bash
# Toy run: a tiny LoRA over the three demo clips in demo/tmp_dataset.
# Meant to check that model, dataloader, TTT branch and optimizer are wired up,
# on one small GPU and in a few seconds. It trains nothing useful.
#
# Everything is shrunk relative to scripts/train/streamttt_4b.sh: a 2B backbone
# instead of 4B, rank-4 LoRA on the LLM instead of a full fine-tune, 16 frames
# at 64 tokens each instead of 480 at 128, and no DeepSpeed.
#
# Measured on one RTX 3090: 12.1 GB peak, ~2.3 s/step. The TTT blocks stay fully
# trainable (--finetune_modules ttt_block, 1.2B params), so their optimizer
# state -- not the activations -- dominates; fewer frames barely helps, but
# `--optim adafactor` brings the peak down to 9.3 GB.
#
# Usage:  bash scripts/train/toy.sh [extra --flags ...]
#         MODEL=Qwen/Qwen3-VL-4B-Instruct MAX_STEPS=5 bash scripts/train/toy.sh
set -euo pipefail

cd "$(dirname "$0")/../.."   # run from the repo root, whatever the caller's cwd

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
DATA_DIR="${DATA_DIR:-demo}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/toy}"
MAX_STEPS="${MAX_STEPS:-2}"
MAX_FRAMES="${MAX_FRAMES:-16}"
VIDEO_PIXELS="${VIDEO_PIXELS:-64}"     # visual tokens per frame (recipe: 128)

export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# --lact_chunk_size is deliberately far below the recipe's 1024: a toy sample is
# only ~1K tokens, and a sequence shorter than one chunk skips the fast-weight
# update entirely (fast_weight_block.py: "we do not update the last chunk"),
# so a large chunk size would leave the TTT update path untested.
torchrun --standalone --nproc_per_node=1 train.py \
    --model_name_or_path "${MODEL}" \
    --model_type streamttt \
    --dataset_use tmp_dataset \
    --data_dir "${DATA_DIR}" \
    --data_seed 42 \
    --bf16 \
    --tf32 True \
    --lora_enable True \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lr_parameterization ttt \
    --ttt_base_lr 0.0001 \
    --ttt_weight_decay headwise \
    --ttt_momentum headwise \
    --use_residual True \
    --num_fw_heads 4 \
    --num_fw_kv_heads 4 \
    --inter_multi 1 \
    --lact_chunk_size 128 \
    --use_gate_for_memory True \
    --sliding_window 512 \
    --video_min_pixels "${VIDEO_PIXELS}" \
    --video_max_pixels "${VIDEO_PIXELS}" \
    --max_frames "${MAX_FRAMES}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 0 \
    --eval_strategy no \
    --save_strategy no \
    --learning_rate 2e-6 \
    --memory_lr 1e-4 \
    --max_grad_norm 1 \
    --lr_scheduler_type constant \
    --warmup_ratio 0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --run_name toy \
    --output_dir "${OUTPUT_DIR}" \
    --report_to none \
    "$@"
