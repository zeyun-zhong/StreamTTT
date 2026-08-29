#!/bin/bash
# Train StreamTTT-4B jointly on offline long-video QA and the real-time QA corpus.
# This is the recipe behind Table 1 of the paper.
#
# Usage:  DATA_DIR=/path/to/StreamT OUTPUT_DIR=/path/to/out bash scripts/train/streamttt_4b.sh
set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the dataset root (see docs/DATA.md)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/streamttt_4b}"
NPROC="${NPROC:-$(python3 -c 'import torch;print(torch.cuda.device_count())')}"
RUN_NAME="${RUN_NAME:-streamttt_4b}"

DATASET_USE="llava_178k_long%50,llava_realtime_perception,activitynet_realtime_qa,qvhighlight_realtime_qa,egotime_realtime_qa,howtocaption_realtime_qa,ego4d_realtime_qa,adt_realtime_qa,activitynet_realtime_caption"

export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

torchrun --standalone --nproc_per_node="${NPROC}" train.py \
    --deepspeed scripts/deepspeed/zero3.json \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --model_type streamttt \
    --dataset_use "${DATASET_USE}" \
    --data_dir "${DATA_DIR}" \
    --data_seed 42 \
    --bf16 \
    --tf32 True \
    --tune_ttt True \
    --tune_llm True \
    --lr_parameterization ttt \
    --ttt_base_lr 0.0001 \
    --ttt_weight_decay headwise \
    --ttt_momentum headwise \
    --use_residual True \
    --num_fw_heads 4 \
    --num_fw_kv_heads 4 \
    --inter_multi 1 \
    --lact_chunk_size 1024 \
    --use_gate_for_memory True \
    --sliding_window 4096 \
    --video_min_pixels 128 \
    --video_max_pixels 128 \
    --max_frames 480 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 512 \
    --save_total_limit 2 \
    --learning_rate 2e-6 \
    --memory_lr 1e-4 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --report_to none
