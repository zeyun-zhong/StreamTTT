#!/bin/bash
# Multi-node SLURM version of scripts/train/streamttt_4b.sh.
# The paper's run used 16 nodes x 4 GPUs. Edit the three CHANGE_ME values below
# for your cluster -- SBATCH directives cannot read shell variables.
#SBATCH --job-name=streamttt_4b
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=28:00:00
set -euo pipefail

# --- adapt to your cluster ---
VENV_ACTIVATE="${VENV_ACTIVATE:-CHANGE_ME/bin/activate}"   # e.g. ~/venvs/streamttt/bin/activate
: "${DATA_DIR:?set DATA_DIR to the dataset root (see docs/DATA.md)}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/outputs/streamttt_4b}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib}
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HOME=${HF_HOME:-$DATA_DIR}

source "${VENV_ACTIVATE}"

DATASET_USE="llava_178k_long%50,llava_realtime_perception,activitynet_realtime_qa,qvhighlight_realtime_qa,egotime_realtime_qa,howtocaption_realtime_qa,ego4d_realtime_qa,adt_realtime_qa,activitynet_realtime_caption"

srun torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    train.py \
    --deepspeed scripts/deepspeed/zero3.json \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --model_type streamttt \
    --dataset_use "${DATASET_USE}" \
    --data_dir "${DATA_DIR}" \
    --data_seed 42 \
    --bf16 --tf32 True \
    --tune_ttt True --tune_llm True \
    --lr_parameterization ttt \
    --ttt_base_lr 0.0001 \
    --ttt_weight_decay headwise \
    --ttt_momentum headwise \
    --use_residual True \
    --num_fw_heads 4 --num_fw_kv_heads 4 --inter_multi 1 \
    --lact_chunk_size 1024 \
    --use_gate_for_memory True \
    --sliding_window 4096 \
    --video_min_pixels 128 --video_max_pixels 128 --max_frames 480 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 --dataloader_pin_memory True --dataloader_persistent_workers True \
    --eval_strategy no --save_strategy steps --save_steps 512 --save_total_limit 2 \
    --learning_rate 2e-6 --memory_lr 1e-4 --max_grad_norm 1 \
    --lr_scheduler_type cosine --warmup_ratio 0.03 --logging_steps 10 \
    --gradient_checkpointing True \
    --run_name streamttt_4b \
    --output_dir "${OUTPUT_DIR}" \
    --report_to none
