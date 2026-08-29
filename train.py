import warnings
warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
    category=UserWarning
)
import os


def setup_triton_cache():
    """Give every rank its own Triton cache dir; a shared one races on multi-GPU nodes."""
    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    cache_dir = f"/tmp/triton_cache_{job_id}_{local_rank}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir


setup_triton_cache()

from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
apply_liger_kernel_to_qwen3_vl(rope=False)

import logging
import pathlib
import sys
import torch
import transformers
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Allow `python /abs/path/to/train.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from streamttt.data.qwen_video_dataset import make_supervised_data_module
from streamttt.models import build_model_tokenizer_processor
from streamttt.train.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from streamttt.train.trainer import StreamTTTTrainer
from transformers import set_seed

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in
                          state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def count_learnable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")
    return trainable_params / all_params


def set_model(model_args, model):
    """Freeze everything, then re-enable exactly what this run is meant to train."""
    for p in model.parameters():
        p.requires_grad = False

    lm = model.language_model
    num_layers = len(lm.layers)
    fw_top = getattr(model_args, "fw_top_percentage", 1.0)
    ttt_start_idx = int(num_layers * (1 - fw_top))

    if model_args.tune_ttt:
        for n, p in lm.named_parameters():
            if "ttt_block" in n or "alpha" in n:
                p.requires_grad = True

    tied = (
        model.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr()
    )

    if model_args.tune_llm:
        if fw_top < 1.0:
            # Train only layers that carry TTT blocks (top fw_top_percentage)
            # plus the final norm. embed_tokens stays frozen.
            for idx in range(ttt_start_idx, num_layers):
                for p in lm.layers[idx].parameters():
                    p.requires_grad = True
            for p in lm.norm.parameters():
                p.requires_grad = True
            # Only unfreeze lm_head when it is independent of embed_tokens.
            # 4B ties them (so unfreezing would also unfreeze the input
            # embedding); 8B does not, so lm_head must be trained explicitly.
            if not tied:
                for p in model.lm_head.parameters():
                    p.requires_grad = True
        else:
            for p in lm.parameters():
                p.requires_grad = True
            for p in model.lm_head.parameters():
                p.requires_grad = True


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    set_seed(training_args.seed)
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model, tokenizer, processor = build_model_tokenizer_processor(model_args, training_args, attn_implementation)

    if not model_args.lora_enable:
        set_model(model_args, model)

    data_args.processor = processor
    data_args.model_name = model_args.model_type
    data_module = make_supervised_data_module(data_args=data_args)
    trainer = StreamTTTTrainer(
        model=model, processing_class=tokenizer, args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    data_args.processor.image_processor.save_pretrained(training_args.output_dir)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
