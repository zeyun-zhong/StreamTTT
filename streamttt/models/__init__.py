import torch
from dataclasses import asdict
from transformers import AutoProcessor, AutoTokenizer, AutoConfig, Qwen3VLForConditionalGeneration

from .modeling_streamttt_qwen3vl import StreamTTTQwen3VLForConditionalGeneration

# Model registry used by `--model_type`.
#   streamttt : StreamTTT — pretrained sliding-window attention with a parallel
#               fast-weight (TTT) memory branch. This is the model of the paper.
#   qwen3     : the unmodified Qwen3-VL backbone, for reference runs.
ALL_MODELS = {
    'streamttt': StreamTTTQwen3VLForConditionalGeneration,
    'qwen3': Qwen3VLForConditionalGeneration,
}


def build_model_tokenizer_processor(model_args, training_args, attn_implementation="flash_attention_2"):
    """
    Returns: model, tokenizer, processor

    All fields of `model_args` are copied onto `config.text_config`, which is how
    the StreamTTT layers pick up their hyper-parameters (fast-weight heads, chunk
    size, sliding-window size, ...).
    """
    torch_dtype = (torch.bfloat16 if training_args.bf16 else None)

    # --- load processor & tokenizer (shared) ---
    processor_name = model_args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Config
    model_args_dict = asdict(model_args)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    for k, v in model_args_dict.items():
        setattr(config.text_config, k, v)

    if model_args.model_type not in ALL_MODELS:
        raise ValueError(
            f"Unknown --model_type {model_args.model_type!r}. "
            f"Available: {sorted(ALL_MODELS)}"
        )
    target_cls = ALL_MODELS[model_args.model_type]
    model = target_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        config=config,
        trust_remote_code=True,
    )
    if hasattr(model, "_custom_init"):
        model._custom_init()

    # Only our own subclass is exported with remote code; registering the stock
    # Qwen3-VL class would break checkpoint saving.
    if model_args.model_type != "qwen3":
        model.__class__.register_for_auto_class("AutoModelForCausalLM")

    # Training-time settings
    model.config.use_cache = False  # important for gradient checkpointing

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _req_grad(_, __, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_req_grad)

    if getattr(model_args, "lora_enable", False):
        from peft import LoraConfig, get_peft_model

        # Currently, we only finetune lora llm layers
        LLM_PREFIXES = "model.language_model"

        llm_targets = []
        for name, mod in model.named_modules():
            # Skip the TTT branch: it is trained in full through
            # `training_args.finetune_modules` (-> `modules_to_save`), and
            # wrapping the same module as a LoRA target too makes peft hold two
            # references to one adapter, which safetensors refuses to save.
            if any(m in name.split(".") for m in training_args.finetune_modules):
                continue
            if name.startswith(LLM_PREFIXES) and name.rsplit(".", 1)[-1] in model_args.lora_target_modules:
                llm_targets.append(name)

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=llm_targets,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=training_args.finetune_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer, processor
