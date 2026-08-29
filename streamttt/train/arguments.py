import transformers
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class ModelArguments:
    """All fields here are copied onto `config.text_config` by
    `streamttt.models.build_model_tokenizer_processor`, which is how the
    StreamTTT layers receive their hyper-parameters.

    The defaults are the released recipe of scripts/train/streamttt_4b.sh."""

    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-VL-4B-Instruct")
    model_type: str = field(default="streamttt", metadata={"help": "streamttt or qwen3"})

    # --- what to train ---
    tune_llm: bool = field(default=True, metadata={"help": "Unfreeze the language model"})
    tune_ttt: bool = field(default=True, metadata={"help": "Unfreeze the TTT branch and the fusion gate"})
    fw_top_percentage: float = field(
        default=1.0,
        metadata={"help": "Fraction of decoder layers (counted from the top) that carry a TTT branch"},
    )

    # --- fast-weight memory branch (paper §3.3, App. A.1) ---
    use_gate_for_memory: bool = field(
        default=True,
        metadata={"help": "Fuse the TTT output through the learnable channel-wise gate tanh(alpha), Eq. (5)"},
    )
    num_fw_heads: Optional[int] = field(default=4, metadata={"help": "Number of fast-weight heads"})
    num_fw_kv_heads: int = field(default=4, metadata={"help": "Number of fast-weight key/value heads"})
    inter_multi: float = field(
        default=1.0, metadata={"help": "Hidden-width multiplier of the fast-weight MLP, Eq. (7)"}
    )
    lact_chunk_size: int = field(default=1024, metadata={"help": "Large-chunk TTT chunk size C, Eq. (3)"})
    ttt_base_lr: float = field(default=1e-4, metadata={"help": "Base learning rate of the inner TTT update"})
    lr_parameterization: str = field(
        default="ttt", metadata={"help": "Inner learning-rate parameterization: 'mamba' or 'ttt'"}
    )
    ttt_weight_decay: str = field(
        default="headwise", metadata={"help": "Input-dependent decay gamma, Eq. (9): 'none' or 'headwise'"}
    )
    ttt_momentum: str = field(
        default="headwise", metadata={"help": "Momentum beta, Eq. (9): 'none' or 'headwise'"}
    )
    use_residual: bool = field(
        default=True, metadata={"help": "Residual readout r_t = q_t + N(f_W(q_t)), Eq. (8)"}
    )

    # --- sliding-window attention (short-range memory) ---
    sliding_window: int = field(
        default=4096, metadata={"help": "Sliding KV cache size L in tokens"}
    )

    # --- streaming over temporal windows (paper §3.4) ---
    output_states: bool = field(
        default=False,
        metadata={"help": "Return the recurrent TTT state, required when one step runs several model forwards"},
    )

    # --- lora (not used in the paper) ---
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[Sequence[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_bias: str = field(default="none")

    print_verbose: bool = field(default=False, metadata={"help": "Verbose logging inside the TTT branch"})


@dataclass
class DataArguments:
    dataset_use: str = field(
        default="tmp_dataset",
        metadata={"help": "Comma-separated dataset keys from streamttt.data; `key%%N` keeps N%% of it"},
    )
    data_dir: str = field(default="demo", metadata={"help": "Root directory holding annotations and videos"})
    video_max_pixels: int = field(default=128)
    video_min_pixels: int = field(default=128)
    video_total_pixels: int = field(default=128000)
    max_frames: int = field(default=480)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=32768,  # This is the default value of the qwen2-vl model
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    memory_lr: Optional[float] = field(
        default=1e-4, metadata={"help": "Separate learning rate for the TTT branch (outer optimizer)"}
    )
    finetune_modules: list[str] = field(default_factory=lambda: ['ttt_block'])
    multi_forward_training: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Inference only: feed each video as a sequence of temporal windows, one forward per "
                    "window, carrying the recurrent state across them "
                    "(see StreamTTTTrainer.prediction_step_multiforward)"
        },
    )
    max_inference_context_window: int = field(
        default=128000, metadata={"help": "Sliding KV cache size at inference; the cache is pruned to this many tokens"}
    )
