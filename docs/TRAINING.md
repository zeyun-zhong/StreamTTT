# Training

## Running

Single node:

```bash
DATA_DIR=/path/to/StreamT OUTPUT_DIR=/path/to/out \
  bash scripts/train/streamttt_4b.sh
```

Multi-node via SLURM — edit the three `CHANGE_ME` values at the top first
(account, partition, venv path), then:

```bash
DATA_DIR=/path/to/StreamT sbatch scripts/train/streamttt_4b.slurm
```

The paper's run used 16 nodes x 4 GPUs with DeepSpeed ZeRO-3 for one epoch over
the joint mixture. `scripts/deepspeed/` also contains ZeRO-2 and a ZeRO-3
CPU-offload config for smaller setups.

Training is a **single forward pass per sample** over the whole sampled video.
Temporal windowing is an inference-time procedure — see
[EVALUATION.md](EVALUATION.md).

## What gets trained

`--tune_ttt True` unfreezes the TTT branch and the fusion gate `alpha`;
`--tune_llm True` unfreezes the language model. Everything else, including the
vision encoder, stays frozen. `--memory_lr` gives the TTT branch its own
(larger) learning rate in the outer optimizer — `2e-6` for the backbone and
`1e-4` for the memory in the released recipe.

`--fw_top_percentage` restricts the TTT branch to the top fraction of decoder
layers; at the default `1.0` every layer carries one.

## Key hyper-parameters

| flag | released value | meaning |
|---|---|---|
| `--model_type` | `streamttt` | `streamttt` or the unmodified `qwen3` backbone |
| `--sliding_window` | `4096` | sliding KV cache size `L`, the short-range memory |
| `--use_gate_for_memory` | `True` | fuse the TTT output through `tanh(α)`, Eq. (5) |
| `--num_fw_heads` / `--num_fw_kv_heads` | `4` / `4` | fast-weight heads |
| `--inter_multi` | `1` | hidden-width multiplier of the fast-weight MLP, Eq. (7) |
| `--lact_chunk_size` | `1024` | large-chunk TTT chunk size `C`, Eq. (3) |
| `--ttt_base_lr` | `1e-4` | base learning rate of the *inner* TTT update |
| `--lr_parameterization` | `ttt` | inner learning-rate parameterization |
| `--ttt_momentum` | `headwise` | momentum `β`, Eq. (9) |
| `--ttt_weight_decay` | `headwise` | input-dependent decay `γ`, Eq. (9) |
| `--use_residual` | `True` | residual readout `r_t = q_t + N(f_W(q_t))`, Eq. (8) |
| `--video_min_pixels` / `--video_max_pixels` | `128` / `128` | per-frame token budget |
| `--max_frames` | `480` | frames sampled per training video |

`--per_device_train_batch_size` must stay `1`. The TTT branch is written for one
video per forward pass — its per-head norm parameters do not broadcast over a
batch dimension — so a larger per-device batch raises a shape error rather than
training incorrectly. Scale with `--gradient_accumulation_steps` and more ranks.

## Checkpoints

The model registers itself for `AutoModelForCausalLM`, so a saved checkpoint
carries a copy of the modeling file and can be reloaded with
`trust_remote_code=True`.
