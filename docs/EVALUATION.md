# Evaluation

All evaluators take a trained checkpoint through `CKPT` and the benchmark root
through `DATA_DIR`, and shard work across GPUs with `torchrun`:

```bash
DATA_DIR=/path/to/benchmarks CKPT=/path/to/streamttt_4b \
  bash scripts/eval/ovo_bench.sh
```

`NPROC` defaults to the visible GPU count and `PROCESSOR` to `CKPT` itself — a
released checkpoint ships a complete `AutoProcessor`. Point `PROCESSOR` at
`Qwen/Qwen3-VL-4B-Instruct` if you are evaluating a checkpoint written by
`train.py`, which only saves the image processor. Any extra flags are forwarded
to the evaluator, so you can narrow a run:

```bash
DATA_DIR=... CKPT=... bash scripts/eval/ovo_bench.sh --splits backward --tasks EPM
```

Long runs write partial results and can be restarted: `--save_every_n_samples N`
checkpoints progress and `--resume` picks it up.

## Streaming inference

For the streaming benchmarks the video is not fed in one pass. `--fps` frames
are sampled, partitioned into contiguous temporal windows of
`--max_window_duration` seconds, and the windows are consumed in order. Between
windows the KV cache is pruned to its most recent `--max_inference_context_window`
tokens while the fixed-size TTT state is carried forward untouched; M-RoPE
positions continue globally so the concatenated windows match the token layout
of a single full-video pass. This is Algorithm 1 in the paper, implemented in
[`streamttt/streaming/windowing.py`](../streamttt/streaming/windowing.py).
Pass `--multi_forward_training` to enable it. OVO-Bench and StreamingBench are
streaming-only and refuse to run without it; Video-MME accepts either mode, and
EgoSchema runs a single full-video forward.

