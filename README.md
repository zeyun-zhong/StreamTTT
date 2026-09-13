# StreamTTT

**Reconciling Real-Time Perception and Long-Term Memory in Streaming VLMs**

[Joya Chen](https://chenjoya.github.io/)<sup>1</sup>, [Zeyun Zhong](https://zeyun-zhong.github.io/)<sup>2</sup>, [Mike Zheng Shou](https://sites.google.com/view/showlab)<sup>1</sup>

<sup>1</sup>National University of Singapore &nbsp;&nbsp; <sup>2</sup>Karlsruhe Institute of Technology

[![arXiv](https://img.shields.io/badge/arXiv-2608.13416-b31b1b.svg)](https://arxiv.org/pdf/2608.13416)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-StreamTTT--4B-ffc107.svg)](https://huggingface.co/zeyun-zhong/StreamTTT-4B)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-RealTimeVideo--Instruct--112K-ffc107.svg)](https://huggingface.co/datasets/zeyun-zhong/RealTimeVideo-Instruct-112K)

---

## Overview

Streaming VLMs trade off real-time perception against long-term memory: a short
context sharpens the current scene but forgets the past, while feeding history
back into attention restores recall and dilutes recent evidence.

**StreamTTT writes long-range history into fast weights outside the attention
context.** Every decoder layer of a pretrained Qwen3-VL runs its untouched
attention path over a short sliding KV cache (4K tokens, dedicated to recent
evidence) in parallel with a test-time-training (TTT) branch whose fast weights
are updated online. A learnable channel-wise gate `tanh(α)`, initialized near
zero, fuses the two, so the model starts at the pretrained function and learns
to use long-term memory. At inference, video is processed as ordered temporal
windows: the KV cache is pruned to its most recent `L` tokens after each window
while the fixed-size TTT state carries forward without eviction, and M-RoPE
positions stay globally continuous.

Trained jointly on offline long-video QA and a new real-time QA corpus,
StreamTTT-4B outperforms the same-scale SimpleStream-4B on OVO-Bench by 0.6
points in real-time perception and 5.3 in backward tracing. It also surpasses
the larger SimpleStream-8B by 0.73 points on StreamingBench's Real-Time Visual
Understanding (RTVU) subset.

## Installation

```bash
git clone https://github.com/zeyun-zhong/StreamTTT.git && cd StreamTTT
conda create -y -n streamttt python=3.12 && conda activate streamttt
pip install -r requirements.txt
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```


DeepSpeed probes for a CUDA 12.x toolkit on import, including in recipes that do
not use it:

```bash
conda install -y -c nvidia cuda-nvcc=12.8
conda env config vars set CUDA_HOME=$CONDA_PREFIX && conda activate streamttt
```

`torchcodec` decodes video through a system FFmpeg (4.x–7.x), which is not a pip
package. If you do not already have one:

```bash
conda install -c conda-forge ffmpeg=6.1
```

## Quick start

The repo ships three short clips in `demo/` so you can check that the model,
dataloader, and training loop are wired up before touching real data:

```bash
bash scripts/train/toy.sh
```

`scripts/train/toy.sh` is the released recipe shrunk to fit one small GPU: a 2B
backbone, a rank-4 LoRA on the LLM, 16 frames per video, two steps, no
DeepSpeed. It takes ~12 GB and a few seconds on an RTX 3090. 

## Documentation

| | |
|---|---|
| [docs/DATA.md](docs/DATA.md) | Dataset download, directory layout, and the `--dataset_use` registry |
| [docs/TRAINING.md](docs/TRAINING.md) | Training recipe and what each hyper-parameter controls |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Running OVO-Bench, StreamingBench, Video-MME, EgoSchema |

## Repository layout

```
streamttt/
├── models/
│   ├── modeling_streamttt_qwen3vl.py   hybrid layer: SWA ∥ TTT + gate (§3.3)
│   └── fast_weight_block.py            fast weights, momentum, decay (App. A.1)
├── streaming/windowing.py              temporal windowing, global M-RoPE (§3.4, Alg. 1)
├── data/                               dataset registry and video dataloader
└── train/                              arguments and trainer
train.py                                training entry point
evaluation/                             per-benchmark evaluators
scripts/                                training and evaluation launchers
```

## Citation

```bibtex
@article{chen2026streamttt,
  title   = {StreamTTT: Reconciling Real-Time Perception and Long-Term Memory in Streaming VLMs},
  author  = {Chen, Joya and Zhong, Zeyun and Shou, Mike Zheng},
  journal = {arXiv preprint arXiv:2608.13416},
  year    = {2026}
}
```

## Acknowledgements

Built on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL). The fast-weight memory
follows [E<sup>2</sup>-TTT](https://arxiv.org/pdf/2608.21308v2); training data comes from
[LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)
and the streaming corpora listed in [docs/DATA.md](docs/DATA.md).

## License

Code and model weights are Apache-2.0 (see [LICENSE](LICENSE)). The released
annotations are CC-BY-4.0, except `adt_realtime_qa`, which derives from Aria
Digital Twin ground truth and is CC-BY-NC-SA-4.0. No video is redistributed —
each source keeps its own terms, some requiring a signed agreement or barring
commercial use; check them before downloading (see [docs/DATA.md](docs/DATA.md)).
