# Data

StreamTTT is trained on a near-balanced mixture of two supervision regimes
(paper §4.1):

- **offline whole-video QA** — ~119K pairs sampled from the 2–3 minute subset of
  LLaVA-Video-178K, which teaches long-range recall;
- **real-time QA** — a 112K corpus in which every question is posed at the
  moment its answer first becomes available, which teaches current-scene
  perception.

Set `DATA_DIR` to the root below; every path in this document is relative to it.

## Download

`zeyun-zhong/RealTimeVideo-Instruct-112K` ships **annotations only** — the
real-time QA JSON files that are this project's contribution. **No video is
redistributed here.** Every video source must be fetched from its own channel
and keeps its own terms; see [License](../README.md#license).

**1. Annotations**

```bash
huggingface-cli download zeyun-zhong/RealTimeVideo-Instruct-112K \
  --repo-type dataset --local-dir $DATA_DIR
```

**2. Videos** — six sources, each with its own gate:

| source | feeds | where to get it |
|---|---|---|
| **LLaVA-Video-178K**<br>Zhang et al. 2024 | `llava_178k_long`<br>`llava_realtime_perception` | <https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K> |
| **ActivityNet**<br>Caba Heilbron et al. 2015 | `activitynet_realtime_qa`<br>`activitynet_realtime_caption` | <http://activity-net.org/download.html> — the official release distributes video **ids**, not files; use the request form on that page |
| **QVHighlights**<br>Lei et al. 2021 | `qvhighlight_realtime_qa` | <https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz><br>repo: <https://github.com/jayleicn/moment_detr> |
| **E.T. Instruct 164K**<br>Liu et al. 2024, [arXiv:2409.18111](https://arxiv.org/abs/2409.18111) | `egotime_realtime_qa`<br>`howtocaption_realtime_qa` | <https://huggingface.co/datasets/PolyU-ChenLab/ET-Instruct-164K> |
| **Ego4D**<br>Grauman et al. 2022, [arXiv:2110.07058](https://arxiv.org/abs/2110.07058) | `ego4d_realtime_qa` | <https://ego4d-data.org/docs/start-here/> — requires signing the Ego4D License Agreement first |
| **Aria Digital Twin**<br>Pan et al. 2023, [arXiv:2306.06362](https://arxiv.org/abs/2306.06362) | `adt_realtime_qa` | <https://www.projectaria.com/datasets/adt/> — download via [`projectaria_tools`](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset) |

The three that are one command:

```bash
huggingface-cli download lmms-lab/LLaVA-Video-178K --repo-type dataset \
  --local-dir $DATA_DIR/LLaVA-Video-178K

huggingface-cli download PolyU-ChenLab/ET-Instruct-164K --repo-type dataset \
  --local-dir $DATA_DIR/ET-Instruct-164K

wget -P $DATA_DIR/QVHighlight/ https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz
```

Ego4D, after approval, via the official CLI — it already writes the `v2/clips/`
layout used below:

```bash
ego4d --output_directory $DATA_DIR/Ego4D --datasets clips
```

ActivityNet and ADT have no public one-liner; follow the links above and place
the results in `$DATA_DIR/ActivityNet/videos/` and `$DATA_DIR/ADT/videos/`.

Only the sources you actually train on are needed; `--dataset_use` selects the
mixture (see [Dataset registry](#dataset-registry)).


## Dataset registry

`--dataset_use` takes a comma-separated list of the keys below, defined in
[`streamttt/data/__init__.py`](../streamttt/data/__init__.py). A key may carry a
sampling suffix: `llava_178k_long%50` keeps a random 50% of it.

| # | key | count | annotation path | video directory |
|---|---|---|---|---|
| 1 | `llava_178k_long` | 6 files | `LLaVA-Video-178K/2_3_m_{academic,youtube}_v0_1/*_processed.json` | `LLaVA-Video-178K/<data_source>/<video>` |
| 2 | `llava_realtime_perception` | 31,619 | `LLaVA-Video-178K/llava_realtime_perception_processed.json` | `LLaVA-Video-178K/<data_source>/<video>` |
| 3 | `activitynet_realtime_qa` | 9,016 | `ActivityNet/qa_ActivityNet.json` | `ActivityNet/videos/<video>` |
| 4 | `qvhighlight_realtime_qa` | 19,577 | `QVHighlight/qa_QVHighlight.json` | `QVHighlight/videos/<video>` |
| 5 | `egotime_realtime_qa` | 6,838 | `ET-Instruct-164K/qa_EgoTimeQA.json` | `ET-Instruct-164K/videos/<data_source>/<video>` |
| 6 | `howtocaption_realtime_qa` | 12,733 | `ET-Instruct-164K/qa_how_to_caption.json` | `ET-Instruct-164K/videos/<data_source>/<video>` |
| 7 | `ego4d_realtime_qa` | 22,493 | `Ego4D/qa_Ego4D.json` | `Ego4D/v2/clips/<video>` |
| 8 | `adt_realtime_qa` | 6,861 | `ADT/qa_ADT.json` | `ADT/videos/<video>` |
| 9 | `activitynet_realtime_caption` | 2,965 | `ActivityNet/caption_activitynet.json` | `ActivityNet/videos/<video>` |
| – | `tmp_dataset` | 3 | `demo/tmp_dataset/videos.json` | `demo/tmp_dataset/videos/` |

Annotation paths resolve as `DATA_DIR / data_path / annotation_path`. Video
paths are assembled per dataset in
[`streamttt/data/qwen_video_dataset.py`](../streamttt/data/qwen_video_dataset.py)
(`get_video_path`).

The 9 keys above are exactly the mixture used for the released model:

```
llava_178k_long%50,llava_realtime_perception,activitynet_realtime_qa,
qvhighlight_realtime_qa,egotime_realtime_qa,howtocaption_realtime_qa,
ego4d_realtime_qa,adt_realtime_qa,activitynet_realtime_caption
```

## Directory layout

The tree you assemble locally. Only the `*.json` annotation files come from
`zeyun-zhong/RealTimeVideo-Instruct-112K`; every video directory is populated
from its own source per [Download](#download).

```
DATA_DIR
├── LLaVA-Video-178K
│   ├── 2_3_m_academic_v0_1
│   │   ├── 2_3_m_academic_mc_v0_1_qa_processed.json      (6,901)
│   │   ├── 2_3_m_academic_oe_v0_1_qa_processed.json      (18,134)
│   │   ├── 2_3_m_academic_v0_1_cap_processed.json        (3,124)
│   │   └── academic_source/…                             (videos)
│   ├── 2_3_m_youtube_v0_1
│   │   ├── 2_3_m_youtube_mc_v0_1_qa_processed.json       (39,967)
│   │   ├── 2_3_m_youtube_oe_v0_1_qa_processed.json       (141,495)
│   │   ├── 2_3_m_youtube_v0_1_cap_processed.json         (24,685)
│   │   └── liwei_youtube_videos/videos/youtube_video_2024/…
│   └── llava_realtime_perception_processed.json
├── ActivityNet
│   ├── qa_ActivityNet.json
│   ├── caption_activitynet.json
│   └── videos/                                           (includes .mkv)
├── QVHighlight
│   ├── qa_QVHighlight.json
│   └── videos/
├── ET-Instruct-164K
│   ├── qa_EgoTimeQA.json
│   ├── qa_how_to_caption.json
│   └── videos/{ego_timeqa,how_to_caption}/
├── Ego4D
│   ├── qa_Ego4D.json
│   └── v2/clips/
└── ADT
    ├── qa_ADT.json
    └── videos/
```

## How the real-time corpus was built

The construction procedure is documented in Appendix B of the paper: proactive
queries from Streamo are relocated to their annotated answer time
(`t_q := t_a`), and additional action, spatial-reasoning, anticipation, and
captioning supervision is derived directly from ground-truth annotations of
EgoTimeQA, Aria Digital Twin, and Ego4D Short-Term Anticipation. The resulting
annotations are what ships in the `zeyun-zhong/RealTimeVideo-Instruct-112K`
dataset above; the generation scripts are not part of this repository.
