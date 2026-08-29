"""Dataset registry.

Each entry maps a `--dataset_use` key to an annotation file (or list of files)
relative to `--data_dir`. Video paths are resolved separately in
`qwen_video_dataset.get_video_path()`. See docs/DATA.md for the directory
layout and download instructions.

A key may carry a sampling suffix, e.g. `llava_178k_long%50` keeps 50% of it.
"""

import re

# --- Demo: three short clips shipped in demo/, for smoke tests ---
tmp_dataset = {
    "data_path": "tmp_dataset",
    "annotation_path": "videos.json"
}

# --- Offline whole-video supervision (paper §4.1 / App. B.1, ~119K QA) ---
llava_178k_long = {
    "data_path": "LLaVA-Video-178K",
    "annotation_path": [
        "2_3_m_academic_v0_1/2_3_m_academic_mc_v0_1_qa_processed.json",
        "2_3_m_academic_v0_1/2_3_m_academic_oe_v0_1_qa_processed.json",
        "2_3_m_academic_v0_1/2_3_m_academic_v0_1_cap_processed.json",
        "2_3_m_youtube_v0_1/2_3_m_youtube_mc_v0_1_qa_processed.json",
        "2_3_m_youtube_v0_1/2_3_m_youtube_oe_v0_1_qa_processed.json",
        "2_3_m_youtube_v0_1/2_3_m_youtube_v0_1_cap_processed.json",
    ]
}

# --- Real-time QA corpus (paper §4.1 / App. B, 112.4K QA) ---
llava_realtime_perception = {
    "data_path": "LLaVA-Video-178K",
    "annotation_path": "llava_realtime_perception_processed.json",
}

activitynet_realtime_qa = {
    "data_path": "ActivityNet",
    "annotation_path": "qa_ActivityNet.json",
}

qvhighlight_realtime_qa = {
    "data_path": "QVHighlight",
    "annotation_path": "qa_QVHighlight.json",
}

egotime_realtime_qa = {
    "data_path": "ET-Instruct-164K",
    "annotation_path": "qa_EgoTimeQA.json",
}

howtocaption_realtime_qa = {
    "data_path": "ET-Instruct-164K",
    "annotation_path": "qa_how_to_caption.json",
}

ego4d_realtime_qa = {
    "data_path": "Ego4D",
    "annotation_path": "qa_Ego4D.json",
}

adt_realtime_qa = {
    "data_path": "ADT",
    "annotation_path": "qa_ADT.json",
}

activitynet_realtime_caption = {
    "data_path": "ActivityNet",
    "annotation_path": "caption_activitynet.json",
}


data_dict = {
    "tmp_dataset": tmp_dataset,
    "llava_178k_long": llava_178k_long,
    "llava_realtime_perception": llava_realtime_perception,
    "activitynet_realtime_qa": activitynet_realtime_qa,
    "qvhighlight_realtime_qa": qvhighlight_realtime_qa,
    "egotime_realtime_qa": egotime_realtime_qa,
    "howtocaption_realtime_qa": howtocaption_realtime_qa,
    "ego4d_realtime_qa": ego4d_realtime_qa,
    "adt_realtime_qa": adt_realtime_qa,
    "activitynet_realtime_caption": activitynet_realtime_caption,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(
                f"Unknown dataset key {dataset_name!r}. Available: {sorted(data_dict)}"
            )
    return config_list
