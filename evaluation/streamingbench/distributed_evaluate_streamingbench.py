from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl, apply_liger_kernel_to_qwen3_vl
apply_liger_kernel_to_qwen2_5_vl()
apply_liger_kernel_to_qwen3_vl(rope=False)

import ast
import csv
import json
import os
import re
import functools
import argparse
import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoConfig, AutoProcessor
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE

from evaluation.utils import (
    load_prediction_checkpoint,
    preprocess_logits_for_metrics,
    save_function_print,
    save_prediction_checkpoint,
)
from streamttt.models import ALL_MODELS
from streamttt.train.trainer import StreamTTTTrainer
from streamttt.train.arguments import TrainingArguments
from streamttt.streaming.windowing import (
    build_model_inputs,
    build_prompt_ids,
    build_token_offsets,
    build_window_prompt_ids,
    build_window_ranges_from_timestamps,
    get_sampled_frame_timestamps,
    prepare_video_only_inputs,
    slice_video_frames,
    slice_video_metadata,
)

IGNORE_KEYS_FOR_PREDICTION = [
    'past_key_values', 'hidden_states', 'attentions', 'rope_deltas', 'states'
]

ANNOTATION_FILE = "StreamingBench/Real_Time_Visual_Understanding.csv"

MCQ_PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with some frames from the video and a multiple-choice question. "
    "Your task is to analyze the video and provide the best answer.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Only give the best option's letter (A, B, C, or D) directly."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_options(raw):
    """Parse the options field from the CSV into a list of strings."""
    if isinstance(raw, list):
        return raw
    raw = raw.strip()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [raw]


def load_annotation_csv(data_path):
    annotation_path = os.path.join(data_path, ANNOTATION_FILE)
    datums = []
    with open(annotation_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            datums.append({
                "question_id": row["question_id"],
                "task_type": row["task_type"],
                "question": row["question"],
                "time_stamp": row["time_stamp"],
                "answer": row["answer"].strip().upper(),
                "options": parse_options(row["options"]),
                "frames_required": row.get("frames_required", ""),
                "temporal_clue_type": row.get("temporal_clue_type", ""),
            })
    return datums


def parse_timestamp_to_seconds(ts: str) -> float:
    """Parse a timestamp like '0:00:10' or '1:23:45' into seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def extract_answer(response):
    if not response or not str(response).strip():
        return None
    s = str(response).strip().upper()
    m = re.search(r"[A-D]", s)
    return m.group(0) if m else None


def score_mcq(response, gt):
    pred = extract_answer(response)
    return 1 if (pred is not None and pred.upper() == gt.upper()) else 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StreamingBenchMultiForwardDataset(Dataset):
    """Multi-forward dataset for StreamingBench Real_Time_Visual_Understanding."""

    def __init__(
        self,
        data_path,
        answer_prefix: str = 'Answer:',
        video_max_pixels=28 * 28 * 768,
        video_min_pixels=28 * 28 * 100,
        video_total_pixels=32000 * 28 * 28 * 0.9,
        max_frames=768,
        fps=None,
        is_qwen3=True,
        processor=None,
        max_window_duration=512,
        task_types=None,
        max_samples=None,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.processor = processor
        self.max_window_duration = max_window_duration
        self.answer_prefix = answer_prefix
        self.video_max_pixels = video_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_total_pixels = video_total_pixels
        self.max_frames = max_frames
        self.fps = fps
        self.is_qwen3 = is_qwen3
        self.image_patch_size = 16 if self.is_qwen3 else 14

        all_datums = load_annotation_csv(data_path)

        if task_types is not None:
            all_datums = [d for d in all_datums if d["task_type"] in task_types]

        self.datums = all_datums if max_samples is None else all_datums[:max_samples]
        print(f"Loaded {len(self.datums)} samples from StreamingBench"
              + (f" (task_types: {task_types})" if task_types else ""))

    def _video_path(self, datum):
        # question_id format: "Real-Time Visual Understanding_sample_1_1"
        # videos live at: real_time_videos/sample_1/video.mp4
        m = re.search(r'_sample_(\d+)_', datum['question_id'])
        sample_dir = f"sample_{m.group(1)}" if m else datum['question_id']
        return os.path.join(self.data_path, "real_time_videos", sample_dir, "video.mp4")

    def __getitem__(self, index):
        di = self.datums[index]
        video_path = self._video_path(di)
        conversation = [{"role": "user", "content": []}]

        content = {
            "type": "video",
            "video": video_path,
            "min_pixels": self.video_min_pixels,
            "max_pixels": self.video_max_pixels,
            "total_pixels": self.video_total_pixels,
            "max_frames": self.max_frames,
            "video_end": parse_timestamp_to_seconds(di["time_stamp"]),
        }
        if self.fps is not None:
            content["fps"] = self.fps
        conversation[0]['content'].append(content)

        options_str = "\n".join(di["options"])
        query = MCQ_PROMPT_TEMPLATE.format(question=di["question"], options=options_str)
        conversation[0]['content'].append({"type": "text", "text": query})
        return conversation

    def data_collator(self, conversations, processor):
        assert len(conversations) == 1, "We assume batch size is 1 for multi-forward testing."
        conversation = conversations[0]
        video_info = conversation[0]["content"][0]
        question = conversation[0]["content"][1]["text"]

        messages = [{"role": "user", "content": [video_info, {"type": "text", "text": question}]}]
        _, video_inputs_data, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=self.is_qwen3,
        )

        if self.is_qwen3:
            videos, video_metadatas = zip(*video_inputs_data)
            videos, video_metadatas = list(videos), list(video_metadatas)
            full_video = videos[0]
            full_metadata = video_metadatas[0]
        else:
            full_video = video_inputs_data[0]
            full_metadata = None

        sampled_frames = int(full_video.shape[0])
        _, sampled_frame_times_sec, video_duration_sec, _ = get_sampled_frame_timestamps(
            video_info["video"], full_metadata, sampled_frames
        )
        temporal_patch_size = getattr(processor.video_processor, "temporal_patch_size", 2)
        window_ranges, _ = build_window_ranges_from_timestamps(
            sampled_frame_times_sec,
            video_duration_sec,
            self.max_window_duration,
            temporal_patch_size=temporal_patch_size,
        )

        window_videos = [
            slice_video_frames(full_video, start, end) for start, end in window_ranges
        ]
        window_metadatas = [
            slice_video_metadata(full_metadata, start, end) for start, end in window_ranges
        ]

        system_user_ids, qa_ids = build_prompt_ids(
            processor, question, "cpu", answer_prefix=self.answer_prefix
        )
        full_video_inputs = prepare_video_only_inputs(
            processor, full_video, full_metadata, video_kwargs, "cpu"
        )
        window_video_inputs = [
            prepare_video_only_inputs(processor, video, metadata, video_kwargs, "cpu")
            for video, metadata in zip(window_videos, window_metadatas)
        ]
        full_v_ids = full_video_inputs["input_ids"]
        window_v_ids = [vi["input_ids"] for vi in window_video_inputs]
        cat_v_ids = torch.cat(window_v_ids, dim=1)
        if not torch.equal(cat_v_ids, full_v_ids):
            raise RuntimeError(
                "Split video tokenization does not reconstruct the full video token block."
            )

        window_ids = build_window_prompt_ids(system_user_ids, qa_ids, window_v_ids)
        full_ids = torch.cat([system_user_ids, full_v_ids, qa_ids], dim=1)
        if not torch.equal(torch.cat(window_ids, dim=1), full_ids):
            raise RuntimeError("Windowed prompt ids do not concatenate back to the full prompt ids.")

        processed_windows = [
            build_model_inputs(ids, video_inputs)
            for ids, video_inputs in zip(window_ids, window_video_inputs)
        ]
        for window in processed_windows:
            window["pixel_values_videos"] = window["pixel_values_videos"].bfloat16()

        del full_video_inputs
        window_token_offsets = build_token_offsets([ids.shape[1] for ids in window_ids])
        window_cache_positions = [
            torch.arange(start, end, dtype=torch.long)
            for start, end in window_token_offsets
        ]
        cumulative_window_attention_masks = [
            torch.ones_like(full_ids)[:, :end]
            for _, end in window_token_offsets
        ]

        return dict(
            processed_windows=processed_windows,
            window_cache_positions=window_cache_positions,
            cumulative_window_attention_masks=cumulative_window_attention_masks,
        )

    def __len__(self):
        return len(self.datums)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def mcq_predict(
    model,
    processor,
    benchmark_path: str,
    letters: list = ['A', 'B', 'C', 'D'],
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = False,
    per_device_eval_batch_size: int = 1,
    dataloader_num_workers: int = 2,
    multi_forward_training: bool = False,
    max_inference_context_window: int = 128000,
    max_window_duration: int = 512,
    task_types=None,
    fps=None,
    max_samples=None,
    checkpoint_path: str | None = None,
    chunk_size: int | None = None,
    resume: bool = False,
    **video_kwargs,
):
    strict_letter_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in letters]

    dataset = StreamingBenchMultiForwardDataset(
        benchmark_path,
        answer_prefix='Answer:',
        processor=processor,
        max_window_duration=max_window_duration,
        task_types=task_types,
        fps=fps,
        max_samples=max_samples,
        **video_kwargs,
    )
    trainer = StreamTTTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir='outputs/', do_predict=True,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            report_to='none', use_liger_kernel=use_liger_kernel,
            multi_forward_training=multi_forward_training,
            max_inference_context_window=max_inference_context_window,
            remove_unused_columns=False,
        ),
        data_collator=functools.partial(dataset.data_collator, processor=processor),
        processing_class=processor,
        preprocess_logits_for_metrics=functools.partial(
            preprocess_logits_for_metrics, strict_letter_ids=strict_letter_ids
        ),
    )
    process_index = trainer.args.process_index

    if chunk_size is None or chunk_size <= 0:
        chunk_size = len(dataset)

    saved_predictions = load_prediction_checkpoint(
        checkpoint_path,
        expected_total=len(dataset),
    ) if (resume and checkpoint_path is not None) else {}

    if process_index == 0 and saved_predictions:
        print(f"Resuming from {len(saved_predictions)}/{len(dataset)} saved predictions at {checkpoint_path}.")

    for chunk_start in range(0, len(dataset), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        chunk_indices = list(range(chunk_start, chunk_end))
        pending_indices = [idx for idx in chunk_indices if idx not in saved_predictions]
        if not pending_indices:
            continue

        chunk_dataset = Subset(dataset, pending_indices)
        chunk_predictions = trainer.predict(
            chunk_dataset,
            ignore_keys=IGNORE_KEYS_FOR_PREDICTION,
        ).predictions

        if process_index == 0:
            if len(chunk_predictions) != len(pending_indices):
                raise RuntimeError(
                    f"Prediction count mismatch for indices {pending_indices[0]}-{pending_indices[-1]}: "
                    f"got {len(chunk_predictions)} predictions for {len(pending_indices)} samples."
                )
            for datum_idx, prediction in zip(pending_indices, chunk_predictions):
                saved_predictions[datum_idx] = int(prediction)
            if checkpoint_path is not None:
                save_prediction_checkpoint(
                    checkpoint_path,
                    predictions_by_index=saved_predictions,
                    total_samples=len(dataset),
                )
            print(f"Saved {len(saved_predictions)}/{len(dataset)} predictions.")

    if process_index != 0:
        return None, dataset.datums, process_index

    missing_indices = [idx for idx in range(len(dataset)) if idx not in saved_predictions]
    if missing_indices:
        raise RuntimeError(f"Missing predictions for indices: {missing_indices[:10]}")

    ordered_predictions = [saved_predictions[idx] for idx in range(len(dataset))]
    return ordered_predictions, dataset.datums, process_index


# ---------------------------------------------------------------------------
# Scoring and reporting
# ---------------------------------------------------------------------------

def assemble_results(datums, letter_idxs_predictions, letters):
    results = []
    for datum, letter_idx in zip(datums, letter_idxs_predictions):
        idx = int(letter_idx)
        predicted_letter = letters[idx] if idx < len(letters) else ""
        results.append({
            "question_id": datum["question_id"],
            "task_type": datum["task_type"],
            "question": datum["question"],
            "time_stamp": datum["time_stamp"],
            "options": datum["options"],
            "response": predicted_letter,
            "ground_truth": datum["answer"],
            "frames_required": datum["frames_required"],
            "temporal_clue_type": datum["temporal_clue_type"],
        })
    return results


def print_streamingbench_results(results: list):
    from collections import defaultdict

    task_to_scores = defaultdict(list)
    for item in results:
        task_to_scores[item["task_type"]].append(
            score_mcq(item["response"], item["ground_truth"])
        )

    all_scores = []
    print("\nStreamingBench Real-Time Visual Understanding Results:")
    print("=" * 60)
    for task_type in sorted(task_to_scores.keys()):
        scores = task_to_scores[task_type]
        acc = 100.0 * sum(scores) / len(scores)
        print(f"  {task_type}: {acc:.2f}% ({sum(scores)}/{len(scores)})")
        all_scores.extend(scores)

    if all_scores:
        overall = 100.0 * sum(all_scores) / len(all_scores)
        print(f"\n{'=' * 60}")
        print(f"Overall Accuracy: {overall:.2f}% ({sum(all_scores)}/{len(all_scores)})")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Distributed evaluation for StreamingBench Real_Time_Visual_Understanding"
    )
    parser.add_argument("--model_type", type=str, default='streamttt')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--processor_name", type=str, required=True)
    parser.add_argument("--benchmark_path", type=str, required=True,
                        help="Directory containing Real_Time_Visual_Understanding.csv and videos/")
    parser.add_argument("--video_max_pixels", type=int, default=768)
    parser.add_argument("--video_min_pixels", type=int, default=100)
    parser.add_argument("--video_total_pixels", type=int, default=256000)
    parser.add_argument("--max_frames", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--multi_forward_training", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_inference_context_window", type=int, default=4096)
    parser.add_argument("--max_window_duration", type=int, default=512)
    parser.add_argument(
        "--task_types",
        nargs="+",
        default=None,
        help="Optional subset of task_types to evaluate (e.g. --task_types Causal_Reasoning Object_Tracking).",
    )
    parser.add_argument(
        "--save_every_n_samples", type=int, default=32,
        help="Save intermediate predictions after each chunk of this many samples.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume evaluation from the intermediate checkpoint if it exists.",
    )

    args = parser.parse_args()

    if not args.multi_forward_training:
        raise ValueError(
            "StreamingBench evaluation is streaming-only: pass --multi_forward_training. "
            "Each question is scored over the video truncated at its `time_stamp`, consumed "
            "as a sequence of temporal windows."
        )

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if hasattr(config.text_config, "sliding_window"):
        if config.text_config.sliding_window != args.max_inference_context_window:
            print(
                f"Saved sliding window ({config.text_config.sliding_window}) differs from "
                f"requested ({args.max_inference_context_window}). Overriding."
            )
    config.text_config.sliding_window = args.max_inference_context_window
    if args.multi_forward_training:
        config.output_states = True
        if hasattr(config, "text_config"):
            config.text_config.output_states = True

    model_cls = ALL_MODELS[args.model_type]
    model = model_cls.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.processor_name, padding_side='left')

    is_qwen3 = True  # both supported model types are Qwen3-VL based
    image_patch_size = 16 if is_qwen3 else 14
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE

    video_kwargs = dict(
        video_max_pixels=args.video_max_pixels * image_factor * image_factor,
        video_min_pixels=args.video_min_pixels * image_factor * image_factor,
        video_total_pixels=args.video_total_pixels * image_factor * image_factor,
        max_frames=args.max_frames,
        is_qwen3=is_qwen3,
    )

    out_dir = 'evaluation/streamingbench/results'
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_multiforward_window{args.max_window_duration}" if args.multi_forward_training else ""
    base_name = (
        f"{os.path.basename(args.model_name_or_path.rstrip('/'))}"
        f"_tpf{args.video_max_pixels}_frame{args.max_frames}{suffix}"
    )
    save_json_path = os.path.join(out_dir, f"{base_name}.json")
    checkpoint_path = os.path.join(out_dir, f"{base_name}.checkpoint.json")

    letters = ['A', 'B', 'C', 'D']
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model,
        processor=processor,
        benchmark_path=args.benchmark_path,
        letters=letters,
        use_liger_kernel=False,
        dataloader_num_workers=args.num_workers,
        per_device_eval_batch_size=1,
        multi_forward_training=args.multi_forward_training,
        max_inference_context_window=args.max_inference_context_window,
        max_window_duration=args.max_window_duration,
        task_types=args.task_types,
        fps=args.fps,
        max_samples=None,
        checkpoint_path=checkpoint_path,
        chunk_size=args.save_every_n_samples,
        resume=args.resume,
        **video_kwargs,
    )

    if process_index == 0:
        results = assemble_results(benchmark_datums, letter_idxs_predictions, letters)

        with open(save_json_path, 'w') as f:
            json.dump(results, f, indent=2)

        save_txt_path = save_json_path.replace('.json', '.txt')

        save_function_print(print_streamingbench_results, save_txt_path, results)

        print(f"\nResults saved to: {save_json_path}")
        print_streamingbench_results(results)


if __name__ == '__main__':
    main()
