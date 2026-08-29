from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl, apply_liger_kernel_to_qwen3_vl
apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
apply_liger_kernel_to_qwen3_vl(rope=False, fused_linear_cross_entropy=False)

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


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

SPLIT_TO_TASKS = {
    "backward": BACKWARD_TASKS,
    "realtime": REAL_TIME_TASKS,
    "forward": FORWARD_TASKS,
}

BR_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Options:\n"
    "{options}\n\n"
    "Respond only with the letter corresponding to your chosen option (e.g., A, B, C). "
    "Do not include any additional text or explanation in your response."
)

HLD_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Options:\n"
    "{options}\n\n"
    "If there is no visual evidence in the video to answer the question, select 'Unable to answer'. "
    "Respond only with the letter corresponding to your chosen option (e.g., A, B, C). "
    "Do not include any additional text or explanation in your response."
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def extract_br_answer(response):
    if not response or not str(response).strip():
        return None
    s = str(response).strip().upper()
    m = re.search(r"\b([A-E])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([1-5])\b", s)
    if m:
        return chr(64 + int(m.group(1)))
    return None


def score_br(response, gt):
    pred = extract_br_answer(response)
    return 1 if (pred is not None and pred.upper() == gt.upper()) else 0


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class OVOBRMultiForwardDataset(Dataset):
    """Multi-forward dataset for backward + realtime (MCQ) tasks."""

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
        splits=("backward", "realtime"),
        tasks=None,
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

        annotation_file = "ovo_bench_new.json"
        with open(os.path.join(data_path, annotation_file)) as f:
            all_datums = json.load(f)

        allowed_tasks = []
        for split in splits:
            assert split in SPLIT_TO_TASKS, f"split must be one of {list(SPLIT_TO_TASKS.keys())}, got '{split}'"
            allowed_tasks.extend(SPLIT_TO_TASKS[split])

        if tasks:
            invalid = [t for t in tasks if t not in allowed_tasks]
            if invalid:
                raise ValueError(f"Tasks {invalid} not found in selected splits {splits}. Available: {allowed_tasks}")
            allowed_tasks = tasks

        self.datums = [d for d in all_datums if d["task"] in allowed_tasks]
        if max_samples is not None:
            self.datums = self.datums[:max_samples]
        print(f"Filtered to splits={splits}: {len(self.datums)} samples (tasks: {allowed_tasks})")

    def __getitem__(self, index):
        di = self.datums[index]
        video_path = os.path.join(self.data_path, "chunked_videos", str(di["id"]))
        conversation = [{"role": "user", "content": []}]

        content = {
            "type": "video",
            "video": f"{video_path}.mp4",
            "min_pixels": self.video_min_pixels,
            "max_pixels": self.video_max_pixels,
            "total_pixels": self.video_total_pixels,
            "max_frames": self.max_frames,
        }
        if self.fps is not None:
            content["fps"] = self.fps
        conversation[0]['content'].append(content)

        options_str = "\n".join(f"{chr(65 + i)}. {s}" for i, s in enumerate(di["options"]))
        template = HLD_PROMPT_TEMPLATE if di["task"] == "HLD" else BR_PROMPT_TEMPLATE
        query = template.format(question=di["question"], options=options_str)
        conversation[0]['content'].append({"type": "text", "text": query})
        return conversation

    def data_collator(self, conversations, processor):
        assert len(conversations) == 1, "We assume batch size is 1 for the multi-forward testing."
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
        window_v_ids = [video_inputs["input_ids"] for video_inputs in window_video_inputs]
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
    letters: list = ['A', 'B', 'C', 'D', 'E'],
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = False,
    per_device_eval_batch_size: int = 1,
    dataloader_num_workers: int = 2,
    multi_forward_training: bool = False,
    max_inference_context_window: int = 128000,
    max_window_duration: int = 512,
    splits=("backward", "realtime"),
    tasks=None,
    fps=None,
    max_samples=None,
    checkpoint_path: str | None = None,
    chunk_size: int | None = None,
    resume: bool = False,
    **video_kwargs,
):
    strict_letter_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in letters]

    dataset = OVOBRMultiForwardDataset(
        benchmark_path,
        answer_prefix='Answer:',
        processor=processor,
        max_window_duration=max_window_duration,
        splits=splits,
        tasks=tasks,
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
        preprocess_logits_for_metrics=functools.partial(preprocess_logits_for_metrics, strict_letter_ids=strict_letter_ids),
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
        gt_letter = chr(65 + datum['gt'])
        results.append({
            "id": str(datum['id']),
            "task": datum['task'],
            "video": datum['video'],
            "question": datum['question'],
            "options": datum['options'],
            "response": predicted_letter,
            "ground_truth": gt_letter,
        })
    return results


def print_ovo_results(results: list):
    from collections import defaultdict

    task_to_scores = defaultdict(list)
    for item in results:
        task_to_scores[item['task']].append(score_br(item['response'], item['ground_truth']))

    category_scores = []
    for section_name, task_group, title in (
        ("backward", BACKWARD_TASKS, "Backward Tracing"),
        ("realtime", REAL_TIME_TASKS, "Real-time Perception"),
    ):
        tasks_in_results = [t for t in task_group if t in task_to_scores]
        if not tasks_in_results:
            continue
        print(f"\n{title}:")
        accs = []
        for task in tasks_in_results:
            scores = task_to_scores[task]
            acc = 100.0 * sum(scores) / len(scores)
            print(f"  {task}: {acc:.2f}% ({sum(scores)}/{len(scores)})")
            accs.append(acc)
        avg = sum(accs) / len(accs)
        category_scores.append(avg)
        print(f"  {title.split()[0]} Avg.: {avg:.2f}%")

    if category_scores:
        total_avg = sum(category_scores) / len(category_scores)
        print(f"\n{'=' * 60}")
        print(f"Total Avg.: {total_avg:.2f}%")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation for OVO-Bench (backward + realtime splits)"
    )
    parser.add_argument("--model_type", type=str, default='streamttt')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--processor_name", type=str, required=True)
    parser.add_argument("--benchmark_path", type=str, required=True)
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
        "--save_every_n_samples", type=int, default=32,
        help="Save intermediate predictions after each chunk of this many samples.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume evaluation from the intermediate checkpoint if it exists.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["backward", "realtime", "forward"],
        default=["backward", "realtime"],
        help="Splits to evaluate. 'forward' (REC, SSR, CRR) is not yet supported.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional subset of tasks to evaluate (e.g. --tasks HLD EPM). Must belong to the selected splits.",
    )

    args = parser.parse_args()

    if not args.multi_forward_training:
        raise ValueError(
            "OVO-Bench evaluation is streaming-only: pass --multi_forward_training. "
            "Each question is scored against its pre-chunked video (chunked_videos/{id}.mp4), "
            "which is cut at the question's `realtime` timestamp."
        )

    if "forward" in args.splits:
        raise NotImplementedError(
            "Forward tasks (REC, SSR, CRR) require free-form generation and are not yet supported. "
            "Use --splits backward realtime."
        )

    # Load model and processor
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

    out_dir = 'evaluation/ovo_bench/results'
    os.makedirs(out_dir, exist_ok=True)
    splits_str = "-".join(sorted(args.splits))
    sufix = f"_multiforward_window{args.max_window_duration}" if args.multi_forward_training else ""
    base_name = (
        f"{os.path.basename(args.model_name_or_path.rstrip('/'))}"
        f"_tpf{args.video_max_pixels}_frame{args.max_frames}_{splits_str}{sufix}_officialprompt"
    )
    save_json_path = os.path.join(out_dir, f"{base_name}.json")
    checkpoint_path = os.path.join(out_dir, f"{base_name}.checkpoint.json")

    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model,
        processor=processor,
        benchmark_path=args.benchmark_path,
        letters=['A', 'B', 'C', 'D', 'E'],
        use_liger_kernel=False,
        dataloader_num_workers=args.num_workers,
        per_device_eval_batch_size=1,
        multi_forward_training=args.multi_forward_training,
        max_inference_context_window=args.max_inference_context_window,
        max_window_duration=args.max_window_duration,
        splits=args.splits,
        tasks=args.tasks,
        fps=args.fps,
        max_samples=None,
        checkpoint_path=checkpoint_path,
        chunk_size=args.save_every_n_samples,
        resume=args.resume,
        **video_kwargs,
    )

    if process_index == 0:
        letters = ['A', 'B', 'C', 'D', 'E']
        results = assemble_results(benchmark_datums, letter_idxs_predictions, letters)

        with open(save_json_path, 'w') as f:
            json.dump(results, f, indent=2)

        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(print_ovo_results, save_txt_path, results)

        print(f"\nResults saved to: {save_json_path}")
        print_ovo_results(results)


if __name__ == '__main__':
    main()
