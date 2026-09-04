from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
apply_liger_kernel_to_qwen3_vl(rope=False)

import ast
import csv
import json
import os
import re
import argparse
from torch.utils.data import Dataset

from evaluation.utils import (
    load_eval_model,
    prepare_multiforward_batch,
    run_mcq_prediction,
    save_function_print,
)

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
        max_window_duration=512,
        task_types=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.max_window_duration = max_window_duration
        self.answer_prefix = answer_prefix
        self.video_max_pixels = video_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_total_pixels = video_total_pixels
        self.max_frames = max_frames
        self.fps = fps

        all_datums = load_annotation_csv(data_path)

        if task_types is not None:
            all_datums = [d for d in all_datums if d["task_type"] in task_types]

        self.datums = all_datums
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
        return prepare_multiforward_batch(
            conversations,
            processor,
            self.answer_prefix,
            max_window_duration=self.max_window_duration,
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
    dataloader_num_workers: int = 2,
    multi_forward_training: bool = True,
    max_inference_context_window: int = 128000,
    max_window_duration: int = 512,
    task_types=None,
    fps=None,
    checkpoint_path: str | None = None,
    chunk_size: int | None = None,
    resume: bool = False,
    **video_kwargs,
):
    dataset = StreamingBenchMultiForwardDataset(
        benchmark_path,
        answer_prefix="Answer:",
        max_window_duration=max_window_duration,
        task_types=task_types,
        fps=fps,
        **video_kwargs,
    )
    return run_mcq_prediction(
        model=model,
        processor=processor,
        dataset=dataset,
        letters=["A", "B", "C", "D"],
        dataloader_num_workers=dataloader_num_workers,
        multi_forward_training=multi_forward_training,
        max_inference_context_window=max_inference_context_window,
        checkpoint_path=checkpoint_path,
        chunk_size=chunk_size,
        resume=resume,
    )


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

    model, processor, video_kwargs = load_eval_model(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        processor_name=args.processor_name,
        max_inference_context_window=args.max_inference_context_window,
        multi_forward_training=args.multi_forward_training,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_total_pixels=args.video_total_pixels,
        max_frames=args.max_frames,
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

    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model,
        processor=processor,
        benchmark_path=args.benchmark_path,
        dataloader_num_workers=args.num_workers,
        multi_forward_training=args.multi_forward_training,
        max_inference_context_window=args.max_inference_context_window,
        max_window_duration=args.max_window_duration,
        task_types=args.task_types,
        fps=args.fps,
        checkpoint_path=checkpoint_path,
        chunk_size=args.save_every_n_samples,
        resume=args.resume,
        **video_kwargs,
    )

    if process_index == 0:
        letters = ['A', 'B', 'C', 'D']
        results = assemble_results(benchmark_datums, letter_idxs_predictions, letters)

        with open(save_json_path, 'w') as f:
            json.dump(results, f, indent=2)

        save_txt_path = save_json_path.replace('.json', '.txt')

        save_function_print(print_streamingbench_results, save_txt_path, results)

        print(f"\nResults saved to: {save_json_path}")
        print_streamingbench_results(results)


if __name__ == '__main__':
    main()
