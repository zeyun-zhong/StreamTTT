from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
apply_liger_kernel_to_qwen3_vl(rope=False, fused_linear_cross_entropy=False)

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


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]

SPLIT_TO_TASKS = {
    "backward": BACKWARD_TASKS,
    "realtime": REAL_TIME_TASKS,
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
        max_window_duration=512,
        splits=("backward", "realtime"),
        tasks=None,
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
    splits=("backward", "realtime"),
    tasks=None,
    fps=None,
    checkpoint_path: str | None = None,
    chunk_size: int | None = None,
    resume: bool = False,
    **video_kwargs,
):
    dataset = OVOBRMultiForwardDataset(
        benchmark_path,
        answer_prefix="Answer:",
        max_window_duration=max_window_duration,
        splits=splits,
        tasks=tasks,
        fps=fps,
        **video_kwargs,
    )
    return run_mcq_prediction(
        model=model,
        processor=processor,
        dataset=dataset,
        letters=["A", "B", "C", "D", "E"],
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
        choices=["backward", "realtime"],
        default=["backward", "realtime"],
        help="OVO-Bench MCQ splits to evaluate.",
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
        dataloader_num_workers=args.num_workers,
        multi_forward_training=args.multi_forward_training,
        max_inference_context_window=args.max_inference_context_window,
        max_window_duration=args.max_window_duration,
        splits=args.splits,
        tasks=args.tasks,
        fps=args.fps,
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
