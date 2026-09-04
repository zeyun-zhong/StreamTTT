from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
apply_liger_kernel_to_qwen3_vl(rope=False, fused_linear_cross_entropy=False)

import json
import os
import argparse
import pandas as pd
from torch.utils.data import Dataset

from evaluation.utils import (
    load_eval_model,
    prepare_multiforward_batch,
    prepare_single_forward_batch,
    run_mcq_prediction,
)


def load_egoschema_datums(data_path, subset=False):
    """Read the EgoSchema MC parquet into a list of plain dict datums.

    Note: ``option`` already carries its letter prefix (e.g. "A. ...") so it is
    kept verbatim and is *not* re-prefixed when building the query.

    When ``subset`` is True, the 500-question Subset parquet is loaded instead of
    the full hidden test set. The Subset carries ground-truth ``answer`` (an index
    string "0"-"4"), so accuracy can be computed locally.
    """
    annotation_file = "Subset/test-00000-of-00001.parquet" if subset else "MC/test-00000-of-00001.parquet"
    df = pd.read_parquet(os.path.join(data_path, annotation_file))
    datums = []
    for row in df.to_dict("records"):
        datums.append({
            "question_idx": str(row["question_idx"]),
            "question": row["question"],
            "video_idx": row["video_idx"],
            "option": list(row["option"]),
            "answer": row["answer"],
        })
    return datums


class EgoSchemaDataset(Dataset):
    def __init__(
            self,
            data_path,
            question_prefix: str = '',
            question_postfix: str = '\nPlease select the correct answer.',
            answer_prefix: str = 'Answer:',
            video_max_pixels=28 * 28 * 768,
            video_min_pixels=28 * 28 * 100,
            video_total_pixels=32000 * 28 * 28 * 0.9,
            max_frames=768,
            fps=None,
            subset=False,
    ):
        super().__init__()
        self.data_path = data_path
        self.subset = subset
        self.datums = load_egoschema_datums(data_path, subset=subset)

        self.question_prefix = question_prefix
        self.question_postfix = question_postfix
        self.answer_prefix = answer_prefix
        self.video_max_pixels = video_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_total_pixels = video_total_pixels
        self.max_frames = max_frames
        self.fps = fps

    def build_query(self, di):
        # options already carry their letter prefix ("A. ...") -> keep verbatim
        return (self.question_prefix +
                di["question"] + '\n' +
                '\n'.join(di["option"]) +
                self.question_postfix)

    def __getitem__(self, index):
        di = self.datums[index]
        video_path = os.path.join(self.data_path, "videos", f"{di['video_idx']}.mp4")
        conversation = [{"role": "user", "content": []}]

        # Video
        content = {
            "type": "video",
            "video": video_path,
            "min_pixels": self.video_min_pixels,
            "max_pixels": self.video_max_pixels,
            "total_pixels": self.video_total_pixels,
            "max_frames": self.max_frames,
        }
        if self.fps is not None:
            content["fps"] = self.fps
        conversation[0]['content'].append(content)

        # Query
        conversation[0]['content'].append({"type": "text", "text": self.build_query(di)})
        return conversation

    def data_collator(self, conversations, processor):
        return prepare_single_forward_batch(
            conversations,
            processor,
            self.answer_prefix,
        )

    def __len__(self):
        return len(self.datums)


class EgoSchemaMultiForwardDataset(EgoSchemaDataset):
    def __init__(
        self,
        data_path,
        question_prefix: str = '',
        question_postfix: str = '\nPlease select the correct answer.',
        answer_prefix: str = 'Answer:',
        video_max_pixels=28 * 28 * 768,
        video_min_pixels=28 * 28 * 100,
        video_total_pixels=32000 * 28 * 28 * 0.9,
        max_frames=768,
        fps=None,
        max_window_duration=512,
        subset=False,
    ):
        super().__init__(
            data_path,
            question_prefix=question_prefix,
            question_postfix=question_postfix,
            answer_prefix=answer_prefix,
            video_max_pixels=video_max_pixels,
            video_min_pixels=video_min_pixels,
            video_total_pixels=video_total_pixels,
            max_frames=max_frames,
            fps=fps,
            subset=subset,
        )
        self.max_window_duration = max_window_duration

    def data_collator(self, conversations, processor):
        return prepare_multiforward_batch(
            conversations,
            processor,
            self.answer_prefix,
            max_window_duration=self.max_window_duration,
        )


def mcq_predict(
    model,
    processor,
    benchmark_path: str,
    letters: list[str] = ['A', 'B', 'C', 'D', 'E'],
    question_prefix: str = '',
    question_postfix: str = '\nPlease select the correct answer.',
    answer_prefix: str = 'Answer:',
    dataloader_num_workers: int = 2,
    multi_forward_training: bool = False,
    max_inference_context_window: int = 128000,
    max_window_duration: int = 512,
    fps=None,
    subset: bool = False,
    checkpoint_path: str | None = None,
    chunk_size: int | None = None,
    resume: bool = False,
    **video_kwargs,
):
    dataset_kwargs = dict(
        data_path=benchmark_path,
        question_prefix=question_prefix,
        question_postfix=question_postfix,
        answer_prefix=answer_prefix,
        fps=fps,
        subset=subset,
        **video_kwargs,
    )
    if multi_forward_training:
        dataset = EgoSchemaMultiForwardDataset(
            max_window_duration=max_window_duration,
            **dataset_kwargs,
        )
    else:
        dataset = EgoSchemaDataset(**dataset_kwargs)

    return run_mcq_prediction(
        model=model,
        processor=processor,
        dataset=dataset,
        letters=letters,
        dataloader_num_workers=dataloader_num_workers,
        multi_forward_training=multi_forward_training,
        max_inference_context_window=max_inference_context_window,
        checkpoint_path=checkpoint_path,
        chunk_size=chunk_size,
        resume=resume,
    )


def build_output_paths(args) -> tuple[str, str]:
    out_dir = 'evaluation/egoschema/results'
    os.makedirs(out_dir, exist_ok=True)
    suffix_extra = f"_multiforward_window{args.max_window_duration}" if args.multi_forward_training else ""
    subset_prefix = "subset_" if args.subset else ""
    base_name = (
        f"{subset_prefix}{os.path.basename(args.model_name_or_path.rstrip('/'))}"
        f"_tpf{args.video_max_pixels}_frame{args.max_frames}{suffix_extra}"
    )
    save_json_path = os.path.join(out_dir, f"{base_name}.json")
    checkpoint_path = os.path.join(out_dir, f"{base_name}.checkpoint.json")
    return save_json_path, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Distributed evaluation for EgoSchema (full test set, MC)"
    )
    parser.add_argument("--model_type", type=str, default='streamttt')
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Path or identifier of the pretrained model"
    )
    parser.add_argument("--processor_name", type=str, required=True)
    parser.add_argument(
        "--benchmark_path", type=str, required=True,
        help="Path to the EgoSchema benchmark (contains MC/ and videos/)"
    )
    parser.add_argument(
        "--uid_map_path", type=str,
        default=None,
        help="Path to uid_to_ego4d.json (defaults to <benchmark_path>/uid_to_ego4d.json), "
             "used to validate the submission uid set.",
    )
    parser.add_argument("--video_max_pixels", type=int, default=768)
    parser.add_argument("--video_min_pixels", type=int, default=100)
    parser.add_argument("--video_total_pixels", type=int, default=256000)
    parser.add_argument("--max_frames", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--multi_forward_training", action="store_true")
    parser.add_argument(
        "--subset", action="store_true",
        help="Evaluate on the 500-question Subset (has ground-truth answers); "
             "accuracy is computed locally instead of writing a leaderboard submission.",
    )
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

    args = parser.parse_args()

    if args.uid_map_path is None:
        args.uid_map_path = os.path.join(args.benchmark_path, "uid_to_ego4d.json")

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
    save_json_path, checkpoint_path = build_output_paths(args)

    # Run distributed prediction
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model,
        processor=processor,
        benchmark_path=args.benchmark_path,
        letters=['A', 'B', 'C', 'D', 'E'],
        dataloader_num_workers=args.num_workers,
        multi_forward_training=args.multi_forward_training,
        max_inference_context_window=args.max_inference_context_window,
        max_window_duration=args.max_window_duration,
        fps=args.fps,
        subset=args.subset,
        checkpoint_path=checkpoint_path,
        chunk_size=args.save_every_n_samples,
        resume=args.resume,
        **video_kwargs
    )

    # Only process rank 0 aggregates and saves the results
    if process_index == 0:
        if args.subset:
            # Subset carries ground-truth answers -> compute accuracy directly.
            correct = 0
            results = []
            for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
                gt_idx = int(datum['answer'])
                pred_idx = int(letter_idx_prediction)
                is_correct = int(gt_idx == pred_idx)
                correct += is_correct
                results.append({
                    "question_idx": datum['question_idx'],
                    "video_idx": datum['video_idx'],
                    "question": datum['question'],
                    "options": datum['option'],
                    "answer": gt_idx,
                    "prediction": pred_idx,
                    "correct": bool(is_correct),
                })
            total = len(benchmark_datums)
            accuracy = correct / total if total else 0.0
            print(f"EgoSchema Subset accuracy: {correct}/{total} = {accuracy:.4f}")
            with open(save_json_path, 'w') as f:
                json.dump(
                    {"accuracy": accuracy, "correct": correct, "total": total, "predictions": results},
                    f,
                )
            print(f"Saved subset results to {save_json_path}")
            return

        # submission: { <question uid> : <predicted answer index 0-4> }
        submission = {
            datum['video_idx']: int(letter_idx_prediction)
            for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions)
        }

        # validate the uid set against uid_to_ego4d.json
        if args.uid_map_path and os.path.exists(args.uid_map_path):
            with open(args.uid_map_path) as f:
                uid_map = json.load(f)
            expected_uids = set(uid_map.keys())
            missing = expected_uids - set(submission.keys())
            extra = set(submission.keys()) - expected_uids
            if missing:
                print(f"WARNING: {len(missing)} uids from uid_to_ego4d.json are missing in predictions, e.g. {list(missing)[:5]}")
            if extra:
                print(f"WARNING: {len(extra)} predicted uids are not in uid_to_ego4d.json, e.g. {list(extra)[:5]}")

        with open(save_json_path, 'w') as f:
            json.dump(submission, f)
        print(f"Saved {len(submission)} predictions to {save_json_path}")


if __name__ == '__main__':
    main()
