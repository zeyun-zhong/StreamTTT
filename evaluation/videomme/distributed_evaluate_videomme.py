from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
apply_liger_kernel_to_qwen3_vl(rope=False, fused_linear_cross_entropy=False)

import json
import os
import argparse
from torch.utils.data import Dataset

from evaluation.videomme.eval_your_results import eval_your_results
from evaluation.utils import (
    load_eval_model,
    prepare_multiforward_batch,
    prepare_single_forward_batch,
    run_mcq_prediction,
    save_function_print,
)


def load_videomme_datums(benchmark_path):
    """Read the VideoMME jsonl into a list of plain dict datums.

    Each line is a JSON value; some exports double-encode the dict as a JSON
    string, so decode a second time when needed.
    """
    lines = open(benchmark_path).readlines()
    datums = [json.loads(line) for line in lines]
    if datums and isinstance(datums[0], str):
        datums = [json.loads(datum) for datum in datums]
    return datums


class VideoMMEDataset(Dataset):
    def __init__(
            self,
            benchmark_path,
            video_root,
            question_prefix: str = '',
            question_postfix: str = '\nPlease select the correct answer.',
            answer_prefix: str = 'Answer:',
            with_subtitles=False,
            video_max_pixels=28 * 28 * 768,
            video_min_pixels=28 * 28 * 100,
            video_total_pixels=32000 * 28 * 28 * 0.9,
            max_frames=768,
            fps=None,
            duration_subset=None,
    ):
        super().__init__()
        self.benchmark_path = benchmark_path
        self.video_root = video_root
        self.with_subtitles = with_subtitles
        self.datums = load_videomme_datums(benchmark_path)
        if duration_subset is not None:
            self.datums = [d for d in self.datums if d.get("duration") == duration_subset]
            if not self.datums:
                raise ValueError(f"No datums found for duration_subset={duration_subset!r}.")

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
        query = (self.question_prefix +
                 di["question"] + '\n' +
                 '\n'.join(di["options"]) +
                 self.question_postfix)
        if self.with_subtitles:
            query = (f"This video's subtitles are listed below:\n{di['subtitles']}\n"
                     f"According to the video and subtitles, " + query)
        return query

    def __getitem__(self, index):
        di = self.datums[index]
        conversation = [{"role": "user", "content": []}]

        # Video
        video_path = os.path.join(self.video_root, di["videoID"] + ".mp4")
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


class VideoMMEMultiForwardDataset(VideoMMEDataset):
    def __init__(
        self,
        benchmark_path,
        video_root,
        question_prefix: str = '',
        question_postfix: str = '\nPlease select the correct answer.',
        answer_prefix: str = 'Answer:',
        with_subtitles=False,
        video_max_pixels=28 * 28 * 768,
        video_min_pixels=28 * 28 * 100,
        video_total_pixels=32000 * 28 * 28 * 0.9,
        max_frames=768,
        fps=None,
        max_window_frames=512,
        duration_subset=None,
    ):
        super().__init__(
            benchmark_path,
            video_root,
            question_prefix=question_prefix,
            question_postfix=question_postfix,
            answer_prefix=answer_prefix,
            with_subtitles=with_subtitles,
            video_max_pixels=video_max_pixels,
            video_min_pixels=video_min_pixels,
            video_total_pixels=video_total_pixels,
            max_frames=max_frames,
            fps=fps,
            duration_subset=duration_subset,
        )
        self.max_window_frames = max_window_frames

    def data_collator(self, conversations, processor):
        return prepare_multiforward_batch(
            conversations,
            processor,
            self.answer_prefix,
            max_window_frames=self.max_window_frames,
        )


def mcq_predict(
    model,
    processor,
    benchmark_path: str,
    video_root: str,
    letters: list[str] = ['A', 'B', 'C', 'D'],
    question_prefix: str = '',
    question_postfix: str = '\nPlease select the correct answer.',
    answer_prefix: str = 'Answer:',
    dataloader_num_workers: int = 2,
    with_subtitles: bool = False,
    multi_forward_training: bool = False,
    max_inference_context_window: int = 128000,
    max_window_frames: int = 512,
    fps=None,
    duration_subset: str | None = None,
    checkpoint_path: str | None = None,
    chunk_size: int | None = None,
    resume: bool = False,
    **video_kwargs,
):
    dataset_kwargs = dict(
        benchmark_path=benchmark_path,
        video_root=video_root,
        question_prefix=question_prefix,
        question_postfix=question_postfix,
        answer_prefix=answer_prefix,
        with_subtitles=with_subtitles,
        fps=fps,
        duration_subset=duration_subset,
        **video_kwargs,
    )
    if multi_forward_training:
        dataset = VideoMMEMultiForwardDataset(
            max_window_frames=max_window_frames,
            **dataset_kwargs,
        )
    else:
        dataset = VideoMMEDataset(**dataset_kwargs)

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


def build_output_paths(args) -> tuple[str, str, str]:
    suffix = 'with_subtitles' if args.with_subtitles else 'no_subtitles'
    out_dir = 'evaluation/videomme/results'
    os.makedirs(out_dir, exist_ok=True)
    suffix_extra = f"_{args.max_inference_context_window}"
    if args.duration_subset is not None:
        suffix_extra += f"_{args.duration_subset}"
    # Mark the inference mode: without it a single-forward and a multi-forward run
    # at the same context window share a name, so `--resume` could pick up a
    # checkpoint written by the other mode.
    if args.multi_forward_training:
        suffix_extra += f"_multiforward_window{args.max_window_frames}"
    base_name = (
        f"{os.path.basename(args.model_name_or_path.rstrip('/'))}_{suffix}"
        f"_tpf{args.video_max_pixels}_frame{args.max_frames}{suffix_extra}"
    )
    save_json_path = os.path.join(out_dir, f"{base_name}.json")
    save_txt_path = save_json_path.replace('.json', '.txt')
    checkpoint_path = os.path.join(out_dir, f"{base_name}.checkpoint.json")
    return save_json_path, save_txt_path, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Distributed evaluation for VideoMME models"
    )
    parser.add_argument("--model_type", type=str, default='streamttt')
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Path or identifier of the pretrained model"
    )
    parser.add_argument("--processor_name", type=str, required=True)
    parser.add_argument(
        "--benchmark_path", type=str, required=True,
        help="Path to the benchmark JSONL file"
    )
    parser.add_argument(
        "--with_subtitles", action="store_true",
        help="Flag to indicate evaluation on subtitles-enabled benchmark"
    )
    parser.add_argument("--video_max_pixels", type=int, default=768)
    parser.add_argument("--video_min_pixels", type=int, default=100)
    parser.add_argument("--video_total_pixels", type=int, default=256000)
    parser.add_argument("--max_frames", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--multi_forward_training", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_inference_context_window", type=int, default=4096)
    parser.add_argument("--max_window_frames", type=int, default=512)
    parser.add_argument(
        "--duration_subset", type=str, default=None, choices=["short", "medium", "long"],
        help="Restrict evaluation to a single VideoMME duration subset (e.g. 'long').",
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
    save_json_path, save_txt_path, checkpoint_path = build_output_paths(args)

    # Videos live in <root>/data, where <root> is the parent of the
    # directory holding the benchmark jsonl, named {videoID}.mp4.
    video_root = os.path.join(os.path.dirname(os.path.dirname(args.benchmark_path)), "data")

    # Run distributed prediction
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model,
        processor=processor,
        benchmark_path=args.benchmark_path,
        video_root=video_root,
        letters=['A', 'B', 'C', 'D'],
        dataloader_num_workers=args.num_workers,
        with_subtitles=args.with_subtitles,
        multi_forward_training=args.multi_forward_training,
        max_inference_context_window=args.max_inference_context_window,
        max_window_frames=args.max_window_frames,
        fps=args.fps,
        duration_subset=args.duration_subset,
        checkpoint_path=checkpoint_path,
        chunk_size=args.save_every_n_samples,
        resume=args.resume,
        **video_kwargs
    )

    # Only process rank 0 for result aggregation and saving
    if process_index == 0:
        video_id_to_results = {}
        for datum, letter_idx_prediction in zip(
            benchmark_datums, letter_idxs_predictions
        ):
            vid = datum['video_id']
            if vid not in video_id_to_results:
                video_id_to_results[vid] = {
                    'video_id': vid,
                    'duration': datum['duration'],
                    'domain': datum['domain'],
                    'sub_category': datum['sub_category'],
                    'questions': [],
                }
            video_id_to_results[vid]['questions'].append({
                "question_id": datum['question_id'],
                "task_type": datum['task_type'],
                "question": datum['question'],
                "options": datum['options'],
                "answer": datum['answer'],
                "response": datum['options'][letter_idx_prediction],
            })

        results = list(video_id_to_results.values())

        # Save JSON
        with open(save_json_path, 'w') as f:
            json.dump(results, f)

        # Save evaluation text report
        video_types = [args.duration_subset] if args.duration_subset is not None else ['short', 'medium', 'long']
        save_function_print(
            eval_your_results,
            save_txt_path,
            save_json_path,
            video_types=video_types,
            return_categories_accuracy=True,
            return_sub_categories_accuracy=True,
            return_task_types_accuracy=True,
        )


if __name__ == '__main__':
    main()
