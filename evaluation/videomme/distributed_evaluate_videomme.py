from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl, apply_liger_kernel_to_qwen3_vl
apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
apply_liger_kernel_to_qwen3_vl(rope=False, fused_linear_cross_entropy=False)

import json
import os
import functools
import argparse
import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoConfig, AutoProcessor
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE

from evaluation.videomme.eval_your_results import eval_your_results
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
    build_window_ranges_from_frames,
    prepare_video_only_inputs,
    slice_video_frames,
    slice_video_metadata,
)


IGNORE_KEYS_FOR_PREDICTION = [
    'past_key_values', 'hidden_states', 'attentions', 'rope_deltas', 'states'
]


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
            is_qwen3=True,
            duration_subset=None,
            **kwargs,
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
        self.is_qwen3 = is_qwen3
        self.image_patch_size = 16 if self.is_qwen3 else 14

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
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        texts = [text + self.answer_prefix for text in texts]
        for attempt in range(10):
            try:
                _, video_inputs, video_kwargs = process_vision_info(conversations, image_patch_size=self.image_patch_size, return_video_kwargs=True, return_video_metadata=self.is_qwen3)
                break
            except Exception:
                print(f"{attempt}-th process_vision_info failed. retry...")
        else:
            raise RuntimeError("process_vision_info failed after 10 retries.")

        if self.is_qwen3:  # qwen3
            videos, video_metadatas = zip(*video_inputs)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:  # qwen2.5
            videos = video_inputs
            video_metadatas = [None]

        inputs = processor(
            text=texts,
            videos=videos[0],
            video_metadata=video_metadatas[0],
            padding=True,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        return inputs

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
        is_qwen3=True,
        processor=None,
        max_window_frames=512,
        duration_subset=None,
        **kwargs,
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
            is_qwen3=is_qwen3,
            duration_subset=duration_subset,
        )
        self.processor = processor
        self.max_window_frames = max_window_frames

    def data_collator(self, conversations, processor):
        assert len(conversations) == 1, "We assume batch size is 1 for the multi-forward testing."
        conversation = conversations[0]
        video_info = conversation[0]["content"][0]
        question = conversation[0]["content"][1]["text"]

        messages = [{"role": "user", "content": [video_info, {"type": "text", "text": question}]}]
        for attempt in range(10):
            try:
                _, video_inputs_data, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=self.image_patch_size,
                    return_video_kwargs=True,
                    return_video_metadata=self.is_qwen3,
                )
                break
            except Exception:
                print(f"{attempt}-th process_vision_info failed. retry...")
        else:
            raise RuntimeError("process_vision_info failed after 10 retries.")

        if self.is_qwen3:
            videos, video_metadatas = zip(*video_inputs_data)
            videos, video_metadatas = list(videos), list(video_metadatas)
            full_video = videos[0]
            full_metadata = video_metadatas[0]
        else:
            full_video = video_inputs_data[0]
            full_metadata = None

        sampled_frames = int(full_video.shape[0])
        temporal_patch_size = getattr(processor.video_processor, "temporal_patch_size", 2)
        window_ranges = build_window_ranges_from_frames(
            sampled_frames,
            self.max_window_frames,
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


def mcq_predict(
    model,
    processor,
    benchmark_path: str,
    video_root: str,
    letters: list[str] = ['A', 'B', 'C', 'D'],
    question_prefix: str = '',
    question_postfix: str = '\nPlease select the correct answer.',
    answer_prefix: str = 'Answer:',
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = False,
    per_device_eval_batch_size: int = 1,
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
    strict_letter_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in letters]

    dataset_cls = VideoMMEMultiForwardDataset if multi_forward_training else VideoMMEDataset
    dataset = dataset_cls(
        benchmark_path, video_root, question_prefix=question_prefix,
        question_postfix=question_postfix, answer_prefix=answer_prefix,
        with_subtitles=with_subtitles,
        processor=processor,
        max_window_frames=max_window_frames,
        fps=fps,
        duration_subset=duration_subset,
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

    # Load model and processor
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # set sliding window for our custom models
    if hasattr(config.text_config, "sliding_window"):
        if config.text_config.sliding_window != args.max_inference_context_window:
            print(f"The saved sliding window does not correspond to the required sliding window. Setting it to {args.max_inference_context_window}.")

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
    processor = AutoProcessor.from_pretrained(
        args.processor_name,
        padding_side='left'
    )

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
        use_liger_kernel=False,
        dataloader_num_workers=args.num_workers,
        per_device_eval_batch_size=1,
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
