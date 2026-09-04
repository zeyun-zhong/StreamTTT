import functools
import hashlib
import json
import os
import sys

import torch
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE
from torch.utils.data import Subset
from transformers import AutoConfig, AutoProcessor

from streamttt.models import ALL_MODELS
from streamttt.streaming.windowing import (
    build_model_inputs,
    build_prompt_ids,
    build_token_offsets,
    build_window_prompt_ids,
    build_window_ranges_from_frames,
    build_window_ranges_from_timestamps,
    get_sampled_frame_timestamps,
    prepare_video_only_inputs,
    slice_video_frames,
    slice_video_metadata,
)
from streamttt.train.arguments import TrainingArguments
from streamttt.train.trainer import StreamTTTTrainer


IGNORE_KEYS_FOR_PREDICTION = [
    "past_key_values", "hidden_states", "attentions", "rope_deltas", "states"
]


def save_function_print(function: callable, save_path: str, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(save_path, 'w') as f:
            sys.stdout = f  
            function(*args, **kwargs)          
    finally:
        sys.stdout = original_stdout


def preprocess_logits_for_metrics(logits, labels, strict_letter_ids):
    return torch.stack([logit[(logit[:, 0] != -100).nonzero()[-1].item(), strict_letter_ids] for logit in logits]).argmax(dim=-1)


def load_prediction_checkpoint(
    checkpoint_path: str,
    expected_total: int | None = None,
    expected_fingerprint: str | None = None,
) -> dict[int, int]:
    """Read a `--resume` checkpoint into {dataset index: predicted letter index}.

    Called on every rank (before the rank-0 branch) so all processes agree on
    which indices are still pending.
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return {}

    with open(checkpoint_path) as f:
        payload = json.load(f)

    total_samples = payload.get("total_samples")
    if expected_total is not None and total_samples not in (None, expected_total):
        raise RuntimeError(
            f"Checkpoint total_samples={total_samples} does not match current dataset size {expected_total}."
        )
    if expected_fingerprint is not None and payload.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(
            "Checkpoint fingerprint does not match this evaluation. "
            "Run without --resume or use a matching checkpoint."
        )

    predictions = {}
    for item in payload.get("predictions", []):
        predictions[int(item["index"])] = int(item["prediction"])
    return predictions


def save_prediction_checkpoint(
    checkpoint_path: str,
    predictions_by_index: dict[int, int],
    total_samples: int,
    fingerprint: str,
) -> None:
    """Atomically write partial predictions so a killed run can `--resume`."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "total_samples": total_samples,
        "completed_samples": len(predictions_by_index),
        "fingerprint": fingerprint,
        "predictions": [
            {"index": idx, "prediction": predictions_by_index[idx]}
            for idx in sorted(predictions_by_index)
        ],
    }
    tmp_path = f"{checkpoint_path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp_path, checkpoint_path)


def load_eval_model(
    *,
    model_type: str,
    model_name_or_path: str,
    processor_name: str,
    max_inference_context_window: int,
    multi_forward_training: bool,
    video_min_pixels: int,
    video_max_pixels: int,
    video_total_pixels: int,
    max_frames: int,
):
    """Load a Qwen3-VL evaluator and convert token budgets to pixel budgets."""
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if getattr(config.text_config, "sliding_window", None) != max_inference_context_window:
        print(
            f"Saved sliding window ({getattr(config.text_config, 'sliding_window', None)}) "
            f"differs from requested ({max_inference_context_window}). Overriding."
        )
    config.text_config.sliding_window = max_inference_context_window
    if multi_forward_training:
        config.output_states = True
        config.text_config.output_states = True

    model = ALL_MODELS[model_type].from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(processor_name, padding_side="left")
    image_factor = 16 * SPATIAL_MERGE_SIZE
    video_kwargs = {
        "video_min_pixels": video_min_pixels * image_factor * image_factor,
        "video_max_pixels": video_max_pixels * image_factor * image_factor,
        "video_total_pixels": video_total_pixels * image_factor * image_factor,
        "max_frames": max_frames,
    }
    return model, processor, video_kwargs


def prepare_single_forward_batch(conversations, processor, answer_prefix: str):
    texts = processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )
    texts = [text + answer_prefix for text in texts]
    for attempt in range(10):
        try:
            _, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            break
        except Exception:
            print(f"{attempt}-th process_vision_info failed. retry...")
    else:
        raise RuntimeError("process_vision_info failed after 10 retries.")

    videos, video_metadatas = zip(*video_inputs)
    return processor(
        text=texts,
        videos=videos[0],
        video_metadata=video_metadatas[0],
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )


def prepare_multiforward_batch(
    conversations,
    processor,
    answer_prefix: str,
    *,
    max_window_duration: int | None = None,
    max_window_frames: int | None = None,
):
    """Build one streaming batch; exactly one window policy must be selected."""
    if (max_window_duration is None) == (max_window_frames is None):
        raise ValueError("Select exactly one of max_window_duration or max_window_frames.")
    if len(conversations) != 1:
        raise ValueError("Multi-forward evaluation requires batch size 1.")

    conversation = conversations[0]
    video_info, text_info = conversation[0]["content"]
    messages = [{"role": "user", "content": [video_info, text_info]}]
    for attempt in range(10):
        try:
            _, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            break
        except Exception:
            print(f"{attempt}-th process_vision_info failed. retry...")
    else:
        raise RuntimeError("process_vision_info failed after 10 retries.")

    videos, video_metadatas = zip(*video_inputs)
    full_video, full_metadata = videos[0], video_metadatas[0]
    temporal_patch_size = getattr(processor.video_processor, "temporal_patch_size", 2)
    if max_window_frames is not None:
        window_ranges = build_window_ranges_from_frames(
            int(full_video.shape[0]),
            max_window_frames,
            temporal_patch_size=temporal_patch_size,
        )
    else:
        _, frame_times, duration, _ = get_sampled_frame_timestamps(
            video_info["video"], full_metadata, int(full_video.shape[0])
        )
        window_ranges, _ = build_window_ranges_from_timestamps(
            frame_times,
            duration,
            max_window_duration,
            temporal_patch_size=temporal_patch_size,
        )

    window_videos = [
        slice_video_frames(full_video, start, end) for start, end in window_ranges
    ]
    window_metadatas = [
        slice_video_metadata(full_metadata, start, end) for start, end in window_ranges
    ]
    system_user_ids, qa_ids = build_prompt_ids(
        processor, text_info["text"], "cpu", answer_prefix=answer_prefix
    )
    full_video_inputs = prepare_video_only_inputs(
        processor, full_video, full_metadata, video_kwargs, "cpu"
    )
    window_video_inputs = [
        prepare_video_only_inputs(processor, video, metadata, video_kwargs, "cpu")
        for video, metadata in zip(window_videos, window_metadatas)
    ]
    full_video_ids = full_video_inputs["input_ids"]
    window_video_ids = [inputs["input_ids"] for inputs in window_video_inputs]
    if not torch.equal(torch.cat(window_video_ids, dim=1), full_video_ids):
        raise RuntimeError("Split video tokenization does not reconstruct the full video token block.")

    window_ids = build_window_prompt_ids(system_user_ids, qa_ids, window_video_ids)
    full_ids = torch.cat([system_user_ids, full_video_ids, qa_ids], dim=1)
    if not torch.equal(torch.cat(window_ids, dim=1), full_ids):
        raise RuntimeError("Windowed prompt ids do not concatenate back to the full prompt ids.")

    processed_windows = [
        build_model_inputs(ids, video_inputs)
        for ids, video_inputs in zip(window_ids, window_video_inputs)
    ]
    for window in processed_windows:
        window["pixel_values_videos"] = window["pixel_values_videos"].bfloat16()

    offsets = build_token_offsets([ids.shape[1] for ids in window_ids])
    return {
        "processed_windows": processed_windows,
        "window_cache_positions": [
            torch.arange(start, end, dtype=torch.long) for start, end in offsets
        ],
        "cumulative_window_attention_masks": [
            torch.ones_like(full_ids)[:, :end] for _, end in offsets
        ],
    }


def prediction_fingerprint(
    dataset,
    model,
    processor,
    letters: list[str],
    multi_forward_training: bool,
    max_inference_context_window: int,
) -> str:
    identity = {
        "schema": 1,
        "samples": [dataset[index] for index in range(len(dataset))],
        "model": {
            "class": type(model).__qualname__,
            "name_or_path": getattr(model.config, "_name_or_path", None),
        },
        "processor": getattr(processor.tokenizer, "name_or_path", None),
        "letters": letters,
        "multi_forward_training": multi_forward_training,
        "max_inference_context_window": max_inference_context_window,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_mcq_prediction(
    *,
    model,
    processor,
    dataset,
    letters: list[str],
    dataloader_num_workers: int,
    multi_forward_training: bool,
    max_inference_context_window: int,
    checkpoint_path: str | None,
    chunk_size: int | None,
    resume: bool,
):
    strict_letter_ids = [
        processor.tokenizer(f": {letter}").input_ids[-1] for letter in letters
    ]
    trainer = StreamTTTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="outputs/",
            do_predict=True,
            per_device_eval_batch_size=1,
            dataloader_num_workers=dataloader_num_workers,
            report_to="none",
            use_liger_kernel=False,
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
    fingerprint = prediction_fingerprint(
        dataset,
        model,
        processor,
        letters,
        multi_forward_training,
        max_inference_context_window,
    )
    if chunk_size is None or chunk_size <= 0:
        chunk_size = len(dataset)

    saved_predictions = (
        load_prediction_checkpoint(
            checkpoint_path,
            expected_total=len(dataset),
            expected_fingerprint=fingerprint,
        )
        if resume and checkpoint_path is not None
        else {}
    )
    if process_index == 0 and saved_predictions:
        print(
            f"Resuming from {len(saved_predictions)}/{len(dataset)} saved predictions "
            f"at {checkpoint_path}."
        )

    for chunk_start in range(0, len(dataset), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        pending_indices = [
            index
            for index in range(chunk_start, chunk_end)
            if index not in saved_predictions
        ]
        if not pending_indices:
            continue

        predictions = trainer.predict(
            Subset(dataset, pending_indices),
            ignore_keys=IGNORE_KEYS_FOR_PREDICTION,
        ).predictions
        if process_index == 0:
            if len(predictions) != len(pending_indices):
                raise RuntimeError(
                    f"Prediction count mismatch for indices "
                    f"{pending_indices[0]}-{pending_indices[-1]}: "
                    f"got {len(predictions)} predictions for "
                    f"{len(pending_indices)} samples."
                )
            saved_predictions.update(
                (index, int(prediction))
                for index, prediction in zip(pending_indices, predictions)
            )
            if checkpoint_path is not None:
                save_prediction_checkpoint(
                    checkpoint_path,
                    predictions_by_index=saved_predictions,
                    total_samples=len(dataset),
                    fingerprint=fingerprint,
                )
            print(f"Saved {len(saved_predictions)}/{len(dataset)} predictions.")

    if process_index != 0:
        return None, dataset.datums, process_index

    missing = [index for index in range(len(dataset)) if index not in saved_predictions]
    if missing:
        raise RuntimeError(f"Missing predictions for indices: {missing[:10]}")
    return [saved_predictions[index] for index in range(len(dataset))], dataset.datums, process_index
