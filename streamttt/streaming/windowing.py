import copy
import inspect
from typing import Any

import numpy as np
import torch

from streamttt.data.custom_torchcodec_backend import get_video_metadata
from streamttt.data.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)


def get_multimodal_model(model):
    if hasattr(model, "model") and hasattr(model.model, "get_rope_index"):
        return model.model
    if hasattr(model, "get_rope_index"):
        return model
    raise AttributeError("Could not find multimodal model with get_rope_index.")


def reset_multimodal_rope_state(model):
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "rope_deltas"):
        inner_model.rope_deltas = None


def metadata_get(metadata: Any, key: str):
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key)


def metadata_set(metadata: Any, key: str, value: Any):
    if isinstance(metadata, dict):
        metadata[key] = value
    else:
        setattr(metadata, key, value)


def slice_video_metadata(metadata: Any, start: int, end: int):
    if metadata is None:
        return None

    out = copy.deepcopy(metadata)
    frame_indices = metadata_get(out, "frames_indices")
    if frame_indices is None:
        raise RuntimeError("Video metadata is missing frames_indices; cannot preserve exact sampling.")

    if torch.is_tensor(frame_indices):
        frame_indices = frame_indices.tolist()
    else:
        frame_indices = list(frame_indices)

    metadata_set(out, "frames_indices", frame_indices[start:end])
    return out


def slice_video_frames(video: Any, start: int, end: int):
    return video[start:end]


def move_tensors_to_device(batch: dict[str, Any], device: str):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def processor_supports_kwarg(processor, kwarg: str) -> bool:
    return kwarg in inspect.signature(processor.__call__).parameters


def prepare_video_only_inputs(processor, video, video_metadata, video_kwargs, device: str):
    kwargs = {
        "text": DEFAULT_VIDEO_TOKEN,
        "videos": video,
        "video_metadata": video_metadata,
        "return_tensors": "pt",
        "padding": False,
        "do_resize": False,
    }
    kwargs.update(video_kwargs)

    if processor_supports_kwarg(processor, "return_mm_token_type_ids"):
        kwargs["return_mm_token_type_ids"] = True

    inputs = processor(**kwargs)
    return move_tensors_to_device(dict(inputs), device)


def build_prompt_ids(processor, prompt: str, device: str, answer_prefix: str = ""):
    tokenizer = processor.tokenizer
    system_user_message = (
        f"{DEFAULT_IM_START_TOKEN}system\n"
        f"{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        f"{DEFAULT_IM_START_TOKEN}user\n"
    )
    qa_message = (
        f"{prompt}{DEFAULT_IM_END_TOKEN}\n"
        f"{DEFAULT_IM_START_TOKEN}assistant\n"
        f"{answer_prefix}"
    )
    system_user_ids = tokenizer(
        system_user_message, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    qa_ids = tokenizer(
        qa_message, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    return system_user_ids, qa_ids


def build_model_inputs(input_ids: torch.Tensor, video_inputs: dict[str, Any]):
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values_videos": video_inputs["pixel_values_videos"],
        "video_grid_thw": video_inputs["video_grid_thw"],
    }
    if video_inputs.get("second_per_grid_ts") is not None:
        model_inputs["second_per_grid_ts"] = video_inputs["second_per_grid_ts"]
    return model_inputs


def build_window_prompt_ids(
    system_user_ids: torch.Tensor,
    qa_ids: torch.Tensor,
    window_v_ids: list[torch.Tensor],
):
    window_ids = []
    for idx, window_v_id in enumerate(window_v_ids):
        if len(window_v_ids) == 1:
            ids = torch.cat([system_user_ids, window_v_id, qa_ids], dim=1)
        elif idx == 0:
            ids = torch.cat([system_user_ids, window_v_id], dim=1)
        elif idx == len(window_v_ids) - 1:
            ids = torch.cat([window_v_id, qa_ids], dim=1)
        else:
            ids = window_v_id
        window_ids.append(ids)
    return window_ids


def build_token_offsets(lengths: list[int]):
    offsets = []
    cursor = 0
    for length in lengths:
        offsets.append((cursor, cursor + length))
        cursor += length
    return offsets


def build_mm_token_type_ids(
    input_ids: torch.Tensor,
    image_token_id: int | None,
    video_token_id: int | None,
):
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    if image_token_id is not None:
        mm_token_type_ids[input_ids == image_token_id] = 1
    if video_token_id is not None:
        mm_token_type_ids[input_ids == video_token_id] = 2
    return mm_token_type_ids


def compute_position_ids_from_scratch(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    second_per_grid_ts: torch.Tensor | list[float] | None,
    mm_token_type_ids: torch.Tensor | None,
):
    mm_model = get_multimodal_model(model)
    get_rope_index = mm_model.get_rope_index

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    sig = inspect.signature(get_rope_index)

    if "image_grid_thw" in sig.parameters:
        kwargs["image_grid_thw"] = image_grid_thw
    if "video_grid_thw" in sig.parameters:
        kwargs["video_grid_thw"] = video_grid_thw
    if "second_per_grid_ts" in sig.parameters:
        kwargs["second_per_grid_ts"] = second_per_grid_ts
    if "mm_token_type_ids" in sig.parameters:
        if mm_token_type_ids is None:
            raise RuntimeError(
                "get_rope_index requires mm_token_type_ids but the processor did not return them."
            )
        kwargs["mm_token_type_ids"] = mm_token_type_ids

    position_ids, rope_deltas = get_rope_index(**kwargs)
    return position_ids, rope_deltas


def recompute_single_window_position(
    model,
    window_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_grid_thw: torch.Tensor,
    second_per_grid_ts: torch.Tensor | list[float] | None,
    image_token_id: int | None,
    video_token_id: int | None,
    running_pos_max: torch.Tensor | None,
):
    mm_token_type_ids = build_mm_token_type_ids(
        window_ids, image_token_id, video_token_id
    )
    pos_local, _ = compute_position_ids_from_scratch(
        model,
        input_ids=window_ids,
        attention_mask=attention_mask,
        image_grid_thw=None,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        mm_token_type_ids=mm_token_type_ids,
    )
    if running_pos_max is None:
        pos_adjusted = pos_local
    else:
        pos_adjusted = pos_local + (running_pos_max + 1).to(pos_local.device)

    next_running_pos_max = (
        pos_adjusted.max(dim=2, keepdim=True).values.max(dim=0, keepdim=True).values
    )
    return pos_adjusted, next_running_pos_max


def get_sampled_frame_timestamps(video_path: str, metadata: Any, sampled_frames: int):
    fps = metadata_get(metadata, "fps")
    total_num_frames = metadata_get(metadata, "total_num_frames")
    frame_indices = metadata_get(metadata, "frames_indices")

    if fps is None or total_num_frames is None:
        fps, total_num_frames = get_video_metadata(video_path)

    if frame_indices is None:
        if sampled_frames < 1:
            raise ValueError("sampled_frames must be >= 1.")
        frame_indices = np.linspace(0, total_num_frames - 1, sampled_frames).round().astype(int).tolist()
    elif torch.is_tensor(frame_indices):
        frame_indices = frame_indices.tolist()
    else:
        frame_indices = list(frame_indices)

    if fps is None or fps <= 0:
        raise ValueError(f"Invalid fps for video split: {fps}.")
    if total_num_frames is None or total_num_frames < 1:
        raise ValueError(f"Invalid total_num_frames for video split: {total_num_frames}.")

    frame_times_sec = np.asarray(frame_indices, dtype=np.float64) / float(fps)
    video_duration_sec = float(total_num_frames) / float(fps)
    return frame_indices, frame_times_sec, video_duration_sec, float(fps)


def build_window_ranges_from_frames(
    num_frames: int,
    window_frames: int,
    temporal_patch_size: int = 2,
):
    """Split ``num_frames`` sampled frames into fixed-size windows of ``window_frames``.

    Unlike the timestamp-based splitter, every window (except possibly the last)
    holds exactly ``window_frames`` frames, so each forward sees a constant number
    of video tokens regardless of how long the video is. This avoids the
    inefficiency of duration-based windows where a long video produces windows
    with very few frames.

    Window boundaries are snapped to multiples of ``temporal_patch_size`` so the
    per-window video tokenization concatenates back to the full video token block.
    """
    if window_frames <= 0:
        raise ValueError(f"--max_window_frames must be > 0, but got {window_frames}.")
    if num_frames < 1:
        raise ValueError("Need at least one sampled frame to build windows.")

    # Snap the window size down to a multiple of the temporal patch size so full
    # windows align with the temporal grid (sampled frame counts are themselves a
    # multiple of the temporal patch size).
    if window_frames % temporal_patch_size != 0:
        window_frames = max(temporal_patch_size, window_frames - (window_frames % temporal_patch_size))

    frame_ranges = []
    start_idx = 0
    while start_idx < num_frames:
        end_idx = min(start_idx + window_frames, num_frames)
        window_len = end_idx - start_idx
        remainder = window_len % temporal_patch_size
        if remainder != 0:
            end_idx = min(end_idx + (temporal_patch_size - remainder), num_frames)
        frame_ranges.append((start_idx, end_idx))
        start_idx = end_idx
    return frame_ranges


def build_window_ranges_from_timestamps(
    frame_times_sec: np.ndarray,
    video_duration_sec: float,
    window_duration_sec: float,
    temporal_patch_size: int = 2,
):
    if window_duration_sec <= 0:
        raise ValueError(
            f"--window_duration must be > 0 seconds, but got {window_duration_sec}."
        )
    if len(frame_times_sec) < 1:
        raise ValueError("Need at least one sampled frame to build windows.")

    split_times = np.arange(
        window_duration_sec, video_duration_sec, window_duration_sec, dtype=np.float64
    )
    frame_ranges = []
    time_ranges = []
    start_idx = 0
    start_time = 0.0

    for split_time in split_times.tolist():
        end_idx = int(np.searchsorted(frame_times_sec, split_time, side="left"))
        window_len = end_idx - start_idx
        remainder = window_len % temporal_patch_size
        if remainder != 0:
            end_idx = min(end_idx + (temporal_patch_size - remainder), len(frame_times_sec))
        if end_idx <= start_idx:
            raise ValueError(
                f"--window_duration={window_duration_sec} seconds produces an empty sampled window "
                f"before t={split_time:.3f}s. Increase the duration or sample more frames."
            )
        frame_ranges.append((start_idx, end_idx))
        time_ranges.append((start_time, float(split_time)))
        start_idx = end_idx
        start_time = float(split_time)

    if start_idx >= len(frame_times_sec):
        if frame_ranges:
            last_start, _ = time_ranges[-1]
            time_ranges[-1] = (last_start, video_duration_sec)
            return frame_ranges, time_ranges
        return [(0, len(frame_times_sec))], [(0.0, video_duration_sec)]

    frame_ranges.append((start_idx, len(frame_times_sec)))
    time_ranges.append((start_time, video_duration_sec))
    return frame_ranges, time_ranges
