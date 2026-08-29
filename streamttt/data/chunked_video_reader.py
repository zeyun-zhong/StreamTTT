"""Memory-efficient, chunked replacement for ``qwen_vl_utils.vision_process.fetch_video``.

The stock ``fetch_video`` decodes an entire video at native resolution into a single
tensor and then runs one ``transforms.functional.resize(...).float()`` over the whole
clip. For long videos (e.g. 1h @ fps=2 => ~7200 frames) the native tensor plus the
float resize buffer can spike to >180GB per process, which is multiplied across
torchrun ranks and gets the DataLoader worker OOM-killed (``killed by signal: Killed``).

This module decodes + resizes in temporal chunks, so only one chunk of native frames
and one chunk of float-resized frames exist at a time. The spatial resize is per-frame
independent, so the concatenated result is bit-identical to the original implementation
(token counts / ``video_grid_thw`` are unchanged).

Importing this module monkeypatches ``qwen_vl_utils.vision_process.fetch_video``.
``process_vision_info`` resolves ``fetch_video`` from the module globals at call time,
so replacing the module attribute is enough for it to take effect everywhere.
"""

import os
import time

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import logging

import qwen_vl_utils.vision_process as vp
from qwen_vl_utils.vision_process import (
    FRAME_FACTOR,
    MODEL_SEQ_LEN,
    SPATIAL_MERGE_SIZE,
    VIDEO_MAX_TOKEN_NUM,
    VIDEO_MIN_TOKEN_NUM,
    calculate_video_frame_range,
    smart_nframes,
    smart_resize,
)

logger = logging.get_logger(__name__)

# Number of frames decoded+resized per chunk. Smaller = less peak memory, slightly
# more decode overhead. Override via VIDEO_DECODE_CHUNK_FRAMES.
DECODE_CHUNK_FRAMES = int(os.environ.get("VIDEO_DECODE_CHUNK_FRAMES", 256))


def _fetch_video_chunked(
    ele,
    image_patch_size=14,
    return_video_sample_fps=False,
    return_video_metadata=False,
):
    """Drop-in replacement for ``fetch_video`` that decodes/resizes in temporal chunks."""
    # Only take over the "video file path" branch; defer the image-frame-list branch
    # to the original implementation.
    if not isinstance(ele.get("video"), str):
        return _ORIG_FETCH_VIDEO(
            ele,
            image_patch_size=image_patch_size,
            return_video_sample_fps=return_video_sample_fps,
            return_video_metadata=return_video_metadata,
        )

    from torchcodec.decoders import VideoDecoder

    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    video_frame_min_pixels = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    video_frame_max_pixels = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor

    num_threads = int(os.environ.get("TORCHCODEC_NUM_THREADS", 8))
    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=num_threads)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    height = decoder.metadata.height  # native resolution, available without decoding
    width = decoder.metadata.width

    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    # Match the stock qwen torchcodec backend (the one active in eval) exactly so the
    # sampled frame indices — and thus the output — are bit-identical.
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    # Target resize dims depend only on native H,W and the pixel budget (constant for
    # the whole clip), so compute once. Mirrors the math in the stock fetch_video.
    min_pixels = ele.get("min_pixels", video_frame_min_pixels)
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
    max_pixels = max(
        min(video_frame_max_pixels, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels = min(ele.get("max_pixels", max_pixels), max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_h, resized_w = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=image_factor
        )
    else:
        resized_h, resized_w = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    # Decode one chunk -> resize to small frames -> free the native chunk.
    resized_chunks = []
    for c in range(0, len(idx), DECODE_CHUNK_FRAMES):
        idx_chunk = idx[c : c + DECODE_CHUNK_FRAMES]
        native = decoder.get_frames_at(indices=idx_chunk).data  # (n, C, H, W) uint8
        small = transforms.functional.resize(
            native,
            [resized_h, resized_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        resized_chunks.append(small)
        del native
    video = torch.cat(resized_chunks, dim=0)
    del resized_chunks

    logger.info(
        f"chunked torchcodec: {video_path=}, {total_frames=}, {video_fps=}, "
        f"chunk={DECODE_CHUNK_FRAMES}, time={time.time() - st:.3f}s"
    )

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchcodec_chunked",
    )
    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


_ORIG_FETCH_VIDEO = vp.fetch_video
# process_vision_info resolves the module-global ``fetch_video`` at call time, so
# replacing the module attribute makes the chunked reader take effect globally.
vp.fetch_video = _fetch_video_chunked
