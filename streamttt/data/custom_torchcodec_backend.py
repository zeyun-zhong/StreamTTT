import torch
import os, time
from transformers import logging
from qwen_vl_utils.vision_process import smart_nframes, calculate_video_frame_range


logger = logging.get_logger(__name__)


def get_video_metadata(video_path):
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(video_path)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    return video_fps, total_frames


def _read_video_torchcodec_safe(
    ele: dict,
) -> (torch.Tensor, float):
    """
    Improves qwen version for robust boundary handling
    read video using torchcodec.decoders.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    # We use end_frame - 1, aiming to avoid error like
    # "Requested next frame while there are no more frames left to decode."
    idx = torch.linspace(start_frame, end_frame - 1, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    logger.info(f"torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchcodec_safe",
    )
    return video, video_metadata, sample_fps

