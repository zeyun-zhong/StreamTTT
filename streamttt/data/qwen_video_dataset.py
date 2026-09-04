"""
Reference:
https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
https://github.com/2U1/Qwen2-VL-Finetune/blob/master/src/dataset/sft_dataset.py
"""

import os
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec_safe"
import copy
import time

import torch
from torch.utils.data import Dataset
from qwen_vl_utils.vision_process import VIDEO_READER_BACKENDS, SPATIAL_MERGE_SIZE
from .custom_torchcodec_backend import _read_video_torchcodec_safe

VIDEO_READER_BACKENDS["torchcodec_safe"] = _read_video_torchcodec_safe

from qwen_vl_utils import process_vision_info

from . import data_list
from .utils import *

def get_llava_video_path(data_dir: str, ann):
    video_path = os.path.join(data_dir, ann["data_path"], ann["data_source"], ann["video"])
    return video_path

def get_tmp_dataset_video_path(data_dir: str, ann):
    video_path = os.path.join(data_dir, ann["data_path"], "videos", ann["video"])
    return video_path

def get_activitynet_qvhighlight_video_path(data_dir: str, ann):
    video_path = os.path.join(data_dir, ann["data_path"], "videos", ann["video"])
    return video_path

def get_etinstruct_video_path(data_dir: str, ann):
    video_path = os.path.join(data_dir, ann["data_path"], "videos", ann["data_source"], ann["video"])
    return video_path

def get_ego4d_video_path(data_dir: str, ann):
    video_path = os.path.join(data_dir, ann["data_path"], "v2", "clips", ann["video"])
    return video_path

def get_adt_video_path(data_dir: str, ann):
    video_path = os.path.join(data_dir, ann["data_path"], "videos", ann["video"])
    return video_path


def get_video_path(data_dir, ann):
    if "llava" in ann["data_path"].lower():
        return get_llava_video_path(data_dir, ann)
    if "tmp" in ann["data_path"].lower():
        return get_tmp_dataset_video_path(data_dir, ann)
    if "activitynet" in ann["data_path"].lower() or "qvhighlight" in ann["data_path"].lower():
        return get_activitynet_qvhighlight_video_path(data_dir, ann)
    if "et-instruct" in ann["data_path"].lower():
        return get_etinstruct_video_path(data_dir, ann)
    if "ego4d" in ann["data_path"].lower():
        return get_ego4d_video_path(data_dir, ann)
    if "adt" in ann["data_path"].lower():
        return get_adt_video_path(data_dir, ann)

    raise NotImplementedError(f"{ann['data_path']} currently not implemented.")


class SupervisedVideoDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args):
        super().__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)

        list_data_dict = []

        for data in dataset_list:
            annotation_path = data["annotation_path"]
            sampling_rate = data["sampling_rate"]
            if isinstance(annotation_path, (list, tuple)):
                annotations = []
                for anno_path in annotation_path:
                    anno_path = os.path.join(data_args.data_dir, data["data_path"], anno_path)
                    annotation = load_annotation(anno_path, sampling_rate)
                    annotations.extend(annotation)
            else:
                annotation_path = os.path.join(data_args.data_dir, data["data_path"], annotation_path)
                annotations = load_annotation(annotation_path, sampling_rate)

            for ann in annotations:
                ann["data_path"] = data["data_path"]

            list_data_dict += annotations

        self.list_data_dict = list_data_dict
        self.data_args = data_args

        self.image_patch_size = 16

        if data_args.video_max_pixels > 768:
            raise ValueError(f"Our current implementation expects pixels to be number of tokens. "
                             f"The max pixel value is 768 while you have {data_args.video_max_pixels}.")

        image_factor = self.image_patch_size * SPATIAL_MERGE_SIZE

        self.video_min_pixel = data_args.video_min_pixels * image_factor * image_factor
        self.video_max_pixel = data_args.video_max_pixels * image_factor * image_factor
        self.video_total_pixel = data_args.video_total_pixels * image_factor * image_factor
        self.max_frame = data_args.max_frames
        self.processor = data_args.processor

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        num_base_retries = 2

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def get_video_info(self, video_path, video_end=None):
        content = {
            "type": "video",
            "video": video_path,
            "min_pixels": self.video_min_pixel,
            "max_pixels": self.video_max_pixel,
            "total_pixels": self.video_total_pixel,
            "max_frames": self.max_frame,
        }
        if video_end is not None:
            content["video_start"] = 0.0
            content["video_end"] = video_end

        messages = [{"role": "user", "content": [content]}]
        _, video_input, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        videos, video_metadatas = zip(*video_input)

        return videos[0], video_metadatas[0], video_kwargs

    def _get_item(self, i) -> dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        video_file = get_video_path(self.data_args.data_dir, sources)
        grid_key = "video_grid_thw"
        pixel_key = "pixel_values_videos"

        rt_time = sources.get("rt_time", None)
        rt_time = float(rt_time) + 1.0 if rt_time is not None else None  # make sure the last frame is included, maybe important for ocr

        video_input, video_metadata, video_kwargs = self.get_video_info(video_file, video_end=rt_time)
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=True))

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Message
        # system message
        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        system_message_input_ids = self.processor.tokenizer(
            system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
        system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

        all_input_ids.append(system_message_input_ids.squeeze(0))
        all_labels.append(system_labels.squeeze(0))

        # conversation (multi-turn)
        video_count = 0
        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            video = video_input if video_count == 0 else None
            inputs = self.processor(
                text=[user_input], videos=video, video_metadata=video_metadata,
                padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
            prompt_input_ids = inputs['input_ids']

            if video is not None:
                if "second_per_grid_ts" in inputs:  # qwen2.5vl uses this, qwen3vl not.
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            response_input_ids = self.processor.tokenizer(
                gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            video_count += user_input.count(DEFAULT_VIDEO_TOKEN)

        assert video_count == 1, "Current implementation only supports 1 video per sample."

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            labels=labels,
        )

        pixel_values = torch.cat(all_pixel_values, dim=0).bfloat16()
        image_thw = torch.cat(all_image_grid_thw, dim=0)
        data_dict[pixel_key] = pixel_values
        data_dict[grid_key] = image_thw
        if all_second_gird:
            data_dict["second_per_grid_ts"] = all_second_gird

        return data_dict


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_second_per_grid_ts = []

        for example in examples:
            keys = example.keys()
            batch_pixel_video_values.append(example["pixel_values_videos"])
            batch_video_thw.append(example["video_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict


def make_supervised_data_module(data_args) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedVideoDataset(data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=data_args.processor.tokenizer.pad_token_id)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
