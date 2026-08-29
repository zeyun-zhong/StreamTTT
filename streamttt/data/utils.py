import re, json, random

IGNORE_INDEX = -100
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_annotation(annotation_path, sampling_rate=1.0):
    file_format = annotation_path.split(".")[-1]
    if file_format == "jsonl":
        annotations = read_jsonl(annotation_path)
    else:
        annotations = json.load(open(annotation_path, "r"))
    if sampling_rate < 1.0:
        annotations = random.sample(
            annotations, int(len(annotations) * sampling_rate)
        )
        print(f"sampling {len(annotations)} examples from dataset {annotation_path}")
    return annotations


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        # replace <video>
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        transformed = re.sub(pattern, replacement, input_string)

        # replace <image>, llava178k has always <image> instead of <video>
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        transformed = re.sub(pattern, replacement, transformed)
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN
        transformed = re.sub(pattern, replacement, input_string)

    return transformed


def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output
