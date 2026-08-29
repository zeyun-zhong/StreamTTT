import json
import os
import sys

import torch


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


def load_prediction_checkpoint(checkpoint_path: str, expected_total: int | None = None) -> dict[int, int]:
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

    predictions = {}
    for item in payload.get("predictions", []):
        predictions[int(item["index"])] = int(item["prediction"])
    return predictions


def save_prediction_checkpoint(
    checkpoint_path: str,
    predictions_by_index: dict[int, int],
    total_samples: int,
) -> None:
    """Atomically write partial predictions so a killed run can `--resume`."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "total_samples": total_samples,
        "completed_samples": len(predictions_by_index),
        "predictions": [
            {"index": idx, "prediction": predictions_by_index[idx]}
            for idx in sorted(predictions_by_index)
        ],
    }
    tmp_path = f"{checkpoint_path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp_path, checkpoint_path)
