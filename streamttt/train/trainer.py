from typing import Union, Any, Optional
from transformers import Trainer
import torch
from torch import nn

from streamttt.streaming.windowing import recompute_single_window_position, reset_multimodal_rope_state


def prune_kv_cache(past_key_values, max_length):
    """
    Optimized prune function for Hugging Face DynamicCache.
    Modifies the cache in-place to keep only the last `max_length` tokens.
    """
    if past_key_values is None:
        return None

    if past_key_values.get_seq_length() <= max_length or past_key_values.layers[0].keys.size(2) <= max_length:
        return past_key_values

    for layer in past_key_values.layers:
        layer.keys = layer.keys[:, :, -max_length:, :].clone()
        layer.values = layer.values[:, :, -max_length:, :].clone()

    return past_key_values


class StreamTTTTrainer(Trainer):

    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            memory_core_parameters = []
            memory_alpha_parameters = []

            if self.args.memory_lr is not None:
                memory_core_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "ttt" in name and not name.endswith(".alpha")
                ]
                memory_alpha_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if name.endswith(".alpha")
                ]

            if self.args.memory_lr is not None:
                special_lr_parameters = memory_core_parameters + memory_alpha_parameters
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

                if memory_core_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in memory_core_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.memory_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in memory_core_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.memory_lr,
                            },
                        ]
                    )

                if memory_alpha_parameters:
                    optimizer_grouped_parameters.append(
                        {
                            "params": [p for n, p in opt_model.named_parameters() if (n in memory_alpha_parameters and p.requires_grad)],
                            "weight_decay": 0.0,
                            "lr": self.args.memory_lr,
                        }
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_grouped_parameters = [
                g for g in optimizer_grouped_parameters
                if g.get("params") and len(g["params"]) > 0
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.args.multi_forward_training:
            return self.prediction_step_multiforward(model, inputs, prediction_loss_only, ignore_keys)

        # Only the final token's logits are used for MCQ scoring. Restrict lm_head to
        # the last position so the stock forward (fused CE disabled in eval) does not
        # materialize the full [1, seq_len, vocab] logits tensor. Skip this when labels
        # are present, since loss computation needs logits over the full sequence.
        if "labels" not in inputs:
            inputs["logits_to_keep"] = 1
        losses, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        logits = logits[:, -1:, :]
        return losses, logits, labels

    def prediction_step_multiforward(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = model.device
        max_context_window = self.args.max_inference_context_window

        # Unpack Inputs (Already processed by Collator)
        processed_windows = inputs.pop("processed_windows")  # The list of tensors
        window_cache_positions = inputs.pop("window_cache_positions")
        cumulative_window_attention_masks = inputs.pop("cumulative_window_attention_masks")

        image_token_id = getattr(model.config, "image_token_id", None)
        video_token_id = getattr(model.config, "video_token_id", None)
        model_config = getattr(model, "model_config", None)
        if image_token_id is None and model_config is not None:
            image_token_id = getattr(model_config, "image_token_id", None)
        if video_token_id is None and model_config is not None:
            video_token_id = getattr(model_config, "video_token_id", None)

        reset_multimodal_rope_state(model)
        past_key_values = None
        states = None
        out = None
        running_pos_max = None
        for i, window_cpu in enumerate(processed_windows):
            window = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in window_cpu.items()
            }
            attention_mask = cumulative_window_attention_masks[i].to(device)
            local_attention_mask = window.get("attention_mask")
            if local_attention_mask is None:
                local_attention_mask = torch.ones_like(window["input_ids"], device=device)
            cache_position = window_cache_positions[i].to(device)
            position_ids, running_pos_max = recompute_single_window_position(
                model,
                window_ids=window["input_ids"],
                attention_mask=local_attention_mask,
                video_grid_thw=window["video_grid_thw"],
                second_per_grid_ts=window.get("second_per_grid_ts"),
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                running_pos_max=running_pos_max,
            )

            model_inputs = dict(window)
            model_inputs["attention_mask"] = attention_mask
            model_inputs["cache_position"] = cache_position
            model_inputs["position_ids"] = position_ids
            model_inputs["use_cache"] = True
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values
            if states is not None:
                model_inputs["states"] = states

            # Only the final token's logits are used for MCQ scoring, so restrict
            # lm_head to the last position. Avoids materializing the full
            # [1, win_len, vocab] logits tensor (~7.5GB for a 24K window), which is
            # the dominant per-window memory spike in multi-forward inference.
            out = model(**model_inputs, logits_to_keep=1)
            past_key_values = out.past_key_values
            states = out.states if hasattr(out, "states") else None
            past_key_values = prune_kv_cache(past_key_values, max_context_window)

        last_logits = out.logits[:, -1:, :]
        last_labels = None

        return (None, last_logits, last_labels)
