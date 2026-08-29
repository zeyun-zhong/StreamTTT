
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from transformers import Cache
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutput

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, Callable

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel,
    Qwen3VLModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextDecoderLayer,
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    Qwen3VLTextRMSNorm,
    Qwen3VLCausalLMOutputWithPast
)

from .fast_weight_block import FastWeightBlock


@dataclass
class StreamTTTCausalLMOutputWithPast(Qwen3VLCausalLMOutputWithPast):
    states: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None


class StreamTTTAttention(nn.Module):
    """
    StreamTTT hybrid layer (paper §3.3, Fig. 2a).

    Wraps a pretrained Qwen3-VL attention module:
      - runs the original sliding-window attention (keeps FA2/window/caching),
        which serves as the short-range memory over recent tokens;
      - runs a parallel fast-weight (TTT) branch that compresses history into a
        fixed-size recurrent state outside the attention context;
      - fuses the two through a learnable channel-wise gate tanh(alpha),
        initialized near zero so the layer starts at the pretrained function.

    Returns the same tuple structure as the original attention.
    """

    def __init__(self, config, layer_idx: int, orig_attn):
        super().__init__()
        self.config = config

        # ---- shared, pretrained projections (do NOT recreate new layers) ----
        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj
        self.q_norm = orig_attn.q_norm
        self.k_norm = orig_attn.k_norm
        self.num_key_value_groups = orig_attn.num_key_value_groups

        self.is_causal = True
        self.rope_scaling = config.rope_scaling
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_idx
        self.sliding_window = config.sliding_window

        # TTT block
        self.ttt_block = FastWeightBlock(config)
        self.output_states = getattr(config, "output_states", False)
        self._last_states = None

        self.alpha = None
        if getattr(config, "use_gate_for_memory", False):
            self.alpha = nn.Parameter(torch.ones(1, config.hidden_size) * 0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [b, s, d]
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,  # (cos, sin)
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        #================= Original Qwen3VL Self-Attention
        query_states_h = query_states.transpose(1, 2)
        key_states_h = key_states.transpose(1, 2)
        value_states_h = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states_h, key_states_h = apply_rotary_pos_emb(query_states_h, key_states_h, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states_h, value_states_h = past_key_values.update(
                key_states_h, value_states_h, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output_h, attn_weights = attention_interface(
            self,
            query_states_h,
            key_states_h,
            value_states_h,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output_h.reshape(*input_shape, -1).contiguous()
        o = self.o_proj(attn_output)

        #==================== Test Time Training
        prev_states = kwargs.get("states", None)
        is_generation = kwargs.get("is_generation", False)
        if prev_states is not None:
            prev_states = prev_states[self.layer_idx]

        ttt_x, recurrent_state = self.ttt_block(
            hidden_states,
            query_states.reshape(*input_shape, -1),
            key_states.reshape(*input_shape, -1),
            value_states.reshape(*input_shape, -1),
            position_embeddings,
            prev_states=prev_states,
            is_generation=is_generation,
        )
        if self.output_states or (past_key_values is not None):
            self._last_states = recurrent_state

        # ---------- mix + shared o_proj ----------
        if self.alpha is not None:
            o = o + F.tanh(self.alpha) * ttt_x
        else:
            o = o + ttt_x

        return o, attn_weights


class StreamTTTDecoderLayer(Qwen3VLTextDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = StreamTTTAttention(config, layer_idx, self.self_attn)


class StreamTTTTextModel(Qwen3VLTextModel):
    def __init__(self, config):
        super().__init__(config)

        num_layers = config.num_hidden_layers
        fw_top_percentage = getattr(config, "fw_top_percentage", 1.0)
        ttt_start_idx = int(num_layers * (1 - fw_top_percentage))

        self.layers = nn.ModuleList(
            [
                StreamTTTDecoderLayer(config, layer_idx) if layer_idx >= ttt_start_idx else Qwen3VLTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        mask_builder = (
            create_sliding_window_causal_mask
            if getattr(self.config, "sliding_window", None) is not None
            else create_causal_mask
        )
        attention_mask = mask_builder(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class StreamTTTModel(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = StreamTTTTextModel._from_config(config.text_config)


def _capture_states_hook(model_ref, layer_idx: int):
    def hook(module, inputs, output):
        s = getattr(module, "_last_states", None)
        if s is not None:
            model_ref._states[layer_idx] = s  # keep autograd refs
    return hook


def attach_state_hooks(vl_text_model):
    vl_text_model._state_hooks = []
    for idx, layer in enumerate(vl_text_model.layers):
        if hasattr(layer, "self_attn"):
            h = layer.self_attn.register_forward_hook(_capture_states_hook(vl_text_model, idx))
            vl_text_model._state_hooks.append(h)


class StreamTTTQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = StreamTTTModel(config)
        self.model_config = config
        self.num_language_layers = config.text_config.num_hidden_layers
        attach_state_hooks(self.model.language_model)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, StreamTTTCausalLMOutputWithPast]:
        self.model.language_model._states = [None] * self.num_language_layers

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        states = getattr(self.model.language_model, "_states", None)

        return StreamTTTCausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            states=states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        states=None,
        is_generation=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["states"] = states
        model_inputs["is_generation"] = is_generation

        return model_inputs

    def _custom_init(self):
        for module in self.model.modules():
            if isinstance(module, StreamTTTAttention):
                if hasattr(module, "alpha") and module.alpha is not None:
                    nn.init.constant_(module.alpha, 0.1)

            if isinstance(module, FastWeightBlock):
                module._init_weights()
