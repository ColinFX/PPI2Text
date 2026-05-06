"""
Modified PairQwen3Model and PairQwen3ForCausalLM, responsible for:

- implementing Interleaved MRoPE (Multimodal Rotary Position Embedding) adapted from Qwen3-VL for 2D protein
  interaction maps: frequency pairs are assigned to [temporal, height, width] channels in an interleaved
  [T,H,W,T,H,W,...,T,T] pattern; 3-channel position_ids encode text positions in all channels identically
  while interaction chunk tokens receive distinct row/col positions in the H/W channels
- implementing partial bidirectional attention logic for interaction tokens (prompt only)
- handling left padding in attention and position id calculations
- supporting 3-channel position_ids of shape [3, B, seq_len] with cache_position-based slicing for generation

Original script: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py

# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3PreTrainedModel, Qwen3RMSNorm, Qwen3RotaryEmbedding
)


class PairQwen3RotaryEmbedding(Qwen3RotaryEmbedding):
    """
    Interleaved Multimodal Rotary Position Embedding adapted from Qwen3-VL.

    Splits the head dimension's frequency pairs across 3 channels (temporal, height, width)
    in an interleaved pattern [T,H,W,T,H,W,...,T,T]. For standard text tokens all channels
    receive the same position, reducing to standard RoPE. For interaction chunk tokens the
    temporal channel stays constant while height/width channels carry 2D grid positions.

    Args:
        config: Qwen3Config used to initialize base RoPE inverse frequencies.
        mrope_section: list of 3 ints [n_t, n_h, n_w] specifying the number of frequency pairs
            allocated to each channel. Must sum to head_dim // 2.
    """

    def __init__(self, config: Qwen3Config, mrope_section: list[int]):
        super().__init__(config)
        self.mrope_section = mrope_section
        assert len(mrope_section) == 3, "mrope_section must have exactly 3 elements [temporal, height, width]"
        assert sum(mrope_section) == self.inv_freq.shape[0], (
            f"mrope_section {mrope_section} (sum={sum(mrope_section)}) must sum to "
            f"inv_freq dimension {self.inv_freq.shape[0]} (= head_dim // 2)"
        )
        mapping = self._create_interleaved_mapping(mrope_section)
        self.register_buffer("channel_mapping", torch.tensor(mapping, dtype=torch.long), persistent=False)

    @staticmethod
    def _create_interleaved_mapping(mrope_section: list[int]) -> list[int]:
        """
        Create the interleaved channel assignment for each frequency pair.
        Pattern: [T,H,W,T,H,W,...] until a channel is exhausted, then remaining channels fill the rest.

        Example for mrope_section=[24, 20, 20]:
            First 60 pairs cycle T,H,W; last 4 pairs are all T.
        """
        total = sum(mrope_section)
        mapping = []
        counters = [0] * len(mrope_section)
        d = 0
        while d < total:
            for c in range(len(mrope_section)):
                if counters[c] < mrope_section[c]:
                    mapping.append(c)
                    counters[c] += 1
                    d += 1
                    if d >= total:
                        break
        return mapping

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        Compute MRoPE cos/sin embeddings.

        Args:
            x: hidden states tensor, used only for dtype and device.
            position_ids: either [B, seq_len] (standard, expanded to 3 identical channels)
                or [3, B, seq_len] (multimodal with T/H/W channels).

        Returns:
            (cos, sin) each of shape [B, seq_len, head_dim].
        """
        if "dynamic" in self.rope_type:
            pid = position_ids[-1] if position_ids.dim() == 3 else position_ids
            self._dynamic_frequency_update(pid, device=x.device)

        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # position_ids: [3, B, seq_len]
        # channel_mapping: [D/2] mapping each frequency dim to a channel index (0/1/2)
        # gather effective positions: for freq dim d, use position_ids[channel_mapping[d], :, :]
        effective_pos = position_ids[self.channel_mapping]  # [D/2, B, seq_len]
        effective_pos = effective_pos.permute(1, 2, 0).float()  # [B, seq_len, D/2]

        inv_freq = self.inv_freq.float()  # [D/2]

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = effective_pos * inv_freq  # [B, seq_len, D/2] element-wise
            emb = torch.cat([freqs, freqs], dim=-1)  # [B, seq_len, D]
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class PairQwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config, mrope_section: list[int]):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        if self.has_sliding_layers:
            raise ValueError("Sliding window attention is not supported by PairQwen3Model.")

        # MODIFICATION: replace standard RoPE with interleaved MRoPE
        self.mrope_section = mrope_section
        self.rotary_emb = PairQwen3RotaryEmbedding(config=config, mrope_section=mrope_section)
        # MODIFICATION END

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        bidirectional_group_ids: torch.LongTensor | None = None,  # ADDITIONAL ARGUMENT
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # MODIFICATION: create a fallback attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], cache_position[0] + inputs_embeds.shape[1]),
                device=inputs_embeds.device,
            )
        # MODIFICATION END

        # MODIFICATION: handle 3-channel position_ids for MRoPE
        if position_ids is not None and position_ids.dim() == 3:
            # 3D MRoPE position_ids: [3, B, full_len]
            # slice for the current query using cache_position
            position_ids = position_ids[:, :, cache_position]  # [3, B, q_len]
        # else: 2D position_ids [B, seq_len], use as-is (MRoPE rotary emb will expand to 3 channels)
        # MODIFICATION END

        # may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids if position_ids.dim() == 2 else position_ids[0],
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            # MODIFICATION: per-group bidirectional attention for interaction tokens
            # Tokens within the same non-zero group attend bidirectionally; tokens in
            # different groups follow causal masking (earlier groups visible to later
            # groups, not vice versa). Groups: 1=protA, 2=protB, 3=pair_map.
            #
            # `bidirectional_group_ids` covers the prompt only (length = prompt_len).
            # During decoding, q_start := kv_len - q_len >= prompt_len, so all query
            # tokens are generated text (group 0) and the same-group check is False
            # everywhere → the block is a no-op. Skip it on decode steps and, on
            # prefill, build the [B, q_len, kv_len] mask (never the full kv×kv square).
            if bidirectional_group_ids is not None:
                kv_len = attention_mask.shape[1]
                q_len = inputs_embeds.shape[1]
                prompt_group_len = bidirectional_group_ids.shape[1]
                q_start = kv_len - q_len

                if q_start < prompt_group_len:
                    overlap = min(prompt_group_len - q_start, q_len)
                    q_group_ids = bidirectional_group_ids.new_zeros(
                        (bidirectional_group_ids.shape[0], q_len)
                    )
                    q_group_ids[:, :overlap] = bidirectional_group_ids[:, q_start:q_start + overlap]

                    pad_len = kv_len - prompt_group_len
                    if pad_len > 0:
                        kv_group_ids = torch.nn.functional.pad(
                            bidirectional_group_ids, (0, pad_len), value=0
                        )
                    else:
                        kv_group_ids = bidirectional_group_ids

                    q_same_group_mask = (
                        (q_group_ids.unsqueeze(2) == kv_group_ids.unsqueeze(1))
                        & (q_group_ids.unsqueeze(2) > 0)
                    )
                    valid_mask = (
                        attention_mask[:, -q_len:].unsqueeze(2).bool()
                        & attention_mask.unsqueeze(1).bool()
                    )
                    q_same_group_mask = q_same_group_mask & valid_mask

                    causal_mask = causal_mask_mapping["full_attention"]
                    if causal_mask is None:
                        min_dtype = torch.finfo(inputs_embeds.dtype).min
                        causal_mask = torch.full(
                            (inputs_embeds.shape[0], 1, q_len, kv_len),
                            fill_value=min_dtype,
                            device=inputs_embeds.device,
                            dtype=inputs_embeds.dtype,
                        )
                        causal_mask = torch.triu(causal_mask, diagonal=kv_len - q_len + 1)

                    causal_mask = causal_mask.masked_fill(q_same_group_mask.unsqueeze(1), 0.0)
                    causal_mask_mapping["full_attention"] = causal_mask
            # MODIFICATION END

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Gradient checkpointing is handled automatically by
            # GradientCheckpointingLayer.__call__ in each Qwen3DecoderLayer
            # when gradient_checkpointing_enable() has been called on the model.
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids if position_ids.dim() == 2 else position_ids[0],
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class PairQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _supports_static_cache = False  # we use dynamic KV cache

    @staticmethod
    def _supports_inputs_embeds_in_generate() -> bool:
        """Enable generation from inputs_embeds (used by PairEsm3Qwen3ForCausalLM.generate)."""
        return True

    def __init__(self, config: Qwen3Config, mrope_section: list[int]):
        super().__init__(config)
        self.model = PairQwen3Model(config, mrope_section=mrope_section)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        Override to preserve 3D position_ids across generation steps.
        The base class would attempt to slice 3D position_ids with 2D indexing, which breaks.
        Instead, we pass the full 3D tensor through and let PairQwen3Model.forward handle slicing
        via cache_position.

        `inputs_embeds` is declared explicitly so that transformers' generate() recognises
        that this model supports generation from embeddings (it inspects the signature).
        """
        position_ids = kwargs.get("position_ids", None)
        is_3d = position_ids is not None and position_ids.dim() == 3

        if is_3d:
            # temporarily remove to prevent base class from slicing incorrectly
            kwargs["position_ids"] = None

        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, **kwargs
        )

        if is_3d:
            model_inputs["position_ids"] = position_ids

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        """
        Override to preserve custom tensors across generation steps.
        - position_ids: the base class would destructively replace the 3D tensor with a
          sliced/incremented version, but we need the full [3, B, prompt_len + max_new_tokens]
          tensor intact so that PairQwen3Model.forward can slice it using cache_position.
        - bidirectional_group_ids: the base class does not know about this kwarg; explicitly
          preserve it to avoid breakage if a future transformers version strips unknown kwargs.
        """
        position_ids = model_kwargs.get("position_ids", None)
        bidirectional_group_ids = model_kwargs.get("bidirectional_group_ids", None)
        is_3d = position_ids is not None and position_ids.dim() == 3

        if is_3d:
            # temporarily remove to prevent base class from mangling
            model_kwargs["position_ids"] = None

        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)

        if is_3d:
            # restore the full 3D tensor (forward handles slicing via cache_position)
            model_kwargs["position_ids"] = position_ids
        if bidirectional_group_ids is not None:
            model_kwargs["bidirectional_group_ids"] = bidirectional_group_ids

        return model_kwargs

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        bidirectional_group_ids: torch.LongTensor | None = None,  # ADDITIONAL ARGUMENT
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        bidirectional_group_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Per-token group assignment for structured bidirectional attention. 0 for standard
            causal tokens, positive integers (1=protA, 2=protB, 3=pair_map) for interaction tokens.
            Tokens within the same group attend bidirectionally; tokens in different groups follow
            causal masking (earlier groups visible to later groups, not vice versa).
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            bidirectional_group_ids=bidirectional_group_ids,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
