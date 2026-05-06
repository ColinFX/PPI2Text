"""
Assembled PairEsm3Qwen3ForCausalLM model.

Bridges two ESM3-encoded proteins into a pair-interaction representation and feeds it to the Qwen3 decoder as 2D
chunked tokens with interleaved MRoPE. Pipeline:
    OpenESM3 + SequenceCompressor + CrossAttentionBlock * N + PairMapConstructor + PairMapToTokens + PairQwen3

Special tokens boi/eoi use standalone trainable embeddings; roi placeholders in the prompt are replaced with the
per-protein and pair-map tokens. Each sample carries its own number of interaction chunks derived from its own
compressed protein lengths, so the decoder input is batch-independent.

Training (forward, teacher forcing):
    * input_ids, attention_mask, labels: (B, prompt_len+answer_len) — whole chat template, -100 for prompt/padding
    * protein_a, protein_b: dict[str, Tensor] — ESM3 encoder inputs
    * compressed_len_a/b: (B,) long — per-sample compressed protein length

Inference (generate):
    * inputs, attention_mask: (B, prompt_len) — prompt part only, left-padded
    * protein_a, protein_b, compressed_len_a/b — same as training

Each protein dict has keys: sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens, function_tokens,
residue_annotation_tokens, average_plddt, per_res_plddt, structure_coords, chain_id, sequence_id, sequence_mask.
The prompt must contain one `<|boi|>...<|eoi|>` span with the right number of `<|roi|>` placeholders.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_pairesm3qwen3 import PairEsm3Qwen3Config
from .modeling_openesm3 import OpenESM3
from .modeling_pairqwen3 import PairQwen3ForCausalLM
from .pretrained_openesm3 import (
    OpenESM3_structure_encoder_v0,
    OpenESM3_structure_decoder_v0,
    OpenESM3_function_decoder_v0,
)


class SequenceCompressor(nn.Module):
    """1D conv compressor reducing protein length ~4x via two stride-2 conv layers with GELU + LayerNorm.

    No positional embedding — ESM3 already encodes position information.
    """

    def __init__(self, d_esm: int, d: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d_esm, d, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.LayerNorm(d)
        self.conv2 = nn.Conv1d(d, d, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.LayerNorm(d)

    @staticmethod
    def compute_compressed_length(length: int) -> int:
        """Compute the output sequence length after two stride-2 conv layers."""
        l1 = (length + 2 * 1 - 4) // 2 + 1
        l2 = (l1 + 2 * 1 - 4) // 2 + 1
        return l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_esm] protein encoder output (padding should be zeroed out beforehand).
        Returns:
            [B, L', d] compressed representation where L' ≈ L / 4.
        """
        x = x.transpose(1, 2)                               # [B, d_esm, L]
        x = F.gelu(self.norm1(self.conv1(x).transpose(1, 2)))  # [B, L1, d]
        x = F.gelu(self.norm2(self.conv2(x.transpose(1, 2)).transpose(1, 2)))  # [B, L', d]
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention + self-attention for one protein attending to another.

    Simplified AlphaFold2 evoformer-style; call bidirectionally per layer: once for (p1, p2), once for (p2, p1).
    Pre-norm with shared LayerNorm for query and key/value in cross-attention.
    """

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(
        self,
        x_self: torch.Tensor,
        x_other: torch.Tensor,
        self_key_padding_mask: torch.BoolTensor | None = None,
        other_key_padding_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_self: [B, L_self, d] representations of the query protein.
            x_other: [B, L_other, d] representations of the other protein.
            self_key_padding_mask: [B, L_self] True at padding positions (MHA convention).
            other_key_padding_mask: [B, L_other] True at padding positions (MHA convention).
        Returns:
            [B, L_self, d] updated representation of the query protein.
        """
        # cross-attention: attend to the other protein
        h = self.norm1(x_self)
        h_other = self.norm1(x_other)
        x_self = x_self + self.cross_attn(h, h_other, h_other, key_padding_mask=other_key_padding_mask)[0]

        # self-attention within this protein
        h = self.norm2(x_self)
        x_self = x_self + self.self_attn(h, h, h, key_padding_mask=self_key_padding_mask)[0]

        return x_self


class PairMapConstructor(nn.Module):
    """Pair representation via learned outer product with element-wise multiplication.

    Captures multiplicative interactions between two protein representations, similar to AlphaFold2's outer product
    mean. Input: x1 [B, L1', d], x2 [B, L2', d] → [B, L1', L2', d_pair].
    """

    def __init__(self, d: int, d_pair: int, d_mid: int | None = None):
        super().__init__()
        d_mid = d_mid if d_mid is not None else d_pair
        self.proj = nn.Sequential(
            nn.Linear(3 * d, d_mid),
            nn.GELU(),
            nn.Linear(d_mid, d_pair),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: [B, L1', d] cross-attention enriched protein 1 representation.
            x2: [B, L2', d] cross-attention enriched protein 2 representation.
        Returns:
            [B, L1', L2', d_pair] pair interaction map.
        """
        L1 = x1.size(1)
        L2 = x2.size(1)

        x1_exp = x1.unsqueeze(2).expand(-1, -1, L2, -1)  # [B, L1', L2', d]
        x2_exp = x2.unsqueeze(1).expand(-1, L1, -1, -1)  # [B, L1', L2', d]

        pair_input = torch.cat([
            x1_exp, x2_exp, x1_exp * x2_exp
        ], dim=-1)  # [B, L1', L2', 3d]

        return self.proj(pair_input)  # [B, L1', L2', d_pair]


class PairMapToTokens(nn.Module):
    """2D pair map → decoder tokens via adaptive avg pooling to a fixed grid + LayerNorm + 2-layer MLP.

    Pipeline: [B, H, W, d_pair] → per-sample adaptive_avg_pool2d to (target_h, target_w) → LayerNorm
              → Linear → GELU → Linear → RMSNorm → [B, N_tokens, d_qwen].

    Last linear is zero-initialized so pair-map tokens start as zeros — the model begins from the per-protein-token
    baseline and gradually learns to use pair-map info. RMSNorm matches text embedding scale (~1.0 RMS).
    """

    def __init__(self, d_pair: int, d_qwen: int, d_mid: int | None = None, target_h: int = 16, target_w: int = 16):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        d_mid = d_mid if d_mid is not None else d_qwen
        self.norm = nn.LayerNorm(d_pair)
        self.proj = nn.Sequential(
            nn.Linear(d_pair, d_mid),
            nn.GELU(),
            nn.Linear(d_mid, d_qwen),
            nn.RMSNorm(d_qwen),
        )
        # zero-init last layer so pair-map tokens start as zeros (D1)
        nn.init.zeros_(self.proj[-2].weight)
        nn.init.zeros_(self.proj[-2].bias)

    def forward(
            self, pair_map: torch.Tensor, valid_h: torch.Tensor, valid_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pair_map: [B, H, W, d_pair] pair map (padding regions zeroed out).
            valid_h: [B] per-sample valid pair map height (compressed_len_a).
            valid_w: [B] per-sample valid pair map width (compressed_len_b).
        Returns:
            tokens: [B, target_h * target_w, d_qwen] pair map tokens.
        """
        B = pair_map.shape[0]
        tokens_list = []
        for i in range(B):
            h_i = int(valid_h[i].item())
            w_i = int(valid_w[i].item())
            valid_map = pair_map[i, :h_i, :w_i, :]                 # [h_i, w_i, d_pair]
            valid_map = valid_map.unsqueeze(0).permute(0, 3, 1, 2)  # [1, d_pair, h_i, w_i]
            pooled = F.adaptive_avg_pool2d(valid_map, (self.target_h, self.target_w))
            tokens_list.append(pooled)
        x = torch.cat(tokens_list, dim=0)   # [B, d_pair, target_h, target_w]
        x = x.permute(0, 2, 3, 1)           # [B, target_h, target_w, d_pair]
        x = self.norm(x)
        x = x.reshape(B, self.target_h * self.target_w, -1)
        return self.proj(x)                  # [B, target_h * target_w, d_qwen]


class PairEsm3Qwen3ForCausalLM(PreTrainedModel):
    """Dual-stream PPI description generator: per-protein tokens for context, pair-map tokens for interaction.

    Pipeline:
        protein A/B → ESM3 → SequenceCompressor → CrossAttention(xN)
            → ProteinProjector → per-protein tokens
            → PairMapConstructor → AdaptPool + MLP → pair-map tokens
            → splice into [text] <|boi|> [prot_A] [prot_B] [pair_map] <|eoi|> [text] → PairQwen3 → text

    Per-protein tokens use 1D sequential positions; pair-map tokens use 2D MRoPE.
    """
    config_class = PairEsm3Qwen3Config

    def __init__(
            self,
            config: PairEsm3Qwen3Config | None = None,
            esm_encoder: OpenESM3 | None = None,
            qwen_decoder: PairQwen3ForCausalLM | None = None,
            **kwargs
    ):
        if config is not None and qwen_decoder is not None:
            # preferred path: use provided pretrained components with explicit config;
            # esm_encoder is optional — pass None when using pre-computed embeddings
            super().__init__(config)
            self.esm_encoder = esm_encoder
            # re-init the rotary embedding with the MRoPE section from config
            # (the pretrained qwen_decoder was loaded with standard RoPE)
            self.qwen_decoder = qwen_decoder
            from model.modeling_pairqwen3 import PairQwen3RotaryEmbedding
            self.qwen_decoder.model.rotary_emb = PairQwen3RotaryEmbedding(
                config=qwen_decoder.config, mrope_section=config.mrope_section
            )
            self.qwen_decoder.model.mrope_section = config.mrope_section
        elif config is not None:
            # config-only path: build all components from scratch
            super().__init__(config)
            self.esm_encoder = OpenESM3(
                config.esm3_d_model,
                config.esm3_n_heads,
                config.esm3_v_heads,
                config.esm3_n_layers,
                OpenESM3_structure_encoder_v0,
                OpenESM3_structure_decoder_v0,
                OpenESM3_function_decoder_v0,
                config.esm3_tokenizers,
            )
            self.qwen_decoder = PairQwen3ForCausalLM(
                config.qwen_config,
                mrope_section=config.mrope_section,
            )
        else:
            # infer config from components
            config = PairEsm3Qwen3Config(
                esm3_d_model=esm_encoder.d_model,
                esm3_n_heads=esm_encoder.n_heads,
                esm3_v_heads=esm_encoder.v_heads,
                esm3_n_layers=esm_encoder.n_layers,
                esm3_tokenizers=esm_encoder.tokenizers,
                qwen_config=qwen_decoder.config,
                **kwargs
            )
            super().__init__(config)
            self.esm_encoder = esm_encoder
            self.qwen_decoder = qwen_decoder

        # interaction pipeline modules (shared compressor for both proteins)
        self.compressor = SequenceCompressor(config.esm3_d_model, config.compress_d)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(config.compress_d, config.n_cross_heads)
            for _ in range(config.n_cross_layers)
        ])
        self.pair_map_constructor = PairMapConstructor(
            config.compress_d, config.d_pair, d_mid=config.d_pair_constructor_mid
        )
        self.pair_map_to_tokens = PairMapToTokens(
            config.d_pair, config.qwen_config.hidden_size,
            d_mid=config.d_pair_map_mid,
            target_h=config.pair_map_target_h, target_w=config.pair_map_target_w,
        )

        # per-protein projector: compressed protein tokens → decoder embedding space (A1);
        # using compressed (compress_d) keeps decoder sequence length manageable (~50-466 tokens per protein vs
        # ~210-1866 from raw ESM3); RMSNorm on output matches text embedding scale (~1.0 RMS)
        hidden_size = config.qwen_config.hidden_size
        self.protein_projector = nn.Sequential(
            nn.Linear(config.compress_d, hidden_size),
            nn.RMSNorm(hidden_size),
        )

        # standalone trainable word embeddings for boi and eoi special tokens
        hidden_size = config.qwen_config.hidden_size
        embed_weight = self.qwen_decoder.model.embed_tokens.weight
        self.boi_embed = nn.Embedding(1, hidden_size, dtype=embed_weight.dtype)
        self.eoi_embed = nn.Embedding(1, hidden_size, dtype=embed_weight.dtype)

        if embed_weight.device.type != "meta":
            with torch.no_grad():
                self.boi_embed.weight.copy_(embed_weight[config.boi_id])
                self.eoi_embed.weight.copy_(embed_weight[config.eoi_id])

    def _encode_protein(
            self, protein_inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ESM3 encoder for one protein, returning (encoder_hidden_states, sequence_mask).

        Encoder is always frozen, so we wrap in torch.no_grad() to skip building a graph through 48 layers and
        geometric attention. Proteins are processed one at a time to avoid O(B*N^2) memory from geometric attention.
        """
        if self.esm_encoder is None:
            raise RuntimeError(
                "ESM3 encoder is not loaded; pass protein dicts with 'embedding' key or provide an encoder."
            )
        batch_size = protein_inputs["sequence_tokens"].shape[0]
        outputs = []
        with torch.no_grad():
            for i in range(batch_size):
                single_output = self.esm_encoder.batch_forward(
                    sequence_tokens=protein_inputs["sequence_tokens"][i:i+1],
                    structure_tokens=protein_inputs["structure_tokens"][i:i+1],
                    ss8_tokens=protein_inputs["ss8_tokens"][i:i+1],
                    sasa_tokens=protein_inputs["sasa_tokens"][i:i+1],
                    function_tokens=protein_inputs["function_tokens"][i:i+1],
                    residue_annotation_tokens=protein_inputs["residue_annotation_tokens"][i:i+1],
                    average_plddt=protein_inputs["average_plddt"][i:i+1],
                    per_res_plddt=protein_inputs["per_res_plddt"][i:i+1],
                    structure_coords=protein_inputs["structure_coords"][i:i+1],
                    chain_id=protein_inputs["chain_id"][i:i+1],
                    sequence_id=protein_inputs["sequence_id"][i:i+1],
                )
                outputs.append(single_output[0])
        return torch.cat(outputs, dim=0), protein_inputs["sequence_mask"]

    @staticmethod
    def _compute_compressed_mask(mask: torch.Tensor, compressed_length: int) -> torch.BoolTensor:
        """Approximate a boolean mask for the compressed sequence using adaptive max pooling."""
        mask_float = mask.float().unsqueeze(1)  # [B, 1, L]
        compressed = F.adaptive_max_pool1d(mask_float, compressed_length)  # [B, 1, L']
        return compressed.squeeze(1).bool()  # [B, L']

    def _run_interaction_pipeline(
            self,
            x1: torch.Tensor, mask1: torch.Tensor,
            x2: torch.Tensor, mask2: torch.Tensor,
            compressed_len_a: torch.LongTensor,
            compressed_len_b: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full interaction pipeline from encoder outputs to dual-stream tokens.

        Returns (protein_a_tokens [B, L1, d_qwen], protein_b_tokens [B, L2, d_qwen],
        pair_map_tokens [B, target_h * target_w, d_qwen]).
        """
        # zero out padding before compression
        x1_masked = x1 * mask1.unsqueeze(-1).float()
        x2_masked = x2 * mask2.unsqueeze(-1).float()

        # compress (shared compressor for both proteins)
        x1_comp = self.compressor(x1_masked)  # [B, L1', d]
        x2_comp = self.compressor(x2_masked)  # [B, L2', d]

        # Build compressed masks from exact per-sample compressed_len.
        # adaptive_max_pool1d masks are batch-dependent: pool window boundaries shift with total input length,
        # so boundary positions flip valid/masked across different batch compositions. Using compressed_len
        # directly produces consistent masks regardless of batch padding.
        B = x1_comp.shape[0]
        device = x1_comp.device
        L1 = x1_comp.shape[1]
        L2 = x2_comp.shape[1]
        range1 = torch.arange(L1, device=device).unsqueeze(0)
        range2 = torch.arange(L2, device=device).unsqueeze(0)
        valid1 = range1 < compressed_len_a.unsqueeze(1)  # [B, L1'] True = valid
        valid2 = range2 < compressed_len_b.unsqueeze(1)  # [B, L2'] True = valid
        mask1_comp = ~valid1  # MHA convention: True = masked
        mask2_comp = ~valid2

        # zero beyond compressed_len before cross-attention for clean padding
        x1_comp = x1_comp * valid1.unsqueeze(-1).float()
        x2_comp = x2_comp * valid2.unsqueeze(-1).float()

        # Snapshot post-compressor, pre-cross-attention tensors for the per-protein stream
        # (pair-unaware per-protein representation).
        # Aliasing (no .clone()) is safe: the CA loop reassigns x1_comp/x2_comp via
        # `xN_new = layer(...); xN_comp = xN_new * ...` without in-place writes into the snapshot.
        x1_pre_ca = x1_comp
        x2_pre_ca = x2_comp

        # cross-attention (bidirectional per layer);
        # zero after each layer: pre-norm LayerNorm converts zeros → bias (non-zero), so padded positions otherwise
        # accumulate batch-dependent values through residual connections
        for layer in self.cross_attn_layers:
            x1_new = layer(x1_comp, x2_comp, self_key_padding_mask=mask1_comp, other_key_padding_mask=mask2_comp)
            x2_new = layer(x2_comp, x1_comp, self_key_padding_mask=mask2_comp, other_key_padding_mask=mask1_comp)
            x1_comp = x1_new * valid1.unsqueeze(-1).float()
            x2_comp = x2_new * valid2.unsqueeze(-1).float()

        # project compressed proteins (pre-CA snapshot) to decoder embedding space
        protein_a_tokens = self.protein_projector(x1_pre_ca)  # [B, L1', d_qwen]
        protein_b_tokens = self.protein_projector(x2_pre_ca)  # [B, L2', d_qwen]

        # pair map construction + pooling
        pair_map = self.pair_map_constructor(x1_comp, x2_comp)  # [B, L1', L2', d_pair]
        # zero pair map at positions where either protein is padding
        valid_2d = (valid1.unsqueeze(2) & valid2.unsqueeze(1)).unsqueeze(-1)  # [B, L1', L2', 1]
        pair_map = pair_map * valid_2d.float()
        pair_map_tokens = self.pair_map_to_tokens(pair_map, compressed_len_a, compressed_len_b)

        return protein_a_tokens, protein_b_tokens, pair_map_tokens

    def embed_decoder_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed text tokens via the Qwen decoder, replacing boi/eoi tokens with the learnable embeddings."""
        inputs_embeds = self.qwen_decoder.get_input_embeddings()(input_ids)  # (bsz, seq_len, hidden_size)

        dummy_idx = torch.tensor(0, dtype=torch.long, device=inputs_embeds.device)
        boi_tensor = self.boi_embed(dummy_idx)
        eoi_tensor = self.eoi_embed(dummy_idx)

        boi_mask = (input_ids == self.config.boi_id).unsqueeze(-1)
        inputs_embeds = torch.where(boi_mask, boi_tensor, inputs_embeds)
        eoi_mask = (input_ids == self.config.eoi_id).unsqueeze(-1)
        inputs_embeds = torch.where(eoi_mask, eoi_tensor, inputs_embeds)

        return inputs_embeds

    def prepare_decoder_inputs(
            self,
            input_ids: torch.LongTensor,
            protein_a_tokens: torch.FloatTensor,
            protein_b_tokens: torch.FloatTensor,
            pair_map_tokens: torch.FloatTensor,
            compressed_len_a: torch.LongTensor,
            compressed_len_b: torch.LongTensor,
            attention_mask: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed text tokens and replace roi_id placeholders with dual-stream tokens.

        ROI partitioning: [protein A | protein B | pair map].
        """
        batch_size, seq_len = input_ids.shape
        n_pair = self.config.pair_map_target_h * self.config.pair_map_target_w
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.long, device=input_ids.device
            )

        inputs_embeds = self.embed_decoder_tokens(input_ids)

        for i in range(batch_size):
            roi_positions = (input_ids[i] == self.config.roi_id).nonzero(as_tuple=True)[0]
            n_a = int(compressed_len_a[i].item())
            n_b = int(compressed_len_b[i].item())
            expected = n_a + n_b + n_pair
            if roi_positions.shape[0] != expected:
                raise ValueError(
                    f"Sample {i}: expected {expected} roi placeholders but found {roi_positions.shape[0]}"
                )
            dtype = inputs_embeds.dtype
            inputs_embeds[i, roi_positions[:n_a]] = protein_a_tokens[i, :n_a].to(dtype)
            inputs_embeds[i, roi_positions[n_a:n_a + n_b]] = protein_b_tokens[i, :n_b].to(dtype)
            inputs_embeds[i, roi_positions[n_a + n_b:]] = pair_map_tokens[i].to(dtype)

        return inputs_embeds, attention_mask

    def build_roi_group_ids(
            self,
            input_ids: torch.LongTensor,
            compressed_len_a: torch.LongTensor,
            compressed_len_b: torch.LongTensor,
    ) -> torch.LongTensor:
        """Assign each token to a bidirectional group based on its ROI rank.

        0 = text (causal), 1 = protein A, 2 = protein B, 3 = pair map. Partitioning: first n_a ROIs = protA,
        next n_b = protB, rest = pair. Returns [B, seq_len] long tensor.
        """
        roi_mask = (input_ids == self.config.roi_id)
        roi_rank = roi_mask.long().cumsum(dim=-1) * roi_mask.long()  # 1-indexed for ROIs

        n_a = compressed_len_a.unsqueeze(1)  # [B, 1]
        n_b = compressed_len_b.unsqueeze(1)  # [B, 1]

        group_ids = torch.zeros_like(input_ids)
        group_ids[roi_mask & (roi_rank <= n_a)] = 1
        group_ids[roi_mask & (roi_rank > n_a) & (roi_rank <= n_a + n_b)] = 2
        group_ids[roi_mask & (roi_rank > n_a + n_b)] = 3
        return group_ids

    def build_position_ids(
            self,
            attention_mask: torch.LongTensor,
            group_ids: torch.LongTensor,
            compressed_len_a: torch.LongTensor,
            compressed_len_b: torch.LongTensor,
    ) -> torch.LongTensor:
        """Build 3-channel MRoPE position IDs from per-token group assignments.

        Pair-map 2D H/W positions are scaled to the true compressed protein lengths so pair tokens span the
        protein-coordinate range correctly. Returns [3, B, seq_len] long tensor.
        """
        B, seq_len = group_ids.shape
        device = group_ids.device
        target_h = self.config.pair_map_target_h
        target_w = self.config.pair_map_target_w

        is_roi = (group_ids > 0)
        is_prot_a = (group_ids == 1)
        is_prot_b = (group_ids == 2)
        is_pair = (group_ids == 3)

        # ROI counts per group
        n_a_roi = is_prot_a.sum(dim=-1)  # [B]
        n_b_roi = is_prot_b.sum(dim=-1)  # [B]

        # base text positions: cumulative count of non-ROI attended tokens
        non_roi = ~is_roi
        increments = non_roi.long() * attention_mask.long()
        text_positions = (increments.cumsum(dim=-1) - 1).clamp(min=0)
        text_positions.masked_fill_(attention_mask == 0, 0)

        # last sequence position of each group
        pos_range = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        last_prot_a = (is_prot_a.long() * pos_range).max(dim=-1).values  # [B]
        last_prot_b = (is_prot_b.long() * pos_range).max(dim=-1).values  # [B]
        last_pair = (is_pair.long() * pos_range).max(dim=-1).values  # [B]

        # text offsets after each ROI group
        non_roi_attended = non_roi & attention_mask.bool()
        t_pos = text_positions.clone()
        t_pos += ((pos_range > last_prot_a.unsqueeze(1)) & non_roi_attended).long() * n_a_roi.unsqueeze(1)
        t_pos += ((pos_range > last_prot_b.unsqueeze(1)) & non_roi_attended).long() * n_b_roi.unsqueeze(1)
        t_pos += ((pos_range > last_pair.unsqueeze(1)) & non_roi_attended).long()

        # initialize all channels with text positions
        position_ids = t_pos.unsqueeze(0).expand(3, -1, -1).clone()  # [3, B, seq_len]

        # first sequence position of each group (argmax on 0/1 tensor gives first True)
        batch_idx = torch.arange(B, device=device)
        first_prot_a = is_prot_a.long().argmax(dim=-1)  # [B]
        first_prot_b = is_prot_b.long().argmax(dim=-1)  # [B]
        first_pair = is_pair.long().argmax(dim=-1)  # [B]

        base_a = text_positions[batch_idx, first_prot_a] + 1  # [B]
        base_b = text_positions[batch_idx, first_prot_b] + n_a_roi + 1  # [B]
        base_pair = text_positions[batch_idx, first_pair] + n_a_roi + n_b_roi + 1  # [B]

        # within-group rank (0-indexed cumulative count within each group)
        prot_a_rank = is_prot_a.long().cumsum(dim=-1) - 1  # [B, seq_len]
        prot_b_rank = is_prot_b.long().cumsum(dim=-1) - 1
        pair_rank = is_pair.long().cumsum(dim=-1) - 1

        # protein A: 1D sequential, same in all 3 channels
        pos_a = base_a.unsqueeze(1) + prot_a_rank
        for ch in range(3):
            position_ids[ch] = torch.where(is_prot_a, pos_a, position_ids[ch])

        # protein B: 1D sequential, same in all 3 channels
        pos_b = base_b.unsqueeze(1) + prot_b_rank
        for ch in range(3):
            position_ids[ch] = torch.where(is_prot_b, pos_b, position_ids[ch])

        # pair map: 2D MRoPE (protein-coordinate-aligned); scaling uses the true compressed_len_a/b so pair_map_only
        # (no inline protein ROIs → n_a_roi=n_b_roi=0) still spans the correct 2D range
        pair_row = pair_rank // target_w  # [B, seq_len]
        pair_col = pair_rank % target_w
        len_a_f = compressed_len_a.to(torch.float32).unsqueeze(1)
        len_b_f = compressed_len_b.to(torch.float32).unsqueeze(1)
        pos_pair_t = base_pair.unsqueeze(1).expand_as(pair_rank)
        pos_pair_h = base_a.unsqueeze(1) + ((pair_row.float() + 0.5) * len_a_f / target_h).round().long()
        pos_pair_w = base_b.unsqueeze(1) + ((pair_col.float() + 0.5) * len_b_f / target_w).round().long()
        position_ids[0] = torch.where(is_pair, pos_pair_t, position_ids[0])
        position_ids[1] = torch.where(is_pair, pos_pair_h, position_ids[1])
        position_ids[2] = torch.where(is_pair, pos_pair_w, position_ids[2])

        return position_ids

    def forward(
            self,
            # chat template text inputs
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.LongTensor | None = None,
            labels: torch.LongTensor | None = None,
            # protein inputs (each is a dict with ESM3 encoder input tensors)
            protein_a: dict[str, torch.Tensor] | None = None,
            protein_b: dict[str, torch.Tensor] | None = None,
            # per-sample compressed protein lengths from the collator
            compressed_len_a: torch.LongTensor | None = None,
            compressed_len_b: torch.LongTensor | None = None,
            # behavior control arguments
            use_cache: torch.BoolTensor | None = None,
            cache_position: torch.LongTensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            return_decoder_inputs: bool = False,
    ) -> dict[str, torch.Tensor] | CausalLMOutputWithPast:
        """Compute encoder and interaction pipeline outputs, then pass to the decoder.

        compressed_len_a, compressed_len_b: [B] per-sample compressed lengths. Required.
        """
        # encode both proteins (or use pre-computed embeddings if available)
        if "embedding" in protein_a:
            x1, mask1 = protein_a["embedding"], protein_a["sequence_mask"]
        else:
            x1, mask1 = self._encode_protein(protein_a)
        if "embedding" in protein_b:
            x2, mask2 = protein_b["embedding"], protein_b["sequence_mask"]
        else:
            x2, mask2 = self._encode_protein(protein_b)

        # run the dual-stream interaction pipeline
        protein_a_tokens, protein_b_tokens, pair_map_tokens = (
            self._run_interaction_pipeline(
                x1, mask1, x2, mask2, compressed_len_a, compressed_len_b
            )
        )

        # prepare decoder inputs: embed text and replace roi placeholders with dual-stream tokens
        inputs_embeds, attention_mask = self.prepare_decoder_inputs(
            input_ids, protein_a_tokens, protein_b_tokens, pair_map_tokens,
            compressed_len_a, compressed_len_b, attention_mask,
        )

        # build per-group bidirectional IDs (0=text, 1=protA, 2=protB, 3=pair_map)
        group_ids = self.build_roi_group_ids(input_ids, compressed_len_a, compressed_len_b)

        # build 3-channel MRoPE position IDs
        position_ids = self.build_position_ids(
            attention_mask, group_ids, compressed_len_a, compressed_len_b
        )

        if return_decoder_inputs:
            return {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "bidirectional_group_ids": group_ids,
            }

        # qwen decoder forward
        decoder_output = self.qwen_decoder(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            bidirectional_group_ids=group_ids,
        )

        return decoder_output

    def generate(
        self,
        inputs: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        protein_a: dict[str, torch.Tensor] | None = None,
        protein_b: dict[str, torch.Tensor] | None = None,
        compressed_len_a: torch.LongTensor | None = None,
        compressed_len_b: torch.LongTensor | None = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> GenerateOutput | torch.LongTensor:
        """Run inference given a prompt and two protein inputs.

        `inputs` is the prompt only. Output drops the prompt because the decoder receives inputs_embeds.
        Generation behavior is controlled via kwargs — see `GenerationMixin.generate`.
        """
        prepared = self(
            input_ids=inputs,
            attention_mask=attention_mask,
            protein_a=protein_a,
            protein_b=protein_b,
            compressed_len_a=compressed_len_a,
            compressed_len_b=compressed_len_b,
            use_cache=False,
            return_decoder_inputs=True,
        )

        # Extend 3-channel position_ids for the maximum generation length.
        # Generated tokens are text (group 0), and build_position_ids assigns text tokens the same value (t_pos)
        # across all 3 MRoPE channels. The next text position is (max t_pos over attended text tokens) + 1. We
        # take the max specifically over the T channel of attended text tokens because pair_h/pair_w are scaled to
        # compressed protein length, not to text cumsum.
        position_ids_3d = prepared["position_ids"]  # [3, B, prompt_len]
        B = inputs.shape[0]
        device = position_ids_3d.device

        group_ids_prompt = prepared["bidirectional_group_ids"]  # [B, prompt_len]
        attention_mask_prompt = prepared["attention_mask"]  # [B, prompt_len]
        text_mask = (group_ids_prompt == 0) & attention_mask_prompt.bool()  # [B, prompt_len]
        t_channel = position_ids_3d[0]  # [B, prompt_len]
        max_text_pos = t_channel.masked_fill(~text_mask, -1).max(dim=-1).values  # [B]

        gen_positions = torch.zeros(3, B, max_new_tokens, dtype=torch.long, device=device)
        for i in range(B):
            start = int(max_text_pos[i].item()) + 1
            gen_positions[:, i, :] = torch.arange(
                start, start + max_new_tokens, device=device
            ).unsqueeze(0).expand(3, -1)
        full_position_ids = torch.cat([position_ids_3d, gen_positions], dim=-1)

        return self.qwen_decoder.generate(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            position_ids=full_position_ids,
            bidirectional_group_ids=prepared["bidirectional_group_ids"],
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
