"""
Configuration class for the assembled PairEsm3Qwen3ForCausalLM model.

PairEsm3Qwen3Config = ESM3-arguments + interaction-pipeline-arguments + MRoPE-arguments + Qwen3Config
"""

from esm.tokenization import TokenizerCollection
from transformers import PretrainedConfig
from transformers.models.qwen3 import Qwen3Config


class PairEsm3Qwen3Config(PretrainedConfig):
    """
    Configuration class of PairEsm3Qwen3ForCausalLM model.

    Interaction pipeline parameters:
        compress_d: hidden dimension after sequence compression.
        n_cross_layers: number of cross-attention layers between the two compressed protein representations.
        n_cross_heads: number of attention heads in each cross-attention layer.
        d_pair: hidden dimension of the pair map representation.
        d_pair_constructor_mid: middle hidden dimension in PairMapConstructor MLP (3*compress_d → mid → d_pair).
        d_pair_map_mid: middle hidden dimension in PairMapToTokens MLP (d_pair → d_pair_map_mid → d_qwen).
        pair_map_target_h: fixed output grid height for PairMapToTokens adaptive pooling.
        pair_map_target_w: fixed output grid width for PairMapToTokens adaptive pooling.

    MRoPE parameters:
        mrope_section: allocation of frequency pairs to [temporal, height, width] channels for interleaved MRoPE.
            Must sum to head_dim // 2 of the Qwen3 decoder. Default [24, 20, 20] for head_dim=128.

    Special tokens:
        boi_id: begin of interaction embeddings, default to `<|boi|>` in the tokenizer.
        eoi_id: end of interaction embeddings, default to `<|eoi|>` in the tokenizer.
        roi_id: representation of interaction as placeholder, default to `<|roi|>` in the tokenizer.
    """
    model_type = "pair_esm3qwen3"

    def __init__(
            self,
            # esm3 arguments
            esm3_d_model: int = 1536,
            esm3_n_heads: int = 24,
            esm3_v_heads: int = 256,
            esm3_n_layers: int = 48,
            esm3_tokenizers: TokenizerCollection | None = None,
            # interaction pipeline
            compress_d: int = 1024,
            n_cross_layers: int = 2,
            n_cross_heads: int = 8,
            d_pair: int = 1024,
            d_pair_constructor_mid: int = 2048,
            d_pair_map_mid: int = 1536,
            pair_map_target_h: int = 16,
            pair_map_target_w: int = 16,
            # mrope
            mrope_section: list[int] | None = None,
            # qwen decoder
            qwen_config: Qwen3Config | None = None,
            # special tokens
            boi_id: int = 151669,
            eoi_id: int = 151670,
            roi_id: int = 151671,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.esm3_d_model = esm3_d_model
        self.esm3_n_heads = esm3_n_heads
        self.esm3_v_heads = esm3_v_heads
        self.esm3_n_layers = esm3_n_layers
        self.esm3_tokenizers = esm3_tokenizers

        self.compress_d = compress_d
        self.n_cross_layers = n_cross_layers
        self.n_cross_heads = n_cross_heads
        self.d_pair = d_pair
        self.d_pair_constructor_mid = d_pair_constructor_mid
        self.d_pair_map_mid = d_pair_map_mid
        self.pair_map_target_h = pair_map_target_h
        self.pair_map_target_w = pair_map_target_w

        self.mrope_section = mrope_section if mrope_section is not None else [24, 20, 20]
        self.qwen_config = qwen_config

        self.boi_id = boi_id
        self.eoi_id = eoi_id
        self.roi_id = roi_id
