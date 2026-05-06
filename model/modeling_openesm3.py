"""
Modified ESM3 model to facilitate loading from local weights and integration with Qwen3.

Original script: https://github.com/evolutionaryscale/esm/blob/main/esm/models/esm3.py
License: https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement
"""


from __future__ import annotations

import einops
from functools import partial
from typing import Callable, List, Tuple

import torch

from esm.models.esm3 import EncodeInputs, ESM3
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.tokenization import TokenizerCollectionProtocol
from esm.utils.constants import esm3 as C
from esm.utils.constants.models import ESM3_OPEN_SMALL, normalize_model_name
from esm.utils.misc import rbf
from esm.utils.structure.affine3d import build_affine3d_from_coordinates


class OpenEncodeInputs(EncodeInputs):
    def forward(
        self,
        sequence_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        average_plddt: torch.Tensor,
        per_res_plddt: torch.Tensor,
        ss8_tokens: torch.Tensor,
        sasa_tokens: torch.Tensor,
        function_tokens: torch.Tensor,
        residue_annotation_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modified EncodeInputs to return additional intermediate states for potential deepstack.
        Returns: aggregate_embed, structure_per_res_plddt, sequence_embed, structure_embed, ss8_embed, sasa_embed
        """
        sequence_embed = self.sequence_embed(sequence_tokens)

        rbf_16_fn = partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        # the `masked_fill(padding_mask.unsqueeze(2), 0)` for the two below is unnecessary
        # as pad tokens never even interact with the "real" tokens (due to sequence_id)
        plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt))
        structure_per_res_plddt = self.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt)
        )

        # structure + "structural features" embeds
        structure_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)

        # "functional" features embeds
        function_embed = torch.cat(
            [
                embed_fn(funcs)
                for embed_fn, funcs in zip(
                    self.function_embed, function_tokens.unbind(-1)
                )
            ],
            -1,
        )

        # residue embeds
        B, L, N = residue_annotation_tokens.shape
        residue_embed = self.residue_embed(
            einops.rearrange(
                residue_annotation_tokens, "B L N -> (B L) N", B=B, L=L, N=N
            )
        )
        residue_embed = einops.rearrange(residue_embed, "(B L) D -> B L D", B=B, L=L)

        aggregate_embed = (
            sequence_embed
            + plddt_embed
            + structure_per_res_plddt
            + structure_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )
        return aggregate_embed, structure_per_res_plddt, sequence_embed, structure_embed, ss8_embed, sasa_embed


class OpenESM3(ESM3):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int,
        n_layers: int,
        structure_encoder_fn: Callable[[str | None, torch.device | str], StructureTokenEncoder],
        structure_decoder_fn: Callable[[str | None, torch.device | str], StructureTokenDecoder],
        function_decoder_fn: Callable[[str | None, torch.device | str], FunctionTokenDecoder],
        tokenizers: TokenizerCollectionProtocol,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            v_heads=v_heads,
            n_layers=n_layers,
            structure_encoder_fn=structure_encoder_fn,
            structure_decoder_fn=structure_decoder_fn,
            function_decoder_fn=function_decoder_fn,
            tokenizers=tokenizers
        )
        self.encoder = OpenEncodeInputs(d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.v_heads = v_heads
        self.n_layers = n_layers
        self.local_dir: str = None

    @classmethod
    def from_pretrained(
        cls, model_name: str = ESM3_OPEN_SMALL, local_dir: str | None = None, device: torch.device | None = None
    ) -> OpenESM3:
        from .pretrained_openesm3 import load_local_open_model

        model_name = normalize_model_name(model_name)
        if not model_name:
            raise ValueError(f"Model name {model_name} is not a valid ESM3 model name.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_local_open_model(model_name, local_dir=local_dir, device=device)
        model = model.to(torch.bfloat16)  # force bf16 dtype aligning with pretrained weights
        assert isinstance(model, OpenESM3)
        model.local_dir = local_dir
        return model

    def get_structure_encoder(self) -> StructureTokenEncoder:
        if self._structure_encoder is None:
            self._structure_encoder = self.structure_encoder_fn(self.local_dir, self.device)  # allow local_dir
        return self._structure_encoder

    def get_structure_decoder(self) -> StructureTokenDecoder:
        if self._structure_decoder is None:
            self._structure_decoder = self.structure_decoder_fn(self.local_dir, self.device)  # allow local_dir
        return self._structure_decoder

    def get_function_decoder(self) -> FunctionTokenDecoder:
        if self._function_decoder is None:
            self._function_decoder = self.function_decoder_fn(self.local_dir, self.device)  # allow local_dir
        return self._function_decoder

    def batch_forward(
        self,
        sequence_tokens: torch.Tensor,  # (B, L)
        structure_tokens: torch.Tensor,  # (B, L)
        ss8_tokens: torch.Tensor,  # (B, L)
        sasa_tokens: torch.Tensor,  # (B, L)
        function_tokens: torch.Tensor,  # (B, L, 8)
        residue_annotation_tokens: torch.Tensor,  # (B, L, 16)
        average_plddt: torch.Tensor,  # (B, L)
        per_res_plddt: torch.Tensor,  # (B, L)
        structure_coords: torch.Tensor,  # (B, L, 3, 3)
        chain_id: torch.Tensor,  # (B, L), for masking in both self attention and geometric attention
        sequence_id: torch.Tensor,  # (B, L), for masking in geometric attention
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for protein embedding.
        Return an additional list of tensors, each of shape (B, L, D), as intermediate states for potential deepstack,
        in the order of:
            - per-residue pLDDT embedding,
            - sequence embeddding,
            - structure embedding,
            - secondary structure embedding,
            - SASA embedding,
            - transformer stack layer 1 output,
            - transformer stack layer 2 output,
            - ...
            - transformer stack layer N output (before normalization).
        """
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        encoder_outputs = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        aggregate_embed = encoder_outputs[0]
        x_postnorm, x_prenorm, hidden_states = self.transformer(
            aggregate_embed, sequence_id, affine, affine_mask, chain_id
        )
        return x_postnorm, list(encoder_outputs[1:]) + hidden_states
