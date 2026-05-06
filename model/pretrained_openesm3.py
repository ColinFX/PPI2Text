"""
Modified ESM3 pretrained loading pipeline to allow loading the model from a local directory.

Original script: https://github.com/evolutionaryscale/esm/blob/main/esm/pretrained.py
License: https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement
"""

import os
from typing import Callable

import torch
import torch.nn as nn

from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
)

from .modeling_openesm3 import OpenESM3
from .tokenization_openesm3 import get_openesm3_model_tokenizers

ModelBuilder = Callable[[torch.device | str], nn.Module]


def OpenESM3_structure_encoder_v0(local_dir: str | None = None, device: torch.device | str = "cpu"):
    """Modified from esm.pretrained.ESM3_structure_encoder_v0 to allow loading the model from a local directory."""
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    if local_dir is not None:
        state_dict = torch.load(
            os.path.join(local_dir, "data/weights/esm3_structure_encoder_v0.pth"),
            map_location=device,
        )
    else:
        state_dict = torch.load(
            data_root("esm3") / "data/weights/esm3_structure_encoder_v0.pth",
            map_location=device,
        )
    model.load_state_dict(state_dict)
    return model


def OpenESM3_structure_decoder_v0(local_dir: str | None = None, device: torch.device | str = "cpu"):
    """Modified from esm.pretrained.ESM3_structure_decoder_v0 to allow loading the model from a local directory."""
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    if local_dir is not None:
        state_dict = torch.load(
            os.path.join(local_dir, "data/weights/esm3_structure_decoder_v0.pth"),
            map_location=device,
        )
    else:
        state_dict = torch.load(
            data_root("esm3") / "data/weights/esm3_structure_decoder_v0.pth",
            map_location=device,
    )
    model.load_state_dict(state_dict)
    return model


def OpenESM3_function_decoder_v0(local_dir: str | None = None, device: torch.device | str = "cpu"):
    """Modified from esm.pretrained.ESM3_function_decoder_v0 to allow loading the model from a local directory."""
    with torch.device(device):
        model = FunctionTokenDecoder().eval()
    if local_dir is not None:
        state_dict = torch.load(
            os.path.join(local_dir, "data/weights/esm3_function_decoder_v0.pth"),
            map_location=device,
        )
    else:
        state_dict = torch.load(
            data_root("esm3") / "data/weights/esm3_function_decoder_v0.pth",
            map_location=device,
        )
    model.load_state_dict(state_dict)
    return model


def OpenESM3_sm_open_v0(local_dir: str | None = None, device: torch.device | str = "cpu"):
    """Modified from esm.pretrained.ESM3_sm_open_v0 to allow loading the model from a local directory."""
    with torch.device(device):
        model = OpenESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=OpenESM3_structure_encoder_v0,
            structure_decoder_fn=OpenESM3_structure_decoder_v0,
            function_decoder_fn=OpenESM3_function_decoder_v0,
            tokenizers=get_openesm3_model_tokenizers(ESM3_OPEN_SMALL, local_dir=local_dir),
        ).eval()
    if local_dir is not None:
        state_dict = torch.load(
            os.path.join(local_dir, "data/weights/esm3_sm_open_v1.pth"),
            map_location=device,
        )
    else:
        state_dict = torch.load(
            data_root("esm3") / "data/weights/esm3_sm_open_v1.pth",
            map_location=device,
        )
    model.load_state_dict(state_dict)
    return model


LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    ESM3_OPEN_SMALL: OpenESM3_sm_open_v0,
    ESM3_STRUCTURE_ENCODER_V0: OpenESM3_structure_encoder_v0,
    ESM3_STRUCTURE_DECODER_V0: OpenESM3_structure_decoder_v0,
    ESM3_FUNCTION_DECODER_V0: OpenESM3_function_decoder_v0,
}


def load_local_open_model(
    model_name: str, local_dir: str | None = None, device: torch.device = torch.device("cpu")
) -> nn.Module:
    """Modified from esm.pretrained.load_local_model to allow loading the model from a local directory."""
    if model_name not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in local model registry.")
    return LOCAL_MODEL_REGISTRY[model_name](local_dir=local_dir, device=device)
