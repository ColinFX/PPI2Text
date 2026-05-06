"""
Modified ESM3 tokenizer loading pipeline to allow loading the tokenizer collection from a local directory.

Original script: https://github.com/evolutionaryscale/esm/blob/main/esm/tokenization/__init__.py
License: https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement
"""

import os

from esm.utils.constants.models import ESM3_OPEN_SMALL, normalize_model_name
from esm.utils.constants.esm3 import RESID_CSV

from esm.tokenization import TokenizerCollection
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.tokenization.residue_tokenizer import ResidueAnnotationsTokenizer
from esm.tokenization.sasa_tokenizer import SASADiscretizingTokenizer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.tokenization.ss_tokenizer import SecondaryStructureTokenizer
from esm.tokenization.structure_tokenizer import StructureTokenizer


def get_openesm3_model_tokenizers(
        model: str = ESM3_OPEN_SMALL,
        local_dir: str | None = None
    ) -> TokenizerCollection:
    """
    Modified from esm.tokenization.get_esm3_model_tokenizers to allow loading the tokenizer collection from a local
    directory.
    """
    if normalize_model_name(model) == ESM3_OPEN_SMALL:
        return TokenizerCollection(
            sequence=EsmSequenceTokenizer(),
            structure=StructureTokenizer(),
            secondary_structure=SecondaryStructureTokenizer(kind="ss8"),
            sasa=SASADiscretizingTokenizer(),
            function=InterProQuantizedTokenizer(),
            residue_annotations=ResidueAnnotationsTokenizer(
                csv_path=os.path.join(local_dir, RESID_CSV) if local_dir is not None else None
            ),
        )
    else:
        raise ValueError(f"Unknown model: {model}")
