"""
Modified ESMProtein class to facilitate loading pLDDT scores from predicted PDB file.

Original script: https://github.com/evolutionaryscale/esm/blob/main/esm/sdk/api.py
License: https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement
"""

from __future__ import annotations

import torch

from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain


class OpenESMProtein(ESMProtein):
    @classmethod
    def from_protein_chain(
        cls, protein_chain: ProteinChain, with_annotations: bool = False
    ) -> OpenESMProtein:
        """OpenESMProtein.from_pdb -> OpenESMProtein.from_protein_chain"""
        if with_annotations:
            return OpenESMProtein(
                sequence=protein_chain.sequence,
                sasa=protein_chain.sasa().tolist(),
                function_annotations=None,
                coordinates=torch.tensor(protein_chain.atom37_positions),
                plddt=protein_chain.confidence,  # expected np.array(float32) of shape (L,)
            )
        else:
            return OpenESMProtein(
                sequence=protein_chain.sequence,
                secondary_structure=None,
                sasa=None,
                function_annotations=None,
                coordinates=torch.tensor(protein_chain.atom37_positions),
                plddt=protein_chain.confidence,  # expected np.array(float32) of shape (L,)
            )
