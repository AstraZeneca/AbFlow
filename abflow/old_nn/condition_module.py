"""
Condition module for AbFlow.
"""

import torch
import torch.nn as nn

from ..nn.modules.pairformer import PairformerStack


class ConditionModule(nn.Module):
    """
    Condition module based on Pairformer from AlphaFold3.
    """

    def __init__(self, c_s: int, c_z: int, n_block: int):

        super().__init__()

        self.pairformer_stack = PairformerStack(
            c_s,
            c_z,
            n_block=n_block,
        )

    def forward(
        self,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
    ):
        """
        Single cycle conditioning.
        """

        s_i, z_ij = self.pairformer_stack(s_i, z_ij)

        return s_i, z_ij
