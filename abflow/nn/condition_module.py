"""
Condition module for AbFlow.
"""

import torch
import torch.nn as nn

from ..nn.pairformer import PairformerStack


class ConditionModule(nn.Module):
    """
    Condition module based on Pairformer from AlphaFold3.
    """

    def __init__(self, c_s: int, c_z: int, n_block: int, params: dict):

        super().__init__()

        pf_params = params["Pairformer"]
        tmo_c_hidden = pf_params["TriangleMultiplicationOutgoing"]["c_hidden"]
        tmi_c_hidden = pf_params["TriangleMultiplicationIncoming"]["c_hidden"]
        tasn_c_hidden = pf_params["TriangleAttentionStartingNode"]["c_hidden"]
        tasn_N_head = pf_params["TriangleAttentionStartingNode"]["N_head"]
        taen_c_hidden = pf_params["TriangleAttentionEndingNode"]["c_hidden"]
        taen_N_head = pf_params["TriangleAttentionEndingNode"]["N_head"]
        apb_c_hidden = pf_params["AttentionPairBias"]["c_hidden"]
        apb_N_head = pf_params["AttentionPairBias"]["N_head"]

        self.pairformer_stack = PairformerStack(
            c_s,
            c_z,
            n_block=n_block,
            tmo_c_hidden=tmo_c_hidden,
            tmi_c_hidden=tmi_c_hidden,
            tasn_c_hidden=tasn_c_hidden,
            tasn_N_head=tasn_N_head,
            taen_c_hidden=taen_c_hidden,
            taen_N_head=taen_N_head,
            apb_c_hidden=apb_c_hidden,
            apb_N_head=apb_N_head,
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
