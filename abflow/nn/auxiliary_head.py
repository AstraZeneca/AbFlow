"""
Auxiliary heads for confidence estimation and auxiliary loss.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

from .feature_embedder import one_hot
from ..nn.pairformer import PairformerStack


class ConfidenceHead(nn.Module):
    """
    Confidence head to predict the plddt score for each residue.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_block: int,
        params: dict,
    ):

        super().__init__()

        self.linear_no_bias_s_i = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_s_j = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_z = nn.Linear(c_z, c_z, bias=False)
        self.linear_no_bias_d = nn.Linear(11, c_z, bias=False)

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

        self.linear_no_bias_plddt = nn.Linear(c_s, 50, bias=False)

    def forward(
        self,
        s_inputs_i: torch.Tensor,
        z_inputs_ij: torch.Tensor,
        s_i: torch.Tensor,
        z_ij: torch.Tensor,
        x_pred_i: torch.Tensor,
    ):
        """
        Current plddt only inputs one representative atom for each residue. Typically, the C-alpha atom.
        """
        z_ij = (
            z_ij
            + self.linear_no_bias_z(z_inputs_ij)
            + self.linear_no_bias_s_i(s_inputs_i)[:, :, None, :]
            + self.linear_no_bias_s_j(s_inputs_i)[:, None, :, :]
        )

        # Embed pair distances of C-alpha atoms
        d_ij = torch.norm(x_pred_i[:, :, None, :] - x_pred_i[:, None, :, :], dim=-1)

        v_bins = torch.linspace(3.375, 21.375, 10, device=d_ij.device)
        z_ij = z_ij + self.linear_no_bias_d(one_hot(d_ij, v_bins))

        s_post_i, z_post_ij = self.pairformer_stack(s_i, z_ij)
        s_i = s_i + s_post_i
        z_ij = z_ij + z_post_ij

        p_plddt_i = F.softmax(self.linear_no_bias_plddt(s_i), dim=-1)

        return p_plddt_i


class DistogramHead(nn.Module):
    """
    Distogram head to predict the distogram from the pair representations (from condition module) and used to calculate the auxiliary loss between
    and the true distogram (CB-CB distance).
    """

    def __init__(self, c_z: int):

        super().__init__()

        self.linear_no_bias_d = nn.Linear(c_z, 66, bias=False)

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        p_distogram_ij = self.linear_no_bias_d(z_ij + torch.einsum("bijd->bjid", z_ij))
        p_distogram_ij = F.softmax(p_distogram_ij, dim=-1)

        return p_distogram_ij
