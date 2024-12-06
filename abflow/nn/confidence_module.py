import torch.nn as nn
import torch
import torch.nn.functional as F

from .modules.feature_embedder import BinnedOneHotEmbedding
from .modules.pairformer import PairformerStack


class ConfidenceHead(nn.Module):
    """
    Confidence head to predict the plddt score for each residue.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        n_block: int,
    ):

        super().__init__()

        self.linear_no_bias_s_i = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_s_j = nn.Linear(c_s, c_z, bias=False)
        self.linear_no_bias_z = nn.Linear(c_z, c_z, bias=False)

        ca_num_bins = 11
        ca_min = 3.375
        ca_max = 21.375
        ca_bins = torch.linspace(ca_min, ca_max, ca_num_bins - 1)
        self.ca_binned_one_hot = BinnedOneHotEmbedding(ca_bins)
        self.linear_no_bias_d = nn.Linear(ca_num_bins, c_z, bias=False)

        self.pairformer_stack = PairformerStack(
            c_s,
            c_z,
            n_block=n_block,
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

        ca_binned_one_hot = self.ca_binned_one_hot(d_ij)
        z_ij = z_ij + self.linear_no_bias_d(ca_binned_one_hot)

        s_post_i, z_post_ij = self.pairformer_stack(s_i, z_ij)
        s_i = s_i + s_post_i
        z_ij = z_ij + z_post_ij

        p_plddt_i = F.softmax(self.linear_no_bias_plddt(s_i), dim=-1)

        return p_plddt_i
