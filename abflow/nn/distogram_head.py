import torch.nn as nn
import torch
import torch.nn.functional as F


class DistogramHead(nn.Module):
    """
    Distogram head to predict the distogram from the pair representations (from condition module)
    and used to calculate the auxiliary loss between and the true distogram (CB-CB distance).
    """

    def __init__(self, c_z: int):

        super().__init__()

        self.linear_no_bias_d = nn.Linear(c_z, 66, bias=False)

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        p_distogram_ij = self.linear_no_bias_d(z_ij + torch.einsum("bijd->bjid", z_ij))
        p_distogram_ij = F.softmax(p_distogram_ij, dim=-1)

        return p_distogram_ij
