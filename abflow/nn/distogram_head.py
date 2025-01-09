import torch.nn as nn
import torch
import torch.nn.functional as F

from einops import rearrange

from .modules.features import CBDistogramEmbedding


class DistogramHead(nn.Module):
    """
    Distogram head to predict the distogram from the pair representations (from condition module)
    and used to calculate the auxiliary loss between and the true distogram (CB-CB distance).

    CB_distogram is a one-hot pairwise feature indicating the distance between CB atoms (CA for glycine).
    Pairwise distances are discretized into 66 bins: 64 bins between 2.0 and 22.0 Angstroms,
    and two bins for any larger and smaller distances.
    """

    def __init__(self, c_z: int):

        super().__init__()

        self.cb_distogram_embedding = CBDistogramEmbedding(
            num_bins=66, min_dist=2.0, max_dist=22.0
        )
        self.linear_no_bias_d = nn.Linear(c_z, 66, bias=False)

    def forward(self, z_ij: torch.Tensor) -> torch.Tensor:

        p_distogram_ij = self.linear_no_bias_d(
            z_ij + rearrange(z_ij, "b i j d -> b j i d")
        )
        p_distogram_ij = F.softmax(p_distogram_ij, dim=-1)

        return p_distogram_ij

    def get_loss_terms(self, z_ij: torch.Tensor, true_CB_coords: torch.Tensor) -> dict:

        p_distogram_ij = self(z_ij)
        CB_distogram = self.cb_distogram_embedding(true_CB_coords)

        return {"distogram": p_distogram_ij}, {"distogram": CB_distogram}
