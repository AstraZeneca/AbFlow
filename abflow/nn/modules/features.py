import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange


def apply_label_smoothing(
    one_hot_data: torch.Tensor, label_smoothing: float, num_classes: int
) -> torch.Tensor:
    """
    Apply label smoothing to one-hot encoded data.

    :param one_hot_data: One-hot encoded tensor of shape (..., num_classes).
    :param label_smoothing: Label smoothing factor (0 to 1).
    :param num_classes: Number of classes in one-hot encoding.
    :return: Smoothed tensor of shape (..., num_classes).
    """
    smooth_value = label_smoothing / (num_classes - 1)
    smoothed_data = one_hot_data * (1 - label_smoothing) + smooth_value
    return smoothed_data


def express_coords_in_frames(
    CA_coords: torch.Tensor, frame_orients: torch.Tensor, normalize: bool = False
) -> torch.Tensor:
    """
    Express coordinates in the local frame of each residue.

    :param CA_coords: Coordinates of CA atoms, shape (..., N_res, 3).
    :param frame_orients: Local frame orientation for each residue, shape (..., N_res, 3, 3).
    :param normalize: Whether to normalize the displacement vectors.
    :return: CA_vectors: The unit vector between CA atoms in the local frame, shape (..., N_res, N_res, 3).
    """
    CA_CA = CA_coords[..., None, :, :] - CA_coords[..., :, None, :]

    if normalize:
        CA_CA_norm = torch.norm(CA_CA, dim=-1, keepdim=True)
        CA_CA = CA_CA / (CA_CA_norm + 1e-10)

    frame_orients_inv = rearrange(frame_orients, "... m n -> ... n m")
    CA_vectors = torch.einsum("... m n,... j n->... j m", frame_orients_inv, CA_CA)

    return CA_vectors


class OneHotEmbedding(nn.Module):
    """
    Converts input sequences to one-hot encoded format with optional label smoothing.
    """

    def __init__(self, num_classes: int, label_smoothing: float = 0.0):
        """
        :param num_classes: Number of classes for one-hot encoding.
        :param label_smoothing: Label smoothing factor (0 to 1).
        """
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, x_i: torch.Tensor) -> torch.Tensor:
        """
        :param x_i: Input tensor of shape (...) with integer class indices.
        :return: One-hot encoded tensor of shape (..., num_classes) with label smoothing applied.
        """
        one_hot_data = F.one_hot(x_i, num_classes=self.num_classes)
        if self.label_smoothing > 0:
            return apply_label_smoothing(
                one_hot_data, self.label_smoothing, self.num_classes
            )
        return one_hot_data


class DihedralEmbedding(nn.Module):
    """
    Converts dihedral angles into their cosine and sine representations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, dihedrals: torch.Tensor) -> torch.Tensor:

        cos_angles = torch.cos(dihedrals)
        sin_angles = torch.sin(dihedrals)
        embeddings = torch.cat([cos_angles, sin_angles], dim=-1)

        return embeddings


class BinnedOneHotEmbedding(nn.Module):
    """
    Converts input sequences to binned one-hot encoded format.
    If values fall outside the specified bin range, they are assigned to the first or last category respectively.

    Example:
        >>> x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        >>> v_bins = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> result = one_hot(x, v_bins)
        >>> print(result)
        tensor([[[1., 0., 0., 0.],
                 [1., 0., 0., 0.],
                 [0., 1., 0., 0.]],
                [[0., 0., 1., 0.],
                 [0., 0., 0., 1.],
                 [0., 0., 0., 1.]]])
    """

    def __init__(self, v_bins: torch.Tensor):
        """
        :param v_bins: A tensor containing bin boundaries.
        """
        super().__init__()
        self.register_buffer("v_bins", v_bins)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Input tensor to be one-hot encoded based on bin boundaries.
        :return: One-hot encoded tensor where each value is binned according to self.v_bins.
        """

        bin_indices = torch.bucketize(data, self.v_bins) - 1

        bin_indices = torch.clamp(bin_indices, min=0, max=len(self.v_bins) - 2)

        p = torch.zeros(
            (*data.size(), len(self.v_bins) - 1), device=data.device, dtype=data.dtype
        )
        while len(bin_indices.shape) < len(p.shape):
            bin_indices = bin_indices.unsqueeze(-1)
        p.scatter_(dim=-1, index=bin_indices, value=1.0)

        return p


class CBDistogramEmbedding(nn.Module):
    """
    A one-hot pairwise feature indicating the distance between CB atoms (or CA for glycine).
    Pairwise distances are discretized into 38 bins, of equal width between 3.25 Ang and
    50.75 Ang; larger or smaller distances are assigned to the first or last bin respectively.
    """

    def __init__(
        self, num_bins: int = 38, min_dist: float = 3.25, max_dist: float = 50.75
    ):
        """
        :param num_bins: Number of bins to discretize distances.
        :param min_dist: Minimum distance for binning.
        :param max_dist: Maximum distance for binning.
        """

        super().__init__()
        bins = torch.linspace(min_dist, max_dist, num_bins + 1)
        self.binned_one_hot = BinnedOneHotEmbedding(bins)

    def forward(self, CB_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert pairwise CB atom distances to a one-hot encoded distance matrix.

        :param CB_coords: Input tensor of CB atom coordinates with shape (..., N_res, 3).
        :return: CB_distogram: A one-hot encoded distance matrix, shape (..., N_res, N_res, num_bins).
        """

        dist_matrix = torch.cdist(CB_coords, CB_coords, p=2)
        CB_distogram = self.binned_one_hot(dist_matrix)
        return CB_distogram


class CAUnitVectorEmbedding(nn.Module):
    """
    Converts CA atom coordinates to unit vectors in the local frame of each residue.

    The unit vector of the displacement of the CA atom of all residues is calculated within
    the local frame of each residue.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, CA_coords: torch.Tensor, frame_orients: torch.Tensor
    ) -> torch.Tensor:
        """
        :param CA_coords: Coordinates of CA atoms, shape (..., N_res, 3).
        :param frame_orients: Local frame orientation for each residue, shape (..., N_res, 3, 3).
        :return: CA_unit_vectors: The unit vector between CA atoms in the local frame, shape (..., N_res, N_res, 3).
        """
        return express_coords_in_frames(CA_coords, frame_orients, normalize=True)


class RelativePositionEncoding(nn.Module):

    def __init__(self, rmax: int):
        """
        :param rmax: Maximum relative position to encode.
        """

        super().__init__()
        self.rmax = rmax
        v_bins_pos = torch.linspace(0, 2 * rmax + 1, 2 * rmax + 2)
        self.rel_pos_binned_one_hot = BinnedOneHotEmbedding(v_bins_pos)

    def forward(self, res_index: torch.Tensor, chain_id: torch.Tensor) -> torch.Tensor:
        """
        :param res_index: Residue index tensor of shape (..., N_res).
        :param chain_id: Chain ID tensor of shape (..., N_res).
        :return: a_rel_pol_ij: One-hot encoded relative position tensor of shape (..., N_res, N_res, 2 * rmax + 1).
        """

        b_same_chain_ij = torch.eq(chain_id[..., :, None], chain_id[..., None, :])
        d_res_ij = torch.where(
            b_same_chain_ij,
            torch.clamp(
                res_index[..., :, None] - res_index[..., None, :] + self.rmax,
                0,
                2 * self.rmax,
            ),
            torch.tensor(2 * self.rmax + 1, device=res_index.device),
        )
        a_rel_pol_ij = self.rel_pos_binned_one_hot(d_res_ij)
        return a_rel_pol_ij
