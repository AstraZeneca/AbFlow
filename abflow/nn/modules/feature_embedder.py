import torch
from torch import nn
from einops import rearrange


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
        param x_i: Input tensor of shape (...) with integer class indices.
        return s_i: One-hot encoded tensor of shape (..., num_classes) with label smoothing applied.
        """

        one_hot_data = F.one_hot(x_i, num_classes=self.num_classes).float()
        smooth_value = self.label_smoothing / (self.num_classes - 1)
        s_i = one_hot_data * (1 - self.label_smoothing) + smooth_value

        return s_i


class BinnedOneHotEmbedding(nn.Module):
    """
    Converts input sequences to binned one-hot encoded format.

    By default, includes -inf and inf as boundaries, resulting in N+1 bins for N boundaries.
    If concat_inf is set to False, ensure x falls within [min(v_bins), max(v_bins)].
    This function will raise an error if x contains nan values.

    Example:
            >>> x = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
            >>> v_bins = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> result = one_hot(x, v_bins)
            >>> print(result)
            tensor([[[1., 0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 0.]],
                            [[0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 1.]]])
    """

    def __init__(self, v_bins: torch.Tensor, concat_inf: bool = True):
        """

        :param v_bins: A tensor containing bin boundaries.
        :param concat_inf: If True, includes -inf and inf as boundaries for the bins.
        """
        super().__init__()
        if concat_inf:
            v_bins = torch.cat(
                [
                    torch.tensor([float("-inf")], device=v_bins.device),
                    v_bins,
                    torch.tensor([float("inf")], device=v_bins.device),
                ]
            )
        self.register_buffer("v_bins", v_bins)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Input tensor to be one-hot encoded based on bin boundaries.
        :return: One-hot encoded tensor where each value is binned according to self.v_bins.
        """

        v_bins = self.v_bins.to(data.device)

        bin_indices = torch.bucketize(data, v_bins) - 1
        bin_indices[torch.isclose(data, v_bins[0])] = 0

        p = torch.zeros((*data.size(), len(v_bins) - 1), device=data.device)
        while len(bin_indices.shape) < len(p.shape):
            bin_indices = bin_indices.unsqueeze(-1)
        p.scatter_(dim=-1, index=bin_indices, value=1.0)

        return p


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

        # Calculate CA unit displacement vectors
        CA_CA = CA_coords[..., None, :, :] - CA_coords[..., :, None, :]
        CA_CA_norm = torch.norm(CA_CA, dim=-1, keepdim=True)
        d = CA_CA / (CA_CA_norm + 1e-10)

        frame_orients_inv = rearrange(frame_orients, "... m n -> ... n m")
        CA_unit_vectors = torch.einsum("... m n,... j n->... j m", frame_orients_inv, d)

        return CA_unit_vectors


class CBDistogramEmbedding(nn.Module):
    """
    A one-hot pairwise feature indicating the distance between CB atoms (or CA for glycine).
    Pairwise distances are discretized into 40 bins. 38 bins are of equal width between 3.25 Ang and
    50.75 Ang; two more bins contain any smaller and larger distances.
    """

    def __init__(
        self, num_bins: int = 40, min_dist: float = 3.25, max_dist: float = 50.75
    ):
        """
        :param num_bins: Number of bins to discretize distances.
        :param min_dist: Minimum distance for binning.
        :param max_dist: Maximum distance for binning.
        """

        super().__init__()
        bins = torch.linspace(min_dist, max_dist, num_bins - 1)
        self.binned_one_hot = BinnedOneHotEmbedding(bins)

    def forward(self, CB_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert pairwise CB atom distances to a one-hot encoded distance matrix.
        :param CB_coords: Input tensor of CB atom coordinates with shape (..., N_res, 3).
        :return: CB_distogram: A one-hot encoded distance matrix, shape (..., N_res, N_res, num_bins).
        """

        dist_matrix = torch.cdist(CB_coords, CB_coords, p=-1)
        CB_distogram = self.binned_one_hot(dist_matrix)
        return CB_distogram
