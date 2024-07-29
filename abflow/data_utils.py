"""
Contains utility functions for data manipulation.
"""

import torch
import scipy.spatial.transform as transform

from .nn.rigid_utils import Rigid, Rotation


def expand_mask(mask: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    Expand the mask to match the shape of the data tensor.
    """

    while mask.dim() < data.dim():
        mask = mask.unsqueeze(-1)
    return mask.expand_as(data)


def apply_mask(
    data_1: torch.Tensor, data_2: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Combine two tensors using a mask. Select elements from data_2 where the mask is 1,
    and from data_1 where the mask is 0.
    """

    expanded_mask = expand_mask(mask, data_1)
    return data_1 * (1 - expanded_mask) + data_2 * expanded_mask


def mask_data(
    data: torch.Tensor, mask_value: float, mask: torch.Tensor
) -> torch.Tensor:
    """
    Mask the data by setting specified positions to a mask value.
    """

    mask_value_data = torch.full_like(data, mask_value)
    masked_data = apply_mask(data, mask_value_data, mask)
    return masked_data


def inv_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Invert a mask of 0s and 1s.
    """

    return (~mask.bool()).long()


def combine_coords(*coord_args: list[torch.Tensor], coord_dim: int = 3) -> torch.Tensor:
    """
    Combines a set of identically shaped 3-D coordinates into a tensor of shape (..., 3). The combination
    preserves ordering along the 2nd-to-last dimension among the input arguments.

    Args:
        coord_args (torch.Tensor): List of tensors to combine of same shape (..., 3).
        coord_dim (int): Dimensionality of the coordinate vectors.

    Returns:
        torch.Tensor: Combined tensor of shape (..., 3).
    """

    outshape = coord_args[0].shape[: len(coord_args[0].shape) - 2] + (-1, coord_dim)
    combined_coords = torch.cat(coord_args, dim=-1).view(outshape)

    return combined_coords


def safe_div(
    numerator: torch.Tensor, denominator: torch.Tensor, default_value: float = 0.0
) -> torch.Tensor:
    """
    Safely divide two tensors element-wise. If the denominator is zero, use the default value.
    """

    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.tensor(default_value, device=denominator.device),
    )


def create_rigid(rots: torch.Tensor, trans: torch.Tensor) -> Rigid:
    """
    Create a Rigid object from rotation matrices and translations.
    """

    rots = Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


def random_rotmat(size: int) -> torch.Tensor:
    """
    Generate random rotation matrices.

    Args:
        size (int): Number of rotation matrices to generate.

    Returns:
        torch.Tensor: Random rotation matrices of shape (size, 3, 3).
    """
    rotation_matrices = transform.Rotation.random(size).as_matrix()
    return torch.tensor(rotation_matrices, dtype=torch.float32)


def random_rigid_batch(N_batch: int, N_res: int) -> Rigid:
    """
    Generate a batch of random Rigid objects.

    Args:
        N_batch (int): Number of batches.
        N_res (int): Number of residues.

    Returns:
        Rigid: Random rigid transformations with
                rotations of shape (N_batch, N_res, 3, 3) and
                translations of shape (N_batch, N_res, 3).
    """
    rotations = random_rotmat(N_batch * N_res).reshape(N_batch, N_res, 3, 3)
    translations = torch.rand((N_batch, N_res, 3))
    return create_rigid(rotations, translations)


def random_rigid_global(N_batch: int) -> Rigid:
    """
    Generate random global Rigid objects for each batch.
    Same rigid transformation for all residues.

    Args:
        N_batch (int): Number of batches.
        N_res (int): Number of residues.

    Returns:
        Rigid: Random rigid transformations with
                rotations of shape (N_batch, 1, 3, 3) and
                translations of shape (N_batch, 1, 3).
    """
    rotations = random_rotmat(N_batch)[:, None, :, :]
    translations = torch.rand((N_batch, 3))[:, None, :]
    return create_rigid(rotations, translations)


def random_single_repr(N_batch: int, N_res: int, c_s: int) -> torch.Tensor:
    """
    Generate random single representations for each residue.

    Args:
        N_batch (int): Number of batches.
        N_res (int): Number of residues.
        c_s (int): Number of channels in the single representation.

    Returns:
        torch.Tensor: Random single representation of shape (N_batch, N_res, c_s).
    """
    return torch.rand((N_batch, N_res, c_s))


def random_pair_repr(N_batch: int, N_res: int, c_z: int) -> torch.Tensor:
    """
    Generate random pair representations for each residue pair.

    Args:
        N_batch (int): Number of batches.
        N_res (int): Number of residues.
        c_p (int): Number of channels in the pair representation.

    Returns:
        torch.Tensor: Random pair representation of shape (N_batch, N_res, N_res, c_z).
    """
    return torch.rand((N_batch, N_res, N_res, c_z))
