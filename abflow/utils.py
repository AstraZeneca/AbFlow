"""
Contains utility functions for data manipulation.
"""

import torch
import scipy.spatial.transform as transform
from typing import List

from .nn.rigid_utils import Rigid, Rotation


def expand_mask(mask: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    Expand the mask to match the shape of the data tensor.
    """

    while mask.dim() < data.dim():
        mask = mask.unsqueeze(-1)
    return mask.expand_as(data)


def combine_masks(masks: List[torch.Tensor], data: torch.Tensor) -> torch.Tensor:
    """
    Combine a list of boolean masks for data with shape (N_batch, ...).
    Masks may have fewer dimensions, but will be expanded to the shape of the data before combining.

    Args:
        masks: A list of boolean masks with shape less than data.
        data: The data tensor with shape (N_batch, ...).

    Returns:
        combined_mask: A boolean tensor with the same shape as dimensions of `data`.
    """

    combined_mask = torch.ones(data.shape, dtype=torch.bool, device=data.device)

    if masks is not None:
        for mask in masks:
            expanded_mask = expand_mask(mask, data)
            combined_mask = combined_mask & expanded_mask

    return combined_mask


def average_data(data: torch.Tensor, masks: List[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply masks and compute the average over the masked elements for data with shapes (N_batch, ...).
    Dimensionality of data is assumed to be at least 2.

    Args:
        data: The data tensor to apply the mask to, shape (N_batch, ...).
        masks: A list of boolean masks with shape less than data.

    Returns:
        torch.Tensor: Averaged values after applying the mask, shape (N_batch,).
    """
    mask = combine_masks(masks, data)

    masked_data = data * mask
    sum_masked_data = masked_data.sum(dim=list(range(1, data.dim())))
    sum_mask = mask.sum(dim=list(range(1, mask.dim())))
    average_data = sum_masked_data / (sum_mask + 1e-9)

    return average_data


def apply_mask(
    data_1: torch.Tensor, data_2: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Combine two tensors using a mask. Select elements from data_2 where the mask is True,
    and from data_1 where the mask is False.
    """

    expanded_mask = expand_mask(mask, data_1)
    return data_1 * (~expanded_mask) + data_2 * expanded_mask


def mask_data(
    data: torch.Tensor, mask_value: float, mask: torch.Tensor
) -> torch.Tensor:
    """
    Mask the data by setting specified positions to a mask value.
    """

    mask_value_data = torch.full_like(data, mask_value)
    masked_data = apply_mask(data, mask_value_data, mask)
    return masked_data


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


def create_rigid(rots: torch.Tensor, trans: torch.Tensor) -> Rigid:
    """
    Create a Rigid object from rotation matrices and translations.
    """

    rots = Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)
