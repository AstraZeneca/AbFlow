"""
Contains utility functions for data manipulation.
"""

import torch
from typing import List

from ..rigid import Rigid, Rotation
import torch


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
    """

    combined_mask = torch.ones(data.shape, dtype=torch.bool, device=data.device)

    if masks is not None:
        for mask in masks:
            expanded_mask = expand_mask(mask, data)
            combined_mask = combined_mask & expanded_mask

    return combined_mask


def average_data(
    data: torch.Tensor, masks: List[torch.Tensor] = None, eps: float = 0.0
) -> torch.Tensor:
    """
    Applies masks to the data tensor and computes the average over all dimensions except
    the first (batch) dimension. Returns NaN if the mask is all False.

    :param data: The data tensor to be averaged, assumed to have at least one dimension,
                 with shape (N_batch, ...).
    :param masks: A list of boolean masks, each of which must be broadcastable to the shape of `data`.
                  If no masks are provided, all data elements are included in the average.
    :param eps: A small value added to the denominator to prevent division by zero.
                Default is set to 0 to return NaN when the mask is all False.

    :return: A tensor of shape (N_batch,), containing the averaged values for each batch
             after applying the masks.
    """
    mask = combine_masks(masks, data)

    masked_data = data * mask
    sum_masked_data = masked_data.sum(dim=list(range(1, data.dim())))
    sum_mask = mask.sum(dim=list(range(1, mask.dim())))
    average_data = sum_masked_data / (sum_mask + eps)

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
    data: torch.Tensor, mask_value: float, mask: torch.Tensor, in_place: bool = False
) -> torch.Tensor:
    """
    Masks 'data' by setting positions where 'mask' is True to 'mask_value'.
    If 'in_place' is True, modifies 'data' in place and returns it.
    Otherwise, returns a new tensor with those positions masked.
    """

    mask = expand_mask(mask, data)

    if in_place:
        # In-place masking
        data[mask] = mask_value
        return data
    else:
        # Out-of-place masking (original data remains unchanged)
        masked_data = data.clone()
        masked_data[mask] = mask_value
        return masked_data


def combine_coords(*coord_args: list[torch.Tensor], coord_dim: int = 3) -> torch.Tensor:
    """
    Combines a set of identically shaped 3-D coordinates into a tensor of shape (..., 3). The combination
    preserves ordering along the 2nd-to-last dimension among the input arguments.

    :param coord_args: List of tensors to combine of same shape (..., 3).
    :param coord_dim: Dimensionality of the coordinate vectors.
    :return: Combined tensor of shape (..., 3).
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
