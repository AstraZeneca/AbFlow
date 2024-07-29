import torch

from ..data_utils import safe_div


def combine_masks(masks: list[torch.Tensor], data: torch.Tensor) -> torch.Tensor:
    """
    Combine a list of masks each with shape (N_batch, N_res), for data with shapes (N_batch, N_res, ...).
    """

    combined_mask = torch.ones(data.shape[:2], dtype=torch.long, device=data.device)

    if masks is not None:
        for mask in masks:
            combined_mask = combined_mask & mask

    return combined_mask.long()


def average_data_2d(
    data: torch.Tensor, masks_dim_1: list[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply masks and compute the average over the masked elements for data with shapes (N_batch, N_res).

    Args:
        data (torch.Tensor): The data tensor to apply the mask to, shape (N_batch, N_res).
        masks_dim_1 (torch.Tensor, optional): Mask for the first dimension, shape (N_batch, N_res).

    Returns:
        torch.Tensor: Averaged values after applying the mask, shape (N_batch,).
    """

    masks_dim_1 = combine_masks(masks_dim_1, data)

    masked_data = data * masks_dim_1
    sum_masked_data = masked_data.sum(dim=-1)
    sum_mask = masks_dim_1.sum(dim=-1)

    return safe_div(sum_masked_data, sum_mask)


def average_data_3d(
    data: torch.Tensor,
    masks_dim_1: list[torch.Tensor] = None,
    masks_dim_2: list[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply masks and compute the average over the masked elements for data with shapes (N_batch, N_res, N_res).

    Args:
        data (torch.Tensor): The data tensor to apply the mask to, shape (N_batch, N_res, N_res).
        masks_dim_1 (torch.Tensor, optional): Mask for the first dimension, shape (N_batch, N_res).
        masks_dim_2 (torch.Tensor, optional): Mask for the second dimension, shape (N_batch, N_res).

    Returns:
        torch.Tensor: Averaged values after applying the mask, shape (N_batch,).
    """

    masks_dim_1 = combine_masks(masks_dim_1, data)[:, :, None]
    masks_dim_2 = combine_masks(masks_dim_2, data)[:, None, :]
    combined_mask = masks_dim_1 * masks_dim_2

    masked_data = data * combined_mask
    sum_masked_data = masked_data.sum(dim=(-1, -2))
    sum_mask = combined_mask.sum(dim=(-1, -2))

    return safe_div(sum_masked_data, sum_mask)


def concat_dicts(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Concatenate a list of dictionaries of tensors.
    The tensors must have the same shape in all dictionaries.

    Args:
        dicts (list of dict): List of dictionaries with tensor values.

    Returns:
        dict: Dictionary with concatenated tensors along the first dimension.

    Example:
        >>> dicts = [
        ...     {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])},
        ...     {"a": torch.tensor([5, 6]), "b": torch.tensor([7, 8])},
        ...     {"a": torch.tensor([9, 10]), "b": torch.tensor([11, 12])},
        ... ]
        >>> concat_dicts(dicts)
        {'a': tensor([ 1,  2,  5,  6,  9, 10]),
         'b': tensor([ 3,  4,  7,  8, 11, 12])}
    """

    keys = dicts[0].keys()
    concatenated_dict = {key: torch.cat([d[key] for d in dicts], dim=0) for key in keys}
    return concatenated_dict
